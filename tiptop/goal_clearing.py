"""Add the ``on(blocker, table)`` atoms a goal needs but never says out loud.

An instruction like "pack the coffee pods onto the wooden tray" names only the objects it wants
moved. It does not mention the banana already lying on the tray -- a human reads "onto the tray" as
"onto the *clear* tray" and moves the banana out of the way first. cuTAMP cannot: its BFS
(``cutamp/task_planning/search.py``) is goal-directed and shortest-first, and collisions are
continuous costs rather than operator preconditions, so nothing in the symbolic layer knows the
banana is in the way. The 4-pod goal is satisfied at depth 16, a skeleton that also clears the
banana is depth 20, and ``num_initial_plans`` cuts the generator off long before BFS gets there.

The optimizer does not simply give up, which is what makes this failure mode hard to read. Measured
on ``runs/prpl/tamp/4_pack_coffee_pods_vae_stroke/failure/2026-08-16_22-55-22``, all ten skeletons
came back with the pod-vs-banana collision terms at *zero* and ``wooden_tray_support`` as the
dominant violation -- 0.189 summed over 4 placements, 4.7 cm each, against a banana whose top sits
4.6 cm above the tray. Lifting the pods clear of the banana is cheaper than sliding them, so the
conflict surfaces as a broken placement rather than as the collision it actually is.

**The rule is the same geometric support relation the scene reset is built on**
(:func:`tiptop.scene_reset.supporting_surfaces`), asked a different question: not "what is stacked"
but "what is stacked *on a surface this goal wants to fill*". Anything that is, and that the goal
does not already move, gets an ``on(obj, table)`` atom appended. Nothing is keyed on a label, which
perception is not stable enough to provide (see ``scene_reset``'s module docstring on the same
point).

Two deliberate restrictions, both inherited from :func:`tiptop.scene_reset.build_reset_goal`:

* **Leaves only.** An object that is itself holding something up is never cleared, so the arm cannot
  be asked to carry a loaded container across the table.
* **To the table, never to another object.** The table is the one surface always present in the TAMP
  environment (``tiptop_run.create_tamp_environment`` appends it unconditionally), and naming it in a
  goal atom is also what triggers ``TABLE_PLACEMENT_RAISE`` -- without a goal atom the table's
  collision box stays ``TABLE_BOX_CLEARANCE`` below the true plane and the clearing placement would
  be commanded ~2 cm *into* the tabletop, with no collision to reject it.

Opt-in per task via ``clear_goal_surfaces: true`` in a ``cfg/tamp/*.yml``'s ``tamp_overrides``,
because the rule is about *room* and cannot measure it: a goal that puts a fork on a plate where a
knife already sits does not necessarily need the knife gone. Only a task whose goal surface is
genuinely full should ask for this.

**Appending the atom to ONE goal is not enough, which is why the flag plans two.** cuTAMP is free to
order a 5-object goal however BFS enumerates it, and it does not know the banana has to go FIRST --
so most orderings place pods onto a still-occupied tray and fail exactly as before. Worse, the
enumeration order depends on ``PYTHONHASHSEED``, which is unset: the 5-object goal yields only 5
skeletons (against 12 for the 4-pod one) and the banana's pick position within them swings per
PROCESS. Measured over seeds 0-15: 7/16 put it first, and seed 1 ([4,3,3,3,3]) failed 6/6 across
every ``placement_shrink_dist``. Live confirmation in
``failure/2026-08-16_23-48-17``: the atom was added correctly and all five skeletons still came back
with the banana 4th or 5th.

So :func:`tiptop_run.plan_clear_then_task` plans the clearing as its own goal, reads where that left
the blocker off the plan itself (:func:`placed_poses` -- no re-perception, which mid-episode would
mean parking the arm at the capture pose for seconds), then plans the task against that updated
world and CONCATENATES the two. Both halves run inside one recorded episode. Measured on the two hash
seeds where the single-goal form fails outright, the second phase then solves 3/3 on its first
skeleton.
"""

import logging
from dataclasses import replace

import numpy as np
from scipy.spatial.transform import Rotation

from tiptop.scene_reset import DEFAULT_XY_MARGIN, supporting_surfaces

_log = logging.getLogger(__name__)


def resolve_clear_goal_surfaces(overrides: dict | None) -> bool:
    """Whether a rollout should clear blocked goal surfaces, from cfg/tamp ``tamp_overrides``.

    Off unless a config opts in with ``clear_goal_surfaces: true``, so every existing task plans
    exactly the goal Gemini grounded. Same shape as :func:`trajectory_blending.resolve_blend_config`
    and the other ``tamp_overrides`` knobs: unknown keys are ignored by ``apply_cost_overrides``, so
    this rides the same JSON the data-collection server already writes.
    """
    return bool((overrides or {}).get("clear_goal_surfaces"))


def _on_args(atom: dict) -> tuple[str, str] | None:
    """``(movable, surface)`` for a well-formed ``on`` atom, else None."""
    if atom.get("predicate") != "on" or len(atom.get("args", [])) != 2:
        return None
    movable, surface = atom["args"]
    return movable, surface


def goal_surfaces(detected_atoms: list[dict] | None, object_meshes: dict) -> set[str]:
    """Perceived objects this goal wants to place something onto.

    Restricted to things in ``object_meshes`` on purpose: the table is a legitimate goal surface but
    cannot be blocked in the sense that matters here -- there is nowhere to clear it *to*, and it is
    big enough that something else resting on it is beside the point rather than in the way.
    """
    surfaces = set()
    for atom in detected_atoms or []:
        args = _on_args(atom)
        if args and args[1] in object_meshes:
            surfaces.add(args[1])
    return surfaces


def goal_movables(detected_atoms: list[dict] | None) -> set[str]:
    """Objects the goal already moves, by any predicate.

    These need no clearing atom even when they are sitting on a goal surface: the plan picks them up
    anyway, and a second atom for the same object would be either redundant (``on(x, tray)`` plus
    ``on(x, table)`` is unsatisfiable) or a contradiction the BFS would grind against forever.
    """
    return {atom["args"][0] for atom in detected_atoms or [] if atom.get("args")}


def blocking_objects(
    object_meshes: dict, detected_atoms: list[dict] | None, xy_margin: float = DEFAULT_XY_MARGIN
) -> dict[str, str]:
    """Map each object in the way to the goal surface it is occupying: ``{blocker: surface}``.

    An object blocks iff it currently rests on a surface the goal wants to fill, the goal does not
    already move it, and it is a leaf of the support tree (nothing is resting on *it*).
    """
    surfaces = goal_surfaces(detected_atoms, object_meshes)
    if not surfaces:
        return {}
    already_moved = goal_movables(detected_atoms)
    supported_by = supporting_surfaces(object_meshes, xy_margin)
    holding_something_up = set(supported_by.values())
    return {
        label: surface
        for label, surface in supported_by.items()
        if surface in surfaces and label not in already_moved and label not in holding_something_up
    }


def build_clearing_goal(
    object_meshes: dict,
    table_cuboid,
    detected_atoms: list[dict] | None,
    xy_margin: float = DEFAULT_XY_MARGIN,
) -> tuple[list[dict], set[str]]:
    """``(atoms, extra_surface_labels)`` for phase one: get the blockers off, and nothing else.

    The instruction's own atoms are deliberately NOT included. They are what phase two plans, once
    the blockers have moved -- putting both in one goal is the form that loses to BFS's enumeration
    order (see the module docstring). Empty atoms means nothing is in the way and the caller should
    plan the task directly, exactly as it does today.

    ``extra_surface_labels`` is load-bearing here in a way it is not for a task goal. A clearing goal
    names nothing but the table, and ``create_tamp_environment`` infers surfaces from the second
    argument of its ``on`` atoms -- so without this the tray the blocker is coming off would classify
    as MOVABLE and the planner would be free to pick the tray up instead. Same reasoning, and the
    same fix, as ``scene_reset.build_reset_goal``.
    """
    blockers = blocking_objects(object_meshes, detected_atoms, xy_margin)
    atoms = [{"predicate": "on", "args": [label, table_cuboid.name]} for label in sorted(blockers)]
    return atoms, goal_surfaces(detected_atoms, object_meshes)


def drop_return_to_initial(plan: list[dict]) -> list[dict]:
    """``plan`` without the trailing ``GoToInitial`` cuTAMP appends to every plan.

    Only meaningful for a plan something else continues from. cuTAMP ends each plan by driving back
    to the configuration it started at, which between two concatenated plans is a round trip to the
    home pose that nothing asked for: the arm finishes the clearing, returns home, then leaves again
    for the first pick. Dropping it lets the next plan's opening ``MoveFree`` go straight from where
    the clearing left the arm to where the task needs it -- one motion instead of two.

    Trailing only, so a ``GoToInitial`` anywhere else (there is none today) would be left alone.
    """
    trimmed = list(plan)
    while trimmed and str(trimmed[-1].get("label", "")).startswith("GoToInitial"):
        trimmed.pop()
    return trimmed


def final_configuration(plan: list[dict]) -> np.ndarray | None:
    """Where the arm ends up after ``plan`` -- the last waypoint of its last trajectory step.

    This is what the next plan has to start from. cuTAMP normally ends a plan with ``GoToInitial``,
    which brings it back to the configuration it started at, but that is not relied on here.
    """
    for step in reversed(plan):
        if step.get("type") == "trajectory":
            positions = step["plan"].position
            return positions[-1].cpu().numpy() if hasattr(positions, "cpu") else np.asarray(positions[-1])
    return None


def placed_poses(plan: list[dict]) -> dict[str, np.ndarray]:
    """Where ``plan`` leaves each object it placed: ``{label: 4x4 world_from_obj}``.

    Read straight off the Place gripper steps, which cuTAMP stamps with the pose it updated the
    collision world to (``motion_solver``). That is the only point the answer is exact.
    Reconstructing it from the trajectory instead looks easy and is wrong: cuTAMP forms the
    object-to-hand attachment at the START of the Place operator rather than at the grasp, and a
    fixed 5 cm tool-axis approach offset is in play. Measured against a real clearing plan, a
    grasp-config reconstruction put the banana 4.5 cm ABOVE the table it had just been set down on --
    identically with blending on and off, which is what ruled blending out as the cause.

    Later placements win, so an object moved twice reports where it finally came to rest.
    """
    poses = {}
    for step in plan:
        if step.get("type") == "gripper" and step.get("placed_object") is not None:
            poses[step["placed_object"]] = np.asarray(step["world_from_obj"], dtype=float)
    return poses


def pose_to_matrix(pose) -> np.ndarray:
    """cuRobo's ``[x, y, z, qw, qx, qy, qz]`` as a 4x4. Note wxyz, where scipy wants xyzw."""
    p = np.asarray(pose, dtype=float)
    mat = np.eye(4)
    mat[:3, 3] = p[:3]
    mat[:3, :3] = Rotation.from_quat([p[4], p[5], p[6], p[3]]).as_matrix()
    return mat


def matrix_to_pose(mat: np.ndarray) -> list[float]:
    """Inverse of :func:`pose_to_matrix`."""
    qx, qy, qz, qw = Rotation.from_matrix(np.asarray(mat)[:3, :3]).as_quat()
    return [float(mat[0][3]), float(mat[1][3]), float(mat[2][3]), float(qw), float(qx), float(qy), float(qz)]


def move_meshes(object_meshes: dict, poses: dict[str, np.ndarray]) -> dict:
    """``object_meshes`` with each placed object at the world pose the plan leaves it in.

    Perception hands over geometry with the centre in ``pose[:3]`` and the vertices re-centred on it
    (``scene_reset.world_aabb``), so writing the new world pose in moves and reorients the object as
    the rigid body it is. Objects with no entry in ``poses`` are passed through untouched, which is
    right -- the plan did not move them.
    """
    moved = dict(object_meshes)
    for label, world_from_obj in poses.items():
        obj = object_meshes.get(label)
        if obj is None:
            continue
        moved[label] = replace(obj, pose=matrix_to_pose(np.asarray(world_from_obj, dtype=float)))
    return moved

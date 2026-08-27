"""Divide one instruction's goal between the two YAM arms.

This is a HARNESS-level split, not part of TAMP. cuTAMP plans one kinematic chain and has no
operator for choosing an arm, so something outside it has to decide who picks what — the same
position taken by ``droid-sim-evals/eval/yam_tiptop_eval.py::_split_by_arm``, which this mirrors on
real hardware.

**The rule is the geometry the reachability sweep measured.** With the base column at
``(0.24, 0, 0.0551)`` the left arm is mounted at ``y = +0.24`` and the right at ``y = -0.24``
(``cutamp/robots/assets/yam_description/bimanual_yam.urdf``); the left reaches ``y >= -0.17`` and the
right ``y <= +0.17``, so splitting at the midline gives each arm objects it can definitely reach and
puts the ambiguous middle ones on whichever side they lean.

**One difference from the sim, and it matters.** The sim harness reads object poses from the
simulator; here they come from perception, and perception runs again before the second arm plans —
by which time the first arm has already put its objects on the target. Those objects can land on the
*other* arm's side of the midline, and a split on position alone would then hand them to the second
arm, which would dutifully pick them off the plate and put them back. :func:`arm_goal_builder`
therefore drops any goal atom that the scene already satisfies, using the same geometric support
relation the scene reset is built on (:func:`tiptop.scene_reset.supporting_surfaces`) rather than
anything keyed on a label, which perception is not stable enough to provide.
"""

import logging

import numpy as np

from tiptop.scene_reset import supporting_surfaces
from tiptop.yam import ARMS

_log = logging.getLogger(__name__)

# How far past the midline each arm can still reach, per the module docstring's reachability sweep
# (left reaches y >= base_y - REACH_MARGIN, right reaches y <= base_y + REACH_MARGIN). This is wider
# than the split boundary itself: a surface near the midline is legitimately reachable by whichever
# arm owns the movable, which is why atom_crosses_midline checks reach, not just object_side.
REACH_MARGIN = 0.17


def object_side(centroid_y: float, base_y: float = 0.0) -> str:
    """Which arm owns an object at this ``y``. Matches ``_split_by_arm`` in the sim harness."""
    return "left" if float(centroid_y) >= base_y else "right"


def split_by_arm(object_meshes: dict, base_y: float = 0.0) -> dict[str, list[str]]:
    """Assign each perceived object to the arm on its side of the robot's midline."""
    assignment: dict[str, list[str]] = {arm: [] for arm in ARMS}
    for label, mesh in object_meshes.items():
        centroid_y = float(np.asarray(mesh.pose, dtype=float)[1])
        assignment[object_side(centroid_y, base_y)].append(label)
    return {arm: sorted(labels) for arm, labels in assignment.items()}


def already_satisfied(atom: dict, supported_by: dict[str, str]) -> bool:
    """True when the scene already places ``atom``'s object on the surface it asks for."""
    if atom.get("predicate") != "on" or len(atom.get("args", [])) != 2:
        return False
    movable, surface = atom["args"]
    return supported_by.get(movable) == surface


def atom_crosses_midline(atom: dict, meshes: dict, base_y: float = 0.0) -> bool:
    """True when the arm that owns ``atom``'s movable cannot reach ``atom``'s surface.

    The split below only inspects the movable's side (see the module docstring) and hands the whole
    atom to whichever arm owns it — which is fine when the surface is near the midline (both arms
    reach it), but silently wrong when the surface sits well past the *other* arm's reach: that arm
    would grind against a target it can never physically satisfy, failing every particle rather than
    raising a clear error. This checks reach (:data:`REACH_MARGIN`), not just which side an object
    leans on, so a shared surface near the midline is correctly NOT flagged. A goal that genuinely
    fails this needs a handover between arms, which the sequential-bimanual harness does not do.
    """
    if atom.get("predicate") != "on" or len(atom.get("args", [])) != 2:
        return False
    movable, surface = atom["args"]
    if movable not in meshes or surface not in meshes:
        return False  # e.g. the table: it isn't one-sided, so it can't be out of reach this way
    movable_y = float(np.asarray(meshes[movable].pose, dtype=float)[1])
    surface_y = float(np.asarray(meshes[surface].pose, dtype=float)[1])
    owner = object_side(movable_y, base_y)
    reachable = surface_y >= base_y - REACH_MARGIN if owner == "left" else surface_y <= base_y + REACH_MARGIN
    return not reachable


def arm_goal_builder(arm: str, base_y: float = 0.0, protect_surfaces: bool = True):
    """A ``goal_builder`` for :func:`tiptop_run.run_perception` restricting the goal to ``arm``.

    Called with ``(processed_scene, detected_atoms)`` and returns ``(atoms, extra_surface_labels)``.
    Gemini's grounding of the FULL instruction is reused as-is — only filtered — so the two arms
    always work toward the same goal and neither needs its own sub-instruction re-grounded (which is
    where the sim harness has to fight the VLM renaming objects between calls).

    ``protect_surfaces`` designates every surface named anywhere in the original goal as a surface,
    including in atoms this arm is not doing. Without it, an arm whose own atoms have all been
    dropped as already-satisfied would leave the target plate classified movable, and the planner
    would be free to pick the plate up.
    """
    if arm not in ARMS:
        raise ValueError(f"arm must be one of {ARMS}, got {arm!r}")

    def build(processed_scene, detected_atoms):
        meshes = processed_scene.object_meshes
        assignment = split_by_arm(meshes, base_y)
        mine = set(assignment[arm])
        supported_by = supporting_surfaces(meshes)
        # create_tamp_environment refuses a goal naming anything it cannot see, and rightly so for a
        # single-arm run. Here the second arm perceives AFTER the first has moved things, so an
        # object can genuinely be missing this time round — occluded, or merged into the pile it was
        # placed on. Dropping that atom with a warning keeps the rest of the episode collectable;
        # failing the whole rollout over it would not.
        known = set(meshes) | {processed_scene.table_cuboid.name}

        kept, satisfied, other, unknown, impossible = [], [], [], [], []
        for atom in detected_atoms or []:
            args = atom.get("args", [])
            missing = [a for a in args if a not in known]
            if missing:
                unknown.append((atom, missing))
                continue
            if atom_crosses_midline(atom, meshes, base_y):
                impossible.append(atom)
                continue
            subject = args[0] if args else None
            if subject is not None and subject not in mine:
                other.append(atom)
                continue
            if already_satisfied(atom, supported_by):
                satisfied.append(atom)
                continue
            kept.append(atom)

        surfaces = None
        if protect_surfaces:
            surfaces = {
                a["args"][1]
                for a in (detected_atoms or [])
                if a.get("predicate") == "on" and len(a.get("args", [])) == 2 and a["args"][1] in meshes
            }

        _log.info(
            f"{arm} arm goal: {[a['args'] for a in kept] or '(nothing to do)'}"
            + (f"; {[a['args'] for a in other]} belong to the other arm" if other else "")
            + (f"; {[a['args'] for a in satisfied]} already satisfied" if satisfied else "")
            + (f"; {[a['args'] for a in impossible]} need a handover" if impossible else "")
            + (f"; protected surfaces {sorted(surfaces)}" if surfaces else "")
        )
        for atom, missing in unknown:
            _log.warning(
                f"{arm} arm: dropping goal {atom.get('predicate')}({', '.join(atom.get('args', []))}) — "
                f"perception did not reconstruct {missing}. Move it by hand if it should have been placed."
            )
        for atom in impossible:
            movable, surface = atom["args"]
            _log.error(
                f"{arm} arm: dropping goal on({movable}, {surface}) — '{movable}' and '{surface}' are "
                f"on opposite sides of the midline (base_y={base_y}), so no single arm can reach both. "
                "This scene needs a handover between arms, which the sequential-bimanual harness "
                "(robot.arms: [left, right]) does not support — use a bimanual_yam_dual handover "
                "config instead, or keep the movable and its target on the same side."
            )
        return kept, surfaces

    return build


def handover_goal_builder(base_y: float = 0.0):
    """A ``goal_builder`` for a SIMULTANEOUS dual-arm handover episode (``bimanual_yam_dual``).

    Unlike :func:`arm_goal_builder`, there is ONE embodiment here, not two arms each owning a side
    of the scene — so this does not split by side at all. What it does instead: cuTAMP's handover
    operator list (``handover_tamp_operators`` = ``[MoveFree, PickGiver, MoveHoldingGiver, Handover,
    MoveHoldingTaker, PlaceTaker]``) has no plain Pick/Place operator in it, so a handover session
    can only ever pursue ONE ``on(movable, surface)`` goal per episode — this is a domain constraint,
    not a simplification to relax later. This builder therefore selects exactly the atom that
    :func:`atom_crosses_midline` flags as actually needing a handover (which arm reaches the
    movable, and which arm reaches its target, are opposite hands) and drops everything else with a
    clear reason, the same "log and drop" philosophy :func:`arm_goal_builder` uses for atoms it
    can't ground or can't reach.
    """

    def build(processed_scene, detected_atoms):
        meshes = processed_scene.object_meshes
        supported_by = supporting_surfaces(meshes)
        known = set(meshes) | {processed_scene.table_cuboid.name}

        handover, other, satisfied, unknown = [], [], [], []
        for atom in detected_atoms or []:
            args = atom.get("args", [])
            missing = [a for a in args if a not in known]
            if missing:
                unknown.append((atom, missing))
                continue
            if already_satisfied(atom, supported_by):
                satisfied.append(atom)
                continue
            if atom_crosses_midline(atom, meshes, base_y):
                handover.append(atom)
            else:
                other.append(atom)

        if len(handover) > 1:
            _log.warning(
                f"handover: {len(handover)} atoms need a handover "
                f"({[a['args'] for a in handover]}), but one bimanual_yam_dual episode can only plan "
                f"one (cuTAMP's handover_tamp_operators has no second Pick/Place) — keeping "
                f"{handover[0]['args']} and dropping the rest for this episode."
            )
            handover = handover[:1]

        surfaces = {
            a["args"][1]
            for a in (detected_atoms or [])
            if a.get("predicate") == "on" and len(a.get("args", [])) == 2 and a["args"][1] in meshes
        }

        _log.info(
            f"handover goal: {[a['args'] for a in handover] or '(nothing needs a handover)'}"
            + (
                f"; {[a['args'] for a in other]} don't need a handover (no operator to plan them "
                "alongside one, dropped this episode)"
                if other
                else ""
            )
            + (f"; {[a['args'] for a in satisfied]} already satisfied" if satisfied else "")
            + (f"; protected surfaces {sorted(surfaces)}" if surfaces else "")
        )
        for atom, missing in unknown:
            _log.warning(
                f"handover: dropping goal {atom.get('predicate')}({', '.join(atom.get('args', []))}) — "
                f"perception did not reconstruct {missing}. Move it by hand if it should have been placed."
            )
        return handover, surfaces

    return build

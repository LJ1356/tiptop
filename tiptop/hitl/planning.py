"""Turning one phase into something TiPToP can plan, and checking it could ever be planned.

There is no outer search any more. The proposer supplies the order, and each robot phase is handed to
cuTAMP as an ordinary goal -- which is all the old breadth-first search ever contributed anyway, since
cuTAMP replanned every segment from its sub-goal regardless. What remains here is the cheap, sound
check that a phase is achievable at all, and the rendering that lets ``create_tamp_environment``
consume a phase unchanged.
"""

import logging
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from cutamp.tamp_domain import Holding, On, all_tamp_operators, get_initial_state
from cutamp.task_planning import Atom, State

from tiptop.hitl.structs import DeferredObject, Phase, SceneTypes, TaskSpecification, display_atom

_log = logging.getLogger(__name__)

# How far outside the anchor's footprint a candidate's centroid may sit and still count as resting on
# it. Absorbs depth noise and the centroid of something overhanging an edge, without being anywhere
# near wide enough to let a neighbouring object in: in the run this was written for, the distractor
# missed by 7cm.
_FOOTPRINT_MARGIN = 0.02

# The fluents a cuTAMP goal can be stated over. create_tamp_environment reads exactly on(...) and
# holding(...), and supplies HandEmpty itself.
_GOAL_EXPRESSIBLE = frozenset({On.name, Holding.name})


def initial_state_for(scene_types: SceneTypes, known_true: Sequence[Atom] = ()) -> State:
    """cuTAMP's symbolic initial state for this scene, plus whatever the VLM says already holds.

    Note what cuTAMP's initial state does NOT contain: any ``On`` atom. It is a pure function of the
    object names -- every movable un-picked, the hand empty, nothing anywhere. That is also why an
    object may be picked up in more than one phase: each phase is planned from this same clean state.
    """
    base = get_initial_state(movables=sorted(scene_types.movables), surfaces=sorted(scene_types.surfaces))
    return frozenset(set(base) | set(known_true))


def unachievable_atoms(atoms: frozenset[Atom], initial_state: State) -> list[Atom]:
    """Atoms in a robot phase that no cuTAMP operator can ever make true.

    Sound but not complete, and instant. Its job is to reject a phase before perception is paid for,
    rather than let cuTAMP's own search discover it -- that search has no bound of any kind and, given
    a goal it cannot reach, mints fresh conf/traj symbols forever without ever yielding.
    """
    achievable = {a.name for a in initial_state}
    achievable |= {f.name for op in all_tamp_operators for f in op.add_effects}
    return sorted((a for a in atoms if a not in initial_state and a.name not in achievable), key=str)


def check_robot_phases(spec: TaskSpecification, initial_state: State) -> str | None:
    """Why the plan cannot be carried out, or None if every robot phase is achievable."""
    for i, phase in enumerate(spec.phases):
        if phase.is_human:
            continue
        unachievable = unachievable_atoms(phase.atoms, initial_state)
        if unachievable:
            return (
                f"phase {i} ({phase.description!r}) asks the robot for "
                f"{', '.join(display_atom(a) for a in unachievable)}, which no robot operator can achieve"
            )
    return None


def goal_atoms_to_dicts(atoms: frozenset[Atom]) -> list[dict]:
    """Render a phase's atoms back into the ``{"predicate", "args"}`` form perception emits.

    That is what create_tamp_environment consumes, so a HITL phase goes through exactly the same
    table-alias resolution, unknown-object rejection and environment construction as an ordinary
    Gemini-translated goal -- no second code path to keep in step.
    """
    dicts = []
    for atom in sorted(atoms, key=str):
        if atom.name == On.name:
            dicts.append({"predicate": "on", "args": [atom.values[0], atom.values[1]]})
        elif atom.name == Holding.name:
            dicts.append({"predicate": "holding", "args": [atom.values[0]]})
    return dicts


def phase_objects(phase: Phase) -> set[str]:
    """Every object a phase names, for the label-drift check."""
    return {value for atom in phase.atoms for value in atom.values}


@dataclass(frozen=True)
class ObjectGeometry:
    """Where one detected object is and how big it is, in world coordinates."""

    centroid: tuple[float, float, float]
    extents: tuple[float, float, float]

    @property
    def top_z(self) -> float:
        return self.centroid[2] + self.extents[2] / 2

    def footprint_contains(self, point: Sequence[float], margin: float) -> bool:
        """Whether ``point`` lies over this object's axis-aligned XY footprint."""
        return all(abs(point[axis] - self.centroid[axis]) <= self.extents[axis] / 2 + margin for axis in (0, 1))


def scene_geometry(meshes: Mapping[str, object]) -> dict[str, ObjectGeometry]:
    """Read centroids and extents off a ProcessedScene's object meshes.

    Anything without a usable pose or vertex set is left out rather than defaulted: a wrong box here
    would be used to decide which object the plan is talking about.
    """
    geometry: dict[str, ObjectGeometry] = {}
    for name, mesh in meshes.items():
        pose = getattr(mesh, "pose", None)
        if pose is None or len(pose) < 3:
            continue
        try:
            vertices = np.asarray(mesh.vertices)
        except Exception:
            continue
        if vertices.size == 0:
            continue
        extents = vertices.max(axis=0)[:3] - vertices.min(axis=0)[:3]
        geometry[name] = ObjectGeometry(
            centroid=tuple(float(v) for v in pose[:3]),
            extents=tuple(float(v) for v in extents),
        )
    return geometry


def objects_resting_on(
    anchor: str,
    geometry: Mapping[str, ObjectGeometry],
    pool: Sequence[str],
    margin: float = _FOOTPRINT_MARGIN,
) -> list[str]:
    """Names in ``pool`` whose centroid sits over ``anchor``'s footprint, at or above its top face."""
    anchor_geometry = geometry.get(anchor)
    if anchor_geometry is None:
        return []
    return sorted(
        name
        for name in pool
        if (candidate := geometry.get(name)) is not None
        and anchor_geometry.footprint_contains(candidate.centroid, margin)
        # Both faces of "on": the candidate's middle is not below the anchor's, and it stands proud of
        # the anchor's top. Either alone admits something the anchor is resting on instead.
        and candidate.centroid[2] >= anchor_geometry.centroid[2]
        and candidate.top_z >= anchor_geometry.top_z
    )


def bind_deferred_object(
    deferred: DeferredObject,
    geometry: Mapping[str, ObjectGeometry],
    pool: Sequence[str],
) -> str | None:
    """Which of this pass's new labels is the object a human phase was supposed to create.

    Binding is geometric, and deliberately NOT by name. Perception names objects from the task
    instruction, so on the very task that needs this it emits block-shaped labels for the wrong
    block: the run this was written for produced ``top_jenga_block`` for the brick still on TOP OF THE
    TOWER while the loose one sat undetected on the paper 17cm away. Matching on the name would have
    bound to it, and the robot would have lifted the tower's top block and put it back on the tower --
    a no-op the phase then reports as success. That silent-wrong-success is worse than not binding.

    So the test is the one thing the plan actually asserted about the object: the creating phase said
    it ends up ``On(anchor)``, so the candidate has to be over the anchor. Ambiguity refuses, exactly
    as match_drifted_names does -- there is no reading of two candidates that is safe to guess at.
    """
    candidates = objects_resting_on(deferred.anchor, geometry, pool)
    if len(candidates) == 1:
        return candidates[0]
    if deferred.anchor not in geometry:
        _log.info(f"HITL: cannot bind '{deferred.name}' -- its anchor '{deferred.anchor}' has no geometry this pass")
    else:
        _log.info(
            f"HITL: cannot bind '{deferred.name}' -- {len(candidates)} of this pass's new labels rest "
            f"on '{deferred.anchor}' ({', '.join(candidates) or 'none'}); new labels were "
            f"{', '.join(sorted(pool)) or 'none'}"
        )
    return None


def bind_deferred_objects(
    spec: TaskSpecification,
    geometry: Mapping[str, ObjectGeometry],
    detected: Sequence[str],
    phase_index: int,
) -> dict[str, str]:
    """Map every deferred object that now exists onto the label this pass gave it.

    Only objects whose creating phase has already been carried out are eligible. Before then the
    object is not supposed to exist, so anything sitting on the anchor is something else -- a sheet of
    paper with a drawing on it is enough -- and binding to it would aim the robot leg at the wrong
    thing while the human's work still lay ahead.

    Candidates are drawn from the labels the plan does not ALREADY own, for the same reason the
    label-drift path draws from that pool: binding to a name some other phase is using would fold two
    plan objects into one inside SceneTypes' frozensets.

    A deferred object's OWN name is a candidate, though -- the detector is told to look for it by name
    (see tiptop_run._hitl_detect_hint), so the label it comes back with is often exactly the one the
    plan uses. Excluding it would leave the object unbindable in the very case the hint worked. It
    earns nothing by matching: the geometric test decides, and a wrong object wearing the right name
    is rejected on the same footing as any other.
    """
    owned = set(spec.scene_types.all_names) - set(spec.scene_types.deferred)
    mapping: dict[str, str] = {}
    for deferred in spec.unbound_deferred:
        if phase_index <= deferred.created_by_phase:
            continue
        # Another deferred object's name is not available: two of them must not collapse into one.
        others = set(spec.scene_types.deferred) - {deferred.name}
        pool = sorted(set(detected) - owned - others - set(mapping.values()))
        label = bind_deferred_object(deferred, geometry, pool)
        if label is not None:
            _log.info(f"HITL: '{deferred.name}' now exists -- bound to this pass's '{label}'")
            mapping[deferred.name] = label
    return mapping


def match_drifted_names(missing: Sequence[str], detected: Sequence[str]) -> dict[str, str] | None:
    """Map names a plan uses onto this pass's labels, or None if it cannot be done unambiguously.

    Gemini names objects afresh on every perception pass and the names drift: one pass calls them
    ``toy`` and ``box``, the next ``blue_toy`` and ``cardboard_box``. Mid-task that is fatal -- the
    plan refers to objects this pass did not produce, so it would be thrown away and the whole task
    re-planned from a scene that has already been half-rearranged, asking the human to redo their
    part. Observed doing exactly that.

    The rule is deliberately conservative and needs no extra model call: a name matches when it is a
    whole-word subset of exactly one detected label (or the other way round). Anything ambiguous
    returns None, and the caller re-plans as before rather than guessing which object was meant.
    """
    available = [d for d in detected]
    mapping: dict[str, str] = {}
    for name in missing:
        wanted = set(name.split("_"))
        candidates = [
            d for d in available if wanted <= set(d.split("_")) or set(d.split("_")) <= wanted
        ]
        if len(candidates) != 1:
            _log.info(
                f"HITL: cannot re-bind '{name}' to this pass's labels "
                f"({', '.join(sorted(detected))}): {len(candidates)} candidate(s)"
            )
            return None
        mapping[name] = candidates[0]
        available.remove(candidates[0])
    return mapping

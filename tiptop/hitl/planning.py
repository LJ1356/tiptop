"""Turning one phase into something TiPToP can plan, and checking it could ever be planned.

There is no outer search any more. The proposer supplies the order, and each robot phase is handed to
cuTAMP as an ordinary goal -- which is all the old breadth-first search ever contributed anyway, since
cuTAMP replanned every segment from its sub-goal regardless. What remains here is the cheap, sound
check that a phase is achievable at all, and the rendering that lets ``create_tamp_environment``
consume a phase unchanged.
"""

import logging
from typing import Sequence

from cutamp.tamp_domain import Holding, On, all_tamp_operators, get_initial_state
from cutamp.task_planning import Atom, State

from tiptop.hitl.structs import Phase, SceneTypes, TaskSpecification, display_atom

_log = logging.getLogger(__name__)

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

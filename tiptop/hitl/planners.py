"""Who carries out a phase, and how far one leg of the plan runs.

The proposal stage decides WHICH parts of a task are the robot's and which are the human's, and
nothing else. What each of those two answers then MEANS -- a goal solved by task and motion planning,
or a teleop hand-off -- lives here, one implementation per executor behind one interface.

That split is the point. "The robot's phases are planned by cuTAMP" is an implementation of
:class:`PhasePlanner`, not a fact of the pipeline: a phase is handed to whichever planner claims it,
and the planner decides how many consecutive phases it takes in one leg and what, if anything, it
needs solved for them. Swapping cuTAMP for a different robot planner is registering another one and
naming it in the config (``hitl.robot_planner``); nothing above this module mentions cuTAMP by name.

What a planner does NOT decide is the plan: the phase list, its order and each phase's goal atoms are
fixed by the proposal stage before the arm moves. A planner is asked only how to carry out phases it
has been given.
"""

from dataclasses import dataclass
from typing import Protocol, Sequence

from tiptop.hitl.planning import goal_atoms_to_dicts, phase_objects
from tiptop.hitl.structs import Phase, SceneTypes


@dataclass(frozen=True)
class Leg:
    """One contiguous run of phases, and who is about to carry it out.

    A leg is the unit the rollout loop works in: one leg is one pass through perception, one attempt
    at whatever the planner needs, and one advance of the session. ``phases`` is always non-empty and
    always starts at the session's current phase.

    ``goal`` is the leg's sub-goal in the ``{"predicate", "args"}`` form ``create_tamp_environment``
    consumes -- empty for a planner that does not solve for a symbolic goal at all, which is what a
    teleop leg is.
    """

    planner: str
    executor: str  # "robot" | "human"
    phases: tuple[Phase, ...]
    goal: tuple[dict, ...] = ()

    def __post_init__(self) -> None:
        assert self.phases, "a leg covers at least one phase"
        assert self.executor in ("robot", "human"), self.executor


class PhasePlanner(Protocol):
    """How one kind of phase gets carried out.

    Three questions, and no more: does this planner claim the phase, how far does one leg of it run,
    and who should the audit trail say wrote what ran.
    """

    name: str
    executor: str
    # One sentence for hitl.json's `provenance`, saying what this planner contributes.
    provenance: str
    # One phrase for a phase record's `planned_by`, saying who decided what about that phase.
    authorship: str

    def handles(self, phase: Phase) -> bool:
        """Whether this planner is the one that carries out ``phase``."""
        ...

    def leg(self, phases: Sequence[Phase], scene_types: SceneTypes) -> Leg:
        """The leg starting at ``phases[0]``, which this planner has already claimed.

        ``phases`` is every phase from the current one to the end of the plan, so a planner that can
        carry out several at once decides for itself how many it takes.
        """
        ...


class TeleopPhasePlanner:
    """A human phase: hand the arm over, then check from a photo that it happened.

    One phase per leg, always. Two consecutive human steps are two hand-offs because each one is
    verified on its own -- merging them would make one failed check fail work that was done -- and
    there is no goal for anything to solve: ``tiptop_run`` skips planning entirely for this leg and
    goes straight to the hand-off (see ``_hitl_human_phase``).
    """

    name = "teleop"
    executor = "human"
    provenance = "the teleoperator, following the phase's instructions"
    authorship = "vlm"

    def handles(self, phase: Phase) -> bool:
        return phase.is_human

    def leg(self, phases: Sequence[Phase], scene_types: SceneTypes) -> Leg:
        return Leg(planner=self.name, executor=self.executor, phases=(phases[0],))


class CuTAMPPhasePlanner:
    """A robot phase: its atoms become a cuTAMP goal, and cuTAMP searches for the plan that reaches it.

    The phase says only what must be TRUE at its end. The sequence of picks and places that gets
    there -- the task plan -- is cuTAMP's own breadth-first search over its operators, run per leg
    against a fresh perception pass, and the grasps, placements and trajectories under it are cuTAMP's
    and cuRobo's. Nothing outside this leg is in that problem: a robot leg is planned from scratch
    from the phases it covers, so the search never sees the whole task.

    **Consecutive robot phases are merged into one leg.** A phase only ever says what must be true at
    its end, and cuTAMP's initial state carries no ``On`` atom at all (see planning.initial_state_for)
    -- every robot phase is planned from the same clean state, so nothing symbolic ever enforced the
    proposer's ordering BETWEEN two robot phases. Conjoining them is the same problem stated once, and
    it is what the non-HITL path already does with a two-clause instruction: one plan, one continuous
    motion, and no re-perception in the middle for the object labels to drift across.

    The merge STOPS at a phase that moves an object an earlier phase in the run already moved.
    cuTAMP's ``Pick`` requires and deletes ``HasNotPickedUp(obj)`` (tamp_domain.py), so a single plan
    picks each object at most once: asking for ``On(toy, table)`` and ``On(toy, shelf)`` at once is
    unsatisfiable rather than merely slow. Phases like that are genuinely sequential -- "take the toy
    off the box ... put the toy back in" -- and stay separate legs, which is what the robot->robot
    continuation in tiptop_run's rollout loop carries.
    """

    name = "cutamp"
    executor = "robot"
    provenance = (
        "cuTAMP -- each robot leg is handed to task and motion planning as the goal its phases' "
        "atoms state, and cuTAMP searches for the sequence of picks and places that reaches it "
        "against a fresh perception pass (so cutamp_skeleton is what ran), then solves the grasps "
        "and placements for it; cuRobo makes the trajectories"
    )
    authorship = "vlm (order and sub-goal); cuTAMP (how)"

    def handles(self, phase: Phase) -> bool:
        return not phase.is_human

    def leg(self, phases: Sequence[Phase], scene_types: SceneTypes) -> Leg:
        run: list[Phase] = []
        claimed: set[str] = set()
        for phase in phases:
            if phase.is_human:
                break
            moved = phase_objects(phase) & scene_types.movables
            if run and moved & claimed:
                break
            run.append(phase)
            claimed |= moved
        assert run, "CuTAMPPhasePlanner.leg was given a phase it does not handle"
        atoms = frozenset().union(*(p.atoms for p in run))
        return Leg(
            planner=self.name,
            executor=self.executor,
            phases=tuple(run),
            goal=tuple(goal_atoms_to_dicts(atoms)),
        )


# Robot planners by name, so `hitl.robot_planner` can select one. Human phases are always the teleop
# planner's: a hand-off is what makes a phase the human's in the first place, so there is nothing to
# choose between.
_ROBOT_PLANNERS: dict[str, PhasePlanner] = {}
_TELEOP_PLANNER = TeleopPhasePlanner()


def register_robot_planner(planner: PhasePlanner) -> None:
    """Make a planner selectable as ``hitl.robot_planner``.

    Registration rather than a hard-coded branch is the extension point: a planner that is not cuTAMP
    -- a learned skill sequencer, a scripted routine, another TAMP system -- is this call plus a
    config key, and nothing in session.py or tiptop_run.py changes.
    """
    assert planner.executor == "robot", planner.executor
    _ROBOT_PLANNERS[planner.name] = planner


register_robot_planner(CuTAMPPhasePlanner())


def teleop_planner() -> PhasePlanner:
    """The planner every human phase goes to."""
    return _TELEOP_PLANNER


def robot_planner(name: str) -> PhasePlanner:
    """The registered robot planner called ``name``.

    Raises rather than falling back to cuTAMP: a config that names a planner this build does not have
    is asking for something it will not get, and silently planning the task with a different one is
    the wrong answer to that in a run whose output is a dataset.
    """
    try:
        return _ROBOT_PLANNERS[name]
    except KeyError:
        raise ValueError(
            f"unknown hitl robot_planner {name!r}. Registered: {', '.join(sorted(_ROBOT_PLANNERS)) or '(none)'}"
        ) from None


def planner_for(phase: Phase, robot_planner_name: str) -> PhasePlanner:
    """The planner that carries out ``phase``."""
    planner = robot_planner(robot_planner_name)
    if planner.handles(phase):
        return planner
    assert _TELEOP_PLANNER.handles(phase), f"no planner claims {phase.executor} phases"
    return _TELEOP_PLANNER

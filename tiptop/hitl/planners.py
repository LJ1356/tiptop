"""Who carries out a phase, and how far one leg of the plan runs.

The proposal stage decides WHICH parts of a task are the robot's and which are the human's, and
nothing else. What each of those two answers then MEANS -- a goal solved by task and motion planning,
or a teleop hand-off -- lives here, one implementation per executor behind one interface.

That split is the point. "The robot's phases are planned by cuTAMP" is an implementation of
:class:`PhasePlanner`, not a fact of the pipeline: a phase is handed to whichever planner claims it,
and the planner decides how many consecutive phases it takes in one leg and what, if anything, it
needs solved for them. Swapping cuTAMP for a different robot planner is registering another one and
naming it in the config (``hitl.robot_planner``); nothing above this module mentions cuTAMP by name.

The HUMAN half is selectable the same way (``hitl.policy_type``), and for a reason worth stating: a
phase is the human's because cuTAMP cannot express what it asks for, not because a person is the only
thing that could do it. A policy trained on the teleop legs of earlier runs of this very task can be
handed the same phase -- so a task that once read tamp -> teleop -> tamp runs as tamp -> policy ->
tamp, with the same sub-goals, the same verification afterwards and the same merged episode. What
changes is who drives the arm for one leg; the plan does not know the difference.

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
    and who should the audit trail say wrote what ran. "Planner" is the interface's name rather than
    a claim about the work: a hand-off planner solves for nothing at all, and answers the three
    questions anyway.
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


class HandoffPhasePlanner:
    """Base for the human side: one phase per leg, and nothing to solve.

    One phase per leg, always. Two consecutive human steps are two hand-offs because each one is
    verified on its own -- merging them would make one failed check fail work that was done -- and
    there is no goal for anything to solve: ``tiptop_run`` skips planning entirely for this leg and
    goes straight to the hand-off (see ``_hitl_human_phase``).

    ``executor`` stays ``"human"`` for every subclass, including the ones where no human touches the
    arm. It is the PHASE's executor -- the proposal stage's judgement that this step is not something
    cuTAMP can be given -- and that judgement is what makes the leg a hand-off at all. Who actually
    drives it is ``name``, and ``provenance`` is where the record says so.
    """

    executor = "human"

    def handles(self, phase: Phase) -> bool:
        return phase.is_human

    def leg(self, phases: Sequence[Phase], scene_types: SceneTypes) -> Leg:
        return Leg(planner=self.name, executor=self.executor, phases=(phases[0],))


class TeleopPhasePlanner(HandoffPhasePlanner):
    """A human phase: hand the arm over to a teleoperator, then check from a photo that it happened."""

    name = "human"
    provenance = "the teleoperator, following the phase's instructions"
    authorship = "vlm"


class PolicyPhasePlanner(HandoffPhasePlanner):
    """Base for a human phase driven by a trained policy instead of a person.

    The policy is behaviour-cloned on the TELEOP LEGS of earlier HITL runs of this same task (one
    hitl-baseline project per policy type), so what it imitates is precisely the phase it is being
    handed. ``tiptop_run._run_policy_phase`` releases the arm and the cameras exactly as a teleop
    hand-off does and runs ``droid/scripts/policy_capture.py`` in their place, which serves the
    checkpoint with ``serve_module`` under ``project``'s venv; the phase is then verified by the same
    VLM check, against the same atoms, as if a person had done it.

    It is imitation, not achievement: nothing in a BC policy knows what the phase's atoms say, so the
    leg ends on ``hitl.policy_max_steps`` rather than on success, and the verification that follows is
    the only thing that decides whether it worked.
    """

    # The hitl-baseline/<project> that trained the checkpoint; its .venv is where the server runs.
    project: str
    # The module policy_capture.py runs (`python -m <serve_module>`) to load the checkpoint.
    serve_module: str
    # Whether the checkpoint has a reverse-diffusion step count to set (hitl.policy_num_inference_steps).
    has_inference_steps: bool = False


class DiffusionPolicyPhasePlanner(PolicyPhasePlanner):
    """A human phase driven by a LeRobot ``DiffusionPolicy`` (hitl-baseline/diffusion_policy)."""

    name = "diffusion"
    project = "diffusion_policy"
    serve_module = "hitl_dp.serve"
    has_inference_steps = True
    provenance = (
        "a LeRobot diffusion policy trained on the teleop legs of earlier runs of this task "
        "(hitl-baseline/diffusion_policy), run closed-loop in place of the teleoperator: it drives "
        "the arm for a fixed number of control steps and the phase verification decides whether what "
        "it did counts"
    )
    authorship = "vlm (what the step must achieve); a diffusion policy (the motion)"


class ACTPolicyPhasePlanner(PolicyPhasePlanner):
    """A human phase driven by a LeRobot ``ACTPolicy`` (hitl-baseline/action_chunk_transformer).

    Trained on exactly the diffusion baseline's corpus, so the two differ by the policy and nothing
    else -- which is what makes swapping one for the other a fair comparison of the same task.
    """

    name = "act"
    project = "action_chunk_transformer"
    serve_module = "hitl_act.serve"
    provenance = (
        "a LeRobot ACT (action chunking transformer) policy trained on the teleop legs of earlier runs "
        "of this task (hitl-baseline/action_chunk_transformer), run closed-loop in place of the "
        "teleoperator: it drives the arm for a fixed number of control steps and the phase "
        "verification decides whether what it did counts"
    )
    authorship = "vlm (what the step must achieve); an ACT policy (the motion)"


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


# Planners by name, one registry per side of the plan, so `hitl.robot_planner` and `hitl.policy_type`
# each select from the half they are about. Two registries rather than one because the two halves are
# not interchangeable: a robot planner is asked for a symbolic goal cuTAMP can solve, and a human
# planner is asked for none at all, so a name from the wrong half would be a config that type-checks
# and then cannot run.
_ROBOT_PLANNERS: dict[str, PhasePlanner] = {}
_HUMAN_PLANNERS: dict[str, PhasePlanner] = {}


def register_robot_planner(planner: PhasePlanner) -> None:
    """Make a planner selectable as ``hitl.robot_planner``.

    Registration rather than a hard-coded branch is the extension point: a planner that is not cuTAMP
    -- a learned skill sequencer, a scripted routine, another TAMP system -- is this call plus a
    config key, and nothing in session.py or tiptop_run.py changes.
    """
    assert planner.executor == "robot", planner.executor
    _ROBOT_PLANNERS[planner.name] = planner


def register_human_planner(planner: PhasePlanner) -> None:
    """Make a planner selectable as ``hitl.policy_type``.

    The same extension point on the other side. A new policy is a :class:`PolicyPhasePlanner` here
    naming its project and server module (a server speaking ``hitl_dp.wire``, which is all
    ``tiptop_run._run_policy_phase`` needs); the plan, the hand-off and the verification are already
    written and do not know which one they got.
    """
    assert planner.executor == "human", planner.executor
    _HUMAN_PLANNERS[planner.name] = planner


register_robot_planner(CuTAMPPhasePlanner())
register_human_planner(TeleopPhasePlanner())
register_human_planner(DiffusionPolicyPhasePlanner())
register_human_planner(ACTPolicyPhasePlanner())


def _lookup(registry: dict[str, PhasePlanner], name: str, key: str) -> PhasePlanner:
    """The registered planner called ``name``, or a ValueError naming what there is.

    Raises rather than falling back to the default: a config that names a planner this build does not
    have is asking for something it will not get, and silently carrying the task out with a different
    one is the wrong answer to that in a run whose output is a dataset.
    """
    try:
        return registry[name]
    except KeyError:
        raise ValueError(
            f"unknown hitl {key} {name!r}. Registered: {', '.join(sorted(registry)) or '(none)'}"
        ) from None


def robot_planner(name: str) -> PhasePlanner:
    """The registered robot planner called ``name`` (``hitl.robot_planner``)."""
    return _lookup(_ROBOT_PLANNERS, name, "robot_planner")


def human_planner(name: str) -> PhasePlanner:
    """The registered human-phase planner called ``name`` (``hitl.policy_type``)."""
    return _lookup(_HUMAN_PLANNERS, name, "policy_type")


def teleop_planner() -> PhasePlanner:
    """The one human-phase planner that needs a PERSON, so callers can test for it by identity.

    ``tiptop_run._hitl_human_phase`` has to know whether to block on a prompt or to run a driver, and
    that is a fact about this planner rather than about its config spelling: comparing against this
    singleton keeps the branch correct if ``policy_type: human`` is ever spelled differently.
    """
    return _HUMAN_PLANNERS["human"]


def check_planners(cfg) -> None:
    """Resolve both of a config's planner names, so a typo fails at startup rather than mid-task.

    A bad ``robot_planner`` used to surface on the first leg and a bad ``policy_type`` would surface
    at the first HUMAN phase -- minutes of warm-up and a rollout's worth of arm motion after the run
    began, and with a half-collected trajectory already on disk.
    """
    robot_planner(cfg.robot_planner)
    human_planner(cfg.policy_type)


def planner_for(phase: Phase, robot_planner_name: str, human_planner_name: str) -> PhasePlanner:
    """The planner that carries out ``phase``."""
    planner = robot_planner(robot_planner_name)
    if planner.handles(phase):
        return planner
    human = human_planner(human_planner_name)
    assert human.handles(phase), f"no planner claims {phase.executor} phases"
    return human

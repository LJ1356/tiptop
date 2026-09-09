"""The HITL session: one task's phases, walked one rollout at a time.

A HITL task is not one rollout. Each robot phase is an ordinary TiPToP rollout aimed at that phase's
sub-goal, and each human phase is a teleop leg -- so the plan has to survive the hand-off between
them. tiptop_run is a single long-lived process, so this state lives here and is keyed on the
trajectory it belongs to; anything that ends the trajectory ends the session with it.
"""

import logging
from dataclasses import dataclass, field
from typing import Mapping, Sequence

from cutamp.task_planning import Atom, State
from PIL import Image

from tiptop.hitl.config import HITLConfig
from tiptop.hitl.grounding import (
    DEFAULT_DESCRIPTIONS,
    Verdict,
    classify_initial_state,
    describe_expectations,
    descriptions_for,
)
from tiptop.hitl.planning import (
    ObjectGeometry,
    bind_deferred_objects,
    check_robot_phases,
    goal_atoms_to_dicts,
    initial_state_for,
    phase_objects,
)
from tiptop.hitl.planners import Leg, planner_for, robot_planner, teleop_planner
from tiptop.hitl.proposal import propose_plan
from tiptop.hitl.structs import Phase, TaskSpecification, describe_atom

_log = logging.getLogger(__name__)

# Bumped per session so invented fluents from two tasks in one process cannot collide in cuTAMP's
# process-global atom cache. See structs.session_fluent_name.
_session_counter = 0


def _next_session_tag() -> int:
    global _session_counter
    _session_counter += 1
    return _session_counter


@dataclass
class HITLSession:
    """One task's plan, and how far through it we are."""

    cfg: HITLConfig
    instruction: str
    trajectory_id: str | None
    spec: TaskSpecification
    initial_state: State
    index: int = 0
    verdicts: list[Verdict] = field(default_factory=list)
    initially_true: frozenset[Atom] = frozenset()
    # Phase index -> the task plan cuTAMP actually found for it, filled in as each one runs.
    tamp_plans: dict[int, dict] = field(default_factory=dict)

    @property
    def phases(self) -> tuple[Phase, ...]:
        return self.spec.phases

    @property
    def finished(self) -> bool:
        return self.index >= len(self.phases)

    @property
    def current(self) -> Phase | None:
        return None if self.finished else self.phases[self.index]

    def matches(self, instruction: str, trajectory_id: str | None) -> bool:
        """Whether this session is still the right one for the rollout about to run.

        A different instruction is a different task. A different trajectory means the previous one was
        closed out (labeled, merged) and this is a fresh attempt, which must re-perceive and re-plan
        rather than resume halfway through a plan made for a scene that no longer exists.
        """
        return self.instruction == instruction and self.trajectory_id == trajectory_id

    def next_is_human(self) -> bool:
        """Whether the phase the rollout now running is followed by belongs to a human."""
        return self.current is not None and self.current.is_human

    def is_final_phase(self) -> bool:
        """Whether the phase now current is the LAST one in the plan -- nothing follows it.

        Distinct from ``is_last_leg``, which asks about the whole ``robot_run()`` a leg covers: a leg
        of three coalesced robot phases is the last leg while only its third phase is the final one.
        This is the phase-level question, which is what decides whether a human step is verified at
        all (tiptop_run._hitl_human_phase) and what a hand-back does (``plan_next``).

        False for a finished session: there is no current phase to be the final one.
        """
        return not self.finished and self.index + 1 >= len(self.phases)

    def is_last_leg(self) -> bool:
        """Whether the leg about to run is the last one of the task -- nothing follows it.

        Mirrors ``advance``: a human phase is one step, a robot leg is the whole ``robot_run()`` its
        single cuTAMP goal covers. This is what tells cuTAMP whether to end the plan by driving the
        arm home (``TAMPConfiguration.return_home``). Only the LAST leg should: a home in the middle
        of a task is motion nobody asked for, recorded into the middle of the demonstration, and the
        next leg then plans from home rather than from where this one left off.

        True for a finished session too, so a caller that asks before checking ``finished`` gets the
        conservative answer (plan the return home) rather than the surprising one.
        """
        leg = self.leg()
        return self.index + (len(leg.phases) if leg else 1) >= len(self.phases)

    def planner(self, phase: Phase | None = None):
        """The planner that carries out ``phase``, or the current one.

        The only place the session decides anything about HOW a phase is carried out, and it decides
        it by asking: which planner claims this phase (planners.planner_for). Everything else here --
        how far a leg runs, what goal it carries, who the record credits -- is read off what that
        planner answers.
        """
        phase = self.current if phase is None else phase
        assert phase is not None, "a finished session has no planner"
        return planner_for(phase, self.cfg.robot_planner)

    def leg(self) -> Leg | None:
        """The phases this rollout carries out, and who carries them out. None when finished.

        A leg is however many consecutive phases its planner takes at once. The teleop planner always
        takes exactly one -- each human step is verified on its own. The cuTAMP planner takes a whole
        run of consecutive robot phases as a single goal, stopping where a shared object would make
        one plan unsatisfiable; ``planners.CuTAMPPhasePlanner`` is where that rule and its reasons
        live now.
        """
        if self.finished:
            return None
        return self.planner().leg(self.phases[self.index :], self.spec.scene_types)

    def robot_run(self) -> tuple[Phase, ...]:
        """The consecutive robot phases this leg plans and executes as ONE goal.

        Empty when the plan is finished or the next phase is a human's, which is what callers that
        only care about the robot's side test.
        """
        leg = self.leg()
        return leg.phases if leg is not None and leg.executor == "robot" else ()

    def goal_dicts(self) -> list[dict]:
        """This leg's goal, in the form create_tamp_environment consumes.

        Every phase in the leg, conjoined -- create_tamp_environment turns each ``on(...)`` into its
        own goal atom and cuTAMP satisfies the set, so two pick-and-places are one plan.
        """
        leg = self.leg()
        assert leg is not None and leg.goal, "goal_dicts is only for a leg that has a goal to solve"
        return list(leg.goal)

    def objects_named(self) -> set[str]:
        """Every object the remaining phases refer to, for the label-drift check.

        Intersected with the DETECTED names, not with every name the plan uses: an object a human
        phase has yet to create is missing from perception on purpose, and the drift check treats a
        missing name as a plan that can no longer be executed. Left in, the very first leg of a plan
        with a deferred object would report "perception no longer detects loose_block" and re-plan the
        whole task -- the failure tiptop_run's re-binding path was added to stop. It joins this set the
        moment it is bound (SceneTypes.rebind), and is checked like anything else from then on.
        """
        names: set[str] = set()
        for phase in self.phases[self.index :]:
            names.update(phase_objects(phase))
        return names & self.spec.scene_types.detected

    def objects_needed_now(self) -> set[str]:
        """The subset of objects_named() this leg cannot proceed without.

        The difference is the whole point. objects_named() spans every phase still to come, human
        ones included, so a task whose LAST phase asks a person to cover the bowls with a cloth names
        that cloth on every leg before it -- and a robot phase that never mentions the cloth would be
        abandoned mid-task just because perception missed it once. What a leg actually needs is the
        phase it is about to carry out, plus the surfaces: those are pinned once for the whole task
        (create_tamp_environment's surface_labels) and a surface that drifted away un-rebound would
        quietly become a movable, changing the world geometry between phases.
        """
        names = set(self.spec.scene_types.surfaces)
        for phase in self.robot_run() or ([self.current] if self.current is not None else []):
            names.update(phase_objects(phase))
        return names & self.spec.scene_types.detected

    def robot_movables(self) -> set[str]:
        """The objects cuTAMP may pick up: exactly the movables some ROBOT phase names.

        A scene shared with a person contains the person's things. Perception detects them because
        the instruction mentions them -- "remove a block from the jenga tower USING THE SCREWDRIVER"
        is what makes the screwdriver a labelled object at all -- and every non-surface detection is
        a Movable by default, so cuTAMP's search treats the human's tool as a thing to pick up. On
        the run this was written for, three of the four skeletons it enumerated for "put the block
        back on the tower" opened with Pick(screwdriver), and one of them placed the screwdriver on
        the tower. None of that is wrong by cuTAMP's lights; it was never told whose the tool is.

        Computed over EVERY robot phase, not the leg about to run, for the reason SceneTypes gives
        for surfaces: an object that is a Movable in one leg and a static obstacle in the next
        changes what the search is allowed to do partway through one task. Intersected with the
        detected names so a deferred object still waiting to be bound -- or one the proposer named
        that no pass ever produced, like the `top_block` in this task's own spec -- is not handed to
        create_tamp_environment, whose unknown-object check would reject it.
        """
        names: set[str] = set()
        for phase in self.phases:
            if not phase.is_human:
                names.update(phase_objects(phase))
        return names & set(self.spec.scene_types.movables) & set(self.spec.scene_types.detected)

    def rebind(self, mapping: dict[str, str]) -> None:
        """Rename the plan's objects to this pass's labels, keeping the progress made so far."""
        _log.info(f"HITL: re-binding plan objects to this pass's labels: {mapping}")
        self.spec = self.spec.rebind(mapping)
        self.initial_state = initial_state_for(self.spec.scene_types, self.initially_true)

    def bind_new_objects(self, geometry: Mapping[str, ObjectGeometry], detected: Sequence[str]) -> None:
        """Match any object a human phase has since created onto the label this pass gave it.

        Run on EVERY leg while something is still unbound, not only when the plan's name is missing
        from this pass. Perception names objects from the task instruction, so on exactly the tasks
        that need this it readily emits a label matching the plan's -- for a different object. Trusting
        the name would bind the plan to it; the geometric test is the only thing that can tell them
        apart, so it always runs.
        """
        if not self.spec.unbound_deferred:
            return
        mapping = bind_deferred_objects(self.spec, geometry, detected, self.index)
        if mapping:
            self.rebind(mapping)

    def unbound_needed_now(self) -> list[str]:
        """Deferred objects this leg cannot run without, and which nothing has bound yet."""
        phases = self.robot_run() or ([self.current] if self.current is not None else [])
        wanted = {name for phase in phases for name in phase_objects(phase)}
        return sorted(wanted & self.spec.scene_types.deferred)

    def advance(self) -> Phase | None:
        """Move past the work this leg carried out, and return the phase it started at.

        A human phase is one step. A robot leg is the whole run its single goal covered, so the plan
        does not stop between phases that were planned and executed together.
        """
        phase = self.current
        if phase is not None:
            self.index += len(self.leg().phases)
        return phase

    def record_tamp_plan(self, index: int, plan_out: dict) -> None:
        """Note the task plan that was solved for this leg, against every phase it carried out.

        One skeleton can cover several phases (a leg), so each of them records it, and any phase
        sharing a skeleton also records which ones it was planned with -- otherwise hitl.json reads
        as though each phase had been solved on its own.

        A phase can be recorded TWICE: a leg interrupted by a teleop hand-off records what it had,
        and the leg that resumes it records again, having re-solved the skeleton it was handed. The
        later record wins, since it is the one that describes the plan that actually ran.
        """
        skeleton = (plan_out or {}).get("plan_skeleton") or []
        covered = [index]
        if index == self.index:
            covered = [index + offset for offset in range(max(1, len(self.robot_run())))]
        record = {
            "cutamp_skeleton": [op.name for op in skeleton],
            "cutamp_skeleton_reused": bool((plan_out or {}).get("reused")),
        }
        if len(covered) > 1:
            record["cutamp_skeleton_covers_phases"] = covered
        for i in covered:
            self.tamp_plans[i] = dict(record)

    def phase_record(self, index: int) -> dict:
        """One phase, with exactly what was handed to whoever carried it out."""
        phase = self.phases[index]
        record = {"index": index, **phase.summary()}
        record["planned_by"] = self.planner(phase).authorship
        if not phase.is_human:
            # What TiPToP is actually given. Note it is ATOMS, not a sentence: the ordinary path runs
            # the instruction through Gemini to get these dicts, and a HITL phase substitutes them
            # directly. `tamp_goal` is the literal grounded_atoms list create_tamp_environment
            # consumes; `tamp_goal_description` is the same thing in words, for reading.
            record["tamp_goal"] = goal_atoms_to_dicts(phase.atoms)
            record["tamp_goal_description"] = [
                describe_atom(a, DEFAULT_DESCRIPTIONS) for a in sorted(phase.atoms, key=str)
            ]
            # Present once the phase has run: the skeleton is solved against a fresh perception pass,
            # so this is what executed.
            record.update(self.tamp_plans.get(index, {}))
        return record

    def to_json(self) -> dict:
        phases = [self.phase_record(i) for i in range(len(self.phases))]
        return {
            "instruction": self.instruction,
            "trajectory_id": self.trajectory_id,
            "specification": self.spec.to_json(),
            "initially_true": sorted(str(a) for a in self.initially_true),
            # Who produced which part of all this. The short version: the VLM decides WHAT each phase
            # must achieve and IN WHAT ORDER; cuTAMP decides how the robot's phases are carried out.
            "provenance": {
                "phases_and_their_order": (
                    "vlm -- the instruction is broken into an ordered list of robot and human phases. "
                    "This replaced a symbolic search over the robot's operators plus invented ones, "
                    "which could only ever order a human step AFTER robot work, never before it"
                ),
                "phase_sub_goals": "vlm -- the atoms each robot phase must establish",
                "invented_predicates": "vlm -- name and the natural-language classifier behind it",
                "human_instructions": "vlm -- the text the operator is shown",
                # Who carries out each half, read off the planners themselves rather than named
                # here: which planner has the robot's phases is a config choice (hitl.robot_planner),
                # and a record that hard-codes one is a false statement the moment it is changed.
                "robot_phases": robot_planner(self.cfg.robot_planner).provenance,
                "human_phases": teleop_planner().provenance,
            },
            "phases": phases,
            "phase_index": self.index,
            "verifications": [v.summary() for v in self.verdicts],
        }


async def build_session(
    image: Image.Image,
    instruction: str,
    object_names: Sequence[str],
    table_name: str,
    cfg: HITLConfig,
    trajectory_id: str | None,
) -> tuple[HITLSession | None, str | None]:
    """Propose the plan for this task. Returns (session, failure_reason).

    A plan whose phases are all robot ones needs no human, which is how a task TiPToP already handles
    behaves exactly as it did before.
    """
    tag = _next_session_tag()
    spec = await propose_plan(image, instruction, object_names, table_name, cfg, tag)
    if spec.unrepresented:
        # Printed, not just logged: this is the operator's cue that the run is about to do less than
        # they asked for, and the usual remedy -- put the missing object on the table and start the
        # task again -- is only available to them before the arm moves.
        print("\n" + "=" * 70, flush=True)
        print("PART OF THIS INSTRUCTION IS NOT IN THE PLAN", flush=True)
        print("=" * 70, flush=True)
        for dropped in spec.unrepresented:
            print(f"  - {dropped['clause']}\n      {dropped['reason']}", flush=True)
        print(f"  Objects detected: {', '.join(sorted(object_names))}", flush=True)
        print("=" * 70 + "\n", flush=True)

    initially_true: frozenset[Atom] = frozenset()
    if cfg.classify_initial and spec.invented:
        initially_true = await classify_initial_state(image, spec, cfg)
        if initially_true:
            _log.info(f"HITL: already true before starting: {sorted(str(a) for a in initially_true)}")

    initial_state = initial_state_for(spec.scene_types, initially_true)
    reason = check_robot_phases(spec, initial_state)
    if reason is not None:
        return None, f"HITL planning failed: {reason}"
    if not spec.needs_human:
        _log.info("HITL: the robot can do this whole task on its own; no human phases were proposed")

    return (
        HITLSession(
            cfg=cfg,
            instruction=instruction,
            trajectory_id=trajectory_id,
            spec=spec,
            initial_state=initial_state,
            initially_true=initially_true,
        ),
        None,
    )


def handoff_message(session: HITLSession, phase: Phase) -> str:
    """What the operator is shown when the plan reaches a human phase.

    The instructions come from the phase itself; the expectations come from its atoms, so the human
    knows what will be checked afterwards -- the same list the VLM is about to be asked about.
    """
    descriptions = descriptions_for(session.spec.invented)
    lines = [
        "=" * 70,
        f"HUMAN STEP {session.index + 1} of {len(session.phases)}: {phase.description}",
        "=" * 70,
        phase.instructions,
        "",
        "When you are done, the following should be true:",
    ]
    lines += [f"  - {text}" for text in describe_expectations(phase, descriptions)]
    remaining = len(session.phases) - session.index - 1
    if remaining:
        lines.append("")
        lines.append(f"({remaining} more phase(s) follow, so the robot carries on after this.)")
    lines += [
        "",
        "Press 'Switch to teleop' in the data-collection UI to take the arm, do this by hand, then",
        "hand control back. Running without the UI: type 'done' once you have done it yourself, or",
        "'abort' to give up on this task.",
        "=" * 70,
    ]
    return "\n".join(lines)


def retry_message(missing: Sequence[str], attempts_left: int) -> str:
    """What the operator is shown when the check says the phase is not done."""
    lines = ["", "=" * 70, "The workspace does not look like that step was completed.", "Still expected:"]
    lines += [f"  - {text}" for text in missing]
    if attempts_left > 0:
        lines.append("")
        lines.append("Take the arm again and finish it, or type 'done' if you believe it IS done.")
    lines.append("=" * 70)
    return "\n".join(lines)


def phase_summary(session: HITLSession, phase: Phase) -> dict:
    """The human phase as it goes into the rollout's event stream.

    Carries where the phase sits in the plan as well as what it asks for. The UI has one "next"
    control for the hand-off, and what pressing it does -- carry on with the robot, or close the
    trajectory out -- is the plan's decision, not the operator's: ``is_last_phase`` is that decision.
    """
    return {
        "description": phase.description,
        "expected": sorted(str(a) for a in phase.atoms),
        "phase_index": session.index,
        "n_phases": len(session.phases),
        "is_last_phase": session.is_final_phase(),
    }

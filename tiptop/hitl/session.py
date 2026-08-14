"""The HITL session: one task's phases, walked one rollout at a time.

A HITL task is not one rollout. Each robot phase is an ordinary TiPToP rollout aimed at that phase's
sub-goal, and each human phase is a teleop leg -- so the plan has to survive the hand-off between
them. tiptop_run is a single long-lived process, so this state lives here and is keyed on the
trajectory it belongs to; anything that ends the trajectory ends the session with it.
"""

import logging
from dataclasses import dataclass, field
from typing import Sequence

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
from tiptop.hitl.planning import check_robot_phases, goal_atoms_to_dicts, initial_state_for, phase_objects
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

    def goal_dicts(self) -> list[dict]:
        """The current robot phase's sub-goal, in the form create_tamp_environment consumes."""
        phase = self.current
        assert phase is not None and not phase.is_human, "goal_dicts is only for a robot phase"
        return goal_atoms_to_dicts(phase.atoms)

    def objects_named(self) -> set[str]:
        """Every object the remaining phases refer to, for the label-drift check."""
        names: set[str] = set()
        for phase in self.phases[self.index :]:
            names.update(phase_objects(phase))
        return names & self.spec.scene_types.all_names

    def rebind(self, mapping: dict[str, str]) -> None:
        """Rename the plan's objects to this pass's labels, keeping the progress made so far."""
        _log.info(f"HITL: re-binding plan objects to this pass's labels: {mapping}")
        self.spec = self.spec.rebind(mapping)
        self.initial_state = initial_state_for(self.spec.scene_types, self.initially_true)

    def advance(self) -> Phase | None:
        """Move past the current phase and return the one just completed."""
        phase = self.current
        if phase is not None:
            self.index += 1
        return phase

    def record_tamp_plan(self, index: int, plan_out: dict) -> None:
        """Note the task plan cuTAMP actually found for a robot phase, once it has run."""
        skeleton = (plan_out or {}).get("plan_skeleton") or []
        self.tamp_plans[index] = {
            "cutamp_skeleton": [op.name for op in skeleton],
            "cutamp_skeleton_reused": bool((plan_out or {}).get("reused")),
        }

    def phase_record(self, index: int) -> dict:
        """One phase, with exactly what was handed to whoever carried it out."""
        phase = self.phases[index]
        record = {"index": index, **phase.summary()}
        record["planned_by"] = "vlm (order and sub-goal); cuTAMP (how)" if not phase.is_human else "vlm"
        if not phase.is_human:
            # What TiPToP is actually given. Note it is ATOMS, not a sentence: the ordinary path runs
            # the instruction through Gemini to get these dicts, and a HITL phase substitutes them
            # directly. `tamp_goal` is the literal grounded_atoms list create_tamp_environment
            # consumes; `tamp_goal_description` is the same thing in words, for reading.
            record["tamp_goal"] = goal_atoms_to_dicts(phase.atoms)
            record["tamp_goal_description"] = [
                describe_atom(a, DEFAULT_DESCRIPTIONS) for a in sorted(phase.atoms, key=str)
            ]
            # Present once the phase has run: cuTAMP searches its own skeleton from a fresh perception
            # pass, so this is what executed.
            record.update(self.tamp_plans.get(index, {}))
        return record

    def to_json(self) -> dict:
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
                "robot_phases": (
                    "cuTAMP -- each phase is planned from its tamp_goal against a fresh perception "
                    "pass, so cutamp_skeleton is what ran"
                ),
                "grasps_placements_trajectories": "cuTAMP / cuRobo",
                "human_steps": "the teleoperator, following the phase's instructions",
            },
            "phases": [self.phase_record(i) for i in range(len(self.phases))],
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


def phase_summary(phase: Phase) -> dict:
    """The human phase as it goes into the rollout's event stream."""
    return {"description": phase.description, "expected": sorted(str(a) for a in phase.atoms)}

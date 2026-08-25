"""Human-in-the-loop planning: proposal validation, phase plans, and what reaches TiPToP.

No robot, no GPU, no network -- the VLM's job here is to produce JSON, so the tests drive the parser
with the JSON directly. The running example is the task that forced the phase model:
"pick the toy off the box and place it on the table, open the box, place the toy inside the box".
"""

import asyncio
import json
import re
from unittest import mock

import pytest

from tiptop.hitl import grounding, llm
from tiptop.hitl.config import HITLConfig, load_hitl_config, resolve_hitl_config
from tiptop.hitl.grounding import Verdict
from tiptop.hitl.planning import (
    ObjectGeometry,
    bind_deferred_object,
    bind_deferred_objects,
    match_drifted_names,
    check_robot_phases,
    goal_atoms_to_dicts,
    initial_state_for,
    objects_resting_on,
    unachievable_atoms,
)
from tiptop.hitl.prompts import plan_prompt
from tiptop.hitl.proposal import parse_plan_response
from tiptop.hitl.session import HITLSession, handoff_message
from tiptop.hitl.structs import (
    DeferredObject,
    HITLProposalError,
    describe_atom,
    display_atom,
    display_name,
)

OBJECTS = ["blue_toy", "white_box"]
TABLE = "table"
CFG = HITLConfig(enabled=True)

# The plan the proposer should produce for the three-phase task. The ordering is the whole point:
# the box is opened BEFORE anything is placed in it, which the previous design could not express.
PLAN_RESPONSE = {
    "new_predicates": [
        {"name": "IsOpen", "instructions": "the container {0} is open, so its interior is visible"}
    ],
    "phases": [
        {
            "executor": "robot",
            "description": "take the toy off the box and put it on the table",
            "atoms": [{"predicate": "On", "args": ["blue_toy", "table"]}],
        },
        {
            "executor": "human",
            "description": "open the box",
            "instructions": "Open the white_box and fold its flaps back.",
            "atoms": [{"predicate": "IsOpen", "args": ["white_box"]}],
        },
        {
            "executor": "robot",
            "description": "put the toy inside the box",
            "atoms": [{"predicate": "On", "args": ["blue_toy", "white_box"]}],
        },
    ],
}


def parse(response=None, objects=OBJECTS, tag=1, instruction="do the thing"):
    return parse_plan_response(response or PLAN_RESPONSE, instruction, objects, TABLE, tag)


def _plan(**changes):
    """PLAN_RESPONSE with the phases replaced."""
    return {**PLAN_RESPONSE, **changes}


def _phases(*phases):
    """A plan of just these phases, with no invented predicates left over to be unused."""
    return {"phases": list(phases)}


def test_the_prompt_actually_contains_the_instruction():
    # It did not, for one round of testing, and the model planned from the IMAGE alone -- inventing a
    # plausible-looking task for the scene and ignoring what was asked. Nothing else catches that:
    # the response parses, validates, and plans perfectly well; it is just answering another question.
    from tiptop.hitl.prompts import plan_prompt

    prompt = plan_prompt("open the box and put the toy in it", ["blue_toy", "white_box"])
    assert "open the box and put the toy in it" in prompt
    assert "blue_toy" in prompt and "white_box" in prompt


# --- the phase plan -------------------------------------------------------------------------------


def test_a_human_phase_can_come_before_robot_work():
    # The defect that forced this design: ordering used to flow only one way, so "open the box, THEN
    # put the toy in" planned the placement first and the run ended after the human step.
    spec = parse()
    assert [p.executor for p in spec.phases] == ["robot", "human", "robot"]
    assert [p.description for p in spec.phases][1] == "open the box"
    assert spec.needs_human


def test_an_object_may_be_picked_up_in_more_than_one_phase():
    # cuTAMP's HasNotPickedUp allows one pick per object per PLAN, which is why the toy could not be
    # moved out of the box and back in under a single-plan design. Each phase is its own cuTAMP
    # problem from a fresh perception pass, so the toy appears in phases 0 and 2.
    spec = parse()
    assert goal_atoms_to_dicts(spec.phases[0].atoms) == [{"predicate": "on", "args": ["blue_toy", "table"]}]
    assert goal_atoms_to_dicts(spec.phases[2].atoms) == [{"predicate": "on", "args": ["blue_toy", "white_box"]}]


def test_the_scene_types_are_fixed_once_across_every_phase():
    # Inferring per phase would make white_box a Surface in phase 2 and a Movable in phase 0, so the
    # world geometry cuTAMP plans against would change mid-task.
    spec = parse()
    assert spec.scene_types.surfaces == frozenset({"table", "white_box"})
    assert spec.scene_types.movables == frozenset({"blue_toy"})
    assert [p.type for p in spec.invented[0].fluent.parameters] == ["surface"]


def test_a_robot_phase_cannot_be_asked_for_an_invented_predicate():
    # The rule that decides what is a human phase: cuTAMP has no operator that can make an invented
    # predicate true, so a robot phase asking for one would plan forever and never reach its goal.
    response = _plan(phases=[
        {"executor": "robot", "description": "open the box", "atoms": [{"predicate": "IsOpen", "args": ["white_box"]}]}
    ])
    with pytest.raises(HITLProposalError, match="A robot phase cannot achieve 'IsOpen'"):
        parse(response)


def test_every_robot_phase_is_checked_for_achievability():
    spec = parse()
    initial = initial_state_for(spec.scene_types)
    assert check_robot_phases(spec, initial) is None
    assert unachievable_atoms(spec.phases[0].atoms, initial) == []
    # An invented atom is unachievable by the robot; this is the guard that keeps it out of cuTAMP's
    # unbounded search rather than discovering it there.
    assert unachievable_atoms(spec.phases[1].atoms, initial) == sorted(spec.phases[1].atoms, key=str)


@pytest.mark.parametrize(
    "response, expected",
    [
        (_phases(), "at least one phase"),
        (
            _phases({"executor": "sidekick", "description": "x", "atoms": []}),
            "executor must be 'robot' or 'human'",
        ),
        (
            _phases({"executor": "robot", "description": "nothing", "atoms": []}),
            "has no atoms",
        ),
        (
            _plan(phases=[{"executor": "human", "description": "open it",
                           "atoms": [{"predicate": "IsOpen", "args": ["white_box"]}]}]),
            "needs `instructions`",
        ),
        (
            _phases({"executor": "robot", "description": "x",
                     "atoms": [{"predicate": "On", "args": ["blue_toy", "moon"]}]}),
            "not an object in this scene",
        ),
        (
            _phases({"executor": "robot", "description": "x",
                     "atoms": [{"predicate": "Sideways", "args": ["blue_toy"]}]}),
            "Unknown predicate",
        ),
        (
            _phases({"executor": "robot", "description": "x",
                     "atoms": [{"predicate": "On", "args": ["blue_toy"]}]}),
            "takes 2 argument",
        ),
        (
            {**PLAN_RESPONSE, "new_predicates": [{"name": "On", "instructions": "{0} on {1}"}]},
            "already exists",
        ),
        (
            {**PLAN_RESPONSE,
             "new_predicates": [*PLAN_RESPONSE["new_predicates"], {"name": "Tidy", "instructions": "{0} tidy"}]},
            "no phase uses it",
        ),
        (
            {**PLAN_RESPONSE, "new_predicates": [{"name": "IsOpen", "instructions": "{0} and {1}"}]},
            "placeholders",
        ),
        # '#' is the internal session-suffix separator; a name carrying one would be double-suffixed.
        (
            {**PLAN_RESPONSE, "new_predicates": [{"name": "IsOpen#2", "instructions": "{0} open"}]},
            "'#' is not allowed",
        ),
    ],
)
def test_plan_rejections(response, expected):
    with pytest.raises(HITLProposalError, match=expected):
        parse(response)


def test_an_all_robot_plan_needs_no_human():
    # The degradation guarantee: a task TiPToP already handles produces one robot phase and behaves
    # exactly as it did before HITL existed.
    response = {"phases": [{"executor": "robot", "description": "put the toy in the box",
                            "atoms": [{"predicate": "On", "args": ["blue_toy", "white_box"]}]}]}
    spec = parse(response)
    assert not spec.needs_human
    assert check_robot_phases(spec, initial_state_for(spec.scene_types)) is None
    assert goal_atoms_to_dicts(spec.phases[0].atoms) == [{"predicate": "on", "args": ["blue_toy", "white_box"]}]


def test_a_clause_that_cannot_be_expressed_is_reported_not_dropped():
    # Observed on the rig: "... pick ANOTHER toy, and place it in the box" against a scene holding
    # exactly one toy. The plan came back missing that clause and nothing said so, so the run did most
    # of the task and reported success.
    response = {**PLAN_RESPONSE, "unrepresented": [
        {"clause": "pick another toy", "reason": "only one toy was detected"}
    ]}
    spec = parse(response)
    assert len(spec.phases) == 3, "the clauses that COULD be expressed are still planned"
    assert spec.unrepresented == ({"clause": "pick another toy", "reason": "only one toy was detected"},)
    assert spec.to_json()["unrepresented"][0]["reason"] == "only one toy was detected"


def test_a_fully_expressible_instruction_reports_nothing_unrepresented():
    assert parse().unrepresented == ()


def test_an_invented_predicate_used_inconsistently_is_rejected():
    # Its signature is read off its uses, so the uses have to agree.
    response = {**PLAN_RESPONSE, "phases": [
        *PLAN_RESPONSE["phases"],
        {"executor": "human", "description": "and the toy", "instructions": "open the toy",
         "atoms": [{"predicate": "IsOpen", "args": ["blue_toy"]}]},
    ]}
    with pytest.raises(HITLProposalError, match="inconsistent arguments"):
        parse(response)


def test_invented_fluent_names_are_unique_per_session():
    # cuTAMP interns ground atoms process-globally on (name, values) and compares them by that string
    # alone, so two tasks in one process must not share an invented predicate's name.
    first, second = parse(tag=1).invented[0], parse(tag=2).invented[0]
    assert first.name != second.name
    assert display_name(first.name) == display_name(second.name) == "IsOpen"


# --- the session ----------------------------------------------------------------------------------


def _session(spec=None, trajectory_id="traj-1"):
    spec = spec or parse()
    return HITLSession(
        cfg=CFG,
        instruction=spec.instruction,
        trajectory_id=trajectory_id,
        spec=spec,
        initial_state=initial_state_for(spec.scene_types),
    )


def test_only_the_last_leg_of_a_task_ends_by_driving_the_arm_home():
    # cuTAMP ends every plan at q_home. That is right for a plan that IS the episode and wrong for a
    # leg of one: the bug this guards had the arm drive home after the robot phase of
    # "place the bread on the plate, then open the box and place the bread in the box" -- a return to
    # home recorded into the middle of the demonstration, with the human then handed an arm parked at
    # home rather than where the plan stopped. is_last_leg is what tiptop_run passes to run_planning
    # as return_home.
    session = _session()  # robot, human, robot
    assert not session.is_last_leg(), "two phases still follow this one"
    session.advance()
    assert not session.is_last_leg(), "the human phase is followed by a robot one"
    session.advance()
    assert session.is_last_leg(), "the final robot phase parks the arm"
    session.advance()
    assert session.finished and session.is_last_leg(), "a finished session answers conservatively"


def test_the_last_leg_is_the_whole_run_of_robot_phases_it_covers():
    # A leg can cover several phases (robot_run), so "is this the last phase" is the wrong question:
    # the run below covers phases 0 and 1 of 3, and only the leg after it ends the task.
    session = _sort_session()  # robot, robot, human
    assert len(session.robot_run()) == 2
    assert not session.is_last_leg(), "the human phase still follows the pair"
    session.advance()
    assert session.index == 2 and session.is_last_leg()


def test_the_session_walks_the_phases_in_order():
    session = _session()
    assert not session.next_is_human() and session.goal_dicts()
    session.advance()
    assert session.next_is_human(), "phase 1 is the human's"
    session.advance()
    assert not session.next_is_human(), "phase 2 is the robot's again -- the part that used to be lost"
    assert session.goal_dicts() == [{"predicate": "on", "args": ["blue_toy", "white_box"]}]
    session.advance()
    assert session.finished


def test_the_handoff_message_says_work_remains():
    session = _session()
    session.advance()
    message = handoff_message(session, session.current)
    assert "Open the white_box" in message
    assert "1 more phase(s) follow" in message, "the operator must know the robot is not done"


def test_a_renamed_object_re_binds_instead_of_throwing_the_plan_away():
    # Observed on the rig: phase 0 ran against objects Gemini called "toy" and "box"; the next pass
    # called the same two things "blue_toy" and "cardboard_box". The plan was discarded and the whole
    # task re-planned from a scene already half rearranged, which asked the human to redo the phase
    # they had just finished.
    assert match_drifted_names(["toy", "box"], ["blue_toy", "cardboard_box"]) == {
        "toy": "blue_toy", "box": "cardboard_box",
    }
    # It works the other way round too (the detector dropping an adjective).
    assert match_drifted_names(["blue_toy"], ["toy"]) == {"blue_toy": "toy"}
    # Ambiguity is refused rather than guessed: the caller then re-plans, as before.
    assert match_drifted_names(["toy"], ["blue_toy", "red_toy"]) is None
    assert match_drifted_names(["toy"], ["cardboard_box"]) is None

    session = _session()
    session.advance()
    session.advance()  # phases 0 and 1 done; phase 2 is the robot's again
    session.rebind({"blue_toy": "green_toy", "white_box": "brown_box"})
    assert session.goal_dicts() == [{"predicate": "on", "args": ["green_toy", "brown_box"]}]
    assert session.index == 2, "progress through the plan survives the rename"
    assert session.spec.scene_types.surfaces == frozenset({"table", "brown_box"})


def test_the_session_is_dropped_when_the_trajectory_changes():
    session = _session()
    assert session.matches(session.instruction, "traj-1")
    assert not session.matches(session.instruction, "traj-2")
    assert not session.matches("a different task", "traj-1")


# The second running example: two ROBOT phases back to back, then a human one. The running example
# above always separates its robot phases with a human phase, which is exactly why the rollout loop
# could ship for months only knowing how to continue a trajectory across a robot->human boundary --
# a robot->robot boundary ended the episode, and the arm sorted one toy and stopped.
SORT_OBJECTS = ["blue_bowl", "blue_cloth", "blue_toy", "green_bowl", "green_toy"]
SORT_RESPONSE = {
    "new_predicates": [
        {"name": "AreCoveredBy", "instructions": "the {2} is draped over both {0} and {1}"}
    ],
    "phases": [
        {
            "executor": "robot",
            "description": "put the blue toy in the blue bowl",
            "atoms": [{"predicate": "On", "args": ["blue_toy", "blue_bowl"]}],
        },
        {
            "executor": "robot",
            "description": "put the green toy in the green bowl",
            "atoms": [{"predicate": "On", "args": ["green_toy", "green_bowl"]}],
        },
        {
            "executor": "human",
            "description": "cover both bowls with the cloth",
            "instructions": "Drape the blue_cloth over both bowls.",
            "atoms": [{"predicate": "AreCoveredBy", "args": ["blue_bowl", "green_bowl", "blue_cloth"]}],
        },
    ],
}


def _sort_session():
    spec = parse(SORT_RESPONSE, objects=SORT_OBJECTS, instruction="sort the toys into same color bowls")
    return HITLSession(
        cfg=CFG,
        instruction=spec.instruction,
        trajectory_id="traj-1",
        spec=spec,
        initial_state=initial_state_for(spec.scene_types),
    )


def test_consecutive_robot_phases_are_planned_and_run_as_one_goal():
    # Both toys are sorted by ONE cuTAMP plan and one continuous motion, which is what the non-HITL
    # path does with the same two-clause instruction. Planning them separately worked, but the arm
    # stopped between them to re-perceive and re-plan -- and that second perception pass is where the
    # object labels drift.
    session = _sort_session()
    assert check_robot_phases(session.spec, session.initial_state) is None

    assert [p.description for p in session.robot_run()] == [
        "put the blue toy in the blue bowl",
        "put the green toy in the green bowl",
    ], "the human phase ends the run"
    assert session.goal_dicts() == [
        {"predicate": "on", "args": ["blue_toy", "blue_bowl"]},
        {"predicate": "on", "args": ["green_toy", "green_bowl"]},
    ]

    session.record_tamp_plan(0, {"plan_skeleton": [], "reused": False})
    session.advance()
    assert session.index == 2, "one leg covered both robot phases"
    assert session.next_is_human()

    record = session.to_json()
    assert [p["executor"] for p in record["phases"]] == ["robot", "robot", "human"]
    # Both phases record the skeleton, and say they shared it -- otherwise the audit trail reads as
    # though each had been solved on its own.
    for phase in record["phases"][:2]:
        assert phase["cutamp_skeleton"] == []
        assert phase["cutamp_skeleton_covers_phases"] == [0, 1]


def test_a_run_stops_at_a_phase_moving_an_object_the_run_already_moved():
    # cuTAMP's Pick requires and deletes HasNotPickedUp(obj), so one plan picks each object once:
    # On(blue_toy, table) and On(blue_toy, white_box) at once is unsatisfiable, not slow. Phases like
    # these are genuinely sequential and stay separate legs -- which is what the robot->robot
    # continuation in the rollout loop exists to carry.
    spec = parse(_phases(
        {
            "executor": "robot",
            "description": "take the toy off the box",
            "atoms": [{"predicate": "On", "args": ["blue_toy", "table"]}],
        },
        {
            "executor": "robot",
            "description": "put the toy back on the box",
            "atoms": [{"predicate": "On", "args": ["blue_toy", "white_box"]}],
        },
    ))
    session = HITLSession(
        cfg=CFG,
        instruction=spec.instruction,
        trajectory_id="traj-1",
        spec=spec,
        initial_state=initial_state_for(spec.scene_types),
    )
    assert len(session.robot_run()) == 1, "the second phase moves blue_toy again"
    assert session.goal_dicts() == [{"predicate": "on", "args": ["blue_toy", "table"]}]

    session.record_tamp_plan(0, {"plan_skeleton": [], "reused": False})
    session.advance()
    assert session.index == 1 and not session.finished and not session.next_is_human()
    assert session.goal_dicts() == [{"predicate": "on", "args": ["blue_toy", "white_box"]}]
    # Planned on its own, so no shared-skeleton marker.
    assert "cutamp_skeleton_covers_phases" not in session.to_json()["phases"][0]


def test_a_leg_is_not_gated_on_an_object_only_a_later_human_phase_names():
    # objects_named() spans every remaining phase, so it includes the cloth -- which no robot phase
    # names. Perception missing the cloth used to destroy the whole plan before cuTAMP was ever
    # called, and the operator's next attempt re-sorted the toys from the start.
    session = _sort_session()

    assert "blue_cloth" in session.objects_named()
    assert "blue_cloth" not in session.objects_needed_now()
    # What the leg genuinely cannot proceed without: every object in the run it is about to plan,
    # plus the surfaces, which are pinned for the whole task and would otherwise turn into movables
    # mid-plan. The table is in there like any other surface; the caller subtracts it, exactly as it
    # does for objects_named().
    assert session.objects_needed_now() == {
        "blue_toy", "blue_bowl", "green_toy", "green_bowl", "table",
    }

    # The human phase's own leg does need the cloth -- the relaxation is per-leg, not permanent.
    session.advance()
    assert "blue_cloth" in session.objects_needed_now()


def test_a_name_the_plan_already_owns_is_never_a_re_binding_candidate():
    # The re-binding pool is `detected - spec.scene_types.all_names`, NOT `detected -
    # objects_named()`. objects_named() covers only the phases still to come, so an object named
    # solely by a COMPLETED phase drops out of it while remaining a plan object -- and offering it as
    # a target lets match_drifted_names fold two spec objects into one (SceneTypes.rebind merges them
    # inside a frozenset), pointing this leg at the thing the robot has already put away.
    session = _sort_session()
    session.advance()  # both robot phases done; only the human phase remains

    owned = session.spec.scene_types.all_names
    assert {"blue_toy", "green_toy"} <= owned, "the sorted toys are still the plan's objects"
    assert not ({"blue_toy", "green_toy"} & session.objects_named()), "but no remaining phase names them"

    # So they must not survive into the pool. Only a label the plan does not already own can be a
    # drifted spelling of one it does -- here, everything perception re-detected is already a plan
    # object, and the one genuinely new label is the only candidate.
    detected = {"blue_toy", "green_toy", "blue_bowl", "green_bowl", "table", "red_ball"}
    assert sorted(detected - owned) == ["red_ball"]
    # The rule the guard rests on: a nested pair would otherwise match.
    assert match_drifted_names(["green_toy"], ["toy"]) == {"green_toy": "toy"}


def test_a_finished_session_still_reports_its_pinned_surfaces():
    # objects_needed_now() is read on any leg, including one that finds the plan already complete.
    session = _sort_session()
    while not session.finished:
        session.advance()
    assert session.objects_needed_now() == {"blue_bowl", "green_bowl", "table"}


def test_hitl_json_records_the_plan_and_what_tiptop_was_handed():
    session = _session()
    session.record_tamp_plan(0, {"plan_skeleton": [], "reused": False})
    record = session.to_json()

    assert [p["executor"] for p in record["phases"]] == ["robot", "human", "robot"]
    assert record["phases"][0]["tamp_goal"] == [{"predicate": "on", "args": ["blue_toy", "table"]}]
    assert record["phases"][0]["tamp_goal_description"] == ["blue_toy is resting on top of table"]
    assert record["phases"][0]["cutamp_skeleton"] == []
    assert "tamp_goal" not in record["phases"][1]
    assert record["phases"][1]["instructions"].startswith("Open the white_box")
    assert set(record["provenance"]) >= {"phases_and_their_order", "robot_phases"}
    # Atoms are rendered without the internal session suffix.
    assert record["phases"][1]["atoms"] == ["IsOpen(white_box)"]


# --- verification ---------------------------------------------------------------------------------


def test_only_camera_settleable_atoms_are_put_to_the_vlm():
    # A human phase may also mention the robot's own state. HandEmpty/Holding must not be judged from
    # a photo: the frame is a third-person view chosen because the arm is wherever the operator left
    # it, so the gripper is often out of shot, and the classifier answers false when it cannot see
    # the statement to be true.
    response = _plan(phases=[{
        "executor": "human", "description": "open the box", "instructions": "open it",
        "atoms": [
            {"predicate": "IsOpen", "args": ["white_box"]},
            {"predicate": "On", "args": ["blue_toy", "table"]},
            {"predicate": "HandEmpty", "args": []},
        ],
    }])
    spec = parse(response)
    phase = spec.phases[0]
    asked = []

    async def fake_classify_all(image, atoms, descriptions, cfg):
        asked.extend(atoms)
        return [Verdict(a, describe_atom(a, descriptions), True, "") for a in atoms]

    with mock.patch.object(grounding, "classify_all", fake_classify_all):
        ok, _ = asyncio.run(grounding.verify_phase(None, phase, spec.invented, CFG))
    assert ok
    assert {display_atom(a) for a in asked} == {"IsOpen(white_box)", "On(blue_toy, table)"}

    # But the human is still TOLD about all of it, so they know what is expected.
    shown = grounding.describe_expectations(phase, grounding.descriptions_for(spec.invented))
    assert any("gripper is empty" in line for line in shown)


# --- the reprompt loop ----------------------------------------------------------------------------


class _FakeGemini:
    """A Gemini client that returns canned responses in order and records the prompts it saw."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.prompts = []
        self.aio = self

    @property
    def models(self):
        return self

    async def generate_content(self, model, contents, config):
        self.prompts.append(contents[-1])
        return mock.Mock(text=self._responses.pop(0))


def test_a_rejected_proposal_is_reprompted_with_the_reason():
    # The mechanism the design leans on: a proposal that parses but does not validate is a correction,
    # not a failed run. This is what recovered the errors seen live (a hallucinated object, an atom
    # applied to the wrong type, a robot phase asking for an invented predicate).
    bad = json.dumps(_plan(phases=[{"executor": "robot", "description": "open it",
                                    "atoms": [{"predicate": "IsOpen", "args": ["white_box"]}]}]))
    client = _FakeGemini([bad, json.dumps(PLAN_RESPONSE)])
    with mock.patch.object(llm, "gemini_client", lambda: client):
        spec = asyncio.run(
            llm.query_json("PROMPT", lambda d: parse(d), model="m", schema={}, max_attempts=3)
        )
    assert len(client.prompts) == 2, "the second attempt should have been made"
    assert "A robot phase cannot achieve" in client.prompts[1], "the reprompt must say what was wrong"
    assert [p.executor for p in spec.phases] == ["robot", "human", "robot"]


def test_a_proposal_that_never_validates_raises_the_last_reason():
    bad = json.dumps({"phases": [{"executor": "robot", "description": "x",
                                  "atoms": [{"predicate": "On", "args": ["blue_toy", "moon"]}]}]})
    client = _FakeGemini([bad, bad, bad])
    with mock.patch.object(llm, "gemini_client", lambda: client):
        with pytest.raises(HITLProposalError, match="not an object in this scene"):
            asyncio.run(llm.query_json("PROMPT", lambda d: parse(d), model="m", schema={}, max_attempts=3))
    assert len(client.prompts) == 3, "every attempt should have been used"


# --- config ---------------------------------------------------------------------------------------


def test_config_defaults_to_off_and_rejects_typos():
    assert resolve_hitl_config(None).enabled is False
    assert resolve_hitl_config({}).enabled is False
    assert load_hitl_config('{"enabled": true, "verify_retries": 2}').verify_retries == 2
    with pytest.raises(ValueError, match="unknown hitl config key"):
        resolve_hitl_config({"enable": True})
    with pytest.raises(ValueError, match="must be a mapping"):
        resolve_hitl_config([1, 2])


def test_the_shipped_hitl_configs_resolve():
    # The configs this feature ships with; their hitl blocks have to survive resolve_hitl_config or
    # the run turns the feature off without saying so.
    yaml = pytest.importorskip("yaml")
    from pathlib import Path

    cfg_dir = Path(__file__).resolve().parents[2] / "data-collection" / "cfg" / "tamp"
    if not cfg_dir.exists():
        pytest.skip("data-collection is not checked out beside tiptop")
    shipped = sorted(p for p in cfg_dir.glob("*.yml") if "hitl" in (yaml.safe_load(p.read_text()) or {}))
    assert shipped, "at least one shipped config should carry an hitl block"
    for path in shipped:
        assert resolve_hitl_config(yaml.safe_load(path.read_text())["hitl"]).enabled, path.name


# --- Deferred objects: naming something a human phase has yet to create ---------------------------
#
# The task that forced these: "remove a block from the jenga tower using the screwdriver onto the
# white paper, and then place the block on top of the jenga tower". At plan time the loose block is
# still a brick inside the tower, so no detected label refers to it -- and the proposer's only legal
# move was to invent a predicate, which forces the phase to be a human one. The second clause is a
# plain pick-and-place, so it was teleoperated when the robot should have done it.

# Straight out of the run this was written for -- runs/.../eval/2026-08-17_15-20-59/scene_objects.json.
# The loose block IS on the paper in the image and was NOT detected; `top_jenga_block` is the brick
# still on top of the tower.
JENGA_GEOMETRY = {
    "jenga_tower": ObjectGeometry((0.5392619834, 0.0840336195, 0.0880103576), (0.0893346521, 0.0782887562, 0.1548677220)),
    "screwdriver": ObjectGeometry((0.3708415167, 0.2350496005, -0.0070450752), (0.1020133699, 0.0353481902, 0.0269756437)),
    "white_paper": ObjectGeometry((0.5365092176, -0.0787286520, -0.0139531689), (0.2941993016, 0.2041751835, 0.0075770352)),
    "top_jenga_block": ObjectGeometry((0.5355900793, 0.0936445544, 0.1298197908), (0.0849671634, 0.0704825390, 0.0303186096)),
}

LOOSE_BLOCK = DeferredObject(name="loose_block", created_by_phase=0, anchor="white_paper")


def test_the_towers_own_top_block_is_never_bound_to_the_loose_one():
    # The whole point of binding on geometry. `top_jenga_block` is block-shaped, block-named, and
    # wrong: it sits at the tower's y, 7cm outside the paper's footprint. Binding it would have the
    # robot lift the tower's top block and place it back on the tower -- a no-op reported as success.
    assert bind_deferred_object(LOOSE_BLOCK, JENGA_GEOMETRY, ["top_jenga_block"]) is None
    # Not even when perception gives it the very name the plan is looking for.
    geometry = {**JENGA_GEOMETRY, "loose_block": JENGA_GEOMETRY["top_jenga_block"]}
    assert bind_deferred_object(LOOSE_BLOCK, geometry, ["loose_block"]) is None


def test_a_block_that_really_is_on_the_paper_binds():
    geometry = {**JENGA_GEOMETRY, "wooden_block": ObjectGeometry((0.55, -0.06, 0.0045), (0.075, 0.025, 0.015))}
    assert bind_deferred_object(LOOSE_BLOCK, geometry, ["wooden_block"]) == "wooden_block"


def test_two_candidates_on_the_anchor_refuse_to_bind():
    # Same discipline as match_drifted_names: there is no safe way to guess which one was meant.
    geometry = {
        **JENGA_GEOMETRY,
        "block_a": ObjectGeometry((0.55, -0.06, 0.0045), (0.075, 0.025, 0.015)),
        "block_b": ObjectGeometry((0.50, -0.10, 0.0045), (0.075, 0.025, 0.015)),
    }
    assert bind_deferred_object(LOOSE_BLOCK, geometry, ["block_a", "block_b"]) is None


def test_an_anchor_with_no_geometry_refuses_to_bind():
    assert bind_deferred_object(LOOSE_BLOCK, {}, ["wooden_block"]) is None


JENGA_OBJECTS = ["jenga_tower", "screwdriver", "white_paper"]

# What the proposer should now answer for the jenga task. Before deferred objects existed it had no
# name for the loose block, so it invented a predicate instead -- and an invented predicate forces a
# human phase, so phase 1 was teleoperated.
JENGA_PLAN = {
    "new_objects": [
        {
            "name": "loose_block",
            "created_by_phase": 0,
            "description": "the single wooden block the human pushes out of the tower",
        }
    ],
    "phases": [
        {
            "executor": "human",
            "description": "push a block out of the tower onto the paper",
            "instructions": "Use the screwdriver to push one block out of the jenga_tower and put it on the white_paper.",
            "atoms": [{"predicate": "On", "args": ["loose_block", "white_paper"]}],
        },
        {
            "executor": "robot",
            "description": "put the block on top of the tower",
            "atoms": [{"predicate": "On", "args": ["loose_block", "jenga_tower"]}],
        },
    ],
}


def _jenga(**changes):
    return {**JENGA_PLAN, **changes}


def parse_jenga(response=None):
    return parse_plan_response(
        response or JENGA_PLAN, "remove a block and put it on top", JENGA_OBJECTS, TABLE, 1
    )


def test_a_pick_and_place_of_a_not_yet_existing_object_stays_a_robot_phase():
    # The bug this fixes: with no name to bind the loose block to, the proposer's only legal move was
    # an invented predicate, which forces a human phase -- so a plain pick-and-place was teleoperated.
    spec = parse_jenga()
    assert [p.executor for p in spec.phases] == ["human", "robot"]
    assert goal_atoms_to_dicts(spec.phases[1].atoms) == [
        {"predicate": "on", "args": ["loose_block", "jenga_tower"]}
    ]
    assert not spec.invented, "nothing needs inventing once the object can be named"
    # Typed like any other movable, so the phase validates and cuTAMP can be aimed at it...
    assert "loose_block" in spec.scene_types.movables
    assert check_robot_phases(spec, initial_state_for(spec.scene_types)) is None
    # ...but held apart, because perception has not produced it yet.
    assert spec.scene_types.deferred == frozenset({"loose_block"})
    assert "loose_block" not in spec.scene_types.detected
    deferred = spec.deferred[0]
    assert (deferred.name, deferred.created_by_phase, deferred.anchor) == ("loose_block", 0, "white_paper")


def test_the_first_leg_is_not_abandoned_because_the_new_object_is_not_detected_yet():
    # The drift check runs before anything else and treats a name perception did not produce as a
    # plan that can no longer be executed. An object a human phase has yet to create is missing on
    # purpose, so counting it would re-plan the whole task on its very first leg.
    session = _session(spec=parse_jenga())
    assert "loose_block" not in session.objects_named()
    assert "loose_block" not in session.objects_needed_now()
    assert not (session.objects_named() - set(JENGA_OBJECTS) - {TABLE}), "nothing looks missing yet"


def test_the_robot_leg_refuses_to_run_until_the_new_object_is_found():
    session = _session(spec=parse_jenga())
    session.advance()  # the human has done phase 0
    assert session.unbound_needed_now() == ["loose_block"]
    # Nothing on the paper: binding refuses, and the leg has no goal it could honestly plan.
    session.bind_new_objects(JENGA_GEOMETRY, ["top_jenga_block"])
    assert session.unbound_needed_now() == ["loose_block"]


def test_once_the_block_is_on_the_paper_the_robot_leg_aims_at_it():
    session = _session(spec=parse_jenga())
    session.advance()
    geometry = {**JENGA_GEOMETRY, "wooden_block": ObjectGeometry((0.55, -0.06, 0.0045), (0.075, 0.025, 0.015))}
    session.bind_new_objects(geometry, [*JENGA_OBJECTS, "wooden_block"])
    assert session.unbound_needed_now() == []
    assert session.goal_dicts() == [{"predicate": "on", "args": ["wooden_block", "jenga_tower"]}]
    # Bound means bound: it is an ordinary object now, so a later pass losing it IS a real problem.
    assert "wooden_block" in session.objects_named()


def test_the_humans_screwdriver_is_never_something_the_robot_picks_up():
    # Observed on the rig, 9_pp_screwdriver_jenga: perception detects the screwdriver because the
    # INSTRUCTION names it ("remove a block ... using the screwdriver"), every non-surface detection
    # is a Movable, and so three of the four skeletons cuTAMP enumerated for "put the block back on
    # the tower" opened with Pick(screwdriver) -- one of them placing it on the tower. No robot phase
    # ever mentions it: it is the human's tool, and the plan already says so.
    session = _session(spec=parse_jenga())
    session.advance()
    geometry = {**JENGA_GEOMETRY, "wooden_block": ObjectGeometry((0.55, -0.06, 0.0045), (0.075, 0.025, 0.015))}
    session.bind_new_objects(geometry, [*JENGA_OBJECTS, "wooden_block"])
    assert session.robot_movables() == {"wooden_block"}
    assert "screwdriver" in session.spec.scene_types.movables, "still a Movable to the SPEC..."


def test_an_object_no_pass_ever_detected_is_not_offered_as_a_movable():
    # The proposer may name a movable that perception never produces -- this task's own spec carried
    # a `top_block`. create_tamp_environment rejects a label it has no mesh for, so the set handed to
    # it has to be the detected ones only.
    session = _session(spec=parse_jenga())
    assert "loose_block" not in session.robot_movables(), "still deferred -- no mesh exists for it"
    assert session.robot_movables() == set()


def test_the_movables_the_robot_may_pick_are_the_only_ones_cutamp_is_given():
    # The other half of the same fix: the session decides WHICH, create_tamp_environment is what
    # acts on it. Everything excluded stays in the scene as a static, so the arm still avoids it.
    pytest.importorskip("curobo")
    from curobo.geom.types import Cuboid, Mesh

    from tiptop.tiptop_run import create_tamp_environment

    def mesh(name):
        return Mesh(name=name, pose=[0, 0, 0, 1, 0, 0, 0], vertices=[[0, 0, 0]], faces=[[0, 0, 0]])

    meshes = {n: mesh(n) for n in ["loose_block", "screwdriver", "jenga_tower", "white_paper"]}
    table = Cuboid(name=TABLE, pose=[0, 0, 0, 1, 0, 0, 0], dims=[1, 1, 0.02])
    goal = [{"predicate": "on", "args": ["loose_block", "jenga_tower"]}]
    surfaces = {"jenga_tower", "white_paper", TABLE}

    env, _ = create_tamp_environment(meshes, table, goal, False, surface_labels=surfaces)
    assert {m.name for m in env.movables} == {"loose_block", "screwdriver"}, "the old behaviour"

    env, _ = create_tamp_environment(
        meshes, table, goal, False, surface_labels=surfaces, movable_labels={"loose_block"}
    )
    assert {m.name for m in env.movables} == {"loose_block"}
    assert "screwdriver" in {s.name for s in env.statics}, "avoided, not ignored"


def test_the_object_the_goal_moves_stays_pickable_whatever_the_caller_said():
    # A goal over something typed out of Movable is not a worse plan, it is an unplannable one, so
    # the goal's own object is added back rather than trusted to be in the set.
    pytest.importorskip("curobo")
    from curobo.geom.types import Cuboid, Mesh

    from tiptop.tiptop_run import create_tamp_environment

    def mesh(name):
        return Mesh(name=name, pose=[0, 0, 0, 1, 0, 0, 0], vertices=[[0, 0, 0]], faces=[[0, 0, 0]])

    meshes = {n: mesh(n) for n in ["loose_block", "screwdriver"]}
    table = Cuboid(name=TABLE, pose=[0, 0, 0, 1, 0, 0, 0], dims=[1, 1, 0.02])
    goal = [{"predicate": "on", "args": ["loose_block", TABLE]}]

    env, _ = create_tamp_environment(meshes, table, goal, False, surface_labels={TABLE}, movable_labels=set())
    assert {m.name for m in env.movables} == {"loose_block"}


def test_binding_never_steals_a_label_the_plan_already_owns():
    # Same discipline as the drift path: folding two plan objects into one would aim the leg at
    # something the robot has already dealt with.
    spec = parse_jenga()
    geometry = {**JENGA_GEOMETRY, "screwdriver": ObjectGeometry((0.55, -0.06, 0.0045), (0.075, 0.025, 0.015))}
    assert bind_deferred_objects(spec, geometry, [*JENGA_OBJECTS], 1) == {}


@pytest.mark.parametrize(
    "response,expected",
    [
        (
            _jenga(new_objects=[{"name": "loose_block", "created_by_phase": 1, "description": "d"}]),
            "which is a ROBOT phase",
        ),
        (
            # Used by a phase that runs before it exists.
            {
                "new_objects": [{"name": "loose_block", "created_by_phase": 1, "description": "d"}],
                "phases": [
                    {"executor": "robot", "description": "early", "atoms": [{"predicate": "On", "args": ["loose_block", "table"]}]},
                    {"executor": "human", "description": "make it", "instructions": "do it",
                     "atoms": [{"predicate": "On", "args": ["loose_block", "white_paper"]}]},
                ],
            },
            "comes before phase 1 where it is created",
        ),
        (
            # The creating phase never says where it ends up, so nothing could ever find it.
            {
                "new_objects": [{"name": "loose_block", "created_by_phase": 0, "description": "d"}],
                "phases": [
                    {"executor": "human", "description": "make it", "instructions": "do it",
                     "atoms": [{"predicate": "On", "args": ["screwdriver", "white_paper"]}]},
                    {"executor": "robot", "description": "move it",
                     "atoms": [{"predicate": "On", "args": ["loose_block", "jenga_tower"]}]},
                ],
            },
            "exactly one On(loose_block, <a real object>) atom",
        ),
        (
            _jenga(new_objects=[{"name": "screwdriver", "created_by_phase": 0, "description": "d"}]),
            "already an object in the scene",
        ),
        (
            _jenga(new_objects=[{"name": "loose_block", "created_by_phase": 7, "description": "d"}]),
            "there are only 2 phase(s)",
        ),
        (
            # A new object used as a surface: it would become static world geometry for a task whose
            # first half it does not exist in.
            {
                "new_objects": [{"name": "loose_block", "created_by_phase": 0, "description": "d"}],
                "phases": [
                    {"executor": "human", "description": "make it", "instructions": "do it",
                     "atoms": [{"predicate": "On", "args": ["loose_block", "white_paper"]}]},
                    {"executor": "robot", "description": "stack on it",
                     "atoms": [{"predicate": "On", "args": ["screwdriver", "loose_block"]}]},
                ],
            },
            "cannot be used as a surface",
        ),
    ],
)
def test_new_object_rejections(response, expected):
    with pytest.raises(HITLProposalError, match=re.escape(expected)):
        parse_jenga(response)


def test_the_prompt_offers_new_objects_without_the_old_absolutes_forbidding_them():
    # The prompt's "no others" / "do not invent objects" lines are stated far more forcefully than any
    # new section, so they have to be scoped or they suppress the field outright.
    prompt = plan_prompt("remove a block and put it back", JENGA_OBJECTS)
    assert "new_objects" in prompt and "NEW OBJECTS" in prompt
    assert "exactly these objects RIGHT NOW" in prompt
    assert "or one you declared in `new_objects`" in prompt
    # The worked example has to survive the f-string as real JSON, not doubled braces.
    assert '[{"name": "loose_block", "created_by_phase": 0,' in prompt
    assert "}]" in prompt and "{{" not in prompt


def test_the_detection_hint_is_empty_until_a_plan_is_waiting_on_something():
    from tiptop.perception.gemini import extra_objects_section

    assert extra_objects_section(()) == ""
    section = extra_objects_section(["loose_block: a wooden block (should be on the white_paper)"])
    assert "loose_block" in section and "EXACTLY the name given here" in section


def test_nothing_binds_before_the_human_phase_that_creates_it_has_run():
    # A distractor already on the anchor -- a drawing on the paper, say -- must not be taken for the
    # object the human has not produced yet, which would aim the later leg at the wrong thing.
    session = _session(spec=parse_jenga())
    geometry = {**JENGA_GEOMETRY, "printed_diagram": ObjectGeometry((0.55, -0.06, -0.008), (0.09, 0.06, 0.001))}
    session.bind_new_objects(geometry, [*JENGA_OBJECTS, "printed_diagram"])
    assert session.spec.scene_types.deferred == frozenset({"loose_block"}), "still unbound at phase 0"
    session.advance()  # now the human has actually done it
    session.bind_new_objects(geometry, [*JENGA_OBJECTS, "printed_diagram"])
    assert session.spec.scene_types.deferred == frozenset()


def test_the_detector_returning_the_plans_own_name_still_has_to_pass_the_geometry():
    # The hint asks for the object BY NAME, so the label often comes back exactly as the plan spells
    # it. That must neither be refused out of hand (the object really is there) nor trusted on the
    # strength of the name (perception applies block-ish names to the wrong block).
    session = _session(spec=parse_jenga())
    session.advance()
    on_the_paper = {**JENGA_GEOMETRY, "loose_block": ObjectGeometry((0.55, -0.06, 0.0045), (0.075, 0.025, 0.015))}
    session.bind_new_objects(on_the_paper, [*JENGA_OBJECTS, "loose_block"])
    assert session.unbound_needed_now() == [], "the right object, correctly named, binds"
    assert session.goal_dicts() == [{"predicate": "on", "args": ["loose_block", "jenga_tower"]}]

    # Same name, but it is the brick still on top of the tower: refused.
    other = _session(spec=parse_jenga())
    other.advance()
    on_the_tower = {**JENGA_GEOMETRY, "loose_block": JENGA_GEOMETRY["top_jenga_block"]}
    other.bind_new_objects(on_the_tower, [*JENGA_OBJECTS, "loose_block"])
    assert other.unbound_needed_now() == ["loose_block"]


# --------------------------------------------------------------------------------------------
# The pre-rollout reset (tiptop_run._episode_reset_actions)
# --------------------------------------------------------------------------------------------


def test_a_leg_after_a_human_phase_keeps_the_arm_where_the_human_left_it():
    # The bug this guards: a HITL human phase used to hand back with the FULL reset re-armed, so the
    # robot drove home before replanning -- throwing away the continuation the trajectory is for and
    # making the next phase plan from q_home instead of the hand-off pose.
    from tiptop.tiptop_run import _episode_reset_actions

    for cam in ("external", "hand"):
        do_home, do_gripper, do_capture = _episode_reset_actions(
            resuming_from_handoff=False,
            resuming_after_human_phase=True,
            continuing_trajectory=True,
            perception_cam_key=cam,
        )
        assert not do_home, f"no homing after a human phase ({cam})"
        assert not do_capture, f"no drive to the capture pose after a human phase ({cam})"
        # Kept, unlike a plain hand-off resume: a person may leave the fingers closed, and cuTAMP
        # plans the next phase from a HandEmpty initial state either way.
        assert do_gripper, f"the gripper check still runs after a human phase ({cam})"


def test_a_teleop_handoff_resume_still_moves_the_arm_not_at_all():
    from tiptop.tiptop_run import _episode_reset_actions

    for cam in ("external", "hand"):
        assert _episode_reset_actions(
            resuming_from_handoff=True,
            resuming_after_human_phase=False,
            continuing_trajectory=True,
            perception_cam_key=cam,
        ) == (False, False, False), cam
    # A hand-off taken DURING a human phase is still a hand-off: the operator may be holding
    # something, so the full skip wins over the narrower one.
    assert _episode_reset_actions(
        resuming_from_handoff=True,
        resuming_after_human_phase=True,
        continuing_trajectory=True,
        perception_cam_key="external",
    ) == (False, False, False)


def test_a_robot_to_robot_continuation_does_not_drive_the_arm_home_either():
    # The HITL robot->robot leg used to take the FULL reset, so a task split into two cuTAMP goals
    # drove the arm home between them -- a return to home recorded into the middle of one episode,
    # with the second leg then planning from home rather than from the retract it was handed.
    from tiptop.tiptop_run import _episode_reset_actions

    for cam in ("external", "hand"):
        do_home, do_gripper, do_capture = _episode_reset_actions(
            resuming_from_handoff=False,
            resuming_after_human_phase=False,
            continuing_trajectory=True,
            perception_cam_key=cam,
        )
        assert not do_home, f"no homing between two legs of one trajectory ({cam})"
        # Unlike a human phase, this one DOES take the capture pose under wrist perception: that
        # move buys a guaranteed view of the table, and nothing about it is a reset.
        assert do_capture == (cam == "hand"), cam
        assert do_gripper, cam


def test_an_ordinary_rollout_still_gets_the_whole_reset():
    from tiptop.tiptop_run import _episode_reset_actions

    assert _episode_reset_actions(
        resuming_from_handoff=False,
        resuming_after_human_phase=False,
        continuing_trajectory=False,
        perception_cam_key="hand",
    ) == (True, True, True), "wrist perception drives to the capture pose"
    # A third-person camera is perceived from home instead -- an arm in frame lands in the point
    # cloud, the RANSAC table fit and the grasps.
    assert _episode_reset_actions(
        resuming_from_handoff=False,
        resuming_after_human_phase=False,
        continuing_trajectory=False,
        perception_cam_key="external",
    ) == (True, True, False)

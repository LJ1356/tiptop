"""Human-in-the-loop planning: proposal validation, phase plans, and what reaches TiPToP.

No robot, no GPU, no network -- the VLM's job here is to produce JSON, so the tests drive the parser
with the JSON directly. The running example is the task that forced the phase model:
"pick the toy off the box and place it on the table, open the box, place the toy inside the box".
"""

import asyncio
import json
from unittest import mock

import pytest

from tiptop.hitl import grounding, llm
from tiptop.hitl.config import HITLConfig, load_hitl_config, resolve_hitl_config
from tiptop.hitl.grounding import Verdict
from tiptop.hitl.planning import (
    match_drifted_names,
    check_robot_phases,
    goal_atoms_to_dicts,
    initial_state_for,
    unachievable_atoms,
)
from tiptop.hitl.proposal import parse_plan_response
from tiptop.hitl.session import HITLSession, handoff_message
from tiptop.hitl.structs import HITLProposalError, describe_atom, display_atom, display_name

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

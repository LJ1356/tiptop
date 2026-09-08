"""The VLM-written task plan (hitl.task_plan): the picks and places cuTAMP verifies.

The model is asked for the ORDER only. Everything else -- the MoveFree/MoveHolding alternation the
domain forces, and the q/traj/grasp/pose symbols particle initialization insists on -- is minted by
the expander, so most of what is worth testing here is that the expansion is indistinguishable from
what the search would have produced, and that every rejection is one the model can act on.

No robot, no GPU, no network: the expansion and both gates are pure symbolic code, and the VLM is
faked exactly as test_hitl.py fakes it.

Run with: pytest tests/test_task_plan.py -v
"""

import asyncio
import json
from unittest import mock

import pytest
from cutamp.tamp_domain import (
    HandEmpty,
    Holding,
    MoveFree,
    MoveHolding,
    On,
    Pick,
    Place,
    all_tamp_operators,
    get_initial_state,
)
from cutamp.task_planning import task_plan_generator

from tiptop.hitl import llm
from tiptop.hitl.config import HITLConfig
from tiptop.hitl.prompts import task_plan_prompt
from tiptop.hitl.structs import HITLProposalError
from tiptop.hitl.task_plan import (
    describe_steps,
    expand_task_plan,
    explain_rejection,
    parse_task_plan_response,
    propose_task_plan,
    structural_rejection,
    task_plan_rejection,
)
from tiptop.planning import skeleton_reuse_rejection

MOVABLES = ["block", "cup"]
SURFACES = ["table", "tray"]
INITIAL = get_initial_state(movables=MOVABLES, surfaces=SURFACES)
# What create_tamp_environment builds for a two-clause robot leg: the atoms, plus the HandEmpty it
# adds itself whenever the goal has no holding().
GOAL = frozenset({On.ground("block", "tray"), On.ground("cup", "tray"), HandEmpty.ground()})
CFG = HITLConfig(enabled=True)


def _response(*actions, **extra):
    """('pick', 'cup'), ('place', 'cup', 'tray') -> the JSON shape the model returns."""
    steps = [
        {"action": "pick", "object": a[1]} if a[0] == "pick"
        else {"action": "place", "object": a[1], "surface": a[2]}
        for a in actions
    ]
    return {"reasoning": "because", "steps": steps, **extra}


def _steps(*actions):
    return parse_task_plan_response(_response(*actions), MOVABLES, SURFACES)


def _skeleton(*actions):
    return expand_task_plan(_steps(*actions))


def _goal_rejection(goal=GOAL, initial=INITIAL):
    return lambda skeleton: skeleton_reuse_rejection(skeleton, initial, goal)


# --- the expansion --------------------------------------------------------------------------------


def test_the_expansion_is_the_skeleton_the_search_would_have_found():
    # The load-bearing claim of the whole feature. TAMPOperator.ground is memoized on (operator name,
    # substitutions), so if the expansion is right these are not merely equal to what BFS yields --
    # they are the SAME interned objects run_cutamp would have been handed. That is what makes
    # "cuTAMP verifies the model's plan" mean the same thing as "cuTAMP solves its own plan".
    hand = _skeleton(("pick", "cup"), ("place", "cup", "tray"), ("pick", "block"), ("place", "block", "tray"))
    names = [op.name for op in hand]
    searched = next(
        sk for sk in task_plan_generator(INITIAL, GOAL, all_tamp_operators) if [o.name for o in sk] == names
    )
    assert all(a is b for a, b in zip(hand, searched)), "the expansion must be what the search would produce"


def test_the_expansion_is_the_forced_alternation_with_the_search_s_own_symbols():
    assert [op.name for op in _skeleton(("pick", "cup"), ("place", "cup", "tray"))] == [
        "MoveFree(q0, traj1, q1)",
        "Pick(cup, grasp1, q1)",
        "MoveHolding(cup, grasp1, q1, traj2, q2)",
        "Place(cup, grasp1, pose1, tray, q2)",
    ]


def test_a_second_pick_gets_its_own_grasp_and_carries_on_from_where_the_first_left_off():
    ops = [op.name for op in _skeleton(
        ("pick", "cup"), ("place", "cup", "tray"), ("pick", "block"), ("place", "block", "table")
    )]
    assert ops[4:] == [
        "MoveFree(q2, traj3, q3)",
        "Pick(block, grasp2, q3)",
        "MoveHolding(block, grasp2, q3, traj4, q4)",
        "Place(block, grasp2, pose2, table, q4)",
    ]


def test_the_expansion_satisfies_the_invariants_cutamp_asserts_about_a_skeleton():
    # Reproduces rollout.py's `conf_params[0] == "q0"` and cost_function's conf equalities, which
    # otherwise only fail on the GPU with the arm warm.
    assert structural_rejection(_skeleton(("pick", "cup"), ("place", "cup", "tray"))) is None
    assert structural_rejection(_skeleton(("pick", "cup"))) is None


def test_a_skeleton_that_ends_on_a_move_is_refused_before_the_gpu():
    # Symbolically legal -- skeleton_reuse_rejection returns None for it -- and fatal downstream: the
    # move's q_end is deferred and nothing ever solves for it. The expander cannot produce one; this
    # is the net under the expander, and the reason the structural gate exists at all.
    goal = frozenset({On.ground("cup", "tray"), HandEmpty.ground()})
    trailing = list(_skeleton(("pick", "cup"), ("place", "cup", "tray")))
    trailing.append(MoveFree.ground({"q_start": "q2", "traj": "traj3", "q_end": "q3"}))
    assert skeleton_reuse_rejection(trailing, INITIAL, goal) is None, "the symbolic check does not catch this"
    assert "ends with a move" in structural_rejection(trailing)


def test_an_empty_plan_is_refused():
    assert "empty" in structural_rejection([])


def _malformed(kind):
    """A skeleton ``expand_task_plan`` cannot produce, one per invariant the structural gate mirrors.

    Hand-built rather than expanded, precisely because the expander is what makes each of these
    impossible -- so without them the gate's branches are unexercised and could rot into no-ops.
    """
    move1 = MoveFree.ground({"q_start": "q0", "traj": "traj1", "q_end": "q1"})
    pick = Pick.ground({"obj": "cup", "grasp": "grasp1", "q": "q1"})
    move2 = MoveHolding.ground({"obj": "cup", "grasp": "grasp1", "q_start": "q1", "traj": "traj2", "q_end": "q2"})
    place = Place.ground({"obj": "cup", "grasp": "grasp1", "placement": "pose1", "surface": "tray", "q": "q2"})
    if kind == "first conf is not q0":
        # The one symbol particle initialization seeds with a value; rollout.py asserts it is first.
        return [
            MoveFree.ground({"q_start": "q9", "traj": "traj1", "q_end": "q1"}), pick, move2, place
        ]
    if kind == "action skips the move's conf":
        # The move parks the arm at q1 and the pick reaches from q5, which nothing ever solves for.
        return [move1, Pick.ground({"obj": "cup", "grasp": "grasp1", "q": "q5"}), move2, place]
    if kind == "move starts from nowhere":
        # The second move does not carry on from where the pick happened.
        return [
            move1, pick,
            MoveHolding.ground({"obj": "cup", "grasp": "grasp1", "q_start": "q7", "traj": "traj2", "q_end": "q2"}),
            place,
        ]
    if kind == "symbol means two things":
        # `traj1` used as both a motion and the placement pose. Built so every OTHER invariant still
        # holds -- the configurations line up exactly -- or an earlier check would answer first and
        # this branch would never be reached.
        return [
            move1, pick, move2,
            Place.ground({"obj": "cup", "grasp": "grasp1", "placement": "traj1", "surface": "tray", "q": "q2"}),
        ]
    assert kind == "two moves share one motion"
    return [move1, pick, MoveHolding.ground(
        {"obj": "cup", "grasp": "grasp1", "q_start": "q1", "traj": "traj1", "q_end": "q2"}
    ), place]


@pytest.mark.parametrize(
    "kind, expected",
    [
        ("first conf is not q0", "the first configuration is 'q9'"),
        ("action skips the move's conf", "the picks and places use configurations"),
        ("move starts from nowhere", "the picks and places use configurations"),
        ("symbol means two things", "traj1 stands for two different things"),
        ("two moves share one motion", "two moves share one motion"),
    ],
)
def test_every_shape_cutamp_would_refuse_is_refused_here_instead(kind, expected):
    # Each of these otherwise surfaces as an AssertionError in cuTAMP's RolloutFunction or a
    # RuntimeError in its CostFunction -- on the GPU, after the world has been built, with the arm
    # warm. The gate is only worth having if every branch of it works, so every branch has a case --
    # and the expected MESSAGE is asserted, not merely that something was refused, or a case that
    # silently starts answering from an earlier check would leave its own branch untested.
    assert expected in structural_rejection(_malformed(kind))


# --- the goal gate: what stands between a wrong plan and the arm ------------------------------------


@pytest.mark.parametrize(
    "actions, expected",
    [
        # The wrong surface. cuTAMP never checks a skeleton it is handed against the goal -- its own
        # docstring says so -- so without this gate a plan of this shape is optimized, motion-planned,
        # executed and reported as a success.
        ((("pick", "cup"), ("place", "cup", "table"), ("pick", "block"), ("place", "block", "tray")),
         "does not reach this goal"),
        # Stops half way through the leg's goal.
        ((("pick", "cup"), ("place", "cup", "tray")), "does not reach this goal"),
        # Moves one object twice: cuTAMP's Pick requires and deletes HasNotPickedUp, so one skeleton
        # picks each object up at most once.
        ((("pick", "cup"), ("place", "cup", "tray"), ("pick", "cup"), ("place", "cup", "table")),
         "only once"),
    ],
)
def test_a_plan_that_does_not_solve_the_problem_is_rejected_in_words_the_model_can_use(actions, expected):
    steps = _steps(*actions)
    skeleton = expand_task_plan(steps)
    why = task_plan_rejection(skeleton, steps, _goal_rejection())
    assert why is not None and expected in why
    # Nothing the proposer has never been shown may appear in what it is sent back. Told about
    # `grasp2` or `MoveHolding` it starts writing plans over the alternation lock, which is exactly
    # what prompts.py refuses to let it see.
    assert not any(sym in why for sym in ("grasp1", "grasp2", "traj1", "q0", "q3", "MoveFree", "MoveHolding"))


def test_a_rejection_names_the_model_s_own_steps():
    steps = _steps(("pick", "cup"), ("place", "cup", "tray"), ("pick", "cup"), ("place", "cup", "table"))
    why = task_plan_rejection(expand_task_plan(steps), steps, _goal_rejection())
    assert "step 3 (pick(cup))" in why


def test_a_plan_that_solves_the_problem_is_accepted():
    steps = _steps(("pick", "cup"), ("place", "cup", "tray"), ("pick", "block"), ("place", "block", "tray"))
    assert task_plan_rejection(expand_task_plan(steps), steps, _goal_rejection()) is None


def test_a_holding_goal_accepts_a_plan_that_ends_on_a_pick():
    # create_tamp_environment adds HandEmpty only when the goal has no holding(), so a leg that ends
    # mid-manipulation is legitimate -- and the expansion must not append a trailing move for it.
    steps = _steps(("pick", "cup"), ("place", "cup", "tray"), ("pick", "block"))
    goal = frozenset({On.ground("cup", "tray"), Holding.ground("block")})
    skeleton = expand_task_plan(steps)
    assert task_plan_rejection(skeleton, steps, _goal_rejection(goal=goal)) is None


def test_a_plan_over_an_object_this_scene_no_longer_has_is_rejected():
    steps = _steps(("pick", "cup"), ("place", "cup", "tray"), ("pick", "block"), ("place", "block", "tray"))
    fewer = get_initial_state(movables=["cup"], surfaces=SURFACES)
    why = task_plan_rejection(expand_task_plan(steps), steps, _goal_rejection(initial=fewer))
    assert why is not None and "block" in why


def test_explain_rejection_passes_through_a_message_it_cannot_improve():
    assert explain_rejection("the cached task plan is empty", [], []) == "the cached task plan is empty"


# --- the parser: everything checkable before a skeleton exists --------------------------------------


@pytest.mark.parametrize(
    "actions, expected",
    [
        ((("pick", "spoon"), ("place", "spoon", "tray")), "not an object the robot can pick up"),
        ((("place", "cup", "tray"),), "not holding cup"),
        ((("pick", "cup"), ("pick", "block")), "still holding cup"),
        ((("pick", "cup"), ("place", "cup", "shelf")), "not a surface"),
    ],
)
def test_the_parser_rejects_what_it_can_before_a_skeleton_exists(actions, expected):
    with pytest.raises(HITLProposalError, match=expected):
        parse_task_plan_response(_response(*actions), MOVABLES, SURFACES)


@pytest.mark.parametrize(
    "data, expected",
    [
        ([1, 2], "Expected a JSON object"),
        ({"steps": "pick the cup"}, "'steps' must be a list"),
        ({"steps": ["pick the cup"]}, "must be a JSON object"),
        ({"steps": [{"action": "throw", "object": "cup"}]}, "must be 'pick' or 'place'"),
    ],
)
def test_the_parser_rejects_a_malformed_response(data, expected):
    with pytest.raises(HITLProposalError, match=expected):
        parse_task_plan_response(data, MOVABLES, SURFACES)


def test_describe_steps_reads_as_the_model_wrote_it():
    assert describe_steps(_steps(("pick", "cup"), ("place", "cup", "tray"))) == ["pick(cup)", "place(cup, tray)"]


# --- the prompt -----------------------------------------------------------------------------------


def test_the_prompt_offers_only_this_scene_and_no_cutamp_vocabulary():
    prompt = task_plan_prompt(
        ["On(cup, tray)", "HandEmpty()"], ["put the cup on the tray"], MOVABLES, SURFACES
    )
    assert "On(cup, tray)" in prompt and "put the cup on the tray" in prompt
    for name in MOVABLES + SURFACES:
        assert f"- {name}" in prompt
    # The rule prompts.py exists to keep: a model shown the operator signatures writes plans over the
    # alternation lock, so none of it may appear here either.
    for leaked in ("MoveFree", "MoveHolding", "conf", "traj", "JustMoved", "CanMove", "HasNotPickedUp"):
        assert leaked not in prompt


# --- the call, with the VLM faked ------------------------------------------------------------------


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


def _propose(client, cfg=CFG, goal=GOAL):
    with mock.patch.object(llm, "gemini_client", lambda: client):
        return asyncio.run(
            propose_task_plan(
                image=None,
                goal_state=goal,
                movables=MOVABLES,
                surfaces=SURFACES,
                descriptions=["put them both on the tray"],
                cfg=cfg,
                goal_rejection=_goal_rejection(goal=goal),
            )
        )


def test_a_plan_that_misses_the_goal_is_reprompted_with_the_reason_and_repaired():
    # The mechanism this feature leans on. A wrong ORDER is the failure mode a model actually has,
    # and it is cheap to fix here -- one more call, before a GPU is touched.
    half = json.dumps(_response(("pick", "cup"), ("place", "cup", "tray")))
    whole = json.dumps(
        _response(("pick", "cup"), ("place", "cup", "tray"), ("pick", "block"), ("place", "block", "tray"))
    )
    client = _FakeGemini([half, whole])
    plan = _propose(client)
    assert len(client.prompts) == 2, "the second attempt should have been made"
    assert "does not reach this goal" in client.prompts[1], "the reprompt must say what was wrong"
    assert describe_steps(plan.steps) == ["pick(cup)", "place(cup, tray)", "pick(block)", "place(block, tray)"]
    assert not plan.declined


def test_a_plan_that_never_validates_raises_the_last_reason():
    bad = json.dumps(_response(("pick", "cup"), ("place", "cup", "table")))
    client = _FakeGemini([bad, bad, bad])
    with mock.patch.object(llm, "gemini_client", lambda: client):
        with pytest.raises(HITLProposalError, match="does not reach this goal"):
            _propose(client)
    assert len(client.prompts) == 3, "every attempt should have been used"


def test_the_model_may_decline_and_is_not_reprompted_for_it():
    # A model that has correctly worked out that picks and places cannot reach the goal is right, and
    # asking it twice more costs two calls and ends in the same place. The caller falls back.
    client = _FakeGemini([json.dumps({"reasoning": "the cup has to move twice", "steps": [],
                                      "problem": "the goal needs the cup moved twice"})])
    plan = _propose(client)
    assert plan.declined and "moved twice" in plan.problem
    assert len(client.prompts) == 1, "a refusal is an answer, not a rejected attempt"


def test_an_empty_plan_with_no_reason_given_is_reprompted():
    empty = json.dumps({"reasoning": "", "steps": []})
    whole = json.dumps(
        _response(("pick", "cup"), ("place", "cup", "tray"), ("pick", "block"), ("place", "block", "tray"))
    )
    client = _FakeGemini([empty, whole])
    plan = _propose(client)
    assert len(client.prompts) == 2 and not plan.declined


def test_the_call_is_bounded_in_wall_clock_time():
    # Every other failure here degrades to "plan the leg the old way", but a request that never
    # returns leaves the arm idle with nothing said. Neither the SDK client nor query_json has a
    # deadline of its own, so propose_task_plan imposes one.
    class _Hangs(_FakeGemini):
        async def generate_content(self, model, contents, config):
            await asyncio.sleep(10)

    with pytest.raises(asyncio.TimeoutError):
        _propose(_Hangs([]), cfg=HITLConfig(enabled=True, task_plan_timeout_s=0.05))


# --- how cuTAMP's answer is read back ---------------------------------------------------------------


# Every string cuTAMP can hand back as a failure reason (cutamp/algorithm.py), formatted. This is what
# the verdict is read off, and the two marked ones carry TWO of the markers the classifier looks for --
# which is why it tests specific before general. Copied here so a change to either side shows up as a
# failing test rather than as a run's worth of mislabelled records.
_CUTAMP_FAILURES = [
    ("Motion planning failed for 32/40 satisfying particle(s)", "no_motion_plan"),
    ("Motion planning failed for 4/4 satisfying particle(s) (max attempts reached)", "no_motion_plan"),
    # Both "Motion planning failed" and "satisfying particles":
    ("Motion planning failed for all skeletons with satisfying particles", "no_motion_plan"),
    ("All 1 plan skeleton(s) failed particle initialization", "particle_init_failed"),
    ("No valid plan skeletons found for the given goal", "no_plan"),
    # Both "time budget" and "satisfying particles":
    ("No satisfying particles found after optimizing 0/1 plan(s) (time budget 60s exceeded)", "timed_out"),
    ("No satisfying particles found after optimizing all 1 plan(s)", "no_satisfying_particles"),
]


@pytest.mark.parametrize("reason, expected", _CUTAMP_FAILURES)
def test_every_way_cutamp_can_refuse_the_plan_is_read_back_as_its_own_verdict(reason, expected):
    from tiptop.tiptop_run import _hitl_task_plan_verdict

    assert _hitl_task_plan_verdict({"supplied_plan_failure": reason}) == expected


def test_a_plan_cutamp_solved_is_verified_and_one_it_refused_up_front_says_so():
    from tiptop.tiptop_run import _hitl_task_plan_verdict

    assert _hitl_task_plan_verdict({"reused": True}) == "verified"
    assert _hitl_task_plan_verdict({"rejection": "it does not reach this goal"}) == "rejected_symbolically"
    assert _hitl_task_plan_verdict({}) == "no_plan"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))

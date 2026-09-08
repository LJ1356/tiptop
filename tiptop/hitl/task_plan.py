"""The task-plan stage: a VLM writes the robot leg's picks and places, and cuTAMP verifies them.

Until now a robot phase was handed to cuTAMP as a GOAL and cuTAMP's breadth-first search enumerated
the operator sequences that reach it. That search sees object NAMES and nothing else -- cuTAMP's
symbolic initial state carries no ``On`` atom at all (see planning.initial_state_for) -- so it cannot
know that the lid is closing the box the toy has to go into, and it enumerates the orders that put
the toy in first alongside the ones that do not. It is the continuous layer, several seconds and one
GPU later, that discovers which of them the world admits.

The order is a question about the picture, so it is asked of something that can see the picture. The
model is asked for the ORDER and nothing else -- ``pick(lid)``, ``place(lid, table)``, ``pick(toy)``,
``place(toy, box)``. It is never shown cuTAMP's operators or its ``conf``/``traj``/``grasp`` symbols,
and :func:`expand_task_plan` is what makes that possible: the domain FORCES a ``MoveFree`` before
every ``Pick`` and a ``MoveHolding`` before every ``Place`` (the ``At``/``CanMove``/``JustMoved``
alternation lock in cuTAMP's tamp_domain), and particle initialization requires each move's ``q_end``
to be the very configuration the action after it uses. None of that is a decision, so none of it is
asked for -- the same bargain prompts.py already strikes one level up.

cuTAMP then VERIFIES the plan rather than searching for one: the skeleton is handed to
``run_cutamp(reuse_plan_skeleton=...)``, which solves this scene's grasps, placements and
trajectories for that sequence and nothing else. "Verified" means it found a particle satisfying
every constraint AND a cuRobo trajectory through it.

Nothing here imports ``tiptop.planning`` -- that module pulls in cuRobo at import time, and is itself
imported at module scope by the websocket server, tiptop_h5 and the viz scripts. The one check that
lives there, ``skeleton_reuse_rejection``, arrives as a callback instead, so this module stays
importable (and testable) with nothing but cuTAMP's symbolic half.
"""

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Callable, Sequence

from cutamp.tamp_domain import MoveFree, MoveHolding, Pick, Place
from cutamp.task_planning import PlanSkeleton, State
from cutamp.task_planning.constraints import KinematicConstraint
from cutamp.task_planning.costs import TrajectoryLength
from cutamp.task_planning.search import FABRICABLE_TYPES
from PIL import Image

from tiptop.hitl.config import HITLConfig
from tiptop.hitl.llm import query_json
from tiptop.hitl.prompts import TASK_PLAN_SCHEMA, task_plan_prompt
from tiptop.hitl.structs import HITLProposalError, display_atom

_log = logging.getLogger(__name__)

# Unmet preconditions, in words the proposer has actually been given. Its whole vocabulary is
# pick/place and the three state predicates; a bare `HasNotPickedUp(cup)` names a fluent it has never
# seen, in an operator signature it has never seen, and a rejection it cannot read is a rejection the
# reprompt loop cannot repair.
_PRECONDITION_HINTS = {
    "HasNotPickedUp": "the robot may pick each object up only once in one plan, and this plan picks one up twice",
    "HandEmpty": "the gripper is not empty at that point in the plan",
    "HoldingWithGrasp": "the robot is not holding that object at that point in the plan",
    "Holding": "the robot is not holding that object at that point in the plan",
}


@dataclass(frozen=True)
class TaskPlan:
    """One VLM answer: the picks and places for this leg, already expanded and checked.

    ``problem`` is the model's own refusal -- "the goal needs the toy moved twice, which one plan
    cannot do". It is a legitimate answer, not a rejected one: reprompting a model that has correctly
    said the goal is unreachable only burns attempts, so it is returned rather than raised, with an
    empty ``skeleton``, and the caller falls back to the search.
    """

    steps: tuple[dict, ...] = ()
    skeleton: tuple = ()
    reasoning: str = ""
    problem: str = ""

    @property
    def declined(self) -> bool:
        return not self.skeleton


def parse_task_plan_response(data: Any, movables: Sequence[str], surfaces: Sequence[str]) -> list[dict]:
    """Validate a task-plan response into the normalized steps ``expand_task_plan`` consumes.

    Everything checkable without the scene's geometry is checked here, and every rejection is phrased
    for the model, because ``query_json`` feeds it straight back on the next attempt. The alternative
    to checking here is a KeyError inside the expander or a ValueError deep in particle
    initialization, several seconds later, with the arm warm and an operator watching.
    """
    if not isinstance(data, dict):
        raise HITLProposalError(f"Expected a JSON object, got {type(data).__name__}.")
    steps = data.get("steps")
    if steps is None:
        steps = []
    if not isinstance(steps, list):
        raise HITLProposalError(f"'steps' must be a list, got {type(steps).__name__}.")
    movable_set, surface_set = set(movables), set(surfaces)
    out: list[dict] = []
    held: str | None = None
    for entry in steps:
        if not isinstance(entry, dict):
            raise HITLProposalError(f"Each step must be a JSON object, got {entry!r}.")
        action = str(entry.get("action", "")).strip().lower()
        obj = str(entry.get("object", "")).strip()
        if action not in ("pick", "place"):
            raise HITLProposalError(f"A step's action must be 'pick' or 'place', got {action!r}.")
        if obj not in movable_set:
            # Also how an object typed out of Movable for this task is refused. A scene shared with a
            # person contains the person's things, and HITLSession.robot_movables demotes them to
            # static obstacles; naming one here is a plan cuTAMP could not ground.
            raise HITLProposalError(
                f"'{obj}' is not an object the robot can pick up here. "
                f"It can pick up: {', '.join(sorted(movable_set)) or '(nothing)'}."
            )
        if action == "pick":
            if held is not None:
                raise HITLProposalError(
                    f"The step pick({obj}) comes while the robot is still holding {held}. The gripper "
                    f"holds one object at a time, so put {held} down somewhere first."
                )
            held = obj
            out.append({"action": "pick", "object": obj})
        else:
            surface = str(entry.get("surface", "")).strip()
            if held != obj:
                raise HITLProposalError(
                    f"The step place({obj}, ...) comes when the robot is not holding {obj}. "
                    f"Every place must come straight after a pick of that same object."
                )
            if surface not in surface_set:
                raise HITLProposalError(
                    f"'{surface}' is not a surface things can be put down on here. "
                    f"Available surfaces: {', '.join(sorted(surface_set)) or '(none)'}."
                )
            held = None
            out.append({"action": "place", "object": obj, "surface": surface})
    return out


def expand_task_plan(steps: Sequence[dict]) -> PlanSkeleton:
    """The picks and places as the ground operators cuTAMP solves.

    Two operators per step, and the symbols are minted with the prefixes the search itself uses
    (``_FABRICABLE_TYPE_PREFIXES`` in cuTAMP's task_planning/search.py): ``q`` for configurations,
    then ``traj``, ``grasp``, ``pose``. Every requirement particle initialization imposes is
    discharged by construction:

    * the first ``q_start`` is the literal ``"q0"`` -- the only symbol the initializer seeds with a
      value, and the one ``At(q0)`` in ``get_initial_state``;
    * each move's fresh ``q_end`` is consumed as the ``q`` of the action right after it, so nothing
      is left in ``deferred_params`` at the end (the initializer raises if anything is);
    * one fresh grasp per pick, reused verbatim by that object's ``MoveHolding`` and ``Place``;
    * the ``grasp`` PREFIX is load-bearing rather than cosmetic: run_cutamp finds the M2T2
      confidences it ranks satisfying particles by with ``k.startswith("grasp")``, and silently stops
      ranking by them if it cannot;
    * the sequence never ends on a move. That is symbolically legal and fatal downstream -- the
      move's ``q_end`` is deferred and nothing ever solves for it, so cuTAMP's cost function raises.

    Note the numbering does NOT try to reproduce what a particular search run would have minted. It
    cannot: the search's counter is per-node and shared with every sibling operator it grounds, so
    adding an unrelated button to the scene renumbers a pure pick-and-place. Only the STRUCTURE has
    to match, and it does -- ``TAMPOperator.ground`` is memoized on (operator name, substitutions),
    so for a plan the search would also have found these come back as the identical interned objects.
    """
    skeleton: PlanSkeleton = []
    grasp_of: dict[str, str] = {}
    q_prev, n_move, n_grasp, n_pose = "q0", 0, 0, 0
    for step in steps:
        n_move += 1
        q, traj, obj = f"q{n_move}", f"traj{n_move}", step["object"]
        if step["action"] == "pick":
            n_grasp += 1
            grasp_of[obj] = grasp = f"grasp{n_grasp}"
            skeleton.append(MoveFree.ground({"q_start": q_prev, "traj": traj, "q_end": q}))
            skeleton.append(Pick.ground({"obj": obj, "grasp": grasp, "q": q}))
        else:
            n_pose += 1
            # parse_task_plan_response guarantees this object's pick came first, so the grasp exists.
            grasp = grasp_of[obj]
            skeleton.append(
                MoveHolding.ground({"obj": obj, "grasp": grasp, "q_start": q_prev, "traj": traj, "q_end": q})
            )
            skeleton.append(
                Place.ground(
                    {"obj": obj, "grasp": grasp, "placement": f"pose{n_pose}", "surface": step["surface"], "q": q}
                )
            )
        q_prev = q
    return skeleton


def _dedup(values: Sequence[str]) -> list[str]:
    """``values`` in order, first occurrence only."""
    seen: dict[str, None] = {}
    for value in values:
        seen.setdefault(value, None)
    return list(seen)


def structural_rejection(skeleton: PlanSkeleton) -> str | None:
    """Why cuTAMP would refuse this skeleton's SHAPE, or None if it would accept it.

    This is not a check on the model -- the model never sees a symbol, and ``expand_task_plan``
    cannot produce a skeleton that fails this. A failure here is a bug in the expander, and it is
    caught here because the alternative is an AssertionError inside cuTAMP's RolloutFunction or a
    RuntimeError inside its CostFunction, on the GPU, after the world has been built.

    Mirrors, in order, the three invariants cuTAMP asserts about a skeleton: the first configuration
    must be ``q0`` (rollout.py), the trajectory-length costs must start from it (cost_function.py),
    and the configurations the kinematic constraints name must be exactly the ones the rollout visits
    (cost_function's rollout validation, which is what a trailing move breaks).
    """
    if not skeleton:
        return "the task plan is empty"
    if skeleton[-1].operator.name in (MoveFree.name, MoveHolding.name):
        return "the plan ends with a move, which leaves a configuration nothing solves for"
    confs = _dedup([v for op in skeleton for v, p in zip(op.values, op.operator.parameters) if p.type == "conf"])
    if confs[0] != "q0":
        return f"the first configuration is {confs[0]!r}, not 'q0'"
    kinematic = [c.params[0] for op in skeleton for c in op.constraints if c.type == KinematicConstraint.type]
    lengths: list[str] = []
    for op in skeleton:
        for cost in op.costs:
            if cost.type == TrajectoryLength.type:
                lengths.extend([cost.params[0], cost.params[2]])
    if confs[1:] != kinematic:
        return f"the picks and places use configurations {kinematic}, but the plan visits {confs[1:]}"
    if confs != _dedup(lengths):
        return f"the moves visit configurations {_dedup(lengths)}, but the plan visits {confs}"
    types_by_symbol: dict[str, set[str]] = {}
    trajectories: list[str] = []
    for op in skeleton:
        for value, param in zip(op.values, op.operator.parameters):
            if param.type in FABRICABLE_TYPES:
                types_by_symbol.setdefault(value, set()).add(param.type)
                if param.type == "traj":
                    trajectories.append(value)
    clashing = sorted(symbol for symbol, kinds in types_by_symbol.items() if len(kinds) > 1)
    if clashing:
        return f"{', '.join(clashing)} stands for two different things"
    if len(trajectories) != len(set(trajectories)):
        return "two moves share one motion"
    return None


def describe_steps(steps: Sequence[dict]) -> list[str]:
    """The steps as the model wrote them, for a log line and for hitl.json."""
    return [
        f"pick({s['object']})" if s["action"] == "pick" else f"place({s['object']}, {s['surface']})"
        for s in steps
    ]


def explain_rejection(rejection: str, skeleton: PlanSkeleton, steps: Sequence[dict]) -> str:
    """A skeleton rejection, rewritten in the vocabulary the model was actually given.

    ``skeleton_reuse_rejection`` names ground operators (``Pick(cup, grasp2, q3)``) and domain
    fluents (``HasNotPickedUp``), neither of which the proposer has been shown -- and telling it
    about them now is exactly what the prompt refuses to do, because a model shown the operator
    signatures starts writing plans over the alternation lock. Two operators per step makes the
    mapping back to the model's own words arithmetic.
    """
    words = describe_steps(steps)
    for idx, op in enumerate(skeleton):
        if idx // 2 < len(words):
            rejection = rejection.replace(op.name, f"step {idx // 2 + 1} ({words[idx // 2]})")
    for fluent, hint in _PRECONDITION_HINTS.items():
        if fluent in rejection:
            return f"{rejection}. In plain terms: {hint}."
    return rejection


def task_plan_rejection(
    skeleton: PlanSkeleton,
    steps: Sequence[dict],
    goal_rejection: Callable[[PlanSkeleton], str | None],
) -> str | None:
    """Why this plan cannot be run here, phrased for the model, or None if it can.

    Two gates. ``structural_rejection`` is ours and should never fire. ``goal_rejection`` is
    ``planning.skeleton_reuse_rejection`` bound to this scene -- the same walk cuTAMP's own reuse
    door is gated on: every operator's preconditions hold in turn, and the goal is a subset of the
    state the last one leaves. It is the ONLY thing standing between a plausible-but-wrong plan and
    the arm, because ``run_cutamp`` never checks a skeleton it is handed against the goal (its own
    docstring says so): a plan that puts the toy on the wrong surface is optimized, motion-planned
    and executed, and reported as a success.
    """
    structural = structural_rejection(skeleton)
    if structural is not None:
        _log.error(f"HITL: the expanded task plan is malformed -- {structural}. This is a bug in expand_task_plan")
        return structural
    why = goal_rejection(skeleton)
    return None if why is None else explain_rejection(why, skeleton, steps)


async def propose_task_plan(
    image: Image.Image,
    goal_state: State,
    movables: Sequence[str],
    surfaces: Sequence[str],
    descriptions: Sequence[str],
    cfg: HITLConfig,
    goal_rejection: Callable[[PlanSkeleton], str | None],
    label: str = "robot steps",
) -> TaskPlan:
    """This leg's picks and places, already checked to solve this problem against this scene.

    ``goal_rejection`` is ``planning.skeleton_reuse_rejection`` bound to this environment's initial
    and goal states, passed in rather than imported so this module stays free of cuRobo. Running it
    INSIDE the parse callback is the whole design: a plan that reaches the wrong surface, stops half
    way, or moves one object twice comes back to the model as a sentence about its own steps, and the
    reprompt loop repairs it before a GPU is touched.

    No cache is passed, for the reason grounding never passes one (see cache.ProposalCache). The
    image here is of a table part-way through a task, and the fingerprint is deliberately
    noise-robust: two frames either side of the very rearrangement the order turns on can collide,
    and the answer to "what is in the way now" would then be the answer from before it was moved.

    The call is bounded in wall-clock time. Everything else in this package degrades gracefully when
    a model cannot be reached, but a request that simply never returns leaves the arm idle at a
    prompt nobody is watching, and neither the SDK client nor ``query_json`` sets a deadline.
    """
    goal = sorted(display_atom(atom) for atom in goal_state)
    prompt = task_plan_prompt(goal, list(descriptions), sorted(movables), sorted(surfaces))

    def parse(data: Any) -> TaskPlan:
        steps = parse_task_plan_response(data, movables, surfaces)
        reasoning = str((data or {}).get("reasoning") or "")
        problem = str((data or {}).get("problem") or "").strip()
        if not steps:
            if problem:
                # A model that has correctly worked out the goal is unreachable with picks and places
                # is right, and reprompting it three times to say so again costs three calls and ends
                # in the same place. Returned as an answer; the caller falls back to the search.
                return TaskPlan(reasoning=reasoning, problem=problem)
            raise HITLProposalError(
                "The plan is empty. Give the picks and places in `steps`, or say in `problem` why the "
                "goal cannot be reached with picks and places at all."
            )
        skeleton = expand_task_plan(steps)
        why = task_plan_rejection(skeleton, steps, goal_rejection)
        if why:
            raise HITLProposalError(why)
        return TaskPlan(steps=tuple(steps), skeleton=tuple(skeleton), reasoning=reasoning, problem=problem)

    plan = await asyncio.wait_for(
        query_json(
            prompt,
            parse,
            model=cfg.task_plan_model,
            schema=TASK_PLAN_SCHEMA,
            image=image,
            max_attempts=cfg.max_attempts,
            label=label,
            cache=None,
        ),
        timeout=cfg.task_plan_timeout_s,
    )
    if plan.declined:
        _log.warning(f"HITL: the model would not write a task plan for this leg -- {plan.problem}")
    else:
        _log.info(f"HITL task plan: {' -> '.join(describe_steps(plan.steps))}")
        if plan.reasoning:
            _log.info(f"HITL task plan reasoning: {plan.reasoning}")
    return plan

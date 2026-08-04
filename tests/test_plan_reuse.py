"""Tests for reusing a task plan across a teleop hand-off (planning.skeleton_reuse_rejection).

A rollout that hands the arm to a human keeps its cuTAMP plan skeleton, and the rollout that
resumes replans only the motion for that same skeleton. These cover the check that decides whether
that is still legitimate against the newly perceived scene.

Run with: pytest tests/test_plan_reuse.py -v
"""

import pytest
from cutamp.tamp_domain import HandEmpty, On, all_tamp_operators, get_initial_state
from cutamp.task_planning import task_plan_generator

from tiptop.planning import skeleton_reuse_rejection

MOVABLES = ["block", "cup"]
SURFACES = ["table", "tray"]


def _problem(movables=MOVABLES, surfaces=SURFACES):
    initial = get_initial_state(movables=movables, surfaces=surfaces)
    goal = frozenset({On.ground("block", "tray"), HandEmpty.ground()})
    return initial, goal


def _skeleton(initial, goal):
    """First skeleton the search yields -- what a rollout would have been following."""
    return next(iter(task_plan_generator(initial, goal, operators=all_tamp_operators)))


def test_same_scene_is_reusable():
    """The common case: perception saw the same objects again, so the plan still stands."""
    initial, goal = _problem()
    skeleton = _skeleton(initial, goal)
    assert skeleton, "expected the search to yield a non-empty skeleton"
    # Re-derived rather than reused, exactly as the resumed rollout does it from fresh perception.
    assert skeleton_reuse_rejection(skeleton, *_problem()) is None


def test_missing_object_is_rejected():
    """An object the plan needs went undetected this time -- reject rather than fail deep in cuTAMP."""
    initial, goal = _problem()
    skeleton = _skeleton(initial, goal)
    fewer_initial, _ = _problem(movables=["cup"])
    rejection = skeleton_reuse_rejection(skeleton, fewer_initial, goal)
    assert rejection is not None
    assert "block" in rejection


def test_different_goal_is_rejected():
    """Perception grounded the instruction differently, so the old plan no longer reaches the goal."""
    initial, goal = _problem()
    skeleton = _skeleton(initial, goal)
    other_goal = frozenset({On.ground("cup", "tray"), HandEmpty.ground()})
    rejection = skeleton_reuse_rejection(skeleton, initial, other_goal)
    assert rejection is not None
    assert "does not reach this goal" in rejection


def test_empty_skeleton_is_rejected():
    """Nothing to reuse. Notably cuTAMP reads a falsy skeleton as a failed subgraph, not as a plan."""
    initial, goal = _problem()
    assert skeleton_reuse_rejection([], initial, goal) is not None
    assert skeleton_reuse_rejection(None, initial, goal) is not None


def test_unmet_preconditions_are_rejected():
    """A plan whose first step cannot run from this initial state is not reusable."""
    initial, goal = _problem()
    skeleton = _skeleton(initial, goal)
    # Drop HandEmpty: the domain's pick requires it, so the plan stops being applicable.
    without_hand_empty = frozenset(atom for atom in initial if atom != HandEmpty.ground())
    rejection = skeleton_reuse_rejection(skeleton, without_hand_empty, goal)
    assert rejection is not None
    assert "preconditions" in rejection


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))

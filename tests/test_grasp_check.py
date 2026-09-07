"""Does an open-loop plan notice when the gripper closed on NOTHING?

A cuTAMP plan executes open-loop: it closes where the plan says to, then carries and releases
regardless of what is in the jaws. Measured over the shipped datasets
(analysis_dataset_diff/TELEOP_VS_APEX.md), 32% of picks in
`1_pp_toys_plate_vae_style_timing_apex_learned_posture_prpl` closed on empty air against 8% for
human teleop -- and every one shipped as a successful pick, because nothing checked.

    tiptop/.pixi/envs/default/bin/python -m pytest tiptop/tests/test_grasp_check.py -q
"""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from tiptop import execute_plan as ep

DT = 0.002
WPS = 40
N_OPS = 3


class _Arr:
    def __init__(self, a):
        self._a = a

    def cpu(self):
        return self

    def numpy(self):
        return self._a

    def __len__(self):
        return len(self._a)


class _Plan:
    def __init__(self, n):
        self.position = _Arr(np.zeros((n, 7)))
        self.velocity = _Arr(np.zeros((n, 7)))


class FakeQueue:
    """Stands in for _QueuedArm: accepts segments and reports arrival immediately."""

    available = True

    def __init__(self):
        self.events: list = []
        self._seq = 0

    def submit(self, waypoints, velocities, dt):
        self._seq += 1
        self.events.append(("submit", self._seq, time.monotonic()))
        return {"success": True, "seq": self._seq}

    def times(self):
        return {}

    def wait_arrival(self, seq, lead=0.0, timeout=180.0):
        return {"success": True}

    def wait_done(self, timeout=180.0):
        return {"success": True}

    def abort(self):
        self.events.append(("abort", None, time.monotonic()))

    def close(self):
        pass


class FakeClient:
    """Gripper whose post-close state is scripted per close event."""

    gripper_socket = object()  # _can_overlap_gripper -> True (Robotiq-style)

    def __init__(self, states: list[dict | None]):
        # states[i] is the settled state reported after the i-th CLOSE
        self._states = states
        self._closes = 0
        self._polls = 0
        self.fired: list[str] = []

    def close_gripper(self, **kw):
        self._closes += 1
        self.fired.append("close")
        return {"success": True}

    def open_gripper(self, **kw):
        self.fired.append("open")
        return {"success": True}

    def get_gripper_state(self):
        self._polls += 1
        moving = self._polls % 3 == 1
        idx = max(self._closes - 1, 0)
        base = self._states[idx] if idx < len(self._states) else self._states[-1]
        if base is None:
            return {}
        return {"state": {**base, "is_moving": moving}}


HELD_ROBOTIQ = {"width": 0.011, "is_grasped": True}
EMPTY_ROBOTIQ = {"width": 0.0008, "is_grasped": False}
HELD_WIDTH_ONLY = {"width": 0.011}
EMPTY_WIDTH_ONLY = {"width": 0.0008}


def _plan(n_ops=N_OPS):
    """[traj, close, traj, open] * n_ops -- the shape cuTAMP emits."""
    steps = []
    for i in range(n_ops):
        steps.append({"type": "trajectory", "plan": _Plan(WPS), "dt": DT, "label": f"Pick{i}"})
        steps.append({"type": "gripper", "action": "close", "label": f"Pick{i}"})
        steps.append({"type": "trajectory", "plan": _Plan(WPS), "dt": DT, "label": f"Place{i}"})
        steps.append({"type": "gripper", "action": "open", "label": f"Place{i}"})
    steps.append({"type": "trajectory", "plan": _Plan(WPS), "dt": DT, "label": "GoToInitial"})
    return steps


def _run(states, *, overlap=True, check=True, plan=None):
    q = FakeQueue()
    client = FakeClient(states)
    saved = (ep._QueuedArm, ep.GRIPPER_OVERLAP, ep.GRASP_CHECK)
    ep._QueuedArm = lambda: q
    ep.GRIPPER_OVERLAP = overlap
    ep.GRASP_CHECK = check
    try:
        ep.execute_cutamp_plan(plan or _plan(), client=client)
    finally:
        ep._QueuedArm, ep.GRIPPER_OVERLAP, ep.GRASP_CHECK = saved
    return client


# --------------------------------------------------------------------------------------------- #
# The detector itself                                                                            #
# --------------------------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "state, expected",
    [
        (EMPTY_ROBOTIQ, True),
        (HELD_ROBOTIQ, False),
        (EMPTY_WIDTH_ONLY, True),
        (HELD_WIDTH_ONLY, False),
        # is_grasped wins over width when both are present: the gripper's own object detection is
        # authoritative, and a thin object can sit below the width threshold while still be held.
        ({"width": 0.0008, "is_grasped": True}, False),
        ({"width": 0.030, "is_grasped": False}, True),
        # Undeterminable -> None, never a failure.
        (None, None),
        ({}, None),
        ({"is_moving": False}, None),
    ],
)
def test_grasp_verdict(state, expected):
    verdict, why = ep._grasp_is_empty(state)
    assert verdict is expected, why
    assert why


def test_width_threshold_is_the_validated_one():
    """2.5 mm == 0.97 on the recorded closedness convention 1 - width/GRIPPER_MAX_WIDTH.

    That is the cut validated 12/12 against wrist video; if either constant moves, the offline
    detector and this online one stop agreeing.
    """
    from tiptop.lerobot_capture import GRIPPER_MAX_WIDTH

    closedness = 1.0 - ep.EMPTY_GRASP_WIDTH_M / GRIPPER_MAX_WIDTH
    assert closedness == pytest.approx(0.97, abs=0.005)


# --------------------------------------------------------------------------------------------- #
# End to end through execute_cutamp_plan                                                         #
# --------------------------------------------------------------------------------------------- #
def test_all_grasps_held_runs_to_completion():
    client = _run([HELD_ROBOTIQ])
    assert client.fired == ["close", "open"] * N_OPS


def test_empty_grasp_fails_the_episode():
    with pytest.raises(ep.ExecutionFailure) as exc:
        _run([EMPTY_ROBOTIQ])
    assert "empty grasp" in str(exc.value)
    assert "Pick0" in str(exc.value)


def test_failure_names_the_step_that_actually_closed():
    """Detection lags by one gripper event on the overlapped path -- the message must still blame
    the close, not the open that happened to trigger the adjudication."""
    with pytest.raises(ep.ExecutionFailure) as exc:
        _run([HELD_ROBOTIQ, EMPTY_ROBOTIQ, HELD_ROBOTIQ])
    msg = str(exc.value)
    assert "Pick1" in msg and "Pick0" not in msg and "Place" not in msg


def test_a_trailing_close_is_still_adjudicated():
    """A plan whose last step is an empty close must not report a clean run."""
    plan = [
        {"type": "trajectory", "plan": _Plan(WPS), "dt": DT, "label": "Pick0"},
        {"type": "gripper", "action": "close", "label": "Pick0"},
    ]
    with pytest.raises(ep.ExecutionFailure):
        _run([EMPTY_ROBOTIQ], plan=plan)


def test_width_only_gripper_still_detects():
    """A controller with no gOBJ register (e.g. the YAM) falls back to the width test."""
    with pytest.raises(ep.ExecutionFailure):
        _run([EMPTY_WIDTH_ONLY])
    client = _run([HELD_WIDTH_ONLY])
    assert client.fired == ["close", "open"] * N_OPS


def test_unreadable_gripper_never_fails_an_episode():
    """No signal -> degrade to today's open-loop behaviour rather than failing everything."""
    client = _run([None])
    assert client.fired == ["close", "open"] * N_OPS


def test_check_can_be_disabled():
    client = _run([EMPTY_ROBOTIQ], check=False)
    assert client.fired == ["close", "open"] * N_OPS


def test_blocking_path_detects_too():
    """With overlap off the close is adjudicated inline, not at the next join."""
    with pytest.raises(ep.ExecutionFailure):
        _run([EMPTY_ROBOTIQ], overlap=False)
    client = _run([HELD_ROBOTIQ], overlap=False)
    assert client.fired == ["close", "open"] * N_OPS


def test_open_events_are_not_grasp_checked():
    """Only a close can be empty; a Place open reporting open jaws must not fail the episode."""
    client = _run([HELD_ROBOTIQ])
    assert client.fired.count("open") == N_OPS

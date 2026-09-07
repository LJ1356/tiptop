"""Does execute_cutamp_dual_plan preserve the ONE invariant a real handover cannot survive without?

A Handover's raw cuTAMP steps put the taker's ``close`` immediately before the giver's ``open`` (no
trajectory step between them, both arms already parked at the shared mid-air configuration --
``motion_solver.py``'s ``Handover`` branch). If those two gripper events ever fired concurrently or
out of order, the object would be unsupported for an instant -- dropped, not mishandled data. This is
the dual-arm analogue of ``test_execute_plan_ordering.py``: a fake queue + fake client exercise
``execute_cutamp_dual_plan``'s actual dispatch loop, not just the pieces in isolation.

    tiptop/.pixi/envs/default/bin/python -m pytest tiptop/tests/test_execute_plan_dual_ordering.py -q
"""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from tiptop import execute_plan as ep

DT = 0.002
WPS = 50  # -> 0.1 s of simulated motion per segment
SEG = DT * WPS
GRIPPER_SETTLE_S = 0.05  # how long the fake gripper reports "moving" before settling


class _Arr:
    """Minimal stand-in for the torch tensor a cuTAMP dual plan carries (12-wide)."""

    def __init__(self, a):
        self._a = a

    def cpu(self):
        return self

    def numpy(self):
        return self._a

    def __len__(self):
        return len(self._a)

    @property
    def position(self):
        return self


class _Plan:
    def __init__(self, n, dof=12):
        self.position = _Arr(np.zeros((n, dof)))
        self.velocity = _Arr(np.zeros((n, dof)))


class FakeDualQueue:
    """_DualQueuedArm, simulated: each arm's segments run back-to-back on a virtual clock."""

    available = True

    def __init__(self):
        self.events: list[tuple[str, object, float]] = []
        self._seq = 0
        self._end: dict[int, float] = {}
        self._busy_until = {"left": 0.0, "right": 0.0}
        self._lock = threading.Lock()

    def submit(self, waypoints12, velocities12, dt):
        with self._lock:
            seqs = {}
            for arm in ("left", "right"):
                self._seq += 1
                start = max(self._busy_until[arm], time.monotonic())
                self._busy_until[arm] = start + len(waypoints12) * dt
                self._end[self._seq] = self._busy_until[arm]
                self.events.append(("submit", (arm, self._seq), time.monotonic()))
                seqs[arm] = self._seq
            return seqs

    def wait_arrival(self, seqs, lead=0.0, timeout=180.0):
        out = {}
        for arm, seq in seqs.items():
            time.sleep(max(0.0, self._end[seq] - lead - time.monotonic()))
            with self._lock:
                self.events.append(("arrive", (arm, seq), time.monotonic()))
            out[arm] = {"success": True}
        return out

    def wait_done(self, timeout=180.0):
        time.sleep(max(0.0, max(self._busy_until.values()) - time.monotonic()))
        return {"success": True}

    def abort(self):
        self.events.append(("abort", None, time.monotonic()))

    def close(self):
        pass


class FakeDualClient:
    gripper_socket = object()

    def __init__(self, q: FakeDualQueue):
        self._q = q
        self._polls: dict[str, int] = {}

    def _fire(self, action, arm):
        self._q.events.append(("gripper_fire", (action, arm), time.monotonic()))
        return {"success": True}

    def open_gripper(self, arm=None, **kw):
        return self._fire("open", arm)

    def close_gripper(self, arm=None, **kw):
        return self._fire("close", arm)

    def get_gripper_state(self, arm=None):
        # Reports "moving" for GRIPPER_SETTLE_S after the most recent fire on that arm, then settled.
        fires = [t for kind, (a, ar), t in self._q.events if kind == "gripper_fire" and ar == arm]
        moving = bool(fires) and (time.monotonic() - fires[-1]) < GRIPPER_SETTLE_S
        state = {"width": 0.0, "is_grasped": True, "is_moving": moving}
        self._q.events.append(("gripper_poll", (arm, moving), time.monotonic()))
        return {"state": state}


def _traj(label, n=WPS):
    return {"type": "trajectory", "plan": _Plan(n), "dt": DT, "label": label}


def _gripper(label, action, arm=None, arms=None):
    step = {"type": "gripper", "label": label, "action": action}
    if arm is not None:
        step["arm"] = arm
    if arms is not None:
        step["arms"] = arms
    return step


def _handover_plan():
    """MoveFree, PickGiver(close left), MoveHoldingGiver, Handover(no gripper step itself), the
    taker-close/giver-open pair, MoveHoldingTaker, PlaceTaker(open right) -- the shape
    motion_solver.py's Handover branch actually emits (arm/arms per gripper_step())."""
    return [
        _traj("MoveFree"),
        _gripper("PickGiver", "close", arm="left"),
        _traj("MoveHoldingGiver"),
        _traj("Handover"),  # both arms reach the shared mid-air configuration
        _gripper("Handover_taker_close", "close", arm="right"),
        _gripper("Handover_giver_open", "open", arm="left"),
        _traj("MoveHoldingTaker"),
        _gripper("PlaceTaker", "open", arm="right"),
        _traj("GoToInitial"),
    ]


def _parallel_plan():
    """PickBoth/PlaceBoth shape: one gripper step, both hands, arms=["left","right"]."""
    return [
        _traj("MoveFree"),
        _gripper("PickBoth", "close", arms=["left", "right"]),
        _traj("MoveHoldingBoth"),
        _gripper("PlaceBoth", "open", arms=["left", "right"]),
        _traj("GoToInitial"),
    ]


@pytest.fixture
def run(monkeypatch):
    def _go(plan):
        q = FakeDualQueue()
        monkeypatch.setattr(ep, "_DualQueuedArm", lambda: q)
        timeline: list = []
        ep.execute_cutamp_dual_plan(plan, client=FakeDualClient(q), timeline=timeline)
        return q, timeline

    return _go


def test_step_arms_prefers_plural_then_singular_then_defaults_to_both():
    assert ep._step_arms({"arms": ["left", "right"]}) == ["left", "right"]
    assert ep._step_arms({"arm": "left", "arms": ["left"]}) == ["left"]
    assert ep._step_arms({"arm": "right"}) == ["right"]
    assert sorted(ep._step_arms({})) == ["left", "right"]


def test_handover_taker_close_settles_before_giver_open_fires(run):
    """The safety invariant: close(taker) must be observed settled BEFORE open(giver) is issued --
    not just ordered in the event log, but separated by at least the fake gripper's settle time."""
    q, _ = run(_handover_plan())

    fires = [(action, arm, t) for kind, (action, arm), t in q.events if kind == "gripper_fire"]
    taker_close = next(t for action, arm, t in fires if action == "close" and arm == "right")
    giver_open = next(t for action, arm, t in fires if action == "open" and arm == "left")

    assert giver_open > taker_close, "giver opened before (or simultaneously with) the taker's close"
    assert giver_open - taker_close >= GRIPPER_SETTLE_S * 0.8, (
        f"giver opened only {(giver_open - taker_close) * 1000:.1f}ms after the taker's close fired -- "
        "the open was not gated on the close actually settling"
    )


def test_handover_giver_open_never_overlaps_the_taker_close_actuation(run):
    """Stronger than ordering: the giver's open must not fire while the taker's gripper is still
    reporting is_moving=True -- i.e. no threaded/overlapped firing for this specific pair."""
    q, _ = run(_handover_plan())

    events = q.events
    close_idx = next(i for i, (k, v, _) in enumerate(events) if k == "gripper_fire" and v == ("close", "right"))
    open_idx = next(i for i, (k, v, _) in enumerate(events) if k == "gripper_fire" and v == ("open", "left"))
    assert close_idx < open_idx

    # Every poll of the taker's ("right") gripper between the two fires must show it settled
    # (is_moving False) by the time we get anywhere near the open -- i.e. execution actually waited.
    between = events[close_idx:open_idx]
    right_polls = [moving for kind, (arm, moving), _ in between if kind == "gripper_poll" and arm == "right"]
    assert right_polls, "no poll of the taker's gripper happened between close and open"
    assert right_polls[-1] is False, "the giver's open fired while the taker's gripper was still moving"


def test_parallel_pickboth_fires_both_arms_close_together(run):
    """A PickBoth/PlaceBoth-style step (arms=[left,right], one gripper event) has NO ordering
    constraint -- both hands should fire close in time to each other (threaded), unlike the handover
    pair above."""
    q, _ = run(_parallel_plan())

    fires = [(action, arm, t) for kind, (action, arm), t in q.events if kind == "gripper_fire"]
    close_left = next(t for action, arm, t in fires if action == "close" and arm == "left")
    close_right = next(t for action, arm, t in fires if action == "close" and arm == "right")
    assert abs(close_left - close_right) < GRIPPER_SETTLE_S, (
        "PickBoth's two hands should fire close together (threaded), not hard-sequenced like a handover"
    )


def test_every_trajectory_segment_submits_both_arms(run):
    q, _ = run(_handover_plan())
    submits = [arm for kind, (arm, _seq), _ in q.events if kind == "submit"]
    n_traj = sum(1 for s in _handover_plan() if s["type"] == "trajectory")
    assert submits.count("left") == n_traj
    assert submits.count("right") == n_traj


def test_timeline_has_one_entry_per_step_in_order(run):
    plan = _handover_plan()
    q, timeline = run(plan)
    assert len(timeline) == len(plan)
    stamps = [e["t_start"] for e in timeline]
    assert stamps == sorted(stamps), "timeline entries are out of order"


def test_dual_execution_requires_the_queue(monkeypatch):
    """No blocking fallback exists for dual execution -- it must refuse loudly, not silently drive
    only one arm."""
    unavailable = FakeDualQueue()
    unavailable.available = False
    monkeypatch.setattr(ep, "_DualQueuedArm", lambda: unavailable)
    with pytest.raises(ep.ExecutionFailure, match="trajectory queue"):
        ep.execute_cutamp_dual_plan(_handover_plan(), client=FakeDualClient(unavailable))

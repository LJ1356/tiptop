"""Does each gripper actually fire when the arm ARRIVES, or just when the segment was queued?

This is the regression test for the 2026-07-28 failure. Queueing made trajectory submission
asynchronous, which removed the synchronisation the blocking call had been providing for free: the
whole plan was accepted in ~6 s while the motion took 46 s, so all six gripper actuations happened
while the arm was still on the first segment. The arm never grasped anything.

A unit test on the streamer alone cannot see this -- the streamer behaved correctly. The property
lives in ``execute_cutamp_plan``'s ordering, so that is what is exercised here, against a fake queue
that simulates back-to-back execution and timestamps every event.

    tiptop/.pixi/envs/default/bin/python -m pytest tiptop/tests/test_execute_plan_ordering.py -q
"""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from tiptop import execute_plan as ep

DT = 0.002
WPS = 250                      # -> 0.5 s of simulated motion per segment
SEG = DT * WPS


class _Arr:
    """Minimal stand-in for the torch tensor a cuTAMP plan carries."""

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
    def __init__(self, n):
        self.position = _Arr(np.zeros((n, 7)))
        self.velocity = _Arr(np.zeros((n, 7)))


class FakeQueue:
    """_QueuedArm + shim, simulated: batches run back-to-back on a virtual clock."""

    available = True

    def __init__(self):
        self.events: list[tuple[str, object, float]] = []
        self._seq = 0
        self._end: dict[int, float] = {}
        self._busy_until = 0.0
        self._lock = threading.Lock()

    def submit(self, waypoints, velocities, dt):
        with self._lock:
            self._seq += 1
            start = max(self._busy_until, time.monotonic())
            self._busy_until = start + len(waypoints) * dt
            self._end[self._seq] = self._busy_until
            self.events.append(("submit", self._seq, time.monotonic()))
            return {"success": True, "seq": self._seq}

    def wait_arrival(self, seq, lead=0.0, timeout=180.0):
        time.sleep(max(0.0, self._end[seq] - lead - time.monotonic()))
        with self._lock:
            self.events.append(("arrive", seq, time.monotonic()))
        return {"success": True}

    def wait_done(self, timeout=180.0):
        time.sleep(max(0.0, self._busy_until - time.monotonic()))
        return {"success": True}

    def times(self):
        return {}

    def abort(self):
        self.events.append(("abort", None, time.monotonic()))

    def close(self):
        pass


class FakeClient:
    gripper_socket = object()          # makes _can_overlap_gripper True (Robotiq-style)

    def __init__(self, q: FakeQueue):
        self._q = q

    def _fire(self, action):
        self._q.events.append(("gripper", action, time.monotonic()))
        return {"success": True}

    def close_gripper(self, **kw):
        return self._fire("close")

    def open_gripper(self, **kw):
        return self._fire("open")

    def __init_state(self):
        pass

    def get_gripper_state(self):
        # Report moving once then settled, so _wait_for_gripper_settled returns in ~2 polls
        # (~40 ms). A gripper that never reports moving takes its 0.25 s fallback path, and six of
        # those would pace the whole loop -- masking whether the ARM is what spaces the actuations.
        self._polls = getattr(self, "_polls", 0) + 1
        moving = self._polls % 3 == 1
        return {"state": {"width": 0.0, "is_grasped": True, "is_moving": moving}}


def _plan(n_ops=3):
    """[traj, close, traj, open] * n_ops -- the shape cuTAMP emits."""
    steps = []
    for i in range(n_ops):
        steps.append({"type": "trajectory", "plan": _Plan(WPS), "dt": DT, "label": f"Pick{i}"})
        steps.append({"type": "gripper", "action": "close", "label": f"Pick{i}"})
        steps.append({"type": "trajectory", "plan": _Plan(WPS), "dt": DT, "label": f"Place{i}"})
        steps.append({"type": "gripper", "action": "open", "label": f"Place{i}"})
    steps.append({"type": "trajectory", "plan": _Plan(WPS), "dt": DT, "label": "GoToInitial"})
    return steps


@pytest.fixture
def run(monkeypatch):
    def _go(lead=0.0):
        q = FakeQueue()
        monkeypatch.setattr(ep, "_QueuedArm", lambda: q)
        monkeypatch.setattr(ep, "GRIPPER_LEAD_S", lead)
        timeline: list = []
        ep.execute_cutamp_plan(_plan(), client=FakeClient(q), timeline=timeline)
        return q, timeline
    return _go


def _fire_gaps(q):
    fires = [t for kind, _, t in q.events if kind == "gripper"]
    return [b - a for a, b in zip(fires[:-1], fires[1:])]


def test_consecutive_grippers_are_one_segment_apart(run):
    """The 2026-07-28 signature: all six actuations inside a few seconds of a 46 s episode.

    The plan alternates trajectory/gripper, so consecutive actuations must be separated by roughly
    one segment of ARM MOTION. Firing at submit time instead collapses those gaps to the gripper's
    own actuation time, which is what this measures.
    """
    q, _ = run()
    assert len([t for kind, _, t in q.events if kind == "gripper"]) == 6
    gaps = _fire_gaps(q)
    assert min(gaps) > 0.5 * SEG, (
        f"consecutive gripper fires were {[round(g*1000) for g in gaps]} ms apart against a "
        f"{SEG*1000:.0f} ms segment -- they are firing at submit time, not on arrival"
    )


def test_each_gripper_fires_after_its_own_segment_arrives(run):
    q, _ = run()
    seq_arrived = 0
    for kind, val, _ in q.events:
        if kind == "arrive":
            seq_arrived = max(seq_arrived, int(val))
        elif kind == "gripper":
            # gripper k follows trajectory k, so by the k-th fire the k-th segment must be done
            assert seq_arrived >= 1, "a gripper fired before any segment had been reached"
    fires = [i for i, (k, _, _) in enumerate(q.events) if k == "gripper"]
    for n, idx in enumerate(fires, start=1):
        arrivals_before = sum(1 for k, _, _ in q.events[:idx] if k == "arrive")
        assert arrivals_before >= n, f"gripper #{n} fired after only {arrivals_before} arrivals"


def test_next_segment_is_queued_before_we_block_on_arrival(run):
    """The lookahead: without it, waiting for arrival re-introduces the stop we removed."""
    q, _ = run()
    for i, (kind, val, _) in enumerate(q.events):
        if kind != "arrive":
            continue
        submitted = {int(v) for k, v, _ in q.events[:i] if k == "submit"}
        nxt = int(val) + 1
        # every arrival except the plan's last segment must already have its successor queued
        if nxt <= max(int(v) for k, v, _ in q.events if k == "submit"):
            assert nxt in submitted, (
                f"segment {nxt} was not queued before the arm reached segment {val} -- the arm "
                f"would stall at that gripper event"
            )


def test_lead_fires_the_gripper_before_arrival(run):
    """A nonzero lead must actually move the fire earlier, not just be accepted."""
    q0, _ = run(lead=0.0)
    q1, _ = run(lead=SEG * 0.5)
    first_fire_offset = lambda q: (  # noqa: E731
        [t for k, _, t in q.events if k == "gripper"][0] - [t for k, _, t in q.events if k == "submit"][0]
    )
    assert first_fire_offset(q1) < first_fire_offset(q0) - SEG * 0.2


def test_timeline_has_one_entry_per_step_in_order(run):
    q, timeline = run()
    assert len(timeline) == len(_plan())
    stamps = [e["t_start"] for e in timeline]
    assert stamps == sorted(stamps), "timeline entries are out of order"


def test_these_checks_would_catch_the_2026_07_28_regression(monkeypatch):
    """Guard on the guards: with the arrival wait removed, the spread check must FAIL.

    The broken version differed from this one by exactly that wait -- everything else (queueing,
    lookahead, timeline) was already in place and looked fine. So a suite that still passes without
    it is not testing the thing that broke.
    """
    q = FakeQueue()
    monkeypatch.setattr(ep, "_QueuedArm", lambda: q)
    monkeypatch.setattr(FakeQueue, "wait_arrival",
                        lambda self, seq, lead=0.0, timeout=180.0: {"success": True})
    ep.execute_cutamp_plan(_plan(), client=FakeClient(q), timeline=[])

    gaps = _fire_gaps(q)
    assert min(gaps) <= 0.5 * SEG, (
        f"without the arrival wait the fires were still {[round(g*1000) for g in gaps]} ms apart "
        f"(segment {SEG*1000:.0f} ms), so test_consecutive_grippers_are_one_segment_apart cannot "
        f"detect the regression"
    )


# --- the teleop hand-off, against the QUEUED arm ---------------------------------------------- #
#
# should_stop is the SIGUSR1 checkpoint (tiptop_run._sigusr1_teleop_switch): the operator asks for
# the arm mid-plan, the current step finishes, and the rollout is saved as one leg of a hand-off
# trajectory that a human continues. Both properties below are about the SEAM between those legs.
#
# Queueing is what makes this subtle. _submit_through keeps _TRAJ_LOOKAHEAD segments in flight, so
# when a stop lands the shim is normally already streaming a segment the loop has not walked past --
# invisible to the loop, absent from the timeline, and with any gripper step before it never issued.


def _stop_after(k, plan=None, monkeypatch=None):
    """Run a plan, requesting a stop after the k-th should_stop poll. Returns (queue, timeline)."""
    plan = plan or _plan()
    q = FakeQueue()
    monkeypatch.setattr(ep, "_QueuedArm", lambda: q)
    monkeypatch.setattr(ep, "GRIPPER_LEAD_S", 0.0)
    timeline: list = []
    n = {"i": 0}

    def should_stop():
        n["i"] += 1
        return n["i"] > k

    ep.execute_cutamp_plan(plan, client=FakeClient(q), timeline=timeline, should_stop=should_stop)
    return q, timeline


@pytest.mark.parametrize("k", range(6))
def test_a_handoff_leaves_the_arm_where_the_recorded_episode_ends(k, monkeypatch):
    """Every segment the arm RUNS is one the episode recorded, at every stop point.

    lerobot_capture._flatten_plan keeps the finite-t_wall PREFIX of the plan, so a segment that runs
    without a timeline entry is motion the dataset does not contain: the robot leg ends at the pose
    of its last recorded frame while the human's leg starts a segment further on, and stitching the
    two (collect/merge_trajectory.py) shows an unexplained jump.
    """
    q, timeline = _stop_after(k, monkeypatch=monkeypatch)
    recorded = sum(1 for e in timeline if e["type"] == "trajectory")
    ran = sum(1 for e in q.events if e[0] == "submit")
    assert recorded == ran, (
        f"stopping after poll {k}: the arm ran {ran} segments but the episode recorded {recorded}. "
        f"The lookahead queued a segment the loop never walked past."
    )


@pytest.mark.parametrize("k", range(6))
def test_a_handoff_never_hands_over_mid_grasp(k, monkeypatch):
    """No segment the arm RUNS is missing a gripper command that belongs before it.

    Derived from what the arm did, not from what the timeline says, because the two coming apart is
    the defect. The failure this pins: a stop landing between a trajectory's submit and its poll used
    to break immediately, so the queued NEXT segment ran while the close that belongs before it never
    fired -- the arm diving to the grasp and lifting with open jaws, which is the mid-plan motion the
    checkpoint exists to prevent.
    """
    plan = _plan()
    q, _ = _stop_after(k, plan=plan, monkeypatch=monkeypatch)
    ran = sum(1 for e in q.events if e[0] == "submit")
    traj_idx = [i for i, s in enumerate(plan) if s["type"] == "trajectory"]
    end = traj_idx[ran - 1] if ran else 0        # plan index of the last segment actually executed
    want = [s["action"] for s in plan[:end] if s["type"] == "gripper"]
    fired = [e[1] for e in q.events if e[0] == "gripper"]
    assert fired == want, (
        f"stopping after poll {k}: the arm ran {ran} segments (through plan step {end}), which "
        f"needed the gripper to have fired {want}, but it fired {fired}"
    )

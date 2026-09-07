"""The shim's trajectory queue: does batch N+1 actually join batch N without a gap?

That gap is the whole point of the queue. A test that only checks "both batches ran" passes with
the gap still there, so the assertions here are on the INTERVAL BETWEEN SETPOINTS at the batch
seam, compared against the intervals inside a batch. The fake robot timestamps every setpoint.

Runs without a robot::

    tiptop/.pixi/envs/default/bin/python -m pytest tiptop/tests/test_trajectory_streamer.py -q
"""

from __future__ import annotations

import importlib.util
import statistics
import threading
import time
from pathlib import Path

import pytest

_SHIM = Path(__file__).resolve().parents[1] / "bamboo_polymetis_shim.py"
pytest.importorskip("zmq")
pytest.importorskip("msgpack")
pytest.importorskip("torch")

_spec = importlib.util.spec_from_file_location("bamboo_polymetis_shim", _SHIM)
shim = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(shim)

DT = 0.02          # matches the plan's control step
GRPC_LATENCY = 0.004   # the real update_desired_joint_positions is a ~5 ms round trip
# Starting/stopping a polymetis policy is NOT free -- it is a gRPC call that tears down and rebuilds
# the controller. The fake must charge for it, or a test of "does the seam have a gap" passes even
# with a policy restart sitting in the seam, which is the very thing being fixed. 30 ms is a
# conservative stand-in; the real cost is at least that.
POLICY_LATENCY = 0.030


class FakeRobot:
    """Records when each setpoint was issued, and the policy start/stop sequence."""

    def __init__(self, latency: float = GRPC_LATENCY):
        self.latency = latency
        self.stamps: list[float] = []
        self.events: list[str] = []
        self._lock = threading.Lock()

    def start_joint_impedance(self):
        time.sleep(POLICY_LATENCY)
        with self._lock:
            self.events.append("start")

    def terminate_current_policy(self):
        time.sleep(POLICY_LATENCY)
        with self._lock:
            self.events.append("stop")

    def update_desired_joint_positions(self, q):
        time.sleep(self.latency)
        with self._lock:
            self.stamps.append(time.monotonic())


def _batch(n: int, dt: float = DT):
    return [{"q_goal": [0.0] * 7, "velocity": [0.0] * 7, "duration": dt} for _ in range(n)]


def _intervals(stamps):
    return [b - a for a, b in zip(stamps[:-1], stamps[1:])]


def test_batches_join_without_a_gap():
    """The seam interval must look like any other interval -- that IS the fix."""
    robot = FakeRobot()
    s = shim._TrajectoryStreamer(robot)
    n = 25
    s.submit(_batch(n), DT, terminate_after=False)
    s.submit(_batch(n), DT, terminate_after=False)   # queued before the first finishes
    assert s.wait_idle(timeout=15.0)["success"]

    assert len(robot.stamps) == 2 * n
    iv = _intervals(robot.stamps)
    seam = iv[n - 1]                       # last setpoint of batch 1 -> first of batch 2
    inside = statistics.median(iv[:n - 1] + iv[n:])
    # A policy restart plus a ZMQ round trip costs tens of ms against a 20 ms step; require the
    # seam to sit within half a step of a normal interval.
    assert abs(seam - inside) < DT / 2, f"seam {seam*1000:.1f} ms vs inside {inside*1000:.1f} ms"
    # And the policy must have been started once and never dropped mid-plan.
    assert robot.events == ["start"], robot.events


def test_legacy_path_still_terminates_per_batch():
    """Without queue=True the behaviour must be exactly what it was: stream, then drop the policy."""
    robot = FakeRobot()
    s = shim._TrajectoryStreamer(robot)
    seq = s.submit(_batch(5), DT, terminate_after=True)
    assert s.wait_for(seq, timeout=10.0)["success"]
    assert robot.events == ["start", "stop"], robot.events


def test_post_idle_batch_does_not_burst():
    """A batch arriving after a pause must not race to catch up to a stale deadline clock."""
    robot = FakeRobot()
    s = shim._TrajectoryStreamer(robot)
    s.submit(_batch(5), DT, terminate_after=False)
    assert s.wait_idle(timeout=10.0)["success"]
    time.sleep(0.4)                        # deadline clock now well in the past
    mark = len(robot.stamps)
    s.submit(_batch(5), DT, terminate_after=False)
    assert s.wait_idle(timeout=10.0)["success"]
    iv = _intervals(robot.stamps[mark:])
    assert min(iv) > DT / 2, f"burst after idle: {[round(x*1000) for x in iv]} ms"


def test_abort_stops_the_batch_already_streaming():
    """Clearing the queue is not enough -- the worker has already popped the running batch."""
    robot = FakeRobot()
    s = shim._TrajectoryStreamer(robot)
    s.submit(_batch(200), DT, terminate_after=False)   # ~4 s of motion
    time.sleep(0.3)
    s.abort()
    time.sleep(0.2)
    stopped_at = len(robot.stamps)
    time.sleep(0.3)
    assert len(robot.stamps) == stopped_at, "setpoints kept flowing after abort"
    assert stopped_at < 200


def test_times_report_covers_each_batch():
    """The export re-anchors its timeline on these spans, so they must exist and be ordered."""
    robot = FakeRobot()
    s = shim._TrajectoryStreamer(robot)
    a = s.submit(_batch(10), DT, terminate_after=False)
    b = s.submit(_batch(10), DT, terminate_after=False)
    assert s.wait_idle(timeout=10.0)["success"]
    times = s.times()
    assert set(times) == {str(a), str(b)}
    for span in times.values():
        assert span[1] > span[0]
    assert times[str(b)][0] >= times[str(a)][0]
    assert s.times() == {}, "times() should drain on read"


def test_the_seam_check_would_catch_a_policy_restart():
    """Guard on the guard: with a per-batch policy restart the seam assertion must FAIL.

    Without this, a future change that makes policy start cheap-looking in the fake would turn
    test_batches_join_without_a_gap into a rubber stamp.
    """
    robot = FakeRobot()
    s = shim._TrajectoryStreamer(robot)
    n = 25
    s.submit(_batch(n), DT, terminate_after=True)    # the legacy behaviour: drop the policy here
    s.submit(_batch(n), DT, terminate_after=False)
    assert s.wait_idle(timeout=15.0)["success"]
    iv = _intervals(robot.stamps)
    seam = iv[n - 1]
    inside = statistics.median(iv[:n - 1] + iv[n:])
    assert abs(seam - inside) >= DT / 2, (
        f"a policy restart in the seam cost only {(seam-inside)*1000:.1f} ms -- the fake is not "
        f"charging enough for it, so the seam test cannot detect the regression"
    )

"""The YAM arm server's queue, routing and safety guards — without a robot.

``yam_arm_server`` imports i2rt lazily (inside ``YamArm.__init__``), so everything except the arm
itself is testable here with a fake. What is pinned:

* the queue actually joins batch N+1 onto batch N, which is the entire reason it exists — a cuTAMP
  plan is one batch per operation group, and a gap at the seam is a hard velocity stop at every
  gripper event, exactly what the non-idle training filter then deletes;
* sequence numbers are server-WIDE, so ``trajectory_times`` can be merged across both arms;
* the large-first-step guard, which is what stands between a stale plan and a violent motion on a
  stiff position-controlled arm.

Runs without hardware::

    tiptop/.pixi/envs/default/bin/python -m pytest tiptop/tests/test_yam_arm_server.py -q
"""

from __future__ import annotations

import importlib.util
import threading
import time
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("zmq")
pytest.importorskip("msgpack")

_SERVER = Path(__file__).resolve().parents[1] / "yam_arm_server.py"
_spec = importlib.util.spec_from_file_location("yam_arm_server", _SERVER)
server = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(server)

DT = 0.02  # the plan's control step


class FakeArm:
    """Stands in for YamArm: records every setpoint with the time it was issued."""

    def __init__(self, name: str = "left", start=None):
        self.name = name
        self._q = np.zeros(server.FULL_DOF)
        if start is not None:
            self._q[: server.ARM_DOF] = start
        self._q[server.ARM_DOF] = server.GRIPPER_OPEN
        self.setpoints: list[tuple[float, np.ndarray]] = []
        self.free_drive_calls: list[bool] = []
        self.engage_calls = 0
        self._lock = threading.Lock()

    def read_full(self) -> np.ndarray:
        with self._lock:
            return self._q.copy()

    def set_arm(self, q_arm) -> None:
        with self._lock:
            self._q[: server.ARM_DOF] = np.asarray(q_arm, dtype=float)[: server.ARM_DOF]
            self.setpoints.append((time.monotonic(), self._q.copy()))

    def set_gripper(self, value: float) -> None:
        with self._lock:
            self._q[server.ARM_DOF] = float(np.clip(value, 0.0, 1.0))

    @property
    def gripper_target(self) -> float:
        return float(self._q[server.ARM_DOF])

    def observation(self) -> dict:
        with self._lock:
            return {
                "q": self._q[: server.ARM_DOF].tolist(),
                "dq": [0.0] * server.ARM_DOF,
                "gripper": float(self._q[server.ARM_DOF]),
                "gripper_vel": 0.0,
                "gripper_eff": 0.0,
                "time_sec": time.time(),
            }

    def free_drive(self, hold_gripper: bool = True) -> None:
        self.free_drive_calls.append(hold_gripper)

    def engage(self) -> None:
        self.engage_calls += 1


def _waypoints(n: int, start: float = 0.0, step: float = 0.001) -> list[dict]:
    """A short dense batch, starting where the fake arm already is."""
    return [
        {"q_goal": [start + i * step] * server.ARM_DOF, "velocity": [0.0] * server.ARM_DOF, "duration": DT}
        for i in range(n)
    ]


def _counter():
    state = {"n": 0}

    def alloc():
        state["n"] += 1
        return state["n"]

    return alloc


# -- the queue -----------------------------------------------------------------------------------


def test_consecutive_batches_join_without_a_gap_at_the_seam():
    """The reason the queue exists. Assert on the INTERVAL between setpoints across the batch
    boundary, not merely that both batches ran — the latter passes with the gap still there."""
    arm = FakeArm()
    streamer = server.TrajectoryStreamer(arm, _counter())
    try:
        first = streamer.submit(_waypoints(15), DT)
        second = streamer.submit(_waypoints(15, start=0.015), DT)
        assert streamer.wait_for(second, timeout=10.0)["success"]
    finally:
        streamer.shutdown()

    times = [t for t, _ in arm.setpoints]
    assert len(times) == 30
    gaps = np.diff(times)
    seam = gaps[14]  # between the last setpoint of batch 1 and the first of batch 2
    within = np.median(np.concatenate([gaps[:14], gaps[15:]]))
    assert seam < within * 3, f"seam gap {seam:.4f}s dwarfs the in-batch interval {within:.4f}s"
    assert first < second


def test_the_deadline_resets_after_a_genuine_idle_instead_of_replaying_at_full_rate():
    """Without the reset, a batch arriving after a pause is streamed as fast as the loop can issue
    setpoints, trying to catch up to a stale clock — a burst of motion at full speed."""
    arm = FakeArm()
    streamer = server.TrajectoryStreamer(arm, _counter())
    try:
        seq = streamer.submit(_waypoints(6), DT)
        assert streamer.wait_for(seq, timeout=10.0)["success"]
        time.sleep(0.4)  # queue genuinely idles
        n_before = len(arm.setpoints)
        seq = streamer.submit(_waypoints(6, start=0.006), DT)
        assert streamer.wait_for(seq, timeout=10.0)["success"]
    finally:
        streamer.shutdown()

    second = [t for t, _ in arm.setpoints[n_before:]]
    assert len(second) == 6
    span = second[-1] - second[0]
    # 6 waypoints at 20 ms should take ~0.1 s, not be dumped instantly.
    assert span > 0.05, f"second batch replayed in {span:.4f}s — the deadline did not reset"


def test_abort_stops_a_batch_that_is_already_streaming():
    """Aborts bump an epoch rather than clearing the deque: the worker may already have popped the
    batch, so a cleared queue would not stop the arm."""
    arm = FakeArm()
    streamer = server.TrajectoryStreamer(arm, _counter())
    try:
        seq = streamer.submit(_waypoints(200), DT)  # ~4 s of motion
        time.sleep(0.15)
        streamer.abort()
        assert streamer.wait_for(seq, timeout=5.0)["success"]
    finally:
        streamer.shutdown()
    assert len(arm.setpoints) < 200, "abort did not cut the batch short"


def test_times_reports_the_real_span_and_drains_on_read():
    arm = FakeArm()
    streamer = server.TrajectoryStreamer(arm, _counter())
    try:
        seq = streamer.submit(_waypoints(10), DT)
        assert streamer.wait_for(seq, timeout=10.0)["success"]
        times = streamer.times()
    finally:
        streamer.shutdown()

    assert str(seq) in times
    start, end = times[str(seq)]
    assert end > start
    assert end - start < 5.0
    assert streamer.times() == {}, "times() must drain on read"


def test_wait_arrival_of_seq_zero_returns_immediately():
    """tiptop probes queue support with seq=0, which no batch owns; it must read as already done."""
    arm = FakeArm()
    streamer = server.TrajectoryStreamer(arm, _counter())
    try:
        assert streamer.wait_arrival(0, timeout=1.0)["success"]
    finally:
        streamer.shutdown()


# -- safety --------------------------------------------------------------------------------------


def test_a_trajectory_starting_far_from_the_arm_is_refused():
    """A planned trajectory always starts where the arm is. A large first step on an arm running
    kp 80 is a violent motion, so it is refused rather than clamped."""
    arm = FakeArm(start=np.zeros(server.ARM_DOF))
    streamer = server.TrajectoryStreamer(arm, _counter())
    try:
        far = server.MAX_START_JUMP_RAD + 0.5
        seq = streamer.submit(_waypoints(5, start=far), DT)
        result = streamer.wait_for(seq, timeout=5.0)
    finally:
        streamer.shutdown()

    assert not result["success"]
    assert "refusing" in result["error"]
    assert arm.setpoints == [], "no setpoint may be issued for a refused trajectory"


def test_a_refused_batch_is_reported_through_the_queued_waits_too():
    """The queued execution path never calls wait_for — only wait_arrival and wait_idle. If those
    answered "success" for a batch that was refused, tiptop would carry on to the next plan step and
    actuate the gripper at a pose the arm never reached."""
    arm = FakeArm(start=np.zeros(server.ARM_DOF))
    streamer = server.TrajectoryStreamer(arm, _counter())
    try:
        seq = streamer.submit(_waypoints(5, start=server.MAX_START_JUMP_RAD + 0.5), DT)
        arrival = streamer.wait_arrival(seq, timeout=5.0)
        idle = streamer.wait_idle(timeout=5.0)
    finally:
        streamer.shutdown()

    assert not arrival["success"], "wait_arrival reported a refused batch as arrived"
    assert "refusing" in arrival["error"]
    assert not idle["success"], "wait_idle reported a failed queue as cleanly drained"


def test_a_successful_batch_still_reports_success_through_both_waits():
    arm = FakeArm(start=np.zeros(server.ARM_DOF))
    streamer = server.TrajectoryStreamer(arm, _counter())
    try:
        seq = streamer.submit(_waypoints(5), DT)
        assert streamer.wait_arrival(seq, timeout=5.0)["success"]
        assert streamer.wait_idle(timeout=5.0)["success"]
    finally:
        streamer.shutdown()


def test_move_to_rejects_a_duration_that_would_defeat_the_jump_guard(two_arms):
    """move_to is the one motion path exempt from MAX_START_JUMP_RAD, because parking has to
    travel. A tiny duration would collapse that ramp into a full-distance step."""
    far = [server.MAX_START_JUMP_RAD + 1.0] * server.ARM_DOF
    res = two_arms.handle_control({"command": "move_to", "arm": "left", "q_goal": far, "duration": 0.01})
    assert not res["success"] and "duration" in res["error"]
    assert two_arms.arms["left"].setpoints == []


def test_a_trajectory_starting_at_the_arm_is_accepted():
    arm = FakeArm(start=np.full(server.ARM_DOF, 0.3))
    streamer = server.TrajectoryStreamer(arm, _counter())
    try:
        seq = streamer.submit(_waypoints(5, start=0.3), DT)
        assert streamer.wait_for(seq, timeout=5.0)["success"]
    finally:
        streamer.shutdown()
    assert len(arm.setpoints) == 5


# -- two-arm routing -----------------------------------------------------------------------------


class _TwoArmServer:
    """A YamServer with fake arms, built without touching YamServer.__init__ (which opens CAN)."""

    def __init__(self):
        self.arms = {"left": FakeArm("left"), "right": FakeArm("right")}
        self._seq_lock = threading.Lock()
        self._seq = 0
        self.streamers = {n: server.TrajectoryStreamer(a, self._next_seq) for n, a in self.arms.items()}
        self._active_arm = "left"

    _next_seq = server.YamServer._next_seq
    _arm = server.YamServer._arm
    _streamer = server.YamServer._streamer
    _streamer_for_seq = server.YamServer._streamer_for_seq
    handle_control = server.YamServer.handle_control
    handle_gripper = server.YamServer.handle_gripper
    _gripper_state = server.YamServer._gripper_state

    def close(self):
        for s in self.streamers.values():
            s.shutdown()


@pytest.fixture
def two_arms():
    s = _TwoArmServer()
    yield s
    s.close()


def test_sequence_numbers_are_unique_across_both_arms(two_arms):
    """trajectory_times is merged across arms and keyed by seq; per-arm counters would collide and
    one arm's spans would silently overwrite the other's."""
    left = two_arms.streamers["left"].submit(_waypoints(3), DT)
    right = two_arms.streamers["right"].submit(_waypoints(3), DT)
    assert left != right
    assert two_arms.streamers["left"].owns(left)
    assert not two_arms.streamers["left"].owns(right)
    assert two_arms.streamers["right"].owns(right)


def test_a_seq_addressed_command_routes_to_the_arm_that_issued_it(two_arms):
    right_seq = two_arms.streamers["right"].submit(_waypoints(3), DT)
    routed = two_arms._streamer_for_seq(right_seq, {})
    assert routed is two_arms.streamers["right"]
    # ... even though the ACTIVE arm is the left one.
    assert two_arms._active_arm == "left"


def test_an_arm_less_request_follows_the_active_arm(two_arms):
    assert two_arms._arm({}).name == "left"
    assert two_arms.handle_control({"command": "set_active_arm", "arm": "right"})["success"]
    assert two_arms._arm({}).name == "right"
    # An explicit arm still wins over the session default.
    assert two_arms._arm({"arm": "left"}).name == "left"


def test_set_active_arm_rejects_an_unknown_arm(two_arms):
    res = two_arms.handle_control({"command": "set_active_arm", "arm": "middle"})
    assert not res["success"]
    assert "unknown arm" in res["error"]


def test_unknown_commands_are_reported_not_raised(two_arms):
    res = two_arms.handle_control({"command": "fly"})
    assert not res["success"] and "Unknown control command" in res["error"]
    res = two_arms.handle_gripper({"command": "fly"})
    assert not res["success"] and "Unknown gripper command" in res["error"]


def test_execute_trajectory_rejects_an_empty_batch(two_arms):
    res = two_arms.handle_control({"command": "execute_trajectory", "data": {"arm": "left", "waypoints": []}})
    assert not res["success"] and "empty waypoints" in res["error"]


def test_queued_submit_returns_at_once_with_a_sequence(two_arms):
    res = two_arms.handle_control(
        {"command": "execute_trajectory", "data": {"arm": "left", "waypoints": _waypoints(50), "queue": True}}
    )
    assert res["success"] and res["queued"] and isinstance(res["seq"], int)


# -- gripper -------------------------------------------------------------------------------------


def test_gripper_state_reports_metres_and_the_i2rt_normalisation(two_arms):
    """tiptop's gripper plumbing speaks metres; i2rt speaks a [0,1] opening where 1.0 is OPEN."""
    arm = two_arms.arms["left"]
    arm.set_gripper(server.GRIPPER_OPEN)
    state = two_arms._gripper_state(arm)["state"]
    assert state["normalized"] == pytest.approx(1.0)
    assert state["width"] == pytest.approx(server.GRIPPER_STROKE_M)

    arm.set_gripper(server.GRIPPER_CLOSED)
    state = two_arms._gripper_state(arm)["state"]
    assert state["width"] == pytest.approx(0.0)


def test_open_and_close_drive_the_gripper_to_the_right_ends(two_arms):
    assert two_arms.handle_gripper({"command": "close_gripper", "arm": "left", "blocking": False})["success"]
    assert two_arms.arms["left"].gripper_target == pytest.approx(server.GRIPPER_CLOSED)
    assert two_arms.handle_gripper({"command": "open_gripper", "arm": "left", "blocking": False})["success"]
    assert two_arms.arms["left"].gripper_target == pytest.approx(server.GRIPPER_OPEN)


# -- free drive ----------------------------------------------------------------------------------


class FakeI2rtRobot:
    """The slice of i2rt's MotorChainRobot that ``YamArm.free_drive`` touches."""

    KP = np.array([80.0] * server.ARM_DOF + [40.0])
    KD = np.array([2.0] * server.ARM_DOF + [1.0])
    GRAV_COMP_KD = np.array([0.3] * server.ARM_DOF + [0.1])

    def __init__(self, measured):
        self.measured = np.asarray(measured, dtype=float)
        self.commands: list[dict] = []

    def get_robot_info(self) -> dict:
        return {"kp": self.KP, "kd": self.KD, "grav_comp_kd": self.GRAV_COMP_KD}

    def get_joint_pos(self) -> np.ndarray:
        return self.measured.copy()

    def command_joint_state(self, joint_state: dict) -> None:
        self.commands.append({k: np.asarray(v, dtype=float).copy() for k, v in joint_state.items()})


def _yam_arm_with(robot: FakeI2rtRobot, gripper_target: float) -> "server.YamArm":
    """A YamArm around a fake robot, bypassing __init__ (which opens CAN and calibrates the jaws)."""
    arm = object.__new__(server.YamArm)
    arm.name = "left"
    arm._robot = robot
    arm._lock = threading.Lock()
    arm._target = np.zeros(server.FULL_DOF)
    arm._target[server.ARM_DOF] = gripper_target
    arm._engaged = True
    return arm


def test_free_drive_limps_the_arm_but_keeps_the_gripper_clamped():
    """The whole point of free_drive over relax: a board clamped in the jaws must not be dropped.

    relax zeroes every joint's gains including the gripper's. Hand-guiding needs kp 0 on the six arm
    joints and the configured gains on joint 6, or the calibration board falls out of the gripper the
    moment the operator takes hold of the arm.
    """
    robot = FakeI2rtRobot(measured=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.55])
    arm = _yam_arm_with(robot, gripper_target=server.GRIPPER_CLOSED)

    arm.free_drive()

    (cmd,) = robot.commands
    assert np.allclose(cmd["kp"][: server.ARM_DOF], 0.0)
    assert np.allclose(cmd["kd"][: server.ARM_DOF], FakeI2rtRobot.GRAV_COMP_KD[: server.ARM_DOF])
    assert cmd["kp"][server.ARM_DOF] == pytest.approx(FakeI2rtRobot.KP[server.ARM_DOF])
    assert cmd["kd"][server.ARM_DOF] == pytest.approx(FakeI2rtRobot.KD[server.ARM_DOF])
    assert np.allclose(cmd["vel"], 0.0)
    assert not arm._engaged


def test_free_drive_commands_the_latched_gripper_target_not_the_measured_jaw_width():
    """A board holds the jaws further open than they were told to go. Re-commanding the MEASURED
    width would hand the gripper a target it has already reached, so it would stop squeezing and the
    board would slip; the commanded target is what keeps the clamping force on."""
    robot = FakeI2rtRobot(measured=[0.0] * server.ARM_DOF + [0.42])
    arm = _yam_arm_with(robot, gripper_target=server.GRIPPER_CLOSED)

    arm.free_drive()

    (cmd,) = robot.commands
    assert cmd["pos"][server.ARM_DOF] == pytest.approx(server.GRIPPER_CLOSED)
    # The arm's own target is re-latched to where it actually is, so re-engaging cannot step.
    assert np.allclose(cmd["pos"][: server.ARM_DOF], robot.measured[: server.ARM_DOF])


def test_free_drive_can_release_the_gripper_too():
    robot = FakeI2rtRobot(measured=np.zeros(server.FULL_DOF))
    arm = _yam_arm_with(robot, gripper_target=server.GRIPPER_CLOSED)

    arm.free_drive(hold_gripper=False)

    (cmd,) = robot.commands
    assert np.allclose(cmd["kp"], 0.0)


def test_free_drive_and_hold_route_to_the_named_arm(two_arms):
    assert two_arms.handle_control({"command": "free_drive", "arm": "right"})["success"]
    assert two_arms.arms["right"].free_drive_calls == [True]
    assert two_arms.arms["left"].free_drive_calls == []

    assert two_arms.handle_control({"command": "free_drive", "arm": "right", "hold_gripper": False})["success"]
    assert two_arms.arms["right"].free_drive_calls == [True, False]

    assert two_arms.handle_control({"command": "hold", "arm": "right"})["success"]
    assert two_arms.arms["right"].engage_calls == 1


def test_neutral_posture_matches_cutamps_locked_configuration():
    """The arm that is not planning is collision-checked at exactly this configuration. If these
    two ever drift apart, every plan is validated against a robot that is not there."""
    from cutamp.robots.bimanual_yam import bimanual_yam_neutral_joint_positions

    from tiptop.yam import NEUTRAL_Q

    assert np.allclose(server.NEUTRAL_Q, bimanual_yam_neutral_joint_positions)
    assert np.allclose(NEUTRAL_Q, bimanual_yam_neutral_joint_positions)

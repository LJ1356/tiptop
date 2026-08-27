"""Bimanual episode capture (tiptop.yam.capture).

What is pinned here is the contract ARCHITECTURE.md §3 states and the LeRobot export depends on:
14 numbers per frame in a known column order, MEASURED proprioception that is never a copy of the
command, a strictly BINARY commanded gripper, and a float64 wall-clock master timeline.
"""

import json

import numpy as np
import pytest

from tiptop.yam import ARM_DOF, BIMANUAL_ARM_DOF
from tiptop.yam.capture import (
    ARM_SLICES,
    GRIPPER_INDEX,
    Segment,
    _dense_bimanual,
    _flatten_dual_plan,
    dump_bimanual_episode,
    segment_from_motion,
    segments_from_dual_plan,
)

T0 = 1_780_000_000.0  # a realistic epoch, so float32 truncation would be visible


def _segment(arm: str, n: int, t_start: float, t_end: float, value: float, gripper: float = 0.0) -> Segment:
    return Segment(
        arm=arm,
        positions=np.full((n, ARM_DOF), value, dtype=np.float32),
        velocities=np.full((n, ARM_DOF), 0.5, dtype=np.float32),
        gripper=np.full(n, gripper, dtype=np.float32),
        t_wall=np.linspace(t_start, t_end, n),
    )


def _samples(t_start: float, t_end: float, n: int):
    """A measured trace that is deliberately DIFFERENT from any command, so a copy would show."""
    times = np.linspace(t_start, t_end, n)
    return [
        (
            float(t),
            np.full(BIMANUAL_ARM_DOF, -7.0, dtype=np.float32),
            np.full(BIMANUAL_ARM_DOF, -0.25, dtype=np.float32),
            np.array([0.37, 0.62], dtype=np.float32),
        )
        for t in times
    ]


def test_segment_rejects_a_bad_arm():
    with pytest.raises(ValueError, match="segment arm must be one of"):
        _segment("middle", 3, T0, T0 + 1, 0.1)


def test_segment_rejects_disagreeing_lengths():
    with pytest.raises(ValueError, match="segment arrays disagree"):
        Segment(
            arm="left",
            positions=np.zeros((4, ARM_DOF), dtype=np.float32),
            velocities=np.zeros((3, ARM_DOF), dtype=np.float32),
            gripper=np.zeros(4, dtype=np.float32),
            t_wall=np.linspace(T0, T0 + 1, 4),
        )


def test_each_segment_writes_only_its_own_arms_columns():
    """The arm a segment does not drive holds its last command at ZERO velocity — that is what makes
    a 14-D action honest when only one arm is moving."""
    start_q = np.arange(BIMANUAL_ARM_DOF, dtype=np.float32)
    dense = _dense_bimanual(
        [_segment("left", 5, T0, T0 + 1.0, value=9.0)], q_start=start_q, gripper_start=np.zeros(2)
    )

    left, right = ARM_SLICES["left"], ARM_SLICES["right"]
    assert np.all(dense["positions"][:, left] == 9.0)
    # The right arm keeps exactly what it started at, and never moves.
    assert np.all(dense["positions"][:, right] == start_q[right])
    assert np.all(dense["velocities"][:, right] == 0.0)
    assert np.all(dense["velocities"][:, left] == 0.5)


def test_the_gap_between_two_arms_becomes_stationary_hold_rows():
    """Between the two plans the robot perceives and plans for several seconds. Those frames are in
    the video, so they need a command behind them — a hold, not an interpolation across the gap."""
    left = _segment("left", 5, T0, T0 + 1.0, value=1.0)
    right = _segment("right", 5, T0 + 6.0, T0 + 7.0, value=2.0)  # 5 s gap
    dense = _dense_bimanual([left, right], q_start=np.zeros(BIMANUAL_ARM_DOF), gripper_start=np.zeros(2))

    t = dense["t_wall"]
    in_gap = (t > T0 + 1.0) & (t < T0 + 6.0)
    assert in_gap.sum() > 100, "a 5 s gap should be filled at ~50 Hz"
    # Nothing is commanded to move during the gap, and the left arm holds where its plan left it.
    assert np.all(dense["velocities"][in_gap] == 0.0)
    assert np.all(dense["positions"][in_gap][:, ARM_SLICES["left"]] == 1.0)
    assert np.all(np.diff(t) >= 0), "the dense timeline must be non-decreasing"


def test_segments_are_ordered_by_wall_clock_not_by_argument_order():
    left = _segment("left", 4, T0, T0 + 1.0, value=1.0)
    right = _segment("right", 4, T0 + 2.0, T0 + 3.0, value=2.0)
    dense = _dense_bimanual([right, left], q_start=np.zeros(BIMANUAL_ARM_DOF), gripper_start=np.zeros(2))
    assert dense["t_wall"][0] == pytest.approx(T0)
    assert np.all(np.diff(dense["t_wall"]) >= 0)


def test_dump_writes_the_bimanual_schema(tmp_path):
    segments = [
        _segment("left", 20, T0, T0 + 2.0, value=1.0, gripper=1.0),
        _segment("right", 20, T0 + 3.0, T0 + 5.0, value=2.0, gripper=0.0),
    ]
    npz_path = dump_bimanual_episode(
        tmp_path,
        segments=segments,
        joint_samples=_samples(T0, T0 + 5.0, 150),
        instruction="Place the toys on the plate",
        cameras={"observation.images.top": "top_cam.mp4"},
        q_start=np.zeros(BIMANUAL_ARM_DOF, dtype=np.float32),
        gripper_start=np.zeros(2, dtype=np.float32),
        fps=15,
        record_start=T0 - 0.5,
        record_stop=T0 + 5.5,
    )
    assert npz_path is not None

    with np.load(npz_path) as sd:
        n = len(sd["frame_time"])
        assert sd["joint_position"].shape == (n, BIMANUAL_ARM_DOF)
        assert sd["gripper_position"].shape == (n, 2)
        assert sd["cmd_joint_position"].shape == (n, BIMANUAL_ARM_DOF)
        assert sd["cmd_joint_velocity"].shape == (n, BIMANUAL_ARM_DOF)
        assert sd["cmd_gripper"].shape == (n, 2)

        # frame_time is the master timeline; float32 near this epoch has 128 s resolution.
        assert sd["frame_time"].dtype == np.float64
        assert len(np.unique(sd["frame_time"])) > n // 2

        # Proprioception is MEASURED and must not be a copy of the command.
        assert np.all(sd["joint_position"] == -7.0)
        assert not np.array_equal(sd["joint_position"], sd["cmd_joint_position"])
        assert np.allclose(sd["gripper_position"], np.array([0.37, 0.62]))

        # The commanded gripper stays strictly binary.
        cmd_g = sd["cmd_gripper"]
        assert np.all((cmd_g == 0.0) | (cmd_g == 1.0))
        # Each arm's own plan drove its own gripper column.
        assert cmd_g[:, GRIPPER_INDEX["left"]].max() == 1.0

    meta = json.loads((tmp_path / "_meta.json").read_text())
    assert meta["n_frames"] == n
    assert meta["embodiment"] == "bimanual_yam"
    assert meta["layout"]["joint_columns"] == {"left": [0, 6], "right": [6, 12]}
    assert meta["layout"]["gripper_convention"] == "0 = open, 1 = closed"
    assert sorted(meta["arms_used"]) == ["left", "right"]
    assert meta["record_start"] == pytest.approx(T0 - 0.5)


def test_dump_refuses_when_the_measured_trace_is_missing(tmp_path):
    """Falling back to plan positions as proprioception is the exact bug this capture path exists to
    prevent, so an empty measured trace must produce nothing at all."""
    out = dump_bimanual_episode(
        tmp_path,
        segments=[_segment("left", 10, T0, T0 + 1.0, value=1.0)],
        joint_samples=[],
        instruction="x",
        cameras={},
        q_start=np.zeros(BIMANUAL_ARM_DOF),
        gripper_start=np.zeros(2),
    )
    assert out is None
    assert not (tmp_path / "robot_state.npz").exists()


def test_dump_refuses_a_non_binary_commanded_gripper(tmp_path):
    bad = _segment("left", 10, T0, T0 + 1.0, value=1.0, gripper=0.4)
    with pytest.raises(ValueError, match="not binary"):
        dump_bimanual_episode(
            tmp_path,
            segments=[bad],
            joint_samples=_samples(T0, T0 + 1.0, 30),
            instruction="x",
            cameras={},
            q_start=np.zeros(BIMANUAL_ARM_DOF),
            gripper_start=np.zeros(2),
        )


def test_dump_returns_none_without_any_executed_segment(tmp_path):
    assert (
        dump_bimanual_episode(
            tmp_path,
            segments=[],
            joint_samples=_samples(T0, T0 + 1.0, 30),
            instruction="x",
            cameras={},
            q_start=np.zeros(BIMANUAL_ARM_DOF),
            gripper_start=np.zeros(2),
        )
        is None
    )


def test_segment_from_motion_spreads_a_curobo_move_over_its_measured_span():
    """Parking an arm between the two plans is real commanded motion inside the recorded window."""
    positions = np.linspace(0.0, 1.0, 8).reshape(8, 1) * np.ones((1, ARM_DOF))
    seg = segment_from_motion("right", positions, None, T0, T0 + 2.0, gripper=1.0)
    assert seg is not None
    assert seg.arm == "right"
    assert seg.positions.shape == (8, ARM_DOF)
    assert np.all(seg.velocities == 0.0)  # velocities omitted -> zero, not fabricated
    assert seg.t_wall[0] == pytest.approx(T0)
    assert seg.t_wall[-1] == pytest.approx(T0 + 2.0)
    assert np.all(seg.gripper == 1.0)


def test_segment_from_motion_is_none_for_an_empty_move():
    assert segment_from_motion("left", np.zeros((0, ARM_DOF)), None, T0, T0 + 1.0, gripper=0.0) is None


# -- arm="dual" (simultaneous dual-arm / handover) -----------------------------------------------


def _dual_segment(n: int, t_start: float, t_end: float, value: float, gripper=(0.0, 0.0)) -> Segment:
    return Segment(
        arm="dual",
        positions=np.full((n, BIMANUAL_ARM_DOF), value, dtype=np.float32),
        velocities=np.full((n, BIMANUAL_ARM_DOF), 0.5, dtype=np.float32),
        gripper=np.tile(np.asarray(gripper, dtype=np.float32), (n, 1)),
        t_wall=np.linspace(t_start, t_end, n),
    )


def test_dual_segment_rejects_a_single_arm_shaped_positions_array():
    """arm="dual" must carry BOTH arms' columns -- a 6-wide array here would silently be someone's
    single-arm data mislabeled, not a genuine simultaneous segment."""
    with pytest.raises(ValueError, match="must carry positions/velocities of width"):
        Segment(
            arm="dual",
            positions=np.zeros((4, ARM_DOF), dtype=np.float32),
            velocities=np.zeros((4, ARM_DOF), dtype=np.float32),
            gripper=np.zeros((4, 2), dtype=np.float32),
            t_wall=np.linspace(T0, T0 + 1, 4),
        )


def test_dual_segment_rejects_a_scalar_gripper_channel():
    with pytest.raises(ValueError, match="must carry positions/velocities of width"):
        Segment(
            arm="dual",
            positions=np.zeros((4, BIMANUAL_ARM_DOF), dtype=np.float32),
            velocities=np.zeros((4, BIMANUAL_ARM_DOF), dtype=np.float32),
            gripper=np.zeros(4, dtype=np.float32),  # one channel, not two
            t_wall=np.linspace(T0, T0 + 1, 4),
        )


def test_dense_bimanual_writes_a_dual_segments_columns_directly_not_through_a_slice():
    """A dual segment already carries both arms' numbers -- _dense_bimanual must write them straight
    through, not through ARM_SLICES (which would only place 6 of the 12 columns)."""
    seg = _dual_segment(5, T0, T0 + 1.0, value=3.0, gripper=(1.0, 0.0))
    dense = _dense_bimanual([seg], q_start=np.zeros(BIMANUAL_ARM_DOF), gripper_start=np.zeros(2))

    assert np.all(dense["positions"] == 3.0), "both arms' columns must be written, not just one slice"
    assert np.all(dense["velocities"] == 0.5)
    assert np.all(dense["gripper"][:, GRIPPER_INDEX["left"]] == 1.0)
    assert np.all(dense["gripper"][:, GRIPPER_INDEX["right"]] == 0.0)


def test_segment_from_motion_dual_carries_both_grippers():
    positions = np.linspace(0.0, 1.0, 8).reshape(8, 1) * np.ones((1, BIMANUAL_ARM_DOF))
    seg = segment_from_motion("dual", positions, None, T0, T0 + 2.0, gripper=(1.0, 0.0))
    assert seg is not None
    assert seg.arm == "dual"
    assert seg.positions.shape == (8, BIMANUAL_ARM_DOF)
    assert seg.gripper.shape == (8, 2)
    assert np.all(seg.gripper[:, 0] == 1.0)
    assert np.all(seg.gripper[:, 1] == 0.0)


def _dual_plan(steps: list[dict], q_init=None) -> dict:
    return {"version": "1.4.0", "q_init": (q_init if q_init is not None else [0.0] * BIMANUAL_ARM_DOF), "steps": steps}


def _traj_step(label, n=4, dt=0.02, value=0.0):
    return {
        "type": "trajectory",
        "label": label,
        "positions": np.full((n, BIMANUAL_ARM_DOF), value).tolist(),
        "velocities": np.full((n, BIMANUAL_ARM_DOF), 0.1).tolist(),
        "dt": dt,
    }


def test_flatten_dual_plan_tracks_each_grippers_channel_independently():
    """The regression this exists to prevent: a handover's taker-close and giver-open, if flattened
    through a single shared gripper channel (like the single-arm _flatten_plan), would each stomp the
    other's state. Left closes; right closing later must not also flip left's channel back open, and
    vice versa."""
    steps = [
        _traj_step("MoveFree"),
        {"type": "gripper", "label": "PickGiver", "action": "close", "arm": "left"},
        _traj_step("Handover"),
        {"type": "gripper", "label": "Handover_taker_close", "action": "close", "arm": "right"},
        {"type": "gripper", "label": "Handover_giver_open", "action": "open", "arm": "left"},
        _traj_step("PlaceTaker"),
        {"type": "gripper", "label": "PlaceTaker_open", "action": "open", "arm": "right"},
        _traj_step("GoToInitial"),
    ]
    # 0.5 s per step window, no trajectory between consecutive gripper steps -- so the state DURING
    # step i's own hold rows (not after them; nothing fills the gap to the next step) is what shows
    # whether that step's update leaked into the other channel.
    timeline = [{"t_start": T0 + i, "t_end": T0 + i + 0.5} for i in range(len(steps))]
    dense = _flatten_dual_plan(_dual_plan(steps), timeline=timeline)

    def _state_during(step_idx):
        ts, te = timeline[step_idx]["t_start"], timeline[step_idx]["t_end"]
        in_window = (dense["t_wall"] >= ts) & (dense["t_wall"] <= te)
        assert in_window.any(), f"no rows found in step {step_idx}'s window"
        return dense["gripper"][in_window][-1]

    # During PickGiver's hold rows: left closed, right still open.
    after_pick = _state_during(1)
    assert after_pick[GRIPPER_INDEX["left"]] == 1.0
    assert after_pick[GRIPPER_INDEX["right"]] == 0.0

    # During the taker's close (right=1) -- left must STILL read closed: the taker's close must not
    # have touched the left (giver) channel.
    after_taker_close = _state_during(3)
    assert after_taker_close[GRIPPER_INDEX["right"]] == 1.0
    assert after_taker_close[GRIPPER_INDEX["left"]] == 1.0, "the taker's close corrupted the giver's channel"

    # During the giver's open: left open, right still closed (holding the object) -- the giver's
    # open must not have touched the taker (right) channel.
    after_giver_open = _state_during(4)
    assert after_giver_open[GRIPPER_INDEX["left"]] == 0.0
    assert after_giver_open[GRIPPER_INDEX["right"]] == 1.0, "the giver's open corrupted the taker's channel"


def test_segments_from_dual_plan_round_trips_through_disk(tmp_path):
    steps = [
        _traj_step("MoveFree", value=1.0),
        {"type": "gripper", "label": "PickBoth", "action": "close", "arms": ["left", "right"]},
        _traj_step("MoveHoldingBoth", value=2.0),
    ]
    timeline = [
        {"t_start": T0, "t_end": T0 + 0.08},
        {"t_start": T0 + 0.08, "t_end": T0 + 0.4},
        {"t_start": T0 + 0.4, "t_end": T0 + 0.48},
    ]
    plan_path = tmp_path / "tiptop_plan.json"
    plan_path.write_text(json.dumps(_dual_plan(steps)))

    segments = segments_from_dual_plan(plan_path, timeline)
    assert len(segments) == 1
    seg = segments[0]
    assert seg.arm == "dual"
    assert seg.positions.shape[1] == BIMANUAL_ARM_DOF
    assert seg.gripper.shape[1] == 2
    # Both grippers close together for PickBoth (arms=[left,right]).
    closed = seg.gripper[seg.t_wall > timeline[1]["t_end"]][0]
    assert np.all(closed == 1.0)

"""``--mode manual``: the operator moves the board, the script decides what is worth keeping.

Autonomous capture can trust its own poses — cuRobo put the arm there and told it when it arrived.
Hand-guided capture cannot, so the gating IS the mode, and it is what these tests pin:

* nothing is captured while the arm is moving (the joint read and the frame are taken milliseconds
  apart, and at arm's length a drift between them is millimetres of silent error);
* nothing is captured twice from the same spot — duplicates agree with each other by construction,
  so they flatter the residuals while adding no constraint;
* the arm always ends up holding position again, including when the operator stops with Ctrl-C.

Runs without hardware::

    tiptop/.pixi/envs/default/bin/python -m pytest tiptop/tests/test_calibrate_top_cam_manual.py -q
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation as Rot

from tiptop.scripts.calibrate_top_cam import BoardSpec, _capture_manual, _is_novel, _mat_from_rt

# Fast enough that a test spends milliseconds in the dwell loop rather than seconds.
FAST = dict(still_rad=0.01, still_s=0.02, poll_s=0.001, min_new_trans_mm=30.0, min_new_rot_deg=8.0)
GATES = dict(min_corners=20, max_incidence_deg=55.0, min_coverage=0.25)
SPEC = BoardSpec(kind="charuco", cols=14, rows=9, square_size=0.02, marker_size=0.015, aruco_dict="DICT_5X5_100")


@dataclass
class FakeFrame:
    bgr: np.ndarray


@dataclass
class FakeSample:
    world_from_ee: np.ndarray


def _pose_from_q(q) -> np.ndarray:
    """A stand-in for forward kinematics: distinct joints give distinct end-effector poses."""
    q = np.asarray(q, dtype=float)
    return _mat_from_rt(Rot.from_rotvec(q[3:6]).as_matrix(), q[:3])


class FakeRig:
    """A camera that always sees a well-presented board, and an arm the test scripts by hand.

    ``moves`` is what the operator does: each entry is a joint vector they hold the arm at, for
    ``reads_per_pose`` polls. Once the last one is used up the operator gives up and hits Ctrl-C,
    which is how a real session ends too — you stop when the rotation spread looks right, not when a
    counter runs out.
    """

    K = np.array([[600.0, 0.0, 32.0], [0.0, 600.0, 32.0], [0.0, 0.0, 1.0]])
    dist = np.zeros((5, 1))

    def __init__(self, moves, *, drift_during_capture=0.0, incidence=10.0, coverage=0.5, reads_per_pose=150):
        self.moves = [np.asarray(m, dtype=float) for m in moves]
        self.drift_during_capture = drift_during_capture
        self.incidence = incidence
        self.coverage = coverage
        self.reads_per_pose = reads_per_pose
        self.free_drive_calls: list[bool] = []
        self.observations = 0
        self.gave_up = False
        self._at = 0
        self._reads_here = 0
        self._just_observed = False

    def set_free_drive(self, enabled: bool) -> None:
        self.free_drive_calls.append(enabled)

    def read_q(self) -> np.ndarray:
        self._reads_here += 1
        if self._reads_here > self.reads_per_pose:
            if self._at + 1 >= len(self.moves):
                self.gave_up = True
                raise KeyboardInterrupt
            self._at += 1
            self._reads_here = 0
        # A read straight after a frame is the capture's own re-check: report where the arm has
        # drifted to while the shutter was open.
        drifted, self._just_observed = self._just_observed, False
        return self.moves[self._at] + (self.drift_during_capture if drifted else 0.0)

    def fk(self, q) -> np.ndarray:
        return _pose_from_q(q)

    def observe(self, spec, board):
        self.observations += 1
        self._just_observed = True
        obj_pts = np.zeros((24, 3))
        obj_pts[:, :2] = np.mgrid[0:6, 0:4].T.reshape(-1, 2) * SPEC.square_size
        img_pts = np.linspace(10.0, 50.0, 48).reshape(24, 2)
        cam_from_board = _mat_from_rt(np.eye(3), [0.0, 0.0, 0.6])
        frame = FakeFrame(np.zeros((64, 64, 3), np.uint8))
        return frame, obj_pts, img_pts, cam_from_board, 0.4, self.incidence, self.coverage


def _run(rig, tmp_path, num_poses, **overrides):
    return _capture_manual(
        rig=rig,
        spec=SPEC,
        board=None,
        num_poses=num_poses,
        max_pnp_error_px=1.5,
        run_dir=tmp_path,
        **{**FAST, **GATES, **overrides},
    )


# -- novelty -------------------------------------------------------------------------------------
def test_a_pose_is_new_if_it_differs_in_translation_or_in_rotation():
    at_origin = [FakeSample(_pose_from_q(np.zeros(6)))]
    assert not _is_novel(_pose_from_q([0.01, 0, 0, 0, 0, 0]), at_origin, 30.0, 8.0)
    assert _is_novel(_pose_from_q([0.05, 0, 0, 0, 0, 0]), at_origin, 30.0, 8.0)  # 50 mm away
    assert _is_novel(_pose_from_q([0.01, 0, 0, 0.3, 0, 0]), at_origin, 30.0, 8.0)  # turned 17 deg
    assert _is_novel(_pose_from_q(np.zeros(6)), [], 30.0, 8.0)


def test_novelty_is_judged_against_every_sample_not_just_the_last():
    """A pose 200 mm from the newest sample can still be 5 mm from an earlier one. Comparing only
    against the most recent capture would let a slow loop back through the workspace re-record poses
    that are already in the set."""
    samples = [FakeSample(_pose_from_q(np.zeros(6))), FakeSample(_pose_from_q([0.2, 0, 0, 0, 0, 0]))]
    assert not _is_novel(_pose_from_q([0.005, 0, 0, 0, 0, 0]), samples, 30.0, 8.0)


# -- the capture loop ----------------------------------------------------------------------------
def test_distinct_steady_poses_are_each_captured_once(tmp_path):
    rig = FakeRig([[0, 0, 0, 0, 0, 0], [0.1, 0, 0, 0, 0, 0], [0.2, 0, 0, 0.4, 0, 0]])

    samples = _run(rig, tmp_path, num_poses=3)

    assert [s.index for s in samples] == [0, 1, 2]
    assert np.allclose([s.world_from_ee[0, 3] for s in samples], [0.0, 0.1, 0.2])
    assert not rig.gave_up  # it reached the target count on its own
    # Every sample kept its own frame pair, for the verification image and for --replay.
    assert sorted(p.name for p in (tmp_path / "frames").glob("*.png")) == [
        "000.png",
        "000_detected.png",
        "001.png",
        "001_detected.png",
        "002.png",
        "002_detected.png",
    ]


def test_holding_the_same_spot_does_not_capture_it_again(tmp_path):
    """Standing still is the capture trigger, so without the novelty gate a hand resting on the arm
    would fill the whole set with one pose — and the solve would look excellent on it."""
    rig = FakeRig([[0, 0, 0, 0, 0, 0], [0.005, 0, 0, 0, 0, 0]])

    samples = _run(rig, tmp_path, num_poses=5)

    assert len(samples) == 1
    assert rig.gave_up  # it kept looking rather than settling for a duplicate


def test_a_pose_that_never_settles_is_never_captured(tmp_path):
    """Joints that change on every poll are an arm still in motion. The frame and the joint reading
    would disagree, and nothing about the result would show it — so the camera is not even read."""

    class NeverStill(FakeRig):
        def read_q(self):
            self._reads_here += 1
            if self._reads_here > self.reads_per_pose:
                self.gave_up = True
                raise KeyboardInterrupt
            return np.full(6, self._reads_here * 0.05)

    rig = NeverStill([[0, 0, 0, 0, 0, 0]])

    assert _run(rig, tmp_path, num_poses=1) == []
    assert rig.observations == 0


def test_a_sample_is_dropped_if_the_arm_stirs_while_the_frame_is_taken(tmp_path):
    rig = FakeRig([[0, 0, 0, 0, 0, 0], [0.1, 0, 0, 0, 0, 0]], drift_during_capture=0.05)

    samples = _run(rig, tmp_path, num_poses=2)

    assert samples == []
    assert rig.observations > 0  # it did look, and threw away what it saw


def test_a_badly_presented_board_is_rejected_however_still_the_arm_is(tmp_path):
    rig = FakeRig([[0, 0, 0, 0, 0, 0], [0.1, 0, 0, 0, 0, 0]], incidence=80.0)

    assert _run(rig, tmp_path, num_poses=2) == []


def test_the_arm_is_freed_at_the_start_and_held_again_at_the_end(tmp_path):
    rig = FakeRig([[0, 0, 0, 0, 0, 0], [0.1, 0, 0, 0, 0, 0]])

    _run(rig, tmp_path, num_poses=2)

    assert rig.free_drive_calls == [True, False]


def test_stopping_early_keeps_what_was_captured_and_still_re_engages_the_arm(tmp_path):
    """Ctrl-C is a supported way to finish, so the samples have to survive it — and the arm must not
    be left limp when they do."""
    rig = FakeRig([[0, 0, 0, 0, 0, 0]])

    samples = _run(rig, tmp_path, num_poses=10)

    assert rig.gave_up
    assert len(samples) == 1
    assert rig.free_drive_calls == [True, False]

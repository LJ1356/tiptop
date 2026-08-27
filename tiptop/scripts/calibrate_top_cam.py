"""Calibrate the bimanual YAM's FIXED third-person camera — solve ``world_from_cam`` from a board.

The YAM perceives through a D435 bolted to the base column, not through a wrist camera, so
``calibrate_wrist_cam.py`` does not apply: there is no ``ee_from_cam`` to compose with forward
kinematics. ``cameras.hand.mount: world`` means the calibration entry **is** ``world_from_cam``
(see ``perception.cameras.camera_mount``), and that is what this writes.

The frame is the URDF's ``world`` link — cuRobo's ``base_link`` for both YAM configs — so the pose
this produces is directly comparable to the sim design pose in ``calibration_info_prism.json``
(``[0.39, 0, 0.8551, ...]``, the MolmoAct2 ``top_cam`` mount carried through
``droid-sim-evals``'s ``MOUNT_XYZ``). That entry is a *design* pose, not a measurement; replacing it
with a real one is the point of this script.

**The board must be CLAMPED IN OR TAPED TO the moving gripper.** Everything below rests on the
board and the arm being one rigid body: a board held by hand, standing on the table or leaning
against something carries no information about where the camera is, however cleanly it is detected.
:func:`check_rigidity` tests this before solving and refuses rather than returning a confident wrong
pose. It also has to face the CAMERA — a board turned edge-on projects to a sliver whose corners fit
to a fraction of a pixel with the pose metres out of place, which is why the presentation gates
exist and why ``--mode aim`` is there to set it up.

Five modes:

``handeye`` (default, recommended)
    Board clamped to one gripper. The arm visits N poses, the camera sees the board at each, and
    OpenCV's eye-to-hand formulation solves ``world_from_cam`` **without** anyone having to know
    where the board sits — the unknown ``ee_from_board`` is solved for at the same time. Nothing has
    to be measured by hand.

``manual``
    The same calibration with the poses chosen by hand. The arm's six joints go limp — gravity
    compensated, gripper still clamped — and you push the board around yourself; each time you hold
    still somewhere new, that pose is captured. The board still has to be **in the gripper**, and
    for the same reason as ever: FK of the measured joints is the only thing that ties a board
    observation to the world frame. A board carried in your hand, off the robot, constrains nothing
    at all, however many views of it the camera gets. See :func:`_capture_manual`.

``aim``
    Turns the wrist until the board faces the camera, then stops without calibrating. The same
    servo runs automatically at the start of ``handeye``, so a board clamped in at an awkward angle
    fixes itself rather than failing the run. See :func:`auto_aim`.

``static``
    The board lies in the scene at a pose you already know in world coordinates (a jig, a marked
    spot on the table). One ``solvePnP`` and an inversion. Faster, but only as good as the number
    you type in — use it as a sanity check, not as the primary calibration.

``sweep``
    DROID-style. You set the starting pose the same way as ``manual`` — the arm goes limp, gravity
    compensated, gripper still clamped, and you push the board to a good starting spot — but instead
    of capturing that one pose, pressing Enter locks it in as an ANCHOR and hands control back to the
    arm. From there the arm drives the board through one smooth loop around the anchor — a Lissajous
    pattern in position and orientation, the same shape DROID's ``calibration_traj`` traces — and
    takes a sample at each point along it, rather than the independently-perturbed poses ``handeye``
    visits. It solves with the DANIILIDIS method by default (DROID's fixed choice, rather than
    ``handeye``'s run-all-five-and-keep-the-best) and, before trusting the result, checks it against
    a held-out split of the samples the way DROID's ``is_calibration_accurate`` does — a solve that
    only fits the data it came from is not caught by in-sample residuals alone. See
    :func:`_capture_sweep` and :func:`_holdout_check`.

Eye-to-hand, concretely: OpenCV's ``calibrateHandEye`` solves ``A·X·B = C`` for ``X`` given
``A = gripper2base`` and ``B = target2cam``, returning ``X = cam2gripper`` (camera on the arm,
board fixed). Here the roles swap — the board is on the arm and the camera is fixed — and the
constant is ``ee_from_board``::

    ee_from_board = ee_from_world · world_from_cam · cam_from_board

which is the same ``A·X·B = C`` with ``A = ee_from_world`` (the INVERTED gripper poses) and
``X = world_from_cam``. So feeding inverted FK poses returns exactly the transform we want.

**Safety.** The board on the gripper is not in cuRobo's collision model, and neither is whatever it
is clamped to. Perturbations are small and centred on a pose you choose, but keep the workspace
clear and watch the arm.

Usage::

    # 1. clamp the board in the gripper, then let the arm turn it to face the camera
    pixi run calibrate-top-cam --mode aim

    # 2. left arm holds the board; it aims itself first, and the right arm is parked for you
    DC_WORKSPACE=prism pixi run calibrate-top-cam --arm left

    # ... or move the board yourself: the arm goes limp and each steady new pose is captured
    DC_WORKSPACE=prism pixi run calibrate-top-cam --mode manual --arm left --close-gripper

    # a plain 9x6-inner-corner chessboard instead of the Charuco default
    DC_WORKSPACE=prism pixi run calibrate-top-cam --board-kind chessboard --cols 9 --rows 6 \
        --square-size 0.025

    # DROID-style: hand-pose the board, press Enter, then a Lissajous sweep collects the samples
    DC_WORKSPACE=prism pixi run calibrate-top-cam --mode sweep --arm left --close-gripper

    # re-solve offline from a previous run's samples (no robot, no camera)
    pixi run calibrate-top-cam --replay tiptop/.cache/top_cam_calib/20260802-141530
"""

import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import tyro
from cv2 import aruco
from scipy.spatial.transform import Rotation as Rot

_log = logging.getLogger(__name__)

# Used unless the caller already named a config, matching scripts/yam_commands.py.
DEFAULT_YAM_CONFIG = "tiptop_yam_real.yml"

BoardKind = Literal["charuco", "chessboard"]

# All five OpenCV hand-eye solvers. They disagree by more than their documentation suggests on
# real data, so the script runs every one and keeps whichever reprojects best (see `_solve`).
HAND_EYE_METHODS = {
    "tsai": cv2.CALIB_HAND_EYE_TSAI,
    "park": cv2.CALIB_HAND_EYE_PARK,
    "horaud": cv2.CALIB_HAND_EYE_HORAUD,
    "andreff": cv2.CALIB_HAND_EYE_ANDREFF,
    "daniilidis": cv2.CALIB_HAND_EYE_DANIILIDIS,
}

# The sim design pose this calibration is meant to replace, for the before/after report.
SIM_DESIGN_POSE = np.array([0.39, 0.0, 0.8551, -2.9670597283903604, 0.0, -1.5707963267948966])


# --------------------------------------------------------------------------------------------- #
# Board                                                                                           #
# --------------------------------------------------------------------------------------------- #
@dataclass(frozen=True)
class BoardSpec:
    """Geometry of the calibration target.

    ``cols``/``rows`` mean different things per kind, which is the one genuinely confusing part of
    OpenCV's board APIs: for ``charuco`` they count SQUARES (a 14x9 board has 14x9 squares), for
    ``chessboard`` they count INNER CORNERS (the classic OpenCV board is 9x6 inner corners on a
    10x7 square grid).
    """

    kind: BoardKind
    cols: int
    rows: int
    square_size: float
    marker_size: float
    aruco_dict: str

    def validate(self) -> None:
        if self.cols < 3 or self.rows < 3:
            raise ValueError(f"board is too small to detect: {self.cols}x{self.rows}")
        if self.square_size <= 0:
            raise ValueError(f"square_size must be positive, got {self.square_size}")
        if self.kind == "charuco":
            if not 0 < self.marker_size < self.square_size:
                raise ValueError(f"marker_size must be in (0, square_size), got {self.marker_size}")
            if not hasattr(aruco, self.aruco_dict):
                raise ValueError(f"unknown ArUco dictionary {self.aruco_dict!r}")
        elif self.cols == self.rows:
            # A square inner-corner grid is rotationally ambiguous: findChessboardCorners can order
            # the corners 90 deg apart between frames, which silently corrupts the hand-eye solve.
            raise ValueError("a square chessboard grid is rotationally ambiguous; use cols != rows")

    def to_json(self) -> dict:
        return {
            "kind": self.kind,
            "cols": self.cols,
            "rows": self.rows,
            "square_size": self.square_size,
            "marker_size": self.marker_size,
            "aruco_dict": self.aruco_dict,
        }


def _charuco_board(spec: BoardSpec):
    return aruco.CharucoBoard(
        size=(spec.cols, spec.rows),
        squareLength=spec.square_size,
        markerLength=spec.marker_size,
        dictionary=aruco.getPredefinedDictionary(getattr(aruco, spec.aruco_dict)),
    )


def _detector_params() -> aruco.DetectorParameters:
    params = aruco.DetectorParameters()
    params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
    return params


def detect_board(bgr: np.ndarray, spec: BoardSpec, board) -> tuple[np.ndarray, np.ndarray] | None:
    """Matched ``(object_points (n,3), image_points (n,2))`` for one frame, or None if unusable.

    Object points are in the BOARD frame; image points are pixels in the colour image.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    if spec.kind == "charuco":
        detector = aruco.CharucoDetector(board, aruco.CharucoParameters(), _detector_params())
        charuco_corners, charuco_ids, _, _ = detector.detectBoard(gray)
        if charuco_ids is None or len(charuco_ids) < 6:
            return None
        obj_pts, img_pts = board.matchImagePoints(charuco_corners, charuco_ids)
        if obj_pts is None or len(obj_pts) < 6:
            return None
        return obj_pts.reshape(-1, 3).astype(np.float64), img_pts.reshape(-1, 2).astype(np.float64)

    # No CALIB_CB_FAST_CHECK: its early reject is for video throughput, and a missed detection here
    # costs a whole robot pose.
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    found, corners = cv2.findChessboardCorners(gray, (spec.cols, spec.rows), flags=flags)
    if not found:
        return None
    corners = cv2.cornerSubPix(
        gray, corners, (11, 11), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
    )
    grid = np.mgrid[0 : spec.cols, 0 : spec.rows].T.reshape(-1, 2).astype(np.float64)
    obj_pts = np.zeros((grid.shape[0], 3), dtype=np.float64)
    obj_pts[:, :2] = grid * spec.square_size
    return obj_pts, corners.reshape(-1, 2).astype(np.float64)


# --------------------------------------------------------------------------------------------- #
# Pose maths                                                                                      #
# --------------------------------------------------------------------------------------------- #
def _mat_from_rt(rmat: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    mat = np.eye(4)
    mat[:3, :3] = rmat
    mat[:3, 3] = np.asarray(tvec).reshape(3)
    return mat


def _pose6_from_mat(mat: np.ndarray) -> np.ndarray:
    """``[x, y, z, roll, pitch, yaw]`` in the xyz-Euler convention ``config.load_calibration`` reads."""
    return np.concatenate([mat[:3, 3], Rot.from_matrix(mat[:3, :3]).as_euler("xyz")])


def _mat_from_pose6(pose6) -> np.ndarray:
    pose6 = np.asarray(pose6, dtype=np.float64).reshape(6)
    return _mat_from_rt(Rot.from_euler("xyz", pose6[3:]).as_matrix(), pose6[:3])


def _pose_delta(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """``(translation mm, rotation deg)`` between two 4x4 poses."""
    trans_mm = float(np.linalg.norm(a[:3, 3] - b[:3, 3]) * 1000.0)
    rot_deg = float(np.degrees(Rot.from_matrix(a[:3, :3].T @ b[:3, :3]).magnitude()))
    return trans_mm, rot_deg


def solve_pnp(obj_pts: np.ndarray, img_pts: np.ndarray, K: np.ndarray, dist: np.ndarray) -> tuple[np.ndarray, float]:
    """``cam_from_board`` and the RMS reprojection error, in pixels."""
    ok, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        raise RuntimeError("solvePnP failed on a frame the board was detected in")
    rvec, tvec = cv2.solvePnPRefineLM(obj_pts, img_pts, K, dist, rvec, tvec)
    projected, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, dist)
    err = float(np.sqrt(np.mean(np.sum((projected.reshape(-1, 2) - img_pts) ** 2, axis=1))))
    return _mat_from_rt(cv2.Rodrigues(rvec)[0], tvec), err


# --------------------------------------------------------------------------------------------- #
# Samples                                                                                         #
# --------------------------------------------------------------------------------------------- #
@dataclass
class Sample:
    """One (robot pose, board observation) pair.

    ``world_from_ee`` is FK of the MEASURED joint positions, not the commanded ones — the arm
    settles a few mrad short of its target and hand-eye is sensitive to exactly that.
    """

    index: int
    world_from_ee: np.ndarray  # 4x4
    cam_from_board: np.ndarray  # 4x4
    obj_pts: np.ndarray  # (n, 3) in the board frame
    img_pts: np.ndarray  # (n, 2) pixels
    pnp_error_px: float


def _as_object_array(arrays: list[np.ndarray]) -> np.ndarray:
    """A 1-D object array of point arrays.

    Not ``np.array(arrays, dtype=object)``: when every frame happens to yield the same number of
    corners that collapses to a 3-D object array, and each row then comes back out of ``np.load``
    as object dtype, which OpenCV refuses. Filling an empty object array keeps it ragged-shaped
    whether the data is ragged or not.
    """
    out = np.empty(len(arrays), dtype=object)
    for i, a in enumerate(arrays):
        out[i] = a
    return out


def _save_samples(
    run_dir: Path, samples: list[Sample], K: np.ndarray, dist: np.ndarray, spec: BoardSpec, cam_serial: str
) -> None:
    """Everything ``--replay`` needs to re-solve without the robot or the camera.

    The point arrays are ragged — a Charuco frame yields however many corners it saw — so they go in
    as object arrays and come back out with ``allow_pickle``.
    """
    np.savez(
        run_dir / "samples.npz",
        world_from_ee=np.stack([s.world_from_ee for s in samples]),
        cam_from_board=np.stack([s.cam_from_board for s in samples]),
        obj_pts=_as_object_array([s.obj_pts for s in samples]),
        img_pts=_as_object_array([s.img_pts for s in samples]),
        pnp_error_px=np.array([s.pnp_error_px for s in samples]),
        K=K,
        dist=dist,
        board=json.dumps(spec.to_json()),
        camera_serial=cam_serial,
    )


def _load_samples(run_dir: Path) -> tuple[list[Sample], np.ndarray, np.ndarray, str]:
    data = np.load(run_dir / "samples.npz", allow_pickle=True)
    samples = [
        Sample(
            index=i,
            world_from_ee=data["world_from_ee"][i],
            cam_from_board=data["cam_from_board"][i],
            obj_pts=np.asarray(data["obj_pts"][i], dtype=np.float64),
            img_pts=np.asarray(data["img_pts"][i], dtype=np.float64),
            pnp_error_px=float(data["pnp_error_px"][i]),
        )
        for i in range(len(data["world_from_ee"]))
    ]
    return samples, data["K"], data["dist"], str(data["camera_serial"])


# --------------------------------------------------------------------------------------------- #
# Eye-to-hand solve                                                                               #
# --------------------------------------------------------------------------------------------- #
def solve_eye_to_hand(samples: list[Sample], method: int) -> np.ndarray:
    """``world_from_cam`` from board-on-gripper observations. See the module docstring for why the
    end-effector poses go in inverted."""
    ee_from_world = [np.linalg.inv(s.world_from_ee) for s in samples]
    rmat, tvec = cv2.calibrateHandEye(
        R_gripper2base=[T[:3, :3] for T in ee_from_world],
        t_gripper2base=[T[:3, 3] for T in ee_from_world],
        R_target2cam=[s.cam_from_board[:3, :3] for s in samples],
        t_target2cam=[s.cam_from_board[:3, 3] for s in samples],
        method=method,
    )
    return _mat_from_rt(rmat, tvec)


def _board_offsets(samples: list[Sample], world_from_cam: np.ndarray) -> np.ndarray:
    """``ee_from_board`` per sample. The board is bolted to the gripper, so a correct
    ``world_from_cam`` makes all of these the same transform — their spread IS the error."""
    return np.stack([np.linalg.inv(s.world_from_ee) @ world_from_cam @ s.cam_from_board for s in samples])


def _mean_offset(offsets: np.ndarray) -> np.ndarray:
    mean = np.eye(4)
    mean[:3, :3] = Rot.from_matrix(offsets[:, :3, :3]).mean().as_matrix()
    mean[:3, 3] = offsets[:, :3, 3].mean(axis=0)
    return mean


@dataclass
class Metrics:
    """How well a candidate ``world_from_cam`` explains the data."""

    reproj_rms_px: float  # board points reprojected through the FULL chain, per-sample RMS
    reproj_max_px: float
    offset_trans_rms_mm: float  # spread of ee_from_board across samples
    offset_rot_rms_deg: float
    per_sample_reproj_px: np.ndarray


def _reproject_errors(
    samples: list[Sample], world_from_cam: np.ndarray, mean_offset: np.ndarray, K: np.ndarray, dist: np.ndarray
) -> np.ndarray:
    """Per-sample RMS reprojection error, in pixels, predicting each board pose from
    ``world_from_cam``, the sample's own measured FK, and a SHARED ``ee_from_board`` rather than
    re-solving PnP per frame — so a candidate calibration is on trial, not just the detections."""
    cam_from_world = np.linalg.inv(world_from_cam)
    per_sample = []
    for s in samples:
        predicted = cam_from_world @ s.world_from_ee @ mean_offset
        rvec = cv2.Rodrigues(predicted[:3, :3])[0]
        projected, _ = cv2.projectPoints(s.obj_pts, rvec, predicted[:3, 3], K, dist)
        per_sample.append(float(np.sqrt(np.mean(np.sum((projected.reshape(-1, 2) - s.img_pts) ** 2, axis=1)))))
    return np.array(per_sample)


def evaluate(samples: list[Sample], world_from_cam: np.ndarray, K: np.ndarray, dist: np.ndarray) -> Metrics:
    """Score a candidate calibration end to end.

    ``pnp_error_px`` only says the board was detected cleanly; it is blind to the calibration. This
    instead predicts each frame's board pose from ``world_from_cam``, the measured FK and the single
    shared ``ee_from_board``, and reprojects — so every term in the chain is on trial, in pixels.
    """
    offsets = _board_offsets(samples, world_from_cam)
    mean_offset = _mean_offset(offsets)

    trans_rms_mm = float(np.sqrt(np.mean(np.sum((offsets[:, :3, 3] - mean_offset[:3, 3]) ** 2, axis=1))) * 1000.0)
    rel = np.einsum("ij,njk->nik", mean_offset[:3, :3].T, offsets[:, :3, :3])
    rot_rms_deg = float(np.degrees(np.sqrt(np.mean(Rot.from_matrix(rel).magnitude() ** 2))))

    per_sample = _reproject_errors(samples, world_from_cam, mean_offset, K, dist)

    return Metrics(
        reproj_rms_px=float(np.sqrt(np.mean(per_sample**2))),
        reproj_max_px=float(per_sample.max()),
        offset_trans_rms_mm=trans_rms_mm,
        offset_rot_rms_deg=rot_rms_deg,
        per_sample_reproj_px=per_sample,
    )


def _rotation_defect(mat: np.ndarray) -> str | None:
    """Why ``mat``'s rotation block is not a rotation, or None if it is one.

    Not paranoia: ANDREFF solves for the rotation through a Kronecker-product linear system that
    does not constrain the result to SO(3), so on marginal data it returns a REFLECTION — orthonormal
    but determinant -1. Everything downstream (scipy's ``from_matrix``, the reprojection) then fails
    on a matrix that looks superficially fine.
    """
    if not np.all(np.isfinite(mat)):
        return "non-finite entries"
    rmat = mat[:3, :3]
    if np.linalg.det(rmat) <= 0:
        return f"determinant {np.linalg.det(rmat):+.3f} (a reflection, not a rotation)"
    if not np.allclose(rmat @ rmat.T, np.eye(3), atol=1e-4):
        return "not orthonormal"
    return None


def _solve(
    samples: list[Sample], K: np.ndarray, dist: np.ndarray, method: str
) -> tuple[np.ndarray, Metrics, str, dict[str, Metrics]]:
    """Run the requested solver, or every solver and keep the best-reprojecting one."""
    names = list(HAND_EYE_METHODS) if method == "auto" else [method]
    scored: dict[str, Metrics] = {}
    results: dict[str, np.ndarray] = {}
    for name in names:
        try:
            candidate = solve_eye_to_hand(samples, HAND_EYE_METHODS[name])
        except cv2.error as e:
            _log.warning(f"hand-eye method {name!r} failed: {e}")
            continue
        defect = _rotation_defect(candidate)
        if defect is not None:
            _log.info(f"hand-eye method {name!r} returned {defect}; skipping it")
            continue
        results[name] = candidate
        scored[name] = evaluate(samples, candidate, K, dist)
    if not scored:
        raise RuntimeError(
            f"every hand-eye solver failed on {len(samples)} samples. Usually too few poses or too "
            "little rotation between them; the samples are saved, so --replay can retry."
        )
    best = min(scored, key=lambda n: scored[n].reproj_rms_px)
    return results[best], scored[best], best, scored


def _solve_robust(
    samples: list[Sample], K: np.ndarray, dist: np.ndarray, method: str, outlier_px: float, min_samples: int
) -> tuple[np.ndarray, Metrics, str, dict[str, Metrics], list[Sample]]:
    """Solve, then drop the worst sample and re-solve until everything fits, one at a time.

    One bad pose — a joint reading taken before the arm settled, a board that shifted in the jaws —
    drags the whole solution off, and then *every* sample reprojects badly. Thresholding against
    that first corrupted solution throws out good data along with the bad, so each pass removes only
    the single worst sample and re-solves. Bounded at a third of the set: if that many samples
    really are outliers the problem is systematic (usually ``--square-size`` not matching the
    printout), and quietly calibrating off the remainder would hide it.
    """
    kept = list(samples)
    budget = min(len(samples) - min_samples, len(samples) // 3)
    for _ in range(max(budget, 0) + 1):
        world_from_cam, metrics, chosen, scored = _solve(kept, K, dist, method)
        worst = int(np.argmax(metrics.per_sample_reproj_px))
        if metrics.per_sample_reproj_px[worst] <= outlier_px or budget <= 0:
            if budget <= 0 and metrics.per_sample_reproj_px[worst] > outlier_px:
                _log.warning(
                    f"Stopped rejecting with {metrics.per_sample_reproj_px[worst]:.2f} px still "
                    "outstanding — a third of the samples were already dropped."
                )
            return world_from_cam, metrics, chosen, scored, kept
        _log.info(
            f"Dropping sample #{kept[worst].index} at {metrics.per_sample_reproj_px[worst]:.2f} px "
            f"and re-solving on {len(kept) - 1}"
        )
        kept.pop(worst)
        budget -= 1
    raise AssertionError("unreachable: the rejection loop always returns within its budget")


def presentation(cam_from_board: np.ndarray, img_pts: np.ndarray, image_shape: tuple[int, int]) -> tuple[float, float]:
    """``(incidence degrees, image coverage fraction)`` for one board observation.

    Incidence is the angle between the board's normal and the camera's ray to it: 0 deg is the board
    square-on to the camera, 90 deg is edge-on. A board seen edge-on projects to a sliver, and its
    pose — especially its normal — comes out badly conditioned even when the corners themselves fit
    to a fraction of a pixel. Coverage is the detected corners' bounding-box diagonal over the image
    diagonal; a board occupying a corner of the frame carries little pose information wherever it
    points.
    """
    normal = cam_from_board[:3, 2]
    ray = cam_from_board[:3, 3] / max(np.linalg.norm(cam_from_board[:3, 3]), 1e-9)
    incidence = float(np.degrees(np.arccos(np.clip(abs(normal @ ray), -1.0, 1.0))))
    extent = img_pts.max(axis=0) - img_pts.min(axis=0)
    coverage = float(np.linalg.norm(extent) / np.linalg.norm(image_shape))
    return incidence, coverage


def rigidity_violations(samples: list[Sample], rot_tol_deg: float = 2.0, trans_tol_mm: float = 10.0) -> tuple:
    """Fraction of sample pairs whose relative motions are not consistent with a rigid board.

    Calibration-free, and the fastest way to catch the one setup mistake that makes the whole method
    meaningless: a board that is not actually bolted to the gripper. Between two poses the arm's
    motion ``A`` and the board's motion ``B`` are conjugate (``B = X⁻¹AX`` for the fixed
    ``X = ee_from_board``), so their screw invariants — rotation angle, and translation along the
    screw axis — must be EQUAL, whatever the calibration turns out to be. If the arm turns 10 deg
    and the board turns 80 deg, no ``world_from_cam`` exists that explains the data, and solving
    anyway just produces a confident-looking wrong answer.

    Returns ``(violating fraction, worst rotation mismatch deg, worst screw mismatch mm, median
    ratio of board rotation to arm rotation)``. That last number names the failure: ~1 with the
    other terms small is a properly clamped board, ~0 is a board that never moved at all, and
    anything scattered in between is a board resting against the gripper rather than held by it.
    """
    pairs, bad, worst_rot, worst_trans = 0, 0, 0.0, 0.0
    ratios: list[float] = []
    for i in range(len(samples)):
        for j in range(i + 1, len(samples)):
            arm = np.linalg.inv(samples[i].world_from_ee) @ samples[j].world_from_ee
            board = np.linalg.inv(samples[i].cam_from_board) @ samples[j].cam_from_board
            rot_arm, rot_board = Rot.from_matrix(arm[:3, :3]), Rot.from_matrix(board[:3, :3])
            angle_arm, angle_board = rot_arm.magnitude(), rot_board.magnitude()
            # Below a few degrees the screw axis is numerically meaningless, so only the pairs with
            # real rotation between them can be judged.
            if angle_arm < np.radians(5.0):
                continue
            pairs += 1
            ratios.append(angle_board / angle_arm)
            d_rot = abs(np.degrees(angle_arm - angle_board))
            screw_arm = float(arm[:3, 3] @ (rot_arm.as_rotvec() / angle_arm)) * 1000.0
            screw_board = float(board[:3, 3] @ (rot_board.as_rotvec() / max(angle_board, 1e-9))) * 1000.0
            d_trans = abs(screw_arm - screw_board)
            worst_rot, worst_trans = max(worst_rot, d_rot), max(worst_trans, d_trans)
            bad += d_rot > rot_tol_deg or d_trans > trans_tol_mm
    ratio = float(np.median(ratios)) if ratios else 0.0
    return (bad / pairs if pairs else 0.0), worst_rot, worst_trans, ratio


def check_rigidity(samples: list[Sample]) -> None:
    """Refuse to solve when the board did not move with the gripper."""
    fraction, worst_rot, worst_trans, ratio = rigidity_violations(samples)
    _log.info(
        f"Rigidity check: {fraction * 100:.0f}% of pose pairs inconsistent (worst {worst_rot:.1f} "
        f"deg, {worst_trans:.0f} mm); the board turned {ratio:.2f}x as far as the arm"
    )
    if fraction > 0.5:
        if ratio < 0.25:
            diagnosis = "it barely moved at all, so it is standing in the scene rather than on the arm"
        elif 0.5 < ratio < 2.0:
            diagnosis = (
                "it roughly follows the arm but not exactly, which is what a board RESTING AGAINST "
                "the gripper does — leaned on it, or standing on the table and pushed by it. "
                "Resting is not holding: it has to be gripped or taped"
            )
        else:
            diagnosis = "its motion bears no fixed relation to the arm's"
        raise RuntimeError(
            f"the board did not move rigidly with the gripper — {fraction * 100:.0f}% of pose pairs "
            f"are inconsistent, off by up to {worst_rot:.0f} deg, and it turned {ratio:.2f}x as far "
            f"as the arm did. Diagnosis: {diagnosis}. The two motions have to match exactly, so the "
            "board must be CLAMPED IN THE JAWS OR TAPED TO the moving gripper. If the board is too "
            "big for the jaws, tape it to a flat face of the gripper (it does not need to be "
            "centred or square — the offset is solved for), or print a smaller one."
        )
    if fraction > 0.15:
        _log.warning(
            f"{fraction * 100:.0f}% of pose pairs are inconsistent (worst {worst_rot:.1f} deg). The "
            "board may be shifting in the jaws — tighten it, or expect a soft calibration."
        )


def _holdout_check(
    samples: list[Sample],
    K: np.ndarray,
    dist: np.ndarray,
    method: str,
    test_fraction: float,
    max_rms_px: float,
    seed: int,
) -> None:
    """Solve on a random subset and check the rest still reprojects well, mirroring DROID's
    ``is_calibration_accurate``.

    In-sample residuals cannot catch a solve that only fits the data it was given — with enough
    samples clustered the same way, an eye-to-hand solution can look good on its own set and still
    be wrong. This is a genuine held-out check: ``world_from_cam`` and ``ee_from_board`` are both
    fit from the train split alone, then scored against test samples the fit never saw. It gates
    saving, the same way DROID's check refuses to write a calibration that fails it, but is not the
    calibration that gets saved — the final solve still uses every sample (see the caller).
    """
    if len(samples) < 4:
        _log.warning(f"Only {len(samples)} samples; skipping the train/test holdout check")
        return
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(samples))
    n_test = max(1, round(len(samples) * test_fraction))
    train = [samples[i] for i in sorted(order[n_test:].tolist())]
    test = [samples[i] for i in sorted(order[:n_test].tolist())]
    if len(train) < 3 or not test:
        _log.warning("Too few samples for both a train and a test split; skipping the holdout check")
        return

    try:
        candidate = solve_eye_to_hand(train, HAND_EYE_METHODS[method])
    except cv2.error as e:
        _log.warning(f"Holdout solve failed ({e}); skipping the check")
        return
    if _rotation_defect(candidate) is not None:
        _log.warning("Holdout solve degenerated; skipping the check")
        return

    mean_offset = _mean_offset(_board_offsets(train, candidate))
    per_sample = _reproject_errors(test, candidate, mean_offset, K, dist)
    rms = float(np.sqrt(np.mean(per_sample**2)))
    _log.info(f"Holdout check: {len(train)} train / {len(test)} test samples, {rms:.2f} px RMS on the held-out set")
    if rms > max_rms_px:
        raise RuntimeError(
            f"held-out reprojection error {rms:.2f} px exceeds --outlier-px {max_rms_px:.2f}: a "
            f"calibration solved from {len(train)} of the samples does not generalize to the "
            f"{len(test)} it never saw. Samples are kept in the run directory for --replay; try "
            "more poses, a larger --rot-scale-deg, or check --square-size against the printout."
        )


def _rotation_diversity_deg(samples: list[Sample]) -> float:
    """Largest relative rotation between any two end-effector poses.

    Hand-eye is degenerate under pure translation: with the gripper always at the same orientation
    the rotation part of ``world_from_cam`` is unconstrained. This is the number that says whether
    the motion was rich enough.
    """
    rots = Rot.from_matrix(np.stack([s.world_from_ee[:3, :3] for s in samples]))
    angles = [(rots[i].inv() * rots[j]).magnitude() for i in range(len(samples)) for j in range(i + 1, len(samples))]
    return float(np.degrees(max(angles))) if angles else 0.0


# --------------------------------------------------------------------------------------------- #
# Capture                                                                                         #
# --------------------------------------------------------------------------------------------- #
def _perturbations(num_poses: int, pos_scale: float, rot_scale_deg: float, seed: int) -> list[np.ndarray]:
    """Local end-effector offsets to visit. The first is identity, so the anchor pose is sampled too."""
    rng = np.random.default_rng(seed)
    deltas = [np.eye(4)]
    for _ in range(num_poses - 1):
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        # Angles in [0.5, 1.0] x scale, not [0, 1]: samples clustered near the anchor orientation
        # add nothing to the rotation diversity the solve needs.
        angle = np.radians(rot_scale_deg) * rng.uniform(0.5, 1.0)
        deltas.append(_mat_from_rt(Rot.from_rotvec(axis * angle).as_matrix(), rng.uniform(-pos_scale, pos_scale, 3)))
    return deltas


def _lissajous_deltas(num_points: int, pos_scale: float, rot_scale_deg: float) -> list[np.ndarray]:
    """Local end-effector offsets tracing one loop of DROID's ``calibration_traj`` (third-person
    branch): a fixed Lissajous shape in position and xyz-Euler orientation, evenly sampled over one
    period. Unlike :func:`_perturbations`, consecutive points are close together, so the poses trace
    a continuous sweep around the anchor rather than jumping to independent random ones."""
    angle_scale = np.radians(rot_scale_deg)
    deltas = []
    for t in np.linspace(0.0, 2.0 * np.pi, num_points, endpoint=False):
        pos = pos_scale * np.array([-abs(np.sin(3 * t)), -0.8 * np.sin(2 * t), 0.5 * np.sin(4 * t)])
        rot = angle_scale * np.array([-np.sin(4 * t), np.sin(3 * t), np.sin(2 * t)])
        deltas.append(_mat_from_pose6(np.concatenate([pos, rot])))
    return deltas


def _draw_detection(bgr: np.ndarray, img_pts: np.ndarray, cam_from_board: np.ndarray, K, dist) -> np.ndarray:
    out = bgr.copy()
    for pt in img_pts.astype(int):
        cv2.circle(out, tuple(pt), 3, (0, 255, 0), -1)
    cv2.drawFrameAxes(out, K, dist, cv2.Rodrigues(cam_from_board[:3, :3])[0], cam_from_board[:3, 3], 0.05)
    return out


@dataclass
class Rig:
    """The camera and the arm, behind the handful of operations everything here needs.

    Bundled so every mode drives identical hardware setup — the servo loop has to move the arm, so
    it needs the same motion planner the capture loop does, and ``manual`` needs the same forward
    kinematics without any of the moving.
    """

    serial: str
    K: np.ndarray
    dist: np.ndarray
    read_frame: object  # () -> Frame
    read_q: object  # () -> (6,) MEASURED joint positions
    fk: object  # ((6,)) -> 4x4 world_from_ee
    move_to_pose: object  # (4x4) -> bool, plan and execute; False if unreachable
    set_free_drive: object  # (bool) -> None, hand-guidable arm with the gripper still clamped
    close: object  # () -> None

    def world_from_ee(self) -> np.ndarray:
        """Where the gripper is now, from the MEASURED joints."""
        return self.fk(self.read_q())

    def observe(self, spec: BoardSpec, board) -> tuple | None:
        """One frame with the board solved: ``(frame, obj_pts, img_pts, cam_from_board, err, incidence,
        coverage)``, or None if the board was not detected."""
        frame = self.read_frame()
        detection = detect_board(frame.bgr, spec, board)
        if detection is None:
            return None
        obj_pts, img_pts = detection
        cam_from_board, err = solve_pnp(obj_pts, img_pts, self.K, self.dist)
        incidence, coverage = presentation(cam_from_board, img_pts, frame.bgr.shape[:2][::-1])
        return frame, obj_pts, img_pts, cam_from_board, err, incidence, coverage


def _open_rig(*, flush_frames: int, time_dilation_factor: float, close_gripper: bool, warmup_iters: int) -> Rig:
    """Open the camera, park the idle arm and build the motion planner.

    ``warmup_iters`` is 0 for ``manual``, which never plans a motion: cuRobo is built there only for
    its forward kinematics, and warming up the trajectory optimiser for that is pure waiting.
    """
    import torch
    from curobo.geom.types import WorldConfig
    from curobo.types.math import Pose
    from curobo.types.state import JointState
    from curobo.wrap.reacher.motion_gen import MotionGenPlanConfig

    from tiptop.motion_planning import get_motion_gen
    from tiptop.perception.cameras import camera_mount, get_hand_camera
    from tiptop.utils import get_robot_client
    from tiptop.workspace import workspace_cuboids

    if camera_mount("hand") != "world":
        raise ValueError(
            "cameras.hand.mount is not 'world' in the active config, so its calibration entry is "
            "ee_from_cam, not world_from_cam. This script only calibrates a scene-fixed camera; use "
            "calibrate-wrist-cam for an end-effector one."
        )

    cam = get_hand_camera()
    intrinsics = cam.get_intrinsics()
    K = np.asarray(intrinsics.K_color, dtype=np.float64)
    dist = np.asarray(intrinsics.distortion_color, dtype=np.float64).reshape(-1, 1)
    _log.info(f"Camera {cam.serial}: fx={K[0, 0]:.2f} fy={K[1, 1]:.2f} cx={K[0, 2]:.2f} cy={K[1, 2]:.2f}")

    client = get_robot_client()
    parked = client.park_idle_arm()
    if not parked.get("success", False):
        raise RuntimeError(f"could not park the idle arm: {parked.get('error')}")
    if close_gripper:
        _log.info("Closing the gripper on the board")
        client.close_gripper()

    _log.info("Building the motion planner...")
    motion_gen = get_motion_gen(
        WorldConfig(cuboid=list(workspace_cuboids())), collision_activation_distance=0.01, warmup_iters=warmup_iters
    )
    plan_config = MotionGenPlanConfig(time_dilation_factor=time_dilation_factor)

    def read_q() -> np.ndarray:
        return np.asarray(client.get_joint_positions(), dtype=np.float64)

    def fk(q) -> np.ndarray:
        q_t = torch.tensor(np.asarray(q, dtype=np.float64), dtype=torch.float32, device="cuda")
        return motion_gen.kinematics.get_state(q_t).ee_pose.get_numpy_matrix()[0].astype(np.float64)

    def set_free_drive(enabled: bool) -> None:
        result = client.free_drive(hold_gripper=True) if enabled else client.hold()
        if not result.get("success", False):
            raise RuntimeError(f"could not {'free-drive' if enabled else 'hold'} the arm: {result.get('error')}")

    def read_frame():
        for _ in range(max(0, flush_frames - 1)):
            cam.read_camera()
        return cam.read_camera()

    def move_to_pose(target: np.ndarray) -> bool:
        q_curr = torch.tensor(client.get_joint_positions(), dtype=torch.float32, device="cuda")
        result = motion_gen.plan_single(
            JointState.from_position(q_curr[None]),
            Pose.from_matrix(torch.tensor(target, dtype=torch.float32, device="cuda")),
            plan_config,
        )
        if not bool(result.success.all()):
            _log.debug(f"no plan to the requested pose ({result.status})")
            return False
        plan = result.interpolated_plan
        exec_result = client.execute_joint_impedance_path(
            joint_confs=plan.position.cpu().numpy(),
            joint_vels=plan.velocity.cpu().numpy(),
            durations=[result.interpolation_dt] * plan.position.shape[0],
        )
        if not exec_result["success"]:
            raise RuntimeError(f"could not move the arm: {exec_result.get('error')}")
        return True

    return Rig(cam.serial, K, dist, read_frame, read_q, fk, move_to_pose, set_free_drive, cam.close)


def _prior_world_from_cam(cam_serial: str) -> np.ndarray:
    """A rough ``world_from_cam`` to start aiming from.

    Aiming is circular on the face of it — to know which way to turn the wrist you need the very
    transform you are trying to measure. The way out is that aiming only needs the ROTATION, only
    needs it approximately, and runs closed loop: any guess within ~90 deg gives a corrective step
    in the right direction, and the loop re-measures and repeats. The stored calibration (or the sim
    design pose) is well inside that, and :func:`_refine_aim_prior` replaces it with a real estimate
    as soon as the arm has moved enough to support one.
    """
    from tiptop.config import load_calibration

    try:
        prior = load_calibration(cam_serial)
        _log.info("Aiming from the stored calibration's rotation")
    except (ValueError, FileNotFoundError):
        prior = _mat_from_pose6(SIM_DESIGN_POSE)
        _log.info("No stored calibration for this camera; aiming from the sim design rotation")
    return prior


def _refine_aim_prior(samples: list[Sample], prior: np.ndarray) -> np.ndarray:
    """Re-estimate the aiming rotation from the poses visited so far.

    The servo generates exactly the data a hand-eye solve needs, so once the arm has rotated enough
    the guess can be replaced by a measurement. This is what makes aiming robust to a stored
    calibration that is badly wrong — including one for a camera that has since been remounted.
    """
    if len(samples) < 4 or _rotation_diversity_deg(samples) < 15.0:
        return prior
    try:
        candidate = solve_eye_to_hand(samples, HAND_EYE_METHODS["daniilidis"])
    except cv2.error:
        return prior
    if _rotation_defect(candidate) is not None:
        return prior
    drift = np.degrees(Rot.from_matrix(candidate[:3, :3].T @ prior[:3, :3]).magnitude())
    _log.info(f"Refined the aiming rotation from {len(samples)} poses ({drift:.1f} deg from the guess)")
    return candidate


def _aim_correction(
    cam_from_board: np.ndarray,
    img_pts: np.ndarray,
    K: np.ndarray,
    image_wh: tuple[int, int],
    max_step_deg: float,
    max_step_m: float,
    gain: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """A camera-frame ``(rotation, translation, remaining incidence deg)`` step toward facing the camera.

    Rotation turns the board's normal onto the camera ray; translation slides it toward the middle
    of the frame. Both are damped by ``gain`` and clipped, because the mapping into world commands
    is only approximate — a step that overshoots would oscillate.

    Depth is deliberately left alone. Driving the board closer to raise coverage means driving the
    gripper at the camera and the base column it is bolted to, and neither is in the collision model.
    """
    depth = float(cam_from_board[2, 3])
    ray = cam_from_board[:3, 3] / max(np.linalg.norm(cam_from_board[:3, 3]), 1e-9)
    # A board square-on to the camera has cam_from_board rotation = I, so its +z runs ALONG the ray,
    # away from the viewer — OpenCV's board axes point into the scene, not back at it. Facing the
    # camera therefore means normal parallel to the ray. Either sign leaves the plane square-on, so
    # take whichever is nearer and the correction is always the short way round; aiming at the far
    # one would drag the board through 90 deg, exactly where it is too edge-on to detect at all.
    normal = cam_from_board[:3, 2]
    target = ray if normal @ ray >= 0 else -ray
    axis = np.cross(normal, target)
    sin_a, cos_a = float(np.linalg.norm(axis)), float(normal @ target)
    incidence = float(np.degrees(np.arctan2(sin_a, cos_a)))

    rot = np.eye(3)
    if sin_a > 1e-6:
        step = np.radians(min(incidence * gain, max_step_deg))
        rot = Rot.from_rotvec(axis / sin_a * step).as_matrix()

    # Recentre using the detected corners' centroid, not the board origin: the origin is a corner of
    # the board and can sit outside the frame even when the visible part is nicely placed.
    centroid = img_pts.mean(axis=0)
    offset_px = np.array([image_wh[0] / 2.0, image_wh[1] / 2.0]) - centroid
    shift = np.array([offset_px[0] * depth / K[0, 0], offset_px[1] * depth / K[1, 1], 0.0]) * gain
    if np.linalg.norm(shift) > max_step_m:
        shift = shift / np.linalg.norm(shift) * max_step_m
    return rot, shift, incidence


def _verdict(
    n_corners: int, incidence: float, coverage: float, min_corners: int, max_incidence_deg: float, min_coverage: float
) -> list[str]:
    """What is still wrong with how the board is presented; empty means good."""
    problems = []
    if n_corners < min_corners:
        problems.append(f"only {n_corners} corners")
    if incidence > max_incidence_deg:
        problems.append(f"{incidence:.0f} deg too edge-on")
    if coverage < min_coverage:
        problems.append(f"{coverage * 100:.0f}% of frame, too small")
    return problems


def auto_aim(
    rig: Rig,
    spec: BoardSpec,
    board,
    *,
    min_corners: int,
    max_incidence_deg: float,
    min_coverage: float,
    max_steps: int,
    gain: float = 0.7,
    max_step_deg: float = 20.0,
    max_step_m: float = 0.04,
    settle_s: float = 0.4,
) -> bool:
    """Turn the wrist until the board faces the camera. True if it got there.

    Visual servoing with an approximate Jacobian: each step measures the board, computes the
    correction in the camera frame, maps it into a world-frame wrist command through the current
    rotation guess, and moves. The guess only has to be right enough to point downhill, and it gets
    replaced by a real estimate once the arm has rotated enough (:func:`_refine_aim_prior`).

    Only the wrist turns and the board slides sideways — never toward the camera. See
    :func:`_aim_correction`.
    """
    prior = _prior_world_from_cam(rig.serial)
    samples: list[Sample] = []
    best = None

    for step in range(max_steps):
        observed = rig.observe(spec, board)
        if observed is None:
            _log.info(f"[aim {step + 1}/{max_steps}] no board detected — is it in the camera's view?")
            return False
        frame, obj_pts, img_pts, cam_from_board, err, incidence, coverage = observed
        problems = _verdict(len(obj_pts), incidence, coverage, min_corners, max_incidence_deg, min_coverage)
        _log.info(
            f"[aim {step + 1}/{max_steps}] {len(obj_pts):3d} corners, incidence {incidence:5.1f} deg, "
            f"coverage {coverage * 100:4.1f}%  ->  {'GOOD' if not problems else ' | '.join(problems)}"
        )
        if not problems:
            return True

        world_from_ee = rig.world_from_ee()
        samples.append(Sample(len(samples), world_from_ee, cam_from_board, obj_pts, img_pts, err))
        prior = _refine_aim_prior(samples, prior)
        if best is None or incidence < best:
            best = incidence

        if incidence <= max_incidence_deg:
            # Rotation is already fine; nothing here can fix corner count or coverage, which are
            # about how big the board is and how close it is — both physical setup.
            _log.warning("The board faces the camera but is too small or too partly visible in frame.")
            return False

        rot_c, shift_c, _ = _aim_correction(
            cam_from_board, img_pts, rig.K, frame.bgr.shape[:2][::-1], max_step_deg, max_step_m, gain
        )
        R_wc = prior[:3, :3]
        target = world_from_ee.copy()
        target[:3, :3] = (R_wc @ rot_c @ R_wc.T) @ world_from_ee[:3, :3]
        target[:3, 3] = world_from_ee[:3, 3] + R_wc @ shift_c

        if not rig.move_to_pose(target):
            # Unreachable in one go; a half step usually is reachable and still makes progress.
            half = world_from_ee.copy()
            half[:3, :3] = (R_wc @ Rot.from_rotvec(Rot.from_matrix(rot_c).as_rotvec() * 0.5).as_matrix() @ R_wc.T) @ (
                world_from_ee[:3, :3]
            )
            half[:3, 3] = world_from_ee[:3, 3] + R_wc @ shift_c * 0.5
            if not rig.move_to_pose(half):
                _log.warning(
                    f"Cannot reach a pose that improves on {incidence:.0f} deg incidence — the wrist "
                    "is out of range. Re-clamp the board at a different angle in the jaws, or "
                    "hand-guide the arm somewhere it has more room."
                )
                return False
        time.sleep(settle_s)

    _log.warning(f"Gave up after {max_steps} aiming steps; best incidence reached was {best:.0f} deg")
    return False


def _capture_samples(
    *,
    rig: Rig,
    spec: BoardSpec,
    board,
    num_poses: int,
    pos_scale: float,
    rot_scale_deg: float,
    seed: int,
    settle_s: float,
    max_pnp_error_px: float,
    min_corners: int,
    max_incidence_deg: float,
    min_coverage: float,
    run_dir: Path,
) -> list[Sample]:
    """Drive the arm through perturbed poses around where it is now and collect board observations."""
    frames_dir = run_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    anchor = rig.world_from_ee()
    _log.info(f"Anchor end-effector position in world: {np.round(anchor[:3, 3], 4)}")
    _log.warning("The board is NOT in the collision model. Keep the workspace clear — the arm is about to move.")

    samples: list[Sample] = []
    for i, delta in enumerate(_perturbations(num_poses, pos_scale, rot_scale_deg, seed)):
        # The first perturbation is identity by construction: sample the anchor where the arm
        # already is rather than asking cuRobo to plan a zero-length motion to it.
        if i > 0 and not rig.move_to_pose(anchor @ delta):
            _log.info(f"[{i + 1}/{num_poses}] no plan to this pose; skipping")
            continue

        time.sleep(settle_s)
        world_from_ee = rig.world_from_ee()
        observed = rig.observe(spec, board)
        if observed is None:
            _log.info(f"[{i + 1}/{num_poses}] board not detected; skipping")
            continue
        frame, obj_pts, img_pts, cam_from_board, pnp_err, incidence, coverage = observed

        # A low PnP error is necessary but nowhere near sufficient: a sliver of an edge-on board
        # fits its own corners beautifully and still puts the board pose metres out of place.
        reject = None
        if pnp_err > max_pnp_error_px:
            reject = f"PnP error {pnp_err:.2f} px over {max_pnp_error_px}"
        elif problems := _verdict(len(obj_pts), incidence, coverage, min_corners, max_incidence_deg, min_coverage):
            reject = " | ".join(problems)
        if reject is not None:
            _log.info(f"[{i + 1}/{num_poses}] {reject}; skipping")
            continue

        sample = Sample(len(samples), world_from_ee, cam_from_board, obj_pts, img_pts, pnp_err)
        samples.append(sample)
        cv2.imwrite(str(frames_dir / f"{sample.index:03d}.png"), frame.bgr)
        cv2.imwrite(
            str(frames_dir / f"{sample.index:03d}_detected.png"),
            _draw_detection(frame.bgr, img_pts, cam_from_board, rig.K, rig.dist),
        )
        _log.info(
            f"[{i + 1}/{num_poses}] captured sample {sample.index}: {len(obj_pts)} corners, "
            f"incidence {incidence:.0f} deg, coverage {coverage * 100:.0f}%, {pnp_err:.2f} px"
        )

    return samples


def _is_novel(world_from_ee: np.ndarray, samples: list[Sample], min_trans_mm: float, min_rot_deg: float) -> bool:
    """Whether this pose differs enough from EVERY captured one to be worth another sample.

    Duplicates are not neutral. Hand-eye weights every sample equally, so re-capturing one spot ten
    times pulls the solve toward whatever that spot's errors are, and the residuals look better for
    it — the extra samples agree with each other by construction. A pose has to be new in
    translation or in rotation; being new in neither is the same measurement again.
    """
    return all(
        trans_mm >= min_trans_mm or rot_deg >= min_rot_deg
        for trans_mm, rot_deg in (_pose_delta(s.world_from_ee, world_from_ee) for s in samples)
    )


def _capture_manual(
    *,
    rig: Rig,
    spec: BoardSpec,
    board,
    num_poses: int,
    max_pnp_error_px: float,
    min_corners: int,
    max_incidence_deg: float,
    min_coverage: float,
    still_rad: float,
    still_s: float,
    min_new_trans_mm: float,
    min_new_rot_deg: float,
    poll_s: float,
    run_dir: Path,
) -> list[Sample]:
    """Let the OPERATOR move the board: the arm goes limp and every new, steady pose is captured.

    The data is exactly what :func:`_capture_samples` gathers — the board still rides in the gripper,
    so FK of the measured joints still says where it is, and the eye-to-hand solve is unchanged. Only
    who chooses the poses differs. That matters when the arm cannot plan its way around the board's
    clamp (which is not in the collision model), or when the useful poses are somewhere cuRobo will
    not go.

    Nothing is captured while the arm is moving: the joint reading and the frame are taken a few
    milliseconds apart, and at arm's length a slow drift between them is millimetres of error that
    no residual would reveal. Hence the dwell, and the re-check afterwards that the arm did not stir
    during the capture itself.
    """
    frames_dir = run_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    rig.set_free_drive(True)
    _log.info("=" * 78)
    _log.info(
        f"The arm is now LIMP — take hold of it and move the board around. Up to {num_poses} poses; "
        f"each is captured on its own once you hold still for {still_s:.1f} s somewhere new. TURN "
        "the board as well as sliding it: the solve needs rotation, and a pure translation sweep "
        "cannot constrain the camera's orientation at all. Ctrl-C stops and solves with what you have."
    )
    _log.info("=" * 78)

    samples: list[Sample] = []
    anchor_q, moved_at, last_report = rig.read_q(), time.time(), 0.0
    try:
        while len(samples) < num_poses:
            time.sleep(poll_s)
            q = rig.read_q()
            if float(np.abs(q - anchor_q).max()) > still_rad:
                anchor_q, moved_at = q, time.time()
                continue
            if time.time() - moved_at < still_s:
                continue

            world_from_ee = rig.fk(q)
            observed = rig.observe(spec, board)
            if observed is None:
                reject = "no board detected — is your hand covering it?"
            else:
                frame, obj_pts, img_pts, cam_from_board, pnp_err, incidence, coverage = observed
                problems = _verdict(len(obj_pts), incidence, coverage, min_corners, max_incidence_deg, min_coverage)
                if pnp_err > max_pnp_error_px:
                    problems.append(f"PnP error {pnp_err:.2f} px over {max_pnp_error_px}")
                if problems:
                    reject = " | ".join(problems)
                elif not _is_novel(world_from_ee, samples, min_new_trans_mm, min_new_rot_deg):
                    reject = (
                        f"too close to a pose already captured — move it {min_new_trans_mm:.0f} mm "
                        f"or turn it {min_new_rot_deg:.0f} deg"
                    )
                elif float(np.abs(rig.read_q() - q).max()) > still_rad:
                    reject = "the arm moved while the frame was being taken"
                else:
                    reject = None

            if reject is not None:
                if time.time() - last_report > 1.5:
                    _log.info(f"[{len(samples)}/{num_poses}] held still, but: {reject}")
                    last_report = time.time()
                continue

            sample = Sample(len(samples), world_from_ee, cam_from_board, obj_pts, img_pts, pnp_err)
            samples.append(sample)
            cv2.imwrite(str(frames_dir / f"{sample.index:03d}.png"), frame.bgr)
            cv2.imwrite(
                str(frames_dir / f"{sample.index:03d}_detected.png"),
                _draw_detection(frame.bgr, img_pts, cam_from_board, rig.K, rig.dist),
            )
            # \a rings the terminal bell: hands are on the robot and eyes are on the board, not here.
            _log.info(
                f"\a[{len(samples)}/{num_poses}] CAPTURED: {len(obj_pts)} corners, incidence "
                f"{incidence:.0f} deg, coverage {coverage * 100:.0f}%, {pnp_err:.2f} px. Rotation "
                f"spread so far {_rotation_diversity_deg(samples):.0f} deg"
            )
            last_report = time.time()
    except KeyboardInterrupt:
        _log.info(f"Stopped by hand with {len(samples)} samples")
    finally:
        rig.set_free_drive(False)
        _log.info("The arm is holding position again")

    return samples


def _set_anchor_interactively(rig: Rig) -> np.ndarray:
    """Free-drive the arm so the operator can pose the board by hand, then lock in that pose as a
    sweep's anchor once they press Enter. The DROID equivalent of teleoperating to a starting pose
    and pressing the controller's A button before the scripted sweep takes over."""
    rig.set_free_drive(True)
    _log.info("=" * 78)
    _log.info(
        "The arm is now LIMP — move the board to a good starting pose, facing the camera. Press "
        "Enter here once you are happy with it; the arm will hold there and then sweep around it."
    )
    _log.info("=" * 78)
    try:
        input()
    finally:
        rig.set_free_drive(False)
    anchor = rig.world_from_ee()
    _log.info(f"Anchor end-effector position in world: {np.round(anchor[:3, 3], 4)}")
    return anchor


def _capture_sweep(
    *,
    rig: Rig,
    spec: BoardSpec,
    board,
    num_poses: int,
    pos_scale: float,
    rot_scale_deg: float,
    settle_s: float,
    max_pnp_error_px: float,
    min_corners: int,
    max_incidence_deg: float,
    min_coverage: float,
    run_dir: Path,
) -> list[Sample]:
    """Drive the arm through one Lissajous loop around an operator-chosen anchor, DROID-style.

    The math and rejection logic are identical to :func:`_capture_samples`; only how the poses are
    chosen differs — a continuous sweep (:func:`_lissajous_deltas`) instead of independent random
    perturbations, and an anchor the operator set by hand rather than wherever the arm happened to
    be when aiming finished.
    """
    frames_dir = run_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    anchor = _set_anchor_interactively(rig)
    _log.warning("The board is NOT in the collision model. Keep the workspace clear — the arm is about to sweep.")

    samples: list[Sample] = []
    for i, delta in enumerate(_lissajous_deltas(num_poses, pos_scale, rot_scale_deg)):
        if not rig.move_to_pose(anchor @ delta):
            _log.info(f"[{i + 1}/{num_poses}] no plan to this point on the sweep; skipping")
            continue

        time.sleep(settle_s)
        world_from_ee = rig.world_from_ee()
        observed = rig.observe(spec, board)
        if observed is None:
            _log.info(f"[{i + 1}/{num_poses}] board not detected; skipping")
            continue
        frame, obj_pts, img_pts, cam_from_board, pnp_err, incidence, coverage = observed

        reject = None
        if pnp_err > max_pnp_error_px:
            reject = f"PnP error {pnp_err:.2f} px over {max_pnp_error_px}"
        elif problems := _verdict(len(obj_pts), incidence, coverage, min_corners, max_incidence_deg, min_coverage):
            reject = " | ".join(problems)
        if reject is not None:
            _log.info(f"[{i + 1}/{num_poses}] {reject}; skipping")
            continue

        sample = Sample(len(samples), world_from_ee, cam_from_board, obj_pts, img_pts, pnp_err)
        samples.append(sample)
        cv2.imwrite(str(frames_dir / f"{sample.index:03d}.png"), frame.bgr)
        cv2.imwrite(
            str(frames_dir / f"{sample.index:03d}_detected.png"),
            _draw_detection(frame.bgr, img_pts, cam_from_board, rig.K, rig.dist),
        )
        _log.info(
            f"[{i + 1}/{num_poses}] captured sample {sample.index}: {len(obj_pts)} corners, "
            f"incidence {incidence:.0f} deg, coverage {coverage * 100:.0f}%, {pnp_err:.2f} px"
        )

    return samples


def _watch_presentation(
    rig: Rig,
    spec: BoardSpec,
    board,
    *,
    seconds: float,
    min_corners: int,
    max_incidence_deg: float,
    min_coverage: float,
) -> None:
    """Report how the board is presented, moving nothing, until it is good or time runs out."""
    _log.info(
        f"Watching. Wanted: at least {min_corners} corners, incidence under {max_incidence_deg:.0f} "
        f"deg, coverage over {min_coverage * 100:.0f}%. Ctrl-C to stop."
    )
    deadline = time.time() + seconds
    try:
        while time.time() < deadline:
            observed = rig.observe(spec, board)
            if observed is None:
                _log.info("no board detected")
                continue
            _, obj_pts, _, _, err, incidence, coverage = observed
            problems = _verdict(len(obj_pts), incidence, coverage, min_corners, max_incidence_deg, min_coverage)
            _log.info(
                f"{len(obj_pts):3d} corners, incidence {incidence:5.1f} deg, coverage {coverage * 100:4.1f}%, "
                f"{err:.2f} px  ->  {'GOOD' if not problems else ' | '.join(problems)}"
            )
            if not problems:
                return
    except KeyboardInterrupt:
        _log.info("stopped watching")


def _write_aim_image(rig: Rig, spec: BoardSpec, board, run_dir: Path) -> None:
    observed = rig.observe(spec, board)
    if observed is None:
        return
    frame, _, img_pts, cam_from_board, _, _, _ = observed
    cv2.imwrite(str(run_dir / "aim.png"), _draw_detection(frame.bgr, img_pts, cam_from_board, rig.K, rig.dist))
    _log.info(f"Wrote {run_dir / 'aim.png'}")


def _capture_static(
    *, spec: BoardSpec, board, world_from_board: np.ndarray, num_frames: int, flush_frames: int, run_dir: Path
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str, float]:
    """Average ``cam_from_board`` over a burst of frames and invert through the known board pose."""
    from tiptop.perception.cameras import get_hand_camera

    cam = get_hand_camera()
    intrinsics = cam.get_intrinsics()
    K = np.asarray(intrinsics.K_color, dtype=np.float64)
    dist = np.asarray(intrinsics.distortion_color, dtype=np.float64).reshape(-1, 1)

    poses, errors = [], []
    for _ in range(num_frames):
        for _ in range(max(0, flush_frames - 1)):
            cam.read_camera()
        frame = cam.read_camera()
        detection = detect_board(frame.bgr, spec, board)
        if detection is None:
            continue
        obj_pts, img_pts = detection
        cam_from_board, err = solve_pnp(obj_pts, img_pts, K, dist)
        poses.append(cam_from_board)
        errors.append(err)
        cv2.imwrite(
            str(run_dir / f"static_{len(poses):02d}.png"), _draw_detection(frame.bgr, img_pts, cam_from_board, K, dist)
        )

    if not poses:
        raise RuntimeError(f"the {spec.kind} board was never detected; check lighting and that it is in view")

    stacked = np.stack(poses)
    mean = np.eye(4)
    mean[:3, :3] = Rot.from_matrix(stacked[:, :3, :3]).mean().as_matrix()
    mean[:3, 3] = stacked[:, :3, 3].mean(axis=0)
    _log.info(f"Averaged {len(poses)} detections; PnP error {np.mean(errors):.2f} px")
    cam.close()
    return world_from_board @ np.linalg.inv(mean), K, dist, cam.serial, float(np.mean(errors))


# --------------------------------------------------------------------------------------------- #
# Reporting                                                                                       #
# --------------------------------------------------------------------------------------------- #
def _draw_axes(img: np.ndarray, cam_from_frame: np.ndarray, K, dist, length: float, thickness: int) -> None:
    """Draw a frame's axes, skipping anything behind or wildly outside the image plane."""
    if cam_from_frame[2, 3] <= 0.05:
        return
    try:
        cv2.drawFrameAxes(
            img, K, dist, cv2.Rodrigues(cam_from_frame[:3, :3])[0], cam_from_frame[:3, 3], length, thickness
        )
    except cv2.error:
        pass


def _write_verification(run_dir: Path, samples: list[Sample], world_from_cam: np.ndarray, K, dist) -> None:
    """Draw the world origin and every sampled end-effector frame into the last captured image.

    The most convincing check there is: if the projected gripper frames do not land on the gripper
    in the picture, the calibration is wrong whatever the residuals say. The world origin often
    falls outside the frame (it is 0.24 m behind the base column) — that is fine, the end-effector
    frames are the ones to look at.
    """
    last = run_dir / "frames" / f"{samples[-1].index:03d}.png"
    if not last.exists():
        return
    img = cv2.imread(str(last))
    cam_from_world = np.linalg.inv(world_from_cam)
    _draw_axes(img, cam_from_world, K, dist, 0.15, 3)
    for s in samples:
        _draw_axes(img, cam_from_world @ s.world_from_ee, K, dist, 0.03, 1)
    cv2.putText(
        img,
        "thick axes = world origin, thin = sampled EE poses",
        (15, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
    )
    cv2.imwrite(str(run_dir / "verification.png"), img)


def _report(world_from_cam: np.ndarray, metrics: Metrics | None, samples: list[Sample], cam_serial: str) -> np.ndarray:
    from tiptop.config import load_calibration

    pose6 = _pose6_from_mat(world_from_cam)
    _log.info("=" * 78)
    _log.info(f"world_from_cam for {cam_serial}")
    _log.info(f"  position (m):     {np.round(pose6[:3], 5).tolist()}")
    _log.info(f"  xyz euler (rad):  {np.round(pose6[3:], 6).tolist()}")
    _log.info(f"  view direction:   {np.round(world_from_cam[:3, 2], 4).tolist()}  (optical +z, in world)")
    if metrics is not None:
        _log.info(f"  samples used:     {len(samples)}")
        _log.info(f"  reprojection:     {metrics.reproj_rms_px:.2f} px RMS, {metrics.reproj_max_px:.2f} px worst")
        _log.info(
            f"  ee_from_board:    {metrics.offset_trans_rms_mm:.2f} mm / {metrics.offset_rot_rms_deg:.3f} deg spread"
        )

    try:
        stored = load_calibration(cam_serial)
    except (ValueError, FileNotFoundError):
        stored = None
    if stored is not None:
        _log.info("  vs stored calibration: {:.1f} mm, {:.2f} deg".format(*_pose_delta(stored, world_from_cam)))
    _log.info(
        "  vs sim design pose:    {:.1f} mm, {:.2f} deg".format(
            *_pose_delta(_mat_from_pose6(SIM_DESIGN_POSE), world_from_cam)
        )
    )
    _log.info("=" * 78)
    return pose6


# --------------------------------------------------------------------------------------------- #
# Entry point                                                                                     #
# --------------------------------------------------------------------------------------------- #
def calibrate_top_cam(
    arm: Literal["left", "right"] = "left",
    mode: Literal["handeye", "manual", "static", "aim", "sweep"] = "handeye",
    board_kind: BoardKind = "charuco",
    cols: int | None = None,
    rows: int | None = None,
    square_size: float = 0.020,
    marker_size: float = 0.015,
    aruco_dict: str = "DICT_5X5_100",
    num_poses: int = 24,
    min_samples: int = 8,
    pos_scale: float = 0.05,
    rot_scale_deg: float = 25.0,
    seed: int = 0,
    settle_s: float = 0.5,
    flush_frames: int = 3,
    time_dilation_factor: float = 0.3,
    method: str = "auto",
    max_pnp_error_px: float = 1.5,
    min_corners: int = 20,
    max_incidence_deg: float = 55.0,
    min_coverage: float = 0.25,
    aim_seconds: float = 120.0,
    auto_aim_steps: int = 12,
    still_rad: float = 0.01,
    still_s: float = 1.0,
    min_new_trans_mm: float = 30.0,
    min_new_rot_deg: float = 8.0,
    poll_s: float = 0.05,
    outlier_px: float = 4.0,
    close_gripper: bool = False,
    board_pose: tuple[float, float, float, float, float, float] | None = None,
    static_frames: int = 10,
    holdout_test_fraction: float = 0.3,
    replay: Path | None = None,
    save: bool = True,
):
    """Solve and store ``world_from_cam`` for the YAM's fixed third-person camera.

    Args:
        arm: Which arm holds the board. The other one is parked at the neutral posture first.
        mode: ``handeye`` (the arm moves the board), ``manual`` (you move it — same board on the
            same gripper, but the arm goes limp and you hand-guide it), ``static`` (board at a known
            world pose), ``aim`` — turn the wrist until the board faces the camera, then stop
            without calibrating — or ``sweep`` — you hand-guide it to a starting pose and press
            Enter, then the arm sweeps it through a DROID-style Lissajous loop around that pose.
        board_kind: ``charuco`` is strongly preferred — it survives partial occlusion and is not
            rotationally ambiguous, both of which matter when the gripper covers part of the board.
        cols: Board width. SQUARES for charuco (default 14), INNER CORNERS for chessboard (default 9).
        rows: Board height. SQUARES for charuco (default 9), INNER CORNERS for chessboard (default 6).
        square_size: Checker square edge length, metres. Measure the printout — printer scaling is
            the most common reason a calibration comes out with the right rotation and wrong depth.
        marker_size: ArUco marker edge length, metres. Charuco only.
        num_poses: How many poses to attempt. Unreachable ones and frames with no detection are
            skipped, so the number of samples is lower. In ``manual`` it is where the capture stops
            on its own; Ctrl-C ends it earlier and solves with what has been collected. In ``sweep``
            it is how many points are sampled along the one Lissajous loop.
        min_samples: Refuse to solve with fewer than this many samples.
        pos_scale: Half-width of the translation perturbation box, metres in ``handeye``; the
            Lissajous loop's translation amplitude, metres, in ``sweep``.
        rot_scale_deg: Maximum orientation perturbation, degrees, in ``handeye``; the Lissajous
            loop's orientation amplitude, degrees, in ``sweep``. Bigger is better for conditioning up
            to the point where the board leaves the camera's view.
        time_dilation_factor: Speed of the calibration motions; these are small, so keep it slow.
        method: A key of ``HAND_EYE_METHODS``, or ``auto`` to run all five and keep the best.
        max_pnp_error_px: Reject a frame whose board detection alone does not fit this well.
        min_corners: Reject a frame that detected fewer board corners than this.
        max_incidence_deg: Reject a frame where the board is more edge-on than this. A low
            per-frame fit error does NOT mean the pose is good — a sliver of an edge-on board fits
            its own corners perfectly with its normal metres out of place.
        min_coverage: Reject a frame where the board's bounding box spans less than this fraction
            of the image diagonal.
        aim_seconds: How long to keep reporting once aiming has stopped moving the arm, so you can
            adjust the board by hand and watch the numbers. 0 disables the watch.
        auto_aim_steps: How many servo steps to allow when turning the board to face the camera. 0
            turns automatic aiming off — the run then fails on a poorly presented board instead of
            fixing it. Not used by ``manual``, which never moves the arm.
        still_rad: ``manual`` only. How far any joint may wander and still count as held still, and
            the same tolerance the arm must not exceed while a sample is being taken.
        still_s: ``manual`` only. How long to hold a pose before it is captured.
        min_new_trans_mm: ``manual`` only. A pose counts as new if it is this far from every
            captured one, OR turned by ``min_new_rot_deg``. Being new in neither is a duplicate.
        min_new_rot_deg: ``manual`` only. The rotation half of the same test.
        poll_s: ``manual`` only. How often the joints are read while waiting for you to settle.
        outlier_px: Keep dropping the worst-reprojecting sample and re-solving until none exceeds
            this, up to a third of the set.
        close_gripper: Close the gripper before moving, for a board clamped in the jaws.
        board_pose: ``static`` mode only — ``world_from_board`` as ``x y z roll pitch yaw``.
        holdout_test_fraction: ``sweep`` mode only. Fraction of samples held out to check the solve
            generalizes, DROID-style; ``--outlier-px`` doubles as the acceptance threshold for the
            held-out RMS.
        replay: Re-solve from a previous run directory instead of touching hardware.
        save: Write the result into the active workspace's calibration file.
    """
    from tiptop.config import update_calibration_info, workspace_calib_path
    from tiptop.utils import get_tiptop_cache_dir
    from tiptop.yam import active_arm

    defaults = {"charuco": (14, 9), "chessboard": (9, 6)}[board_kind]
    spec = BoardSpec(
        kind=board_kind,
        cols=cols if cols is not None else defaults[0],
        rows=rows if rows is not None else defaults[1],
        square_size=square_size,
        marker_size=marker_size,
        aruco_dict=aruco_dict,
    )
    spec.validate()
    if method != "auto" and method not in HAND_EYE_METHODS:
        raise ValueError(f"unknown hand-eye method {method!r}; expected 'auto' or one of {list(HAND_EYE_METHODS)}")
    if mode == "sweep" and method == "auto":
        method = "daniilidis"
        _log.info("mode=sweep defaults --method to daniilidis, mirroring DROID's fixed method=4")
    if spec.kind == "chessboard" and replay is None:
        _log.warning(
            "A plain chessboard needs the WHOLE grid visible in every frame and can order its "
            "corners from either end when the board is near 180 deg rotated. Prefer --board-kind "
            "charuco if you have one; if not, keep the board upright and fully in view."
        )

    # update_calibration_info writes to the workspace layer when DC_WORKSPACE is set and to the
    # SHARED default file when it is not — and that default is the Franka rig's. Writing a YAM
    # world_from_cam there would shadow a wrist camera's ee_from_cam for every other robot.
    if save and workspace_calib_path() is None:
        raise ValueError(
            "DC_WORKSPACE is not set, so the result would be written into the shared default "
            "calibration_info.json rather than this rig's layer. Rerun with DC_WORKSPACE=<workspace> "
            "(e.g. prism), or pass --no-save to solve without storing."
        )

    if replay is not None:
        run_dir = Path(replay)
        samples, K, dist, cam_serial = _load_samples(run_dir)
        _log.info(f"Replaying {len(samples)} samples from {run_dir}")
    else:
        run_dir = get_tiptop_cache_dir() / "top_cam_calib" / time.strftime("%Y%m%d-%H%M%S")
        run_dir.mkdir(parents=True, exist_ok=True)
        _log.info(f"Run directory: {run_dir}")
        board = _charuco_board(spec) if spec.kind == "charuco" else None

        if mode == "static":
            if board_pose is None:
                raise ValueError("--mode static needs --board-pose x y z roll pitch yaw (world_from_board)")
            world_from_cam, K, dist, cam_serial, pnp_px = _capture_static(
                spec=spec,
                board=board,
                world_from_board=_mat_from_pose6(board_pose),
                num_frames=static_frames,
                flush_frames=flush_frames,
                run_dir=run_dir,
            )
            pose6 = _report(world_from_cam, None, [], cam_serial)
            _log.info(f"Static-mode PnP error: {pnp_px:.2f} px. Accuracy is bounded by --board-pose.")
            (run_dir / "result.json").write_text(
                json.dumps(
                    {
                        "mode": "static",
                        "camera_serial": cam_serial,
                        "pose": pose6.tolist(),
                        "world_from_cam": world_from_cam.tolist(),
                        "board": spec.to_json(),
                        "board_pose": list(board_pose),
                        "pnp_error_px": pnp_px,
                    },
                    indent=2,
                )
            )
            if save:
                update_calibration_info(cam_serial, pose6)
            return world_from_cam

        with active_arm(arm):
            rig = _open_rig(
                flush_frames=flush_frames,
                time_dilation_factor=time_dilation_factor,
                close_gripper=close_gripper,
                # manual never plans a motion, so cuRobo is built for its kinematics alone.
                # sweep plans one short move per Lissajous point, so it needs the planner warmed up.
                warmup_iters=0 if mode == "manual" else 4,
            )
            try:
                gates = dict(min_corners=min_corners, max_incidence_deg=max_incidence_deg, min_coverage=min_coverage)
                if mode == "manual":
                    # No aiming: aiming turns the wrist, and in this mode the wrist is yours.
                    samples = _capture_manual(
                        rig=rig,
                        spec=spec,
                        board=board,
                        num_poses=num_poses,
                        max_pnp_error_px=max_pnp_error_px,
                        still_rad=still_rad,
                        still_s=still_s,
                        min_new_trans_mm=min_new_trans_mm,
                        min_new_rot_deg=min_new_rot_deg,
                        poll_s=poll_s,
                        run_dir=run_dir,
                        **gates,
                    )
                elif mode == "sweep":
                    # No aiming here either: the anchor is wherever the operator posed the board,
                    # and the sweep is defined relative to it, not to the camera's current framing.
                    samples = _capture_sweep(
                        rig=rig,
                        spec=spec,
                        board=board,
                        num_poses=num_poses,
                        pos_scale=pos_scale,
                        rot_scale_deg=rot_scale_deg,
                        settle_s=settle_s,
                        max_pnp_error_px=max_pnp_error_px,
                        run_dir=run_dir,
                        **gates,
                    )
                else:
                    aimed = auto_aim(rig, spec, board, max_steps=auto_aim_steps, **gates) if auto_aim_steps else False
                    if not aimed and aim_seconds > 0:
                        # Servoing could not finish the job — usually the board is too small in frame
                        # or the wrist has run out of range, neither of which the arm can fix. Keep
                        # reporting so it can be sorted out by hand without restarting the whole run.
                        _watch_presentation(rig, spec, board, seconds=aim_seconds, **gates)
                    if mode == "aim":
                        _write_aim_image(rig, spec, board, run_dir)
                        return None

                    observed = rig.observe(spec, board)
                    if observed is None or _verdict(len(observed[1]), observed[5], observed[6], **gates):
                        raise RuntimeError(
                            "the board is still not well presented, so there is nothing worth "
                            "calibrating from. It must be CLAMPED IN THE GRIPPER with its checkered "
                            f"face toward the camera, at least {min_corners} corners visible, under "
                            f"{max_incidence_deg:.0f} deg incidence and over {min_coverage * 100:.0f}% "
                            "of the frame. Use `--mode aim` to sort the setup out on its own."
                        )
                    samples = _capture_samples(
                        rig=rig,
                        spec=spec,
                        board=board,
                        num_poses=num_poses,
                        pos_scale=pos_scale,
                        rot_scale_deg=rot_scale_deg,
                        seed=seed,
                        settle_s=settle_s,
                        max_pnp_error_px=max_pnp_error_px,
                        run_dir=run_dir,
                        **gates,
                    )
                K, dist, cam_serial = rig.K, rig.dist, rig.serial
            finally:
                rig.close()
        _save_samples(run_dir, samples, K, dist, spec, cam_serial)

    if len(samples) < min_samples:
        raise RuntimeError(
            f"only {len(samples)} usable samples, need {min_samples}. Samples are kept in {run_dir}; "
            "the most common causes are the board leaving the camera's view and unreachable poses."
        )

    # Before anything is solved: did the board actually move with the gripper? A calibration off a
    # board that did not is not a worse calibration, it is a meaningless one.
    check_rigidity(samples)

    diversity = _rotation_diversity_deg(samples)
    _log.info(f"Rotation diversity: {diversity:.1f} deg between the two most different poses")
    if diversity < 20.0:
        _log.warning(
            f"Only {diversity:.1f} deg of rotation across the samples. Hand-eye is poorly "
            "conditioned below ~20 deg — raise --rot-scale-deg and rerun."
        )

    if mode == "sweep":
        _holdout_check(samples, K, dist, method, holdout_test_fraction, outlier_px, seed)

    world_from_cam, metrics, chosen, all_scored, samples = _solve_robust(
        samples, K, dist, method, outlier_px, min_samples
    )
    if method == "auto":
        for name, m in sorted(all_scored.items(), key=lambda kv: kv[1].reproj_rms_px):
            _log.info(f"  {name:<11} {m.reproj_rms_px:6.2f} px RMS   {m.offset_trans_rms_mm:6.2f} mm spread")
    _log.info(f"Using the {chosen!r} solution on {len(samples)} samples")

    pose6 = _report(world_from_cam, metrics, samples, cam_serial)
    _write_verification(run_dir, samples, world_from_cam, K, dist)

    (run_dir / "result.json").write_text(
        json.dumps(
            {
                "mode": mode,
                "camera_serial": cam_serial,
                "arm": arm,
                "method": chosen,
                "pose": pose6.tolist(),
                "world_from_cam": world_from_cam.tolist(),
                "ee_from_board": _mean_offset(_board_offsets(samples, world_from_cam)).tolist(),
                "board": spec.to_json(),
                "num_samples": len(samples),
                "rotation_diversity_deg": diversity,
                "reproj_rms_px": metrics.reproj_rms_px,
                "reproj_max_px": metrics.reproj_max_px,
                "offset_trans_rms_mm": metrics.offset_trans_rms_mm,
                "offset_rot_rms_deg": metrics.offset_rot_rms_deg,
            },
            indent=2,
        )
    )
    _log.info(f"Wrote {run_dir / 'result.json'} and {run_dir / 'verification.png'}")

    if metrics.reproj_rms_px > outlier_px:
        _log.warning(
            f"{metrics.reproj_rms_px:.2f} px RMS is high for a calibration this constrained. Check "
            "--square-size against the actual printout before trusting the result."
        )
    if save:
        update_calibration_info(cam_serial, pose6)
    else:
        _log.info("--no-save: nothing written to the calibration file")
    return world_from_cam


def calibrate_top_cam_entrypoint():
    os.environ.setdefault("TIPTOP_CONFIG", DEFAULT_YAM_CONFIG)

    from tiptop.utils import setup_logging

    setup_logging()
    tyro.cli(calibrate_top_cam)


if __name__ == "__main__":
    calibrate_top_cam_entrypoint()

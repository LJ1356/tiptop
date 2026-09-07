"""Calibrate the FIXED third-person (exterior) camera on the Franka rig — solve ``world_from_cam``.

The counterpart to :mod:`tiptop.scripts.calibrate_wrist_cam`. The wrist camera rides the gripper, so
its calibration entry is ``ee_from_cam`` and the world pose is recomposed per observation as
``FK(q) @ ee_from_cam``. The third-person camera is bolted to the room: its entry **is**
``world_from_cam``, read straight out of ``calibration_info`` with no forward kinematics involved,
which is exactly what ``tiptop_run.py`` does when ``cameras.perception`` is ``external`` (it pins
``mount = "world"`` for that slot). So the two calibrations solve for different transforms and this
is a separate script rather than a flag on that one.

**The board goes IN THE GRIPPER, not on the table.** The wrist camera looks *at* a fixed board; here
the camera is fixed and the *board* has to move, and the only thing tying a board observation to the
robot's world frame is forward kinematics of the arm holding it. A board taped to the table carries
no information about where the camera is, however cleanly it is detected. Nothing in this script ever
commands the gripper — the clamp holding the board is left exactly as the operator set it, which is
the same reason DROID's ``scripts/main.py`` runs its calibration under a ``NoGripperVRPolicy``.

Where the joint configuration comes from: DROID drives the arm to a fixed posture before third-person
calibration (``scripts/main.py::THIRD_PERSON_CALIBRATION_JOINTS``) because the GUI's ordinary reset
leaves the board out of this camera's view. Same problem here, same answer —
:data:`THIRD_PERSON_CALIBRATION_JOINTS` is that posture, and the Lissajous sweep is centred on the
end-effector pose the arm reaches there.

Method: DROID's eye-to-hand formulation, ported from ``droid/calibration/calibration_utils.py``'s
``ThirdPersonCameraCalibrator``. OpenCV's ``calibrateHandEye`` solves ``A·X·B = C`` for
``X = cam2gripper``; feeding it the INVERTED end-effector poses (``base2gripper``) swaps the roles so
that ``X`` comes back as ``cam2base`` — the camera's pose in the robot base frame, which is the
``world_from_cam`` this rig's ``calibration_info`` is keyed on. The solve is accuracy-checked against
a held-out split of the samples before anything is written (:meth:`is_calibration_accurate`), so a
sweep the board was barely visible in fails loudly rather than writing a confident wrong pose.

**Safety.** The board on the gripper is not in cuRobo's collision model, and neither is whatever it
is clamped to. The sweep is small and centred on a fixed posture, but keep the workspace clear.

Usage::

    # board clamped in the gripper, facing the third-person camera
    DC_WORKSPACE=prpl pixi run calibrate-third-person-cam

    # check the result: the point cloud should land on the robot model
    pixi run viz-calibration --camera external
"""

import logging
import os
import time

import cv2
import numpy as np
import torch
from curobo.geom.types import WorldConfig
from curobo.types.math import Pose
from curobo.types.state import JointState
from curobo.wrap.reacher.motion_gen import MotionGenPlanConfig
from jaxtyping import Float
from scipy.spatial.transform import Rotation as R

from tiptop.config import tiptop_cfg, update_calibration_info
from tiptop.motion_planning import get_motion_gen, go_to_q
from tiptop.perception.cameras import get_external_camera
from tiptop.scripts.calibrate_wrist_cam import (
    CharucoDetector,
    calibration_traj,
    change_pose_frame,
    euler_to_rmat,
    pose_diff,
    rmat_to_euler,
)
from tiptop.utils import get_robot_client, setup_logging
from tiptop.workspace import workspace_cuboids

_log = logging.getLogger(__name__)

# Same switch calibrate_wrist_cam.py reads: TIPTOP_CALIB_VIZ=1 opens the live OpenCV windows. The
# data-collection server pins it to "0" — a cv2 window nobody can reach would hang the job with the
# arm live.
_VIZ = os.environ.get("TIPTOP_CALIB_VIZ", "0") == "1"

# The posture the arm is driven to before the sweep, so the board in its gripper is square-on to the
# third-person camera. Copied from droid/scripts/main.py::THIRD_PERSON_CALIBRATION_JOINTS, which is
# where it was tuned against this rig's camera mount. Franka's 7 joints.
THIRD_PERSON_CALIBRATION_JOINTS = np.array([-0.0884, 0.0362, -0.2561, -1.9031, -0.0013, 1.7775, 0.7454])

# How far around the Lissajous loop each waypoint steps, and how far round the loop goes (one full
# cycle). Matches the wrist sweep: ~42 waypoints, which is comfortably above the 10-image threshold
# CharucoDetector needs even when a third of the frames miss the board.
STEP_SIZE = 0.15

# How many waypoints may fail to plan before the sweep is called off. A handful of unreachable poses
# costs a few samples; a third of the loop failing means the starting posture is wrong for this rig,
# and finishing the sweep would only produce a solve with nothing behind it.
MAX_UNPLANNABLE_WAYPOINTS = 12


class ThirdPersonCameraCalibrator(CharucoDetector):
    """Eye-to-hand calibration for a camera fixed in the scene, watching a board on the gripper.

    Ported from DROID's ``ThirdPersonCameraCalibrator``. The detection half is shared with the wrist
    calibrator (both subclass :class:`CharucoDetector`); what differs is which transform the
    hand-eye solve is asked for — see the module docstring.
    """

    def __init__(
        self, intrinsics_dict, lin_error_threshold=1e-3, rot_error_threshold=1e-2, train_percentage=0.7, **kwargs
    ):
        self.lin_error_threshold = lin_error_threshold
        self.rot_error_threshold = rot_error_threshold
        self.train_percentage = train_percentage
        super().__init__(intrinsics_dict, **kwargs)

    def calibrate(self, cam_id):
        """``world_from_cam`` as ``[x, y, z, roll, pitch, yaw]`` — what gets written to disk."""
        return self._calibrate_cam_to_base(cam_id=cam_id)

    def _calibrate_cam_to_base(self, cam_id=None, readings=None, gripper_poses=None, target2cam_results=None):
        # Get Calibration Data #
        if cam_id is not None:
            readings, gripper_poses = self._readings_dict[cam_id], self._pose_dict[cam_id]
            self._curr_cam_id = cam_id

        # Get Target2Cam Transformation #
        if target2cam_results is None:
            target2cam_results = self.calculate_target_to_cam(readings)
        if target2cam_results is None:
            return None

        R_target2cam, t_target2cam, successes = target2cam_results
        gripper_poses = np.array(gripper_poses)[successes]

        # Invert the end-effector poses: base2gripper in the slots calibrateHandEye labels
        # gripper2base is what turns "camera on the arm" into "board on the arm".
        t_base2gripper = [
            -R.from_euler("xyz", pose[3:6]).inv().as_matrix() @ np.array(pose[:3]) for pose in gripper_poses
        ]
        R_base2gripper = [R.from_euler("xyz", pose[3:6]).inv().as_matrix() for pose in gripper_poses]

        # Perform Calibration #
        rmat, pos = cv2.calibrateHandEye(
            R_gripper2base=R_base2gripper,
            t_gripper2base=t_base2gripper,
            R_target2cam=R_target2cam,
            t_target2cam=t_target2cam,
            method=4,
        )

        # Return Pose #
        pos = pos.flatten()
        angle = R.from_matrix(rmat).as_euler("xyz")
        return np.concatenate([pos, angle])

    def _calibrate_gripper_to_target(self, cam_id=None, readings=None, gripper_poses=None, target2cam_results=None):
        """The other unknown in the same system: where the board sits in the gripper.

        Not written anywhere — it exists so :meth:`is_calibration_accurate` can predict a held-out
        sample's arm pose from its board observation alone.
        """
        # Get Calibration Data #
        if cam_id is not None:
            readings, gripper_poses = self._readings_dict[cam_id], self._pose_dict[cam_id]
            self._curr_cam_id = cam_id

        # Get Target2Cam Transformation #
        if target2cam_results is None:
            target2cam_results = self.calculate_target_to_cam(readings)
        if target2cam_results is None:
            return None

        R_target2cam, t_target2cam, successes = target2cam_results
        gripper_poses = np.array(gripper_poses)[successes]

        # Calculate Appropriate Transformations #
        t_base2gripper = [
            -R.from_euler("xyz", pose[3:6]).inv().as_matrix() @ np.array(pose[:3]) for pose in gripper_poses
        ]
        R_base2gripper = [R.from_euler("xyz", pose[3:6]).inv().as_matrix() for pose in gripper_poses]

        # Perform Calibration #
        rmat, pos = cv2.calibrateHandEye(
            R_gripper2base=R_target2cam,
            t_gripper2base=t_target2cam,
            R_target2cam=R_base2gripper,
            t_target2cam=t_base2gripper,
            method=4,
        )

        # Return Pose #
        pos = pos.flatten()
        angle = R.from_matrix(rmat).as_euler("xyz")
        return np.concatenate([pos, angle])

    def _calculate_gripper_to_base(self, train_readings, train_gripper_poses, eval_readings=None):
        """Predict each eval frame's arm pose from its board observation and the training solve."""
        if eval_readings is None:
            eval_readings = train_readings

        # Get Eval Target2Cam Transformations #
        eval_results = self.calculate_target_to_cam(eval_readings, train=False)
        if eval_results is None:
            return None
        eval_R_target2cam, eval_t_target2cam, eval_successes = eval_results
        rmats, tvecs = [], []

        # Get Train Target2Cam Transformations #
        train_results = self.calculate_target_to_cam(train_readings)
        if train_results is None:
            return None

        # Use Training Data For Calibrations #
        gripper2target = self._calibrate_gripper_to_target(
            gripper_poses=train_gripper_poses, target2cam_results=train_results
        )
        R_gripper2target = R.from_euler("xyz", gripper2target[3:]).as_matrix()
        t_gripper2target = np.array(gripper2target[:3])

        cam2base = self._calibrate_cam_to_base(gripper_poses=train_gripper_poses, target2cam_results=train_results)
        R_cam2base = R.from_euler("xyz", cam2base[3:]).as_matrix()
        t_cam2base = np.array(cam2base[:3])

        # Calculate Gripper2Base #
        for i in range(len(eval_R_target2cam)):
            R_gripper2cam = eval_R_target2cam[i] @ R_gripper2target
            t_gripper2cam = eval_R_target2cam[i] @ t_gripper2target + eval_t_target2cam[i]

            R_gripper2base = R_cam2base @ R_gripper2cam
            t_gripper2base = R_cam2base @ t_gripper2cam + t_cam2base

            rmats.append(R_gripper2base)
            tvecs.append(t_gripper2base)

        # Return Poses #
        eulers = np.array([R.from_matrix(rmat).as_euler("xyz") for rmat in rmats])
        return np.concatenate([np.array(tvecs), eulers], axis=1), eval_successes

    def is_calibration_accurate(self, cam_id):
        """Held-out check: solve on 70% of the samples, predict the other 30%'s arm poses, compare.

        In-sample residuals cannot catch a solve that only fits the data it came from, which on a
        short sweep is the failure that matters.
        """
        # Set Camera #
        self._curr_cam_id = cam_id

        # Split Into Train / Test #
        readings = self._readings_dict[cam_id]
        if len(readings) == 0:
            _log.error(
                f"{cam_id}: no charuco detections at all — the board was never seen with "
                f">= {self.num_corner_threshold} corners. Is it in the gripper, facing the camera?"
            )
            return False
        poses = np.array(self._pose_dict[cam_id])
        ind = np.random.choice(len(readings), size=len(readings), replace=False)
        num_train = int(len(readings) * self.train_percentage)

        train_ind, test_ind = ind[:num_train], ind[num_train:]
        train_poses, test_poses = poses[train_ind], poses[test_ind]
        train_readings = [readings[i] for i in train_ind]
        test_readings = [readings[i] for i in test_ind]

        _log.info(
            f"{cam_id}: {len(readings)} frames with detections, {len(train_readings)} train / {len(test_readings)} test"
        )
        results = self._calculate_gripper_to_base(train_readings, train_poses, eval_readings=test_readings)
        if results is None:
            _log.error(f"{cam_id}: gripper-to-base solve failed — too few usable detections to check the fit")
            return False
        approx_poses, successes = results
        test_poses = np.array(test_poses)[successes]

        # Calculate Per Dimension Error #
        pose_error = np.array([pose_diff(pose, approx_pose) for pose, approx_pose in zip(test_poses, approx_poses)])
        lin_error = np.linalg.norm(pose_error[:, :3], axis=0) ** 2 / pose_error.shape[0]
        rot_error = np.linalg.norm(pose_error[:, 3:6], axis=0) ** 2 / pose_error.shape[0]

        # Check Calibration Error #
        lin_success = np.all(lin_error < self.lin_error_threshold)
        rot_success = np.all(rot_error < self.rot_error_threshold)
        verdict = "PASS" if (lin_success and rot_success) else "FAIL"
        _log.info(
            f"{cam_id}: lin_error {np.round(lin_error, 6).tolist()} (< {self.lin_error_threshold}?) "
            f"rot_error {np.round(rot_error, 6).tolist()} (< {self.rot_error_threshold}?) -> {verdict}"
        )

        return lin_success and rot_success


def calibrate_third_person_camera():
    """Calibrate the exterior (third-person) camera. The board must be clamped in the gripper."""
    setup_logging()

    # Set up the camera, robot, and calibrator
    cam = get_external_camera()
    cam_id = cam.serial
    intrinsics_dict = cam.get_intrinsics()
    client = get_robot_client()
    calibrator = ThirdPersonCameraCalibrator(intrinsics_dict)
    _log.info(f"Calibrating the third-person camera (s/n {cam_id})")

    # Setup motion planner. Hard-code the time_dilation_factor as the movements are small
    _log.info("Setting up motion planner...")
    world_cfg = WorldConfig(cuboid=list(workspace_cuboids()))
    motion_gen = get_motion_gen(world_cfg, collision_activation_distance=0.01, warmup_iters=4)
    plan_config = MotionGenPlanConfig(time_dilation_factor=0.4)

    # Drive to the calibration posture BEFORE anything else. Wherever the arm was left — home, the
    # capture pose, mid-teleop — the board it is holding is almost certainly not in this camera's
    # view, and the sweep below only perturbs whatever pose it starts from.
    _log.info(f"joints before: {np.round(client.get_joint_positions(), 5).tolist()}")
    _log.info(f"Moving to the third-person calibration posture: {THIRD_PERSON_CALIBRATION_JOINTS.round(4).tolist()}")
    go_to_q(
        q_target=list(THIRD_PERSON_CALIBRATION_JOINTS),
        time_dilation_factor=tiptop_cfg().robot.time_dilation_factor,
        # go_to_q's default dist_tol (0.05 over the whole joint vector) is a loose gate: an arm parked
        # NEAR this posture would skip the move and the sweep would silently centre on the wrong pose.
        # There is nothing to save here -- this runs once, before a multi-minute sweep -- so the gate
        # is tightened to "already there for practical purposes".
        dist_tol=1e-3,
        motion_gen=motion_gen,
    )
    time.sleep(0.5)  # let the arm settle before the pose the whole sweep is centred on is read
    # What the arm ACTUALLY reached, the way droid/scripts/main.py logs it: the sweep is centred on
    # this, so a move that was skipped or fell short is worth seeing in the log rather than inferring
    # from a bad solve at the end.
    _log.info(f"joints after:  {np.round(client.get_joint_positions(), 5).tolist()}")

    if _VIZ:
        # Visualize the camera feed; adjust the board in the gripper, then press 'y'.
        while True:
            frame = cam.read_camera()
            viz_img = calibrator.augment_image(cam_id=cam_id, image=frame.bgr)
            viz_img = cv2.putText(
                viz_img,
                "Board must be IN THE GRIPPER and facing this camera.",
                (15, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )
            viz_img = cv2.putText(
                viz_img, "Press 'y' to continue, 'n' to exit", (15, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
            )
            cv2.imshow("Calibration View", viz_img)
            key = cv2.waitKey(1)
            if key == ord("y"):
                break
            elif key == ord("n"):
                return
    else:
        # Headless (how the data-collection server runs this): no window to confirm in, so take one
        # frame through the detector and say whether the board is visible from here. The sweep runs
        # either way — a few missed frames are normal — but "not detected" before the arm starts
        # moving is the cheapest warning there is.
        frame = cam.read_camera()
        detected = calibrator.process_image(frame.bgr) is not None
        if detected:
            _log.info("Charuco board detected from the calibration posture")
        else:
            _log.warning(
                "Charuco board NOT detected from the calibration posture — the sweep will run, but "
                "check the board is clamped in the gripper and facing the third-person camera"
            )

    def get_q_curr() -> Float[torch.Tensor, "d"]:
        _q_curr = client.get_joint_positions()
        return torch.tensor(_q_curr, dtype=torch.float32, device="cuda")

    def get_mat4x4() -> Float[np.ndarray, "4 4"]:
        return motion_gen.kinematics.get_state(get_q_curr()).ee_pose.get_numpy_matrix()[0]

    # Bad hack for now, flush out the communication channel
    _log.debug("Attempting to flush out the buffer")
    for _ in range(100):
        get_q_curr()
    _log.debug("Flushed out the buffer (I hope)")

    pose_origin_mat4x4 = get_mat4x4()
    pose_origin = np.zeros(6)
    pose_origin[:3] = pose_origin_mat4x4[:3, 3]
    pose_origin[3:] = rmat_to_euler(pose_origin_mat4x4[:3, :3])
    _log.info(
        f"Sweeping around ee pose xyz={pose_origin[:3].round(4).tolist()} rpy={pose_origin[3:].round(4).tolist()}"
    )
    i = 0
    unplannable = 0

    while True:
        # hand_camera=False: the third-person variant of DROID's Lissajous loop. The wrist sweep
        # permutes the same curve into the camera's own frame because there the CAMERA is what has
        # to see round the board; here it is the board that has to be turned, in the base frame.
        calib_pose = calibration_traj(i * STEP_SIZE, hand_camera=False)
        desired_pose = change_pose_frame(calib_pose, pose_origin)
        desired_pose_mat4x4 = np.eye(4)
        desired_pose_mat4x4[:3, 3] = desired_pose[:3]
        desired_pose_mat4x4[:3, :3] = euler_to_rmat(desired_pose[3:])

        desired_pose_pt = torch.tensor(desired_pose_mat4x4, dtype=torch.float32, device="cuda")
        desired_pose_curobo = Pose.from_matrix(desired_pose_pt)

        q_curr = get_q_curr()
        js_curr = JointState.from_position(q_curr[None])
        result = motion_gen.plan_single(js_curr, desired_pose_curobo, plan_config)
        if not result.success:
            # Skip the waypoint rather than aborting the sweep. The excursion here is the FULL
            # base-frame Lissajous (the wrist sweep scales its rotations down by 1.5), centred on a
            # posture chosen to aim a board at the camera rather than to sit mid-workspace, so a
            # waypoint can land against a joint limit where the wrist sweep's would not. The solve
            # needs ~10 usable frames out of 43, so throwing away the samples already collected over
            # one unreachable pose is the wrong trade. Nothing moves; the next waypoint is planned
            # from the same current joint state.
            unplannable += 1
            _log.warning(f"waypoint {i}: no motion plan ({result.status}) — skipping it")
            if unplannable > MAX_UNPLANNABLE_WAYPOINTS:
                raise RuntimeError(
                    f"{unplannable} waypoints could not be planned — the arm cannot sweep from this "
                    "posture. Check that the board and its clamp are not fouling the workspace, and "
                    "that THIRD_PERSON_CALIBRATION_JOINTS still suits this rig."
                )
            i += 1
            if (i * STEP_SIZE) >= (2 * np.pi):
                break
            continue

        plan = result.interpolated_plan
        dt = result.interpolation_dt
        timings = [dt] * plan.position.shape[0]
        full_trajectory = plan.position.cpu().numpy()
        velocities = plan.velocity.cpu().numpy()
        # Arm only. The gripper is never commanded anywhere in this script: it is clamping the board.
        result = client.execute_joint_impedance_path(
            joint_confs=full_trajectory, joint_vels=velocities, durations=timings
        )
        if not result["success"]:
            raise RuntimeError(f"Could not move robot! Error: {result['error']}")

        time.sleep(0.4)  # wait for robot to stabilize

        # The MEASURED pose, not the commanded one — the arm settles a few mrad short of its target
        # and hand-eye is sensitive to exactly that.
        pose_mat4x4 = get_mat4x4()
        pose = np.zeros(6)
        pose[:3] = pose_mat4x4[:3, 3]
        pose[3:] = rmat_to_euler(pose_mat4x4[:3, :3])

        # Add Sample + Augment Images #
        frame = cam.read_camera()
        image = frame.bgr

        cycle_complete = (i * STEP_SIZE) >= (2 * np.pi)
        cycle_prop_complete = 100 * (i * STEP_SIZE) / (2 * np.pi)
        # INFO, not DEBUG: setup_logging() defaults to INFO, and this is the only progress the sweep
        # reports — the data-collection UI parses these lines to drive its progress bar.
        _log.info(f"{cycle_prop_complete:.2f}% calibration complete")

        calibrator.add_sample(cam_id=cam_id, image=image, pose=pose)
        augmented_image = calibrator.augment_image(cam_id=cam_id, image=image)
        augmented_image = cv2.putText(
            augmented_image,
            f"Calibration {cycle_prop_complete:.2f}% complete...",
            (15, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )
        if _VIZ:
            cv2.imshow("Calibration View", augmented_image)
        cv2.waitKey(1)

        # Check if cycle is complete
        if cycle_complete:
            break
        i += 1

    if unplannable:
        _log.warning(f"sweep finished with {unplannable} waypoint(s) skipped for lack of a motion plan")

    success = calibrator.is_calibration_accurate(cam_id)
    if not success:
        raise RuntimeError("Calibration failed as it wasn't accurate enough")

    # Save the calibration. update_calibration_info prints the "Updated calibration for <serial> in
    # <path>" line the data-collection UI reports back to the operator.
    transformation = calibrator.calibrate(cam_id)
    update_calibration_info(cam_id, transformation)
    _log.info(f"Updated calibration info for {cam_id}. world_from_cam: {transformation}")
    _log.info("Remember to take the calibration board out of the gripper before collecting.")


def calibrate_third_person_camera_entrypoint():
    calibrate_third_person_camera()


if __name__ == "__main__":
    calibrate_third_person_camera_entrypoint()

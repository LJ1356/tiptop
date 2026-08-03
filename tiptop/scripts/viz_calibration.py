import logging
import time
from typing import Literal

import numpy as np
import rerun as rr
from curobo.types.base import TensorDeviceType
from cutamp.robots import (
    load_fr3_franka_container,
    load_fr3_robotiq_container,
    load_panda_container,
    load_panda_robotiq_container,
    load_ur5_container,
)

from tiptop.config import load_calibration, tiptop_cfg
from tiptop.perception.cameras import get_external_camera, get_hand_camera
from tiptop.perception.utils import depth_to_xyz
from tiptop.utils import get_robot_client, get_robot_rerun, patch_log_level, setup_logging

_log = logging.getLogger(__name__)


def _load_robot_container(tensor_args: TensorDeviceType):
    """Load the cuTAMP robot container for the configured robot, for forward kinematics."""
    robot_type = tiptop_cfg().robot.type
    with patch_log_level("curobo", logging.ERROR):
        if robot_type == "fr3_robotiq":
            return load_fr3_robotiq_container(tensor_args)
        elif robot_type == "fr3":
            return load_fr3_franka_container(tensor_args)
        elif robot_type == "panda":
            return load_panda_container(tensor_args)
        elif robot_type == "panda_robotiq":
            return load_panda_robotiq_container(tensor_args)
        elif robot_type == "ur5":
            return load_ur5_container(tensor_args)
        else:
            raise ValueError(f"Unknown robot type: {robot_type}")


def viz_calibration(
    camera: Literal["hand", "external"] = "hand",
    rr_spawn: bool = True,
    viz_freq: float = 5.0,
    max_time: float = 60.0,
):
    """
    Visualize camera calibration with robot in rerun: a correct calibration puts the point cloud on
    the robot model.

    Args:
        camera: Which camera to check. "hand" follows the wrist camera through forward kinematics;
            "external" uses the third-person camera's static world_from_cam from calibration_info.
        rr_spawn: Spawn rerun viewer. You should only set to False if you're connecting to remote visualizer.
        viz_freq: Visualization loop frequency in Hz.
        max_time: Maximum visualization time in seconds before automatically stopping. Used to prevent the script from
            running too long and logging and crazy amount of data to rerun.
    """
    setup_logging()
    rr.init("viz_calibration", spawn=rr_spawn)
    # Connect to robot
    client = get_robot_client()
    robot_rr = get_robot_rerun()

    tensor_args = TensorDeviceType()
    if camera == "hand":
        cam = get_hand_camera(depth=True)
        ee_from_cam = load_calibration(cam.serial)
        robot_container = _load_robot_container(tensor_args)
    else:
        cam = get_external_camera(depth=True)
        world_from_cam_static = load_calibration(cam.serial)
    _log.info(f"Checking calibration of the {camera} camera (s/n {cam.serial})")

    start_time = time.perf_counter()
    sleep_time = 1.0 / viz_freq
    _log.warning("Do not keep this script running indefinitely! It logs a **lot** of data to rerun.")
    _log.info(f"Starting visualization loop at {viz_freq} Hz. Ctrl+C to exit.")

    try:
        while True:
            iter_start = time.perf_counter()
            time_elapsed = iter_start - start_time
            if time_elapsed >= max_time:
                _log.info(f"Max run time of {max_time}s reached. Exiting...")
                break

            rr.set_time("elapsed", duration=time_elapsed)

            # Get joint positions, camera pose, and camera frame
            frame = cam.read_camera()
            q_curr = client.get_joint_positions()
            if camera == "hand":
                q_curr_pt = tensor_args.to_device(q_curr)
                world_from_ee = robot_container.kin_model.get_state(q_curr_pt).ee_pose.get_numpy_matrix()[0]
                world_from_cam = world_from_ee @ ee_from_cam
            else:
                world_from_cam = world_from_cam_static

            # Read camera frame
            rgb = frame.rgb
            rgb_map = rgb / 255.0

            depth_m = frame.depth.copy()
            depth_m[depth_m > 5.0] = 0.0
            K = frame.intrinsics
            xyz_map = depth_to_xyz(depth_m, K)

            # Convert point cloud to world frame using camera transform
            xyz_hom = np.ones((xyz_map.shape[0], xyz_map.shape[1], 4))
            xyz_hom[:, :, :3] = xyz_map
            xyz_world = np.einsum("ij,hwj->hwi", world_from_cam, xyz_hom)[:, :, :3]
            valid_mask = (depth_m > 0) & ~np.isnan(xyz_world).any(axis=-1)
            xyz_valid, rgb_valid = xyz_world[valid_mask], rgb_map[valid_mask]

            # Log to rerun
            robot_rr.set_joint_positions(q_curr)
            rr.log(
                "world_from_cam",
                rr.Transform3D(translation=world_from_cam[:3, 3], mat3x3=world_from_cam[:3, :3], axis_length=0.05),
            )

            rr.log("rgb", rr.Image(rgb))
            rr.log("depth", rr.DepthImage(depth_m, meter=1.0))
            rr.log("pcd", rr.Points3D(positions=xyz_valid, colors=rgb_valid))

            # Sleep to maintain desired frequency
            iter_elapsed = time.perf_counter() - iter_start
            remaining_time = sleep_time - iter_elapsed
            if remaining_time > 0:
                time.sleep(remaining_time)
    except KeyboardInterrupt:
        _log.info(f"Detected keyboard interrupt. Exiting...")
    finally:
        client.close()


def viz_calibration_entrypoint():
    import tyro

    tyro.cli(viz_calibration)


if __name__ == "__main__":
    viz_calibration()

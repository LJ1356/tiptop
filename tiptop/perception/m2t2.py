import logging
import os
import time
from functools import cache

import aiohttp
import numpy as np
import requests
from jaxtyping import Float
from scipy.spatial.transform import Rotation

from tiptop.utils import ServerHealthCheckError

_log = logging.getLogger(__name__)


@cache
def m2t2_to_tiptop_transform():
    """4x4 transform to take M2T2 grasp poses to the convention expected by tiptop."""
    # Panda offset
    base_to_tcp = np.eye(4)
    base_to_tcp[2, 3] = 0.1034

    # To tiptop frame with z-up
    to_tiptop_frame = np.eye(4)
    to_tiptop_frame[:3, :3] = Rotation.from_euler("xyz", np.array([np.pi, 0, -np.pi / 2])).as_matrix()
    return base_to_tcp @ to_tiptop_frame


@cache
def jaw_flip_transform() -> Float[np.ndarray, "4 4"]:
    """180 degree rotation about the gripper's approach axis, in the tiptop grasp frame.

    A parallel jaw is symmetric under this rotation: the two fingers simply swap sides, so the
    resulting pose is the SAME physical grasp reached with the wrist rolled over. M2T2 emits only one
    of the two representatives.
    """
    flip = np.eye(4)
    flip[0, 0] = -1.0
    flip[1, 1] = -1.0
    return flip


def augment_flipped_grasps(
    world_from_grasp: Float[np.ndarray, "n 4 4"], grasp_dict: dict
) -> tuple[Float[np.ndarray, "2n 4 4"], dict]:
    """Append each grasp's jaw-flipped twin, keeping ``grasp_dict``'s arrays aligned.

    Free extra reach for a robot with no null space. A 7-DOF arm can usually roll its wrist to the
    representative M2T2 happened to emit, but a 6-DOF arm cannot, so half the physically valid
    grasps are silently unreachable -- measured on a scene-6 run, the bimanual YAM could reach 0 of
    the 3 grasps found on the far toy as emitted and all 3 of their flipped twins. Enabled via
    ``perception.augment_flipped_grasps``.

    ``poses`` is recomputed rather than duplicated so the invariant the saved outputs rely on --
    ``world_from_grasp == poses @ m2t2_to_tiptop_transform()`` -- still holds for the new entries.
    """
    if len(world_from_grasp) == 0:
        return world_from_grasp, grasp_dict

    both = np.concatenate([world_from_grasp, world_from_grasp @ jaw_flip_transform()], axis=0)
    out = dict(grasp_dict)
    out["poses"] = both @ np.linalg.inv(m2t2_to_tiptop_transform())
    for key in ("confidences", "contacts"):
        # The flipped twin is the same physical grasp, so it inherits confidence and contact point.
        if key in grasp_dict and len(grasp_dict[key]):
            out[key] = np.concatenate([grasp_dict[key], grasp_dict[key]], axis=0)
    return both, out


def _build_payload(
    scene_xyz: Float[np.ndarray, "n 3"],
    scene_rgb: Float[np.ndarray, "n 3"],
    grasp_threshold: float,
    num_points: int,
    num_runs: int,
    apply_bounds: bool,
) -> dict:
    return {
        "pointcloud": {
            "points": scene_xyz.tolist(),
            "rgb": scene_rgb.tolist(),
        },
        "num_points": num_points,
        "num_runs": num_runs,
        "mask_thresh": grasp_threshold,
        "apply_bounds": apply_bounds,
    }


def _process_m2t2_response(result: dict, num_grasps: int | None) -> dict:
    """Process M2T2 response and return structured grasp outputs."""
    grasps_list = result.get("grasps", [])
    confidences_list = result.get("grasp_confidence", [])
    contacts_list = result.get("grasp_contacts", [])
    outputs = {}

    for i, (grasps, confidences, contacts) in enumerate(zip(grasps_list, confidences_list, contacts_list)):
        label = f"object_{i}"
        if len(grasps) == 0:
            outputs[label] = {
                "poses": np.array([]).reshape(0, 4, 4),
                "confidences": np.array([]),
                "contacts": np.array([]),
            }
        else:
            poses = np.array(grasps)
            confs = np.array(confidences)
            conts = np.array(contacts)

            if num_grasps is not None and len(poses) > num_grasps:
                top_indices = np.argsort(confs)[-num_grasps:]
                poses = poses[top_indices]
                confs = confs[top_indices]
                conts = conts[top_indices]

            outputs[label] = {
                "poses": poses,
                "confidences": confs,
                "contacts": conts,
            }

    return outputs


def generate_grasps(
    server_url: str,
    scene_xyz: Float[np.ndarray, "n 3"],
    scene_rgb: Float[np.ndarray, "n 3"],
    grasp_threshold: float = 0.035,
    num_grasps: int = 200,
    num_points: int = 16384,
    num_runs: int = 5,
    apply_bounds: bool = True,
):
    """
    Generate grasps from point cloud using M2T2 server (synchronous version).
    Note: the coordinate frame of the grasps are in M2T2's convention.
    """
    start_time = time.perf_counter()
    payload = _build_payload(scene_xyz, scene_rgb, grasp_threshold, num_points, num_runs, apply_bounds)
    endpoint = os.path.join(server_url.rstrip("/"), "predict")

    _log.debug(f"Sending inference request to M2T2 server at {endpoint}")
    response = requests.post(endpoint, json=payload, timeout=500)
    result = response.json()

    outputs = _process_m2t2_response(result, num_grasps)
    duration = time.perf_counter() - start_time
    _log.info(f"M2T2 inference time={duration:.2f}s")
    return outputs


async def generate_grasps_async(
    session: aiohttp.ClientSession,
    server_url: str,
    scene_xyz: Float[np.ndarray, "n 3"],
    scene_rgb: Float[np.ndarray, "n 3"],
    grasp_threshold: float = 0.035,
    num_grasps: int = 200,
    num_points: int = 16384,
    num_runs: int = 5,
    apply_bounds: bool = True,
):
    """
    Generate grasps from point cloud using M2T2 server (async version).
    Note: the coordinate frame of the grasps are in M2T2's convention.
    """
    start_time = time.perf_counter()
    payload = _build_payload(scene_xyz, scene_rgb, grasp_threshold, num_points, num_runs, apply_bounds)
    endpoint = os.path.join(server_url.rstrip("/"), "predict")

    _log.debug(f"Sending inference request to M2T2 server at {endpoint}")
    async with session.post(endpoint, json=payload, timeout=aiohttp.ClientTimeout(total=30.0)) as response:
        result = await response.json()

    outputs = _process_m2t2_response(result, num_grasps)
    duration = time.perf_counter() - start_time
    _log.info(f"M2T2 inference time={duration:.2f}s")
    return outputs


async def check_health_status(session: aiohttp.ClientSession, server_url: str):
    """Calls the M2T2 server health status endpoint."""
    endpoint = os.path.join(server_url.rstrip("/"), "health")
    try:
        async with session.get(endpoint, timeout=5.0) as response:
            response.raise_for_status()
            health_data = await response.json()
            status = health_data["status"]

            if status != "healthy":
                _log.error(f"M2T2 health check failed at {server_url}")
                raise ServerHealthCheckError(f"{server_url} returned status: {status}")

            _log.info(f"✓ M2T2 server is healthy")
    except aiohttp.ClientError as e:
        _log.error(f"Health check failed for M2T2")
        raise ServerHealthCheckError(f"M2T2 is unreachable: {e}") from e

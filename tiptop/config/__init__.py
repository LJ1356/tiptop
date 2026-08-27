import json
import os
import re
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from jaxtyping import Float
from omegaconf import DictConfig, OmegaConf
from scipy.spatial.transform import Rotation

config_dir = Path(__file__).parent
config_assets_dir = config_dir / "assets"
calib_info_path = config_assets_dir / "calibration_info.json"


def _resolve_config_path() -> Path:
    """Which YAML ``tiptop_cfg()`` reads, honouring ``$TIPTOP_CONFIG``.

    Robot type, DOF and camera setup all live in one file, so a second embodiment needs a second
    file rather than a few overrides -- and the entry points are tyro CLIs, which reject the
    ``key=value`` tokens ``OmegaConf.from_cli`` would otherwise pick up. Set ``TIPTOP_CONFIG`` to a
    filename (resolved inside this directory) or an absolute path, e.g.::

        TIPTOP_CONFIG=tiptop_yam.yml pixi run python -m tiptop.tiptop_websocket_server

    Everything that reads ``tiptop_config_path`` follows, including the copy of the config that
    ``recording.save_run_outputs`` drops into each run directory.
    """
    override = (os.environ.get("TIPTOP_CONFIG") or "").strip()
    if not override:
        return config_dir / "tiptop.yml"
    path = Path(os.path.expanduser(override))
    if not path.is_absolute():
        path = config_dir / path
    if not path.exists():
        raise FileNotFoundError(f"TIPTOP_CONFIG={override!r} resolved to {path}, which does not exist")
    return path


tiptop_config_path = _resolve_config_path()

# data-collection scopes each robot's data to a workspace and spawns tiptop with $DC_WORKSPACE set
# (see data-collection/server/lib/sessions.js). Each robot has its own cameras, so a workspace can
# carry its own extrinsics in assets/calibration_info_<workspace>.json, layered over the defaults.
# Same name rule as collect.config.WORKSPACE_RE — it is also what keeps a workspace name from
# escaping the assets dir via path traversal.
WORKSPACE_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


def workspace_calib_path() -> Path | None:
    """Calibration file for the active workspace, or None when running outside one.

    Existence is not checked: reads layer it on only if present, writes create it.
    """
    workspace = (os.environ.get("DC_WORKSPACE") or "").strip()
    if not workspace:
        return None
    if not WORKSPACE_RE.match(workspace):
        raise ValueError(f"invalid DC_WORKSPACE {workspace!r} (allowed: {WORKSPACE_RE.pattern})")
    return config_assets_dir / f"calibration_info_{workspace}.json"


_cached_cfg = None  # Cache for lazy loading


def tiptop_cfg(force_reload: bool = False) -> DictConfig:
    """Load TiPToP config from file."""
    global _cached_cfg
    if _cached_cfg is None or force_reload:
        _cached_cfg = OmegaConf.load(tiptop_config_path)
        # Merge CLI overrides from sys.argv
        cli = OmegaConf.from_cli()
        _cached_cfg = OmegaConf.merge(_cached_cfg, cli)
    return _cached_cfg


@contextmanager
def as_robot_type(robot_type: str):
    """Make ``robot_type`` the active embodiment for the duration of the block.

    ``robot.type`` is read by ~50 call sites to decide which robot they are talking about — cuRobo
    solver selection, the workspace cuboids, the Rerun model, cuTAMP's ``TAMPConfiguration.robot``,
    the hardware client. Swapping it here is what makes all of them agree at once.

    Exists because one embodiment per process is not enough for a bimanual YAM: cuTAMP plans a
    single kinematic chain, so an episode that uses both arms genuinely runs two embodiments in
    sequence (``bimanual_yam_left`` then ``bimanual_yam_right``). Restoring on exit keeps a
    single-embodiment session byte-identical. Not a lock: tiptop plans and executes one arm at a
    time, and nothing else in the process mutates ``robot.type``.
    """
    cfg = tiptop_cfg()
    previous = cfg.robot.type
    cfg.robot.type = robot_type
    try:
        yield robot_type
    finally:
        cfg.robot.type = previous


def load_calibration_info():
    """Camera extrinsics keyed by serial, with the active workspace's entries layered on top.

    Keys are camera serials, so a workspace only needs entries for the cameras it actually has;
    everything else falls through to the defaults.
    """
    if not os.path.exists(calib_info_path):
        raise FileNotFoundError(f"{calib_info_path} not found.")
    with open(calib_info_path, "r") as f:
        calibration_info = json.load(f)

    ws_path = workspace_calib_path()
    if ws_path is not None and ws_path.exists():
        with open(ws_path, "r") as f:
            calibration_info.update(json.load(f))
    return calibration_info


def load_calibration(cam_key: str) -> Float[np.ndarray, "4 4"]:
    """Load camera calibration 4x4 transform for a given camera serial."""
    calibration_dict = load_calibration_info()
    if cam_key not in calibration_dict:
        ws_path = workspace_calib_path()
        searched = f"{calib_info_path}" + (f" or {ws_path}" if ws_path is not None else "")
        raise ValueError(f"{cam_key} not found in {searched}")

    pose_vec = calibration_dict[cam_key]["pose"]
    xyz, rpy = pose_vec[:3], pose_vec[3:]
    cam2frame = np.eye(4)
    cam2frame[:3, :3] = Rotation.from_euler("xyz", rpy).as_matrix()
    cam2frame[:3, 3] = xyz
    return cam2frame


def update_calibration_info(cam_key: str, pose: np.ndarray):
    """Update calibration info with new camera pose.

    Writes to the active workspace's file when there is one, so a re-calibration lands in the same
    layer that load_calibration_info() reads last — writing to the defaults instead would leave the
    workspace entry shadowing the fresh measurement.

    Args:
        cam_key: Camera identifier (e.g., "16779706_left")
        pose: 6DOF pose vector [x, y, z, roll, pitch, yaw]
    """
    import time

    target_path = workspace_calib_path() or calib_info_path

    # Read the target layer alone, not the merged view — merging would copy defaults into it.
    if os.path.exists(target_path):
        with open(target_path, "r") as f:
            calibration_dict = json.load(f)
    else:
        calibration_dict = {}

    # Update with new pose and timestamp
    calibration_dict[cam_key] = {
        "pose": pose.tolist() if isinstance(pose, np.ndarray) else list(pose),
        "timestamp": time.time(),
    }

    # Write back to file
    with open(target_path, "w") as f:
        json.dump(calibration_dict, f, indent=2)

    print(f"Updated calibration for {cam_key} in {target_path}")

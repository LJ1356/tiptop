import logging
from typing import Awaitable, Callable, Protocol

import aiohttp
import numpy as np

from tiptop.config import tiptop_cfg
from tiptop.perception.cameras.frame import Frame
from tiptop.perception.cameras.rs_camera import (
    RealsenseCamera,
    RealsenseFrame,
    rs_infer_depth_async,
)
from tiptop.perception.cameras.zed_camera import (
    ZedCamera,
    ZedFrame,
    zed_infer_depth_async,
)

_log = logging.getLogger(__name__)

# Callable that takes an aiohttp session and a camera frame and returns a float depth map (H, W) in metres.
DepthEstimator = Callable[[aiohttp.ClientSession, Frame], Awaitable[np.ndarray]]


class Camera(Protocol):
    serial: str

    def read_camera(self) -> Frame: ...
    def close(self) -> None: ...


def get_depth_estimator(cam: Camera) -> DepthEstimator:
    """Get the appropriate FoundationStereo depth estimator for the given camera. Call once per camera."""
    if isinstance(cam, ZedCamera):
        intrinsics = cam.get_intrinsics()

        async def _zed_estimate(session: aiohttp.ClientSession, f: Frame) -> np.ndarray:
            return await zed_infer_depth_async(session, f, intrinsics)  # type: ignore[arg-type]

        return _zed_estimate
    elif isinstance(cam, RealsenseCamera):
        intrinsics = cam.get_intrinsics()

        async def _rs_estimate(session: aiohttp.ClientSession, f: Frame) -> np.ndarray:
            return await rs_infer_depth_async(session, f, intrinsics)  # type: ignore[arg-type]

        return _rs_estimate
    else:
        raise ValueError(f"No depth estimator available for camera type: {type(cam).__name__}")


def _get_zed_camera(cam_cfg, depth: bool = False, pointcloud: bool = False) -> ZedCamera:
    """Create a ZedCamera from config."""
    serial = str(cam_cfg.serial)
    flip = cam_cfg.get("flip", False)
    resolution = cam_cfg.get("resolution", "HD720")
    fps = cam_cfg.get("fps", 60)
    return ZedCamera(serial, resolution=resolution, fps=fps, flip=flip, depth=depth, pointcloud=pointcloud)


def _get_realsense_camera(cam_cfg, depth: bool = False) -> RealsenseCamera:
    """Create a RealsenseCamera from config.

    ``record_only: true`` opens colour alone. The IR pair exists to feed FoundationStereo, so a
    camera that is only ever written to an mp4 does not need it — and on a three-camera rig the
    bandwidth it frees is what keeps all three at their configured resolution on one USB controller.
    """
    record_only = bool(cam_cfg.get("record_only", False))
    return RealsenseCamera(
        str(cam_cfg.serial),
        width=int(cam_cfg.get("width", 1280)),
        height=int(cam_cfg.get("height", 720)),
        fps=int(cam_cfg.get("fps", 30)),
        enable_depth=depth,
        enable_ir=not record_only,
    )


def _build_camera(cam_cfg, depth: bool = False) -> Camera:
    cam_type = cam_cfg.type
    if cam_type == "zed":
        return _get_zed_camera(cam_cfg, depth=depth)
    elif cam_type == "realsense":
        return _get_realsense_camera(cam_cfg, depth=depth)
    else:
        raise ValueError(f"Unknown camera type: {cam_type}")


def camera_mount(slot: str = "hand") -> str:
    """How a camera slot's calibration entry should be read: ``"ee"`` (default) or ``"world"``.

    ``ee`` is a camera bolted to the end effector, the DROID/Franka arrangement: its calibration is
    ``ee_from_cam`` and the world pose is recovered per-observation as ``FK(q) @ ee_from_cam``.

    ``world`` is a camera fixed in the scene, which is how the bimanual YAM perceives (the
    third-person D435 on the base column, matching the sim's ``top_cam``). Its calibration entry IS
    ``world_from_cam`` and no forward kinematics are involved — so the pose does not move with the
    arm, and the gripper mask, which exists to erase a wrist camera's own fingers from the point
    cloud, does not apply.
    """
    cam_cfg = tiptop_cfg().cameras.get(slot)
    mount = str((cam_cfg or {}).get("mount", "ee")).lower()
    if mount not in ("ee", "world"):
        raise ValueError(f"cameras.{slot}.mount must be 'ee' or 'world', got {mount!r}")
    return mount


def get_hand_camera(depth: bool = False) -> Camera:
    """The PERCEPTION camera — wrist-mounted on the Franka, world-mounted on the YAM.

    The slot is named ``hand`` for history (on DROID it is the wrist ZED) but what it means is "the
    camera TAMP perceives through"; see :func:`camera_mount` for how its calibration is interpreted.
    """
    return _build_camera(tiptop_cfg().cameras.hand, depth=depth)


def get_external_camera(depth: bool = False) -> Camera:
    """The exterior (third-person) camera — recorded always, and TAMP's perception camera when
    ``cameras.perception`` is ``external``.

    ``depth`` turns on the camera's own depth stream, which perception does not need (it runs
    FoundationStereo over the stereo pair) but ``viz-calibration`` reads directly.
    """
    return _build_camera(tiptop_cfg().cameras.external, depth=depth)


def get_external_camera_2() -> Camera | None:
    """Get the optional second external camera (DROID exterior_2).

    Returns None if no ``cameras.external_2`` is configured or the camera can't be
    opened (e.g. not connected), so single-exterior rigs keep working unchanged.
    """
    cfg = tiptop_cfg()
    cam_cfg = cfg.cameras.get("external_2")
    if cam_cfg is None:
        return None
    try:
        return _build_camera(cam_cfg)
    except Exception as e:
        _log.warning(f"Second external camera (s/n {cam_cfg.get('serial')}) unavailable ({e}); skipping exterior_2")
        return None

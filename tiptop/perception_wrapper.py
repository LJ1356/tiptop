import asyncio
import logging
import time
import warnings

import aiohttp
import numpy as np
from jaxtyping import Bool, Float, UInt8
from PIL import Image

from tiptop.config import tiptop_cfg
from tiptop.perception.cameras import (
    DepthEstimator,
    Frame,
)
from tiptop.perception.m2t2 import generate_grasps_async
from tiptop.perception.utils import depth_to_xyz, get_o3d_pcd

_log = logging.getLogger(__name__)


async def _estimate_smoothed_depth(
    session: aiohttp.ClientSession,
    frames: list[Frame],
    depth_estimator: DepthEstimator,
) -> Float[np.ndarray, "h w"]:
    """Temporally smooth depth by per-pixel median over a burst of frames from a static pose.

    Depth is estimated for each frame, then fused per pixel with a median over frames where the
    pixel is valid (finite and > 0; 0 is FoundationStereo's invalid convention). Pixels invalid in
    every frame stay 0. Estimation is sequential to avoid overloading the single depth server.
    """
    depths = []
    for i, f in enumerate(frames):
        depths.append(await depth_estimator(session, f))
        _log.debug(f"Estimated depth for smoothing frame {i + 1}/{len(frames)}")

    stacked = np.stack(depths, axis=0).astype(np.float32)  # (num_frames, h, w)
    valid = np.isfinite(stacked) & (stacked > 0)
    stacked[~valid] = np.nan

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN pixels -> nan, handled below
        fused = np.nanmedian(stacked, axis=0)
    fused = np.nan_to_num(fused, nan=0.0)

    coverage = float(valid.any(axis=0).mean())
    _log.info(f"Median-fused depth over {len(frames)} frames (valid-pixel coverage {coverage:.1%})")
    return fused


async def detect_and_segment(rgb: UInt8[np.ndarray, "h w 3"], task_instruction: str) -> dict:
    """Run VLM detection and SAM2 segmentation pipeline."""
    rgb_pil = Image.fromarray(rgb)
    rgb_pil_resized = rgb_pil.resize((800, int(800 * rgb_pil.size[1] / rgb_pil.size[0])), Image.Resampling.LANCZOS)
    _log.info(
        f"Resized image from {rgb_pil.size[1]}x{rgb_pil.size[0]} to {rgb_pil_resized.size[1]}x{rgb_pil_resized.size[0]}"
    )

    async def _detect():
        from tiptop.perception.gemini import detect_and_translate_async

        _log.info(f"Starting Gemini object detection")
        _st = time.perf_counter()
        _bboxes, _grounded_atoms = await detect_and_translate_async(rgb_pil_resized, task_instruction)
        _dur = time.perf_counter() - _st
        _log.info(f"Gemini detection took {_dur:.2f}s ({len(_bboxes)} objects, {len(_grounded_atoms)} atoms)")
        return _bboxes, _grounded_atoms

    def _segment(_bboxes: list[dict]):
        from tiptop.perception.sam2 import sam2_segment_objects

        _log.info(f"Starting SAM2 object segmentation with Gemini masks")
        _st = time.perf_counter()
        # TODO: async version of this?
        _masks = sam2_segment_objects(rgb_pil, _bboxes)
        _dur = time.perf_counter() - _st
        _log.info(f"SAM2 segmentation took {_dur:.2f}s ({len(_masks)} masks)")
        return _masks

    bboxes, grounded_atoms = await _detect()

    # Sanitize labels: replace spaces with underscores for downstream compatibility
    for bbox in bboxes:
        bbox["label"] = bbox["label"].replace(" ", "_")
    for atom in grounded_atoms:
        atom["args"] = [arg.replace(" ", "_") for arg in atom["args"]]

    masks = await asyncio.to_thread(_segment, bboxes)

    return {"bboxes": bboxes, "masks": masks, "grounded_atoms": grounded_atoms}


async def predict_depth_and_grasps(
    session: aiohttp.ClientSession,
    frame: Frame,
    world_from_cam: Float[np.ndarray, "4 4"],
    downsample_voxel_size: float,
    depth_estimator: DepthEstimator | None = None,
    robot_mask: Bool[np.ndarray, "h w"] | None = None,
    depth_frames: tuple[Frame, ...] = (),
) -> dict:
    """Predict depth map using FoundationStereo and grasps using M2T2. Uses depth_estimator if provided, otherwise uses frame.depth.

    When depth_frames holds more than one frame, their depth maps are per-pixel median-fused for
    temporal smoothing (see _estimate_smoothed_depth); otherwise a single frame is used.
    """
    cfg = tiptop_cfg()

    # Get depth map — use estimator (e.g. FoundationStereo) or fall back to onboard sensor depth
    if depth_estimator is not None:
        frames = list(depth_frames) if depth_frames else [frame]
        if len(frames) > 1:
            depth_map = await _estimate_smoothed_depth(session, frames, depth_estimator)
        else:
            depth_map = await depth_estimator(session, frame)
    else:
        if frame.depth is None:
            raise RuntimeError(
                "No depth available: depth_estimator is None and frame.depth is not set. "
                "Either provide a depth_estimator or ensure the camera captures hardware depth."
            )
        _log.warning("No depth_estimator provided, falling back to hardware depth")
        depth_map = frame.depth

    # Convert to point cloud in world frame
    K = frame.intrinsics
    xyz_map = depth_to_xyz(depth_map, K)
    xyz_map = xyz_map @ world_from_cam[:3, :3].T + world_from_cam[:3, 3]
    # Pixels that are a real measurement of something that is not the robot itself. Invalid depth
    # (0 is FoundationStereo's convention) projects to the camera centre and masked pixels are
    # zeroed below -- neither is a place anything actually is, so segmentation is handed this mask
    # instead of being left to recognise those coordinates. Matters most for a third-person camera,
    # whose mask covers the whole arm rather than a corner of the wrist's view.
    valid_mask = np.isfinite(xyz_map).all(axis=-1) & (depth_map > 0)
    if robot_mask is not None:
        valid_mask &= ~robot_mask
        # Zeroing is what keeps the robot out of the o3d cloud M2T2 sees: every masked pixel lands
        # on the same point, so the voxel downsample collapses the lot into one.
        xyz_map[robot_mask] = 0.0
    rgb_map = frame.rgb.astype(np.float32) / 255.0  # make it float with [0, 1]

    # Create open3d point cloud and downsample
    pcd = await asyncio.to_thread(
        get_o3d_pcd,
        xyz_map,
        rgb_map,
        downsample_voxel_size,
    )
    xyz_downsampled = np.asarray(pcd.points)
    rgb_downsampled = np.asarray(pcd.colors)

    # Predict grasps using M2T2
    grasps = await generate_grasps_async(
        session,
        cfg.perception.m2t2.url,
        scene_xyz=xyz_downsampled,
        scene_rgb=rgb_downsampled,
        apply_bounds=cfg.perception.m2t2.apply_bounds,
        # Read from config rather than left at generate_grasps' default, so cfg/tamp can retune how
        # many candidates each object gets. See motion_planning._PERCEPTION_OVERRIDE_KEYS.
        grasp_threshold=cfg.perception.m2t2.get("grasp_threshold", 0.035),
        num_runs=cfg.perception.m2t2.get("num_runs", 5),
    )

    return {
        "depth_map": depth_map,
        # (h, w, 3) for xyz, rgb, and valid mask map
        "xyz_map": xyz_map,
        "rgb_map": rgb_map,
        "valid_mask": valid_mask,
        # (n, 3) for downsampled point cloud
        "xyz_downsampled": xyz_downsampled,
        "rgb_downsampled": rgb_downsampled,
        "pcd_downsampled": pcd,
        # grasps
        "grasps": grasps,
    }

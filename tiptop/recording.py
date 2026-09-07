import json
import logging
import shutil
import subprocess
import threading
import time
from contextlib import contextmanager
from functools import cache
from pathlib import Path
from typing import Generator

import cv2
import dill
import numpy as np
import open3d as o3d
import torch
from jaxtyping import Bool, Float, UInt8
from PIL import Image

import tiptop
from tiptop.config import tiptop_cfg, tiptop_config_path
from tiptop.perception.cameras.zed_camera import ZedCamera, convert_svo_to_mp4
from tiptop.perception.utils import get_o3d_pcd
from tiptop.perception.visualization import visualize_detections, visualize_masks
from tiptop.utils import NumpyEncoder
from tiptop.viz_utils import get_heatmap

_log = logging.getLogger(__name__)


@cache
def _get_git_root() -> Path:
    """Return the repository root."""
    return Path(subprocess.check_output(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=Path(__file__).parent,
        stderr=subprocess.DEVNULL,
        encoding="utf-8", errors="replace",
    ).strip())


def _collect_git_info() -> dict:
    """Return git commit hash and dirty status for metadata.

    pixi.lock is excluded from the dirty check as it's not relevant for debugging
    """
    try:
        root = _get_git_root()
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            stderr=subprocess.DEVNULL,
            encoding="utf-8", errors="replace",
        ).strip()
        porcelain = subprocess.check_output(
            ["git", "status", "--porcelain", "--", ".", ":(exclude)pixi.lock"],
            cwd=root,
            stderr=subprocess.DEVNULL,
            encoding="utf-8", errors="replace",
        )
        dirty = bool(porcelain.strip())
        return {"commit": commit, "dirty": dirty, "porcelain": porcelain.strip() if dirty else None}
    except (FileNotFoundError, subprocess.CalledProcessError, UnicodeDecodeError):
        _log.warning("Failed to collect git info", exc_info=True)
        return {"commit": None, "dirty": None, "porcelain": None}


def _get_git_diff() -> str | None:
    """Return the full git diff against HEAD, excluding pixi.lock, or None if unavailable or empty."""
    try:
        root = _get_git_root()
        diff = subprocess.check_output(
            ["git", "diff", "HEAD", "--", ".", ":(exclude)pixi.lock"],
            cwd=root,
            stderr=subprocess.DEVNULL,
            encoding="utf-8", errors="replace",
        )
        return diff if diff else None
    except (FileNotFoundError, subprocess.CalledProcessError, UnicodeDecodeError):
        _log.warning("Failed to get git diff", exc_info=True)
        return None


class _Mp4FrameWriter:
    """Pipe RGB frames straight to ffmpeg (H.264 / yuv420p), the format the ZED path also produces.

    A camera with no SDK-side recorder — a RealSense — is captured by grabbing frames and encoding
    them as they arrive. H.264 rather than OpenCV's default MPEG-4 Part 2 because the
    data-collection UI streams these mp4s into a browser ``<video>`` element, which will not decode
    the latter.
    """

    def __init__(self, path: Path, width: int, height: int, fps: float, crf: int = 20):
        import shutil
        import subprocess

        if shutil.which("ffmpeg") is None:
            raise RuntimeError("ffmpeg not found on PATH, required to record a non-ZED camera to MP4")
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.frames = 0
        self._proc = subprocess.Popen(
            [
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "warning",
                "-f", "rawvideo", "-pix_fmt", "rgb24",
                "-s", f"{width}x{height}", "-r", f"{max(fps, 1.0):g}",
                "-i", "pipe:0",
                "-c:v", "libx264", "-crf", str(crf), "-pix_fmt", "yuv420p",
                # faststart puts the moov atom first so the UI can seek without the whole file.
                "-movflags", "+faststart",
                str(path),
            ],
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        # Drain stderr continuously. It is a 64K OS pipe and this process lives for the whole
        # episode, so a chatty encoder would fill it and then block ffmpeg on write -- which stalls
        # the grab thread feeding it and truncates the recording. Keep only the tail, which is what
        # a failure message needs.
        self._stderr_tail: list[bytes] = []
        threading.Thread(target=self._drain_stderr, name=f"ffmpeg-err-{path.stem}", daemon=True).start()

    def _drain_stderr(self) -> None:
        for line in iter(self._proc.stderr.readline, b""):
            self._stderr_tail.append(line)
            del self._stderr_tail[:-40]

    def write(self, rgb: np.ndarray) -> None:
        try:
            self._proc.stdin.write(np.ascontiguousarray(rgb, dtype=np.uint8).tobytes())
            self.frames += 1
        except BrokenPipeError:
            _log.error(f"ffmpeg closed early while writing {self.path.name}; stopping this recording")
            raise

    def close(self) -> int:
        try:
            if self._proc.stdin is not None:
                self._proc.stdin.close()
        except Exception:
            pass
        self._proc.wait()
        if self._proc.returncode != 0:
            tail = b"".join(self._stderr_tail).decode("utf-8", "replace")[-2000:]
            _log.error(f"ffmpeg failed for {self.path.name}: {tail}")
        return self.frames


def _camera_fps(camera) -> float:
    """Nominal capture rate for a camera, for the mp4 container header.

    Only affects playback speed in a video player: the LeRobot export derives the real rate from
    the frame count over the recording window (``_meta.json``'s record_start/record_stop), so a
    camera that ran slightly off its nominal rate still aligns to the control timeline correctly.
    """
    for slot_cfg in (tiptop_cfg().cameras or {}).values():
        if slot_cfg is not None and str(slot_cfg.get("serial", "")) == str(camera.serial):
            return float(slot_cfg.get("fps", 30))
    return 30.0


@contextmanager
def record_cameras(recordings: list[tuple[object, Path, Path | None]]) -> Generator[dict, None, None]:
    """Context manager for recording multiple cameras simultaneously.

    Each entry is ``(camera, raw_path, mp4_path)``. A **ZED** records to ``raw_path`` (an SVO) via
    the SDK and is converted to ``mp4_path`` on exit. Any **other** camera (a RealSense) has no
    SDK-side recorder, so its frames are grabbed and encoded to ``mp4_path`` as they arrive and
    ``raw_path`` is unused — same yielded window, same output file, so callers and the LeRobot
    export cannot tell the two apart.

    All cameras stop collecting frames at the same time on exit, then MP4
    conversion runs sequentially afterwards.

    Yields a mutable ``window`` dict bracketing the recording with epoch-second wall clocks:
    ``window["t_start"]`` is set once every camera has started, ``window["t_stop"]`` once every
    camera has stopped (before MP4 conversion, which is not part of the recording window). These
    let the LeRobot build map state frames -- whose ``frame_time`` is the master timeline -- onto
    camera frames by wall clock (see ARCHITECTURE.md "Camera <-> state alignment"). Using the
    context manager without ``as`` still works; the yielded dict is simply ignored.

    Args:
        recordings: List of (camera, svo_path, mp4_path) tuples
    """
    started: list[tuple[object, Path, Path | None, threading.Event, threading.Thread]] = []
    writers: dict[str, _Mp4FrameWriter] = {}
    window: dict[str, float] = {}

    def stop_all():
        # Signal all cameras to stop simultaneously so recordings are the same length
        for _, _, _, stop_event, _ in started:
            stop_event.set()
        for camera, _, _, _, thread in started:
            thread.join(timeout=5.0)
            if isinstance(camera, ZedCamera):
                camera.stop_recording()
            _log.info(f"Stopped recording camera {camera.serial}")
        for serial, writer in writers.items():
            frames = writer.close()
            _log.info(f"Wrote {frames} frames to {writer.path.name} (camera {serial})")

    try:
        for camera, svo_path, mp4_path in recordings:
            stop_event = threading.Event()

            if isinstance(camera, ZedCamera):
                # The SDK records the SVO itself; the loop only has to keep grabbing.
                def recording_loop(cam=camera, event=stop_event):
                    while not event.is_set():
                        try:
                            cam.read_camera()
                        except Exception as e:
                            _log.error(f"Error grabbing frame during recording: {e}")
                            break

                camera.start_recording(str(svo_path))
                _log.info(f"Started recording camera {camera.serial} to {svo_path}")
            else:
                # No SDK recorder: grab and encode in the same thread. Encoding a 720p frame to
                # H.264 costs a few ms against a 33 ms frame period, so the grab loop keeps up; if
                # it ever does not, frames are dropped rather than queued, and the export stays
                # correct because it maps frames onto the recording window by count, not by index.
                if mp4_path is None:
                    raise ValueError(f"camera {camera.serial} has no SDK recorder, so mp4_path is required")
                probe = camera.read_camera()
                height, width = probe.rgb.shape[:2]
                writer = _Mp4FrameWriter(mp4_path, width, height, fps=_camera_fps(camera))
                writers[camera.serial] = writer

                def recording_loop(cam=camera, event=stop_event, w=writer, first=probe.rgb):
                    try:
                        w.write(first)
                        while not event.is_set():
                            w.write(cam.read_camera().rgb)
                    except Exception as e:
                        _log.error(f"Error recording camera {cam.serial}: {e}")

                _log.info(f"Started recording camera {camera.serial} to {mp4_path} ({width}x{height})")

            thread = threading.Thread(target=recording_loop, name=f"record-{camera.serial}")
            thread.start()
            started.append((camera, svo_path, mp4_path, stop_event, thread))
    except BaseException:
        # A camera failing to start leaves the earlier ones recording with live grab threads, which
        # then hold the devices and break every subsequent rollout in a warm session. Stop them and
        # let the original error propagate -- no MP4 conversion, it would mask the real failure.
        stop_all()
        raise

    window["t_start"] = time.time()
    try:
        yield window
    finally:
        stop_all()
        # Recording window closes when the cameras stop; the SVO->MP4 conversion below must NOT
        # count as recording time, so stamp t_stop before it.
        window["t_stop"] = time.time()

        # Convert to MP4 after all cameras have stopped. Only the ZEDs need this — a directly
        # encoded camera closed its writer inside stop_all() and its mp4 is already on disk.
        for camera, svo_path, mp4_path, _, _ in started:
            if not isinstance(camera, ZedCamera):
                continue
            if mp4_path is None:
                continue
            actual_svo_path = svo_path
            if not svo_path.exists():
                svo2_path = svo_path.with_suffix(".svo2")
                if svo2_path.exists():
                    _log.debug(f"SVO actually written by ZED SDK to {svo2_path.name}")
                    actual_svo_path = svo2_path
            if actual_svo_path.exists():
                convert_svo_to_mp4(actual_svo_path, mp4_path)
            else:
                raise FileNotFoundError(
                    f"Recording failed: SVO file not found at {svo_path} or {svo_path.with_suffix('.svo2')}"
                )


def save_perception_outputs(
    rgb: UInt8[np.ndarray, "h w 3"],
    intrinsics_matrix: Float[np.ndarray, "3 3"],
    depth_map: Float[np.ndarray, "h w"],
    xyz_map: Float[np.ndarray, "n 3"],
    rgb_map: Float[np.ndarray, "n 3"],
    bboxes: list[dict],
    masks: np.ndarray,
    save_dir: Path,
    robot_mask: Bool[np.ndarray, "h w"] | None = None,
):
    """Save perception outputs to disk.

    Visualization files (rgb.png, bboxes_viz.png, masks_viz.png) are saved at the top
    level of save_dir for quick access. Raw data files go into save_dir/perception/.
    """
    start_time = time.perf_counter()
    save_dir.mkdir(parents=True, exist_ok=True)
    perception_dir = save_dir / "perception"
    perception_dir.mkdir(exist_ok=True)

    # Camera image
    cv2.imwrite(str(save_dir / "rgb.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

    # Intrinsics
    intrinsics_dict = {"intrinsics": intrinsics_matrix.tolist()}
    with open(perception_dir / "intrinsics.json", "w") as f:
        json.dump(intrinsics_dict, f, indent=2)

    # Convert depth from meters to millimeters for uint16 storage
    depth_mm = depth_map * 1000.0
    depth_mm = np.clip(depth_mm, 0, 65535)
    depth_uint16 = depth_mm.astype(np.uint16)
    cv2.imwrite(str(perception_dir / "depth.png"), depth_uint16)

    # Turbo-colormapped depth for quick visual inspection (normalized over valid depth range)
    depth_turbo = get_heatmap(depth_map.reshape(-1), cmap_name="turbo").reshape(*depth_map.shape, 3)
    depth_turbo = (depth_turbo * 255.0).clip(0, 255).astype(np.uint8)
    cv2.imwrite(str(save_dir / "depth_turbo.png"), cv2.cvtColor(depth_turbo, cv2.COLOR_RGB2BGR))

    # Create point cloud and write
    pcd = get_o3d_pcd(xyz_map, rgb_map)
    o3d.io.write_point_cloud(str(perception_dir / "pointcloud.ply"), pcd)

    # Write bboxes
    rgb_pil = Image.fromarray(rgb)
    bbox_viz = visualize_detections(rgb_pil, bboxes, output_path=str(save_dir / "bboxes_viz.png"), show_plot=False)
    with open(perception_dir / "bboxes.json", "w") as f:
        json.dump(bboxes, f, indent=2)

    # Write masks
    masks_viz = visualize_masks(rgb_pil, masks, bboxes)
    cv2.imwrite(str(save_dir / "masks_viz.png"), cv2.cvtColor(masks_viz, cv2.COLOR_RGB2BGR))
    masks_bool = masks > 0.5
    np.savez_compressed(str(perception_dir / "masks.npz"), masks_bool)  # masks are sparse so can compress

    if robot_mask is not None:
        # Filename predates third-person perception, where the mask covers the whole arm rather than
        # just the gripper. Kept so viz-tiptop-run reads old and new runs the same way.
        robot_mask_img = Image.fromarray(robot_mask.astype(np.uint8) * 255)
        robot_mask_img.save(str(perception_dir / "gripper_mask.png"))

    save_dur = time.perf_counter() - start_time
    _log.info(f"Saved perception outputs to {save_dir} in {save_dur:.2f}s")
    return bbox_viz, masks_viz


def save_run_outputs(save_dir: Path, env, grasps: dict) -> None:
    """Save cuTAMP environment, grasps, and run artifacts (config) to disk."""
    # Save cutamp environment and grasps
    perception_dir = save_dir / "perception"
    perception_dir.mkdir(parents=True, exist_ok=True)
    with open(perception_dir / "cutamp_env.pkl", "wb") as f:
        dill.dump(env, f)
    _log.info(f"Saved cutamp env to {perception_dir}/cutamp_env.pkl")

    torch.save(grasps, perception_dir / "grasps.pt")
    _log.info(f"Saved grasps to {perception_dir}/grasps.pt")

    # tiptop config for reproducibility
    shutil.copy2(tiptop_config_path, save_dir / "tiptop.yml")
    _log.info(f"Saved tiptop config to {save_dir}/tiptop.yml")


def save_run_metadata(
    save_dir: Path,
    timestamp: str,
    task_instruction: str | None,
    q_at_capture: np.ndarray | list | None,
    world_from_cam: np.ndarray | list | None,
    perception_duration: float | None,
    grounded_atoms: list[dict] | None,
    planning_success: bool | None,
    planning_failure_reason: str | None,
    planning_duration: float | None,
    cleared: list[str] | None = None,
) -> None:
    """Save structured run metadata to metadata.json.

    ``cleared`` names the objects this episode moved out of the way BEFORE doing what the instruction
    asked (``tiptop.goal_clearing``). Those steps are part of the recorded episode while the
    instruction says nothing about them, so an episode with a non-empty ``cleared`` opens with motion
    the language does not account for -- recorded here so a dataset build can find or filter them
    without re-deriving it from the plan.
    """
    git_info = _collect_git_info()
    if git_info["dirty"]:
        diff = _get_git_diff()
        if diff:
            (save_dir / "git.diff").write_text(diff, encoding="utf-8")

    metadata = {
        "task_instruction": task_instruction,
        "timestamp": timestamp,
        "observation": {
            "q_at_capture": q_at_capture,
            "world_from_cam": world_from_cam,
        },
        "perception": {
            "grounded_atoms": grounded_atoms,
            "duration": round(perception_duration, 3) if perception_duration is not None else None,
        },
        "planning": {
            "success": planning_success,
            "failure_reason": planning_failure_reason,
            "duration": round(planning_duration, 3) if planning_duration is not None else None,
            "cleared": list(cleared or []),
        },
        "version": "1.0.0",
        "tiptop_version": tiptop.__version__,
        "git": git_info,
    }
    with open(save_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, cls=NumpyEncoder)
    _log.info(f"Saved run metadata to {save_dir}/metadata.json")

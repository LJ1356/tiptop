"""Modified from https://github.com/droid-dataset/droid/blob/main/droid/camera_utils/camera_readers/zed_camera.py"""

import json
import logging
import time
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import aiohttp
import cv2
import numpy as np
from jaxtyping import Float, UInt8
from tqdm import tqdm

from tiptop.config import tiptop_cfg
from tiptop.perception.cameras.frame import Frame

_log = logging.getLogger(__name__)


def time_ms():
    return time.time_ns() // 1_000_000


@dataclass(frozen=True, kw_only=True)
class ZedFrame(Frame):
    """Frame from ZED which also includes raw stereo pair and point cloud."""

    left_bgra: UInt8[np.ndarray, "h w 4"]  # BGRA left uint8
    right_bgra: UInt8[np.ndarray, "h w 4"]  # BGRA right uint8
    pointcloud: Float[np.ndarray, "h w 4"] | None = None  # Hardware pointcloud float32 millimeters


@dataclass(frozen=True)
class ZedIntrinsics:
    """Intrinsics for ZED camera."""

    K_left: Float[np.ndarray, "3 3"]  # Left RGB camera matrix
    K_right: Float[np.ndarray, "3 3"]  # Right RGB camera matrix
    distortion_left: Float[np.ndarray, "12"]
    distortion_right: Float[np.ndarray, "12"]
    baseline: float  # Meters


_STEREOLABS_USB_VENDOR_ID = "2b03"


def _sdk_visible_serials() -> dict[str, str]:
    """Serials the ZED SDK can currently enumerate, mapped to a 'model, state' description.

    A ZED whose video interface failed to come up is either absent from this list or reported with
    serial 0, which is why a serial can be correct yet still open as CAMERA NOT DETECTED.
    """
    import pyzed.sl as sl

    devices = {}
    for dev in sl.Camera.get_device_list():
        devices[str(dev.serial_number)] = f"{dev.camera_model}, {dev.camera_state}"
    return devices


def _usb_attached_serials() -> dict[str, str]:
    """Serials of Stereolabs USB devices the kernel sees, mapped to their product string.

    Reads sysfs directly (Linux-only, returns {} elsewhere). Every ZED presents a low-speed HID
    interface carrying the serial plus a separate USB3 video interface; the HID half enumerating
    alone is the signature of a cable/port that isn't giving the camera USB3.
    """
    serials = {}
    for dev_dir in Path("/sys/bus/usb/devices").glob("*/"):
        try:
            if (dev_dir / "idVendor").read_text().strip() != _STEREOLABS_USB_VENDOR_ID:
                continue
            serial = (dev_dir / "serial").read_text().strip()
            product = (dev_dir / "product").read_text().strip()
        except OSError:
            continue
        # The video interface reports a generic serial ("OV0001"); only the HID half carries the real one.
        if serial.isdigit() and serial != "0":
            serials[serial] = product
    return serials


def _open_failure_help(serial: str) -> str:
    """Explain an open failure in terms of what is actually attached, not just the SDK error code."""
    sdk = _sdk_visible_serials()
    if serial in sdk:
        return f"The SDK does see this serial ({sdk[serial]}), so it is attached but not openable."

    usb = _usb_attached_serials()
    lines = [
        f"The ZED SDK does not see s/n {serial}.",
        f"  SDK enumerates: {sdk if sdk else 'no cameras at all'}",
        f"  Stereolabs USB devices attached: {usb if usb else 'none'}",
    ]
    if serial in usb:
        lines += [
            f"  -> s/n {serial} IS attached ({usb[serial]}) but only its HID interface enumerated; its USB3",
            "     video interface never came up. The serial is correct -- editing it will not help.",
            "     Cause is the cable, the port, or the camera itself. To tell them apart: move the camera to",
            "     a USB3 port on the same controller as a working ZED and re-check. A video node appearing",
            "     means cable/port; still missing means the unit.",
        ]
    elif usb:
        lines += [
            f"  -> s/n {serial} is not attached at all. The serials above are what is plugged in; if a camera",
            "     was swapped, update TIPTOP_HAND_CAMERA_ID / TIPTOP_EXTERNAL_CAMERA_ID to match.",
        ]
    else:
        lines.append("  -> No Stereolabs device is attached. Check power and the USB connection.")
    return "\n".join(lines)


def _custom_params(resolution: str, fps: int, flip: bool = False):
    """Build camera init params from explicit resolution and fps strings."""
    import pyzed.sl as sl

    resolution_map = {
        "HD720": sl.RESOLUTION.HD720,
        "HD1080": sl.RESOLUTION.HD1080,
        "HD2K": sl.RESOLUTION.HD2K,
    }
    if resolution not in resolution_map:
        raise ValueError(f"Unknown resolution '{resolution}', expected one of {list(resolution_map)}")

    return dict(
        depth_minimum_distance=0.1,
        camera_resolution=resolution_map[resolution],
        depth_stabilization=False,
        camera_fps=fps,
        camera_image_flip=sl.FLIP_MODE.ON if flip else sl.FLIP_MODE.OFF,
    )


class ZedCamera:
    def __init__(
        self,
        serial: str,
        depth: bool = False,
        pointcloud: bool = False,
        resolution: str = "HD720",
        fps: int = 60,
        flip: bool = False,
        open_retries: int = 3,
        open_retry_delay: float = 2.0,
    ):
        import pyzed.sl as sl

        start_time = time.perf_counter()
        self.serial = serial
        self._enable_depth = depth
        self._enable_pcd = pointcloud
        self._is_recording = False

        # Initialize readers
        self._cam = sl.Camera()
        self._left_img = sl.Mat()
        self._right_img = sl.Mat()
        self._depth = sl.Mat()
        self._pcd = sl.Mat()
        self._runtime = sl.RuntimeParameters()

        # Open the camera. Opening several ZEDs in quick succession can hit a transient
        # USB enumeration/claim race ("can't claim interface", "Unable to capture
        # images"); the SDK's own 5 s retry isn't always enough, so retry from a fresh
        # Camera handle a few times before giving up.
        _log.info(f"Opening ZED camera {self.serial}")
        self._params = _custom_params(resolution, fps, flip=flip)
        sl_params = sl.InitParameters(**self._params)
        sl_params.set_from_serial_number(int(self.serial))
        status = self._cam.open(sl_params)
        attempt = 1
        while status != sl.ERROR_CODE.SUCCESS and attempt < open_retries:
            _log.warning(
                f"ZED camera (s/n: {self.serial}) open failed ({status}); "
                f"retry {attempt}/{open_retries - 1} in {open_retry_delay:.1f}s"
            )
            self._cam.close()
            # "can't claim interface: -6 (LIBUSB_ERROR_BUSY)" means the camera is still claimed
            # by a stale handle from a previous run that didn't release it (crash / kill -9 /
            # interrupted teardown). A hardware reboot of the camera over USB clears that and a
            # wedged video module. Targets this serial only, so sibling cameras are unaffected.
            # Only meaningful when the SDK can actually see the camera: rebooting a serial it can't
            # enumerate just returns INVALID FUNCTION CALL and buries the real error in noise.
            if self.serial in _sdk_visible_serials():
                try:
                    _log.info(f"Rebooting ZED camera {self.serial} over USB to recover...")
                    sl.Camera.reboot(int(self.serial))
                except Exception as e:
                    _log.warning(f"ZED reboot({self.serial}) failed (continuing to retry): {e}")
            else:
                # Full diagnosis is left to the RuntimeError below, so it is not repeated per retry.
                _log.warning(f"ZED camera {self.serial} is not enumerated by the SDK; skipping USB reboot")
            time.sleep(open_retry_delay)
            self._cam = sl.Camera()
            status = self._cam.open(sl_params)
            attempt += 1
        if status != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(
                f"ZED camera (s/n: {self.serial}) failed to open after {attempt} attempt(s): {status}\n"
                f"{_open_failure_help(self.serial)}"
            )

        # Cache the intrinsics
        _ = self.get_intrinsics()
        init_dur = time.perf_counter() - start_time
        _log.info(f"ZED camera (s/n: {self.serial}) initialization complete, took {init_dur:.2f}s")

    @cache
    def get_intrinsics(self) -> ZedIntrinsics:
        calib_params = self._cam.get_camera_information().camera_configuration.calibration_parameters
        baseline_mm = calib_params.get_camera_baseline()
        baseline = baseline_mm / 1000.0

        left = calib_params.left_cam
        K_left = np.array([[left.fx, 0.0, left.cx], [0.0, left.fy, left.cy], [0.0, 0.0, 1.0]])
        distortion_left = np.array(left.disto)

        right = calib_params.right_cam
        K_right = np.array([[right.fx, 0.0, right.cx], [0.0, right.fy, right.cy], [0.0, 0.0, 1.0]])
        distortion_right = np.array(right.disto)

        return ZedIntrinsics(
            K_left=K_left,
            K_right=K_right,
            distortion_left=distortion_left,
            distortion_right=distortion_right,
            baseline=baseline,
        )

    def read_camera(self) -> ZedFrame:
        import pyzed.sl as sl

        # Grab a frame
        err = self._cam.grab(self._runtime)
        if err != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"Failed to grab frame from ZED camera {self.serial}")

        # Retrieve data
        timestamp = self._cam.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_milliseconds()
        self._cam.retrieve_image(self._left_img, sl.VIEW.LEFT)
        left_bgra = self._left_img.get_data()
        self._cam.retrieve_image(self._right_img, sl.VIEW.RIGHT)
        right_bgra = self._right_img.get_data()

        depth = None
        if self._enable_depth:
            self._cam.retrieve_measure(self._depth, sl.MEASURE.DEPTH)
            depth = (self._depth.get_data() / 1000.0).astype(np.float32)  # mm to m

        pointcloud = None
        if self._enable_pcd:
            self._cam.retrieve_measure(self._pcd, sl.MEASURE.XYZ)
            pointcloud = self._pcd.get_data()

        return ZedFrame(
            serial=self.serial,
            timestamp=timestamp,
            rgb=cv2.cvtColor(left_bgra, cv2.COLOR_BGRA2RGB),
            intrinsics=self.get_intrinsics().K_left,
            depth=depth,
            left_bgra=left_bgra,
            right_bgra=right_bgra,
            pointcloud=pointcloud,
        )

    def start_recording(self, filename: str):
        """Start recording camera stream to SVO file using H265 compression."""
        import pyzed.sl as sl

        if not filename.endswith(".svo"):
            raise ValueError("Recording filename must end with .svo")

        # H265 needs NVENC, which requires ZED SDK >= 5.2.3 on Blackwell GPUs -- older SDKs ship a
        # Video Codec SDK that fails to init against driver 590+, and every hardware codec then dies
        # with a bare SVO_RECORDING_ERROR. Fall back to SVO_COMPRESSION_MODE.LOSSLESS (CPU, ~30x
        # larger) if this ever has to run on an older SDK.
        recording_param = sl.RecordingParameters(filename, sl.SVO_COMPRESSION_MODE.H265)
        err = self._cam.enable_recording(recording_param)
        if err != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"Failed to enable recording to {filename} ({err})")
        self._is_recording = True
        _log.info(f"Started recording to {filename}")

    def stop_recording(self):
        """Stop recording camera stream."""
        if self._is_recording:
            self._cam.disable_recording()
            self._is_recording = False
            _log.info("Stopped recording")

    def close(self):
        self.stop_recording()
        self._cam.close()


def zed_infer_depth(
    frame: ZedFrame,
    intrinsics: ZedIntrinsics,
) -> Float[np.ndarray, "h w"]:
    """Estimate depth from Zed frame and intrinsics using FoundationStereo. Synchronous version."""
    from tiptop.perception.foundation_stereo import infer_depth

    cfg = tiptop_cfg()
    K_left = intrinsics.K_left
    depth = infer_depth(
        cfg.perception.foundation_stereo.url,
        left_rgb=cv2.cvtColor(frame.left_bgra, cv2.COLOR_BGRA2RGB),
        right_rgb=cv2.cvtColor(frame.right_bgra, cv2.COLOR_BGRA2RGB),
        fx=K_left[0, 0],
        fy=K_left[1, 1],
        cx=K_left[0, 2],
        cy=K_left[1, 2],
        baseline=intrinsics.baseline,
    )
    return depth


async def zed_infer_depth_async(
    session: aiohttp.ClientSession,
    frame: ZedFrame,
    intrinsics: ZedIntrinsics,
) -> Float[np.ndarray, "h w"]:
    """Estimate depth from Zed frame and intrinsics using FoundationStereo. Async version."""
    from tiptop.perception.foundation_stereo import infer_depth_async

    cfg = tiptop_cfg()
    K_left = intrinsics.K_left
    depth = await infer_depth_async(
        session,
        cfg.perception.foundation_stereo.url,
        left_rgb=cv2.cvtColor(frame.left_bgra, cv2.COLOR_BGRA2RGB),
        right_rgb=cv2.cvtColor(frame.right_bgra, cv2.COLOR_BGRA2RGB),
        fx=K_left[0, 0],
        fy=K_left[1, 1],
        cx=K_left[0, 2],
        cy=K_left[1, 2],
        baseline=intrinsics.baseline,
    )
    return depth


class CorruptSVOError(RuntimeError):
    """Raised when an SVO file is corrupted/truncated and yields no usable video.

    Typically happens when a `tiptop-run` recording is killed mid-write (e.g. a
    crash), leaving the SVO truncated past the point ZED's auto-repair can fix.
    """


class SVOGPUMemoryError(RuntimeError):
    """Raised when the GPU is too full for the ZED SDK to open an SVO for decoding.

    The recording on disk is INTACT -- this is a transient resource failure, not corruption.
    Opening an SVO allocates a CUDA decode context, so it fails when something else is holding
    VRAM. The .svo2 is left in place and converts fine once the GPU frees up; do not discard the
    episode on this error.
    """


def convert_svo_to_mp4(svo_path: Path, mp4_path: Path, crf: int = 20):
    """Convert SVO file to MP4 video (left RGB only) using ffmpeg for efficient compression.

    Raises CorruptSVOError if the SVO cannot be opened or yields zero frames. If the
    SVO opens but truncates partway through, the frames decoded before the corruption
    are still written and a warning is logged (partial recovery).
    """
    import shutil
    import subprocess

    import pyzed.sl as sl

    if not svo_path.exists():
        raise FileNotFoundError(f"SVO file not found: {svo_path}")
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found on PATH, required for SVO to MP4 conversion")

    # Open SVO file
    _log.info(f"Converting {svo_path} to {mp4_path}")
    init_params = sl.InitParameters()
    init_params.set_from_svo_file(str(svo_path))
    init_params.svo_real_time_mode = False
    # Only the left RGB image is extracted, so disable depth to avoid loading the
    # NEURAL depth net onto the GPU (it OOMs when other models are still resident).
    init_params.depth_mode = sl.DEPTH_MODE.NONE
    zed = sl.Camera()
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        # Opening an SVO allocates a CUDA decode context, so a full GPU fails here with a code that
        # says nothing about the file. Reporting that as corruption throws away a perfectly good
        # recording, so keep the two apart: only a genuine file problem is CorruptSVOError.
        if err in (
            sl.ERROR_CODE.NOT_ENOUGH_GPU_MEMORY,
            sl.ERROR_CODE.NO_GPU_DETECTED,
            sl.ERROR_CODE.NO_GPU_COMPATIBLE,
        ):
            raise SVOGPUMemoryError(
                f"Cannot open SVO file -- the GPU is unavailable, the recording itself is fine: "
                f"{svo_path} ({err}). Free VRAM and re-run the conversion; the .svo2 is kept on disk."
            )
        # A truncated/corrupted recording (ZED auto-repair already tried and failed)
        # reports INVALID_SVO_FILE here. Surface it as recoverable so batch
        # conversion can skip this file and still process the others.
        raise CorruptSVOError(f"Cannot open SVO file (corrupted or truncated): {svo_path} ({err})")

    image_size = zed.get_camera_information().camera_configuration.resolution
    width = image_size.width
    height = image_size.height
    fps = zed.get_camera_information().camera_configuration.fps

    # Use ffmpeg with CRF for much smaller files with good quality
    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "warning",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "pipe:0",
        "-c:v",
        "libx264",
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
        str(mp4_path),
    ]
    proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    # Pipe frames from SVO to ffmpeg
    left_image = sl.Mat()
    rt_param = sl.RuntimeParameters()
    nb_frames = zed.get_svo_number_of_frames()
    frames_written = 0
    truncated = False
    # Per-frame hardware capture times (epoch seconds), one per written frame. Used by
    # the LeRobot export to align camera frames to the control timeline by timestamp.
    timestamps_s: list[float] = []
    try:
        with tqdm(total=nb_frames, desc="Converting SVO to MP4", unit="frame") as pbar:
            while True:
                err = zed.grab(rt_param)
                if err == sl.ERROR_CODE.SUCCESS:
                    zed.retrieve_image(left_image, sl.VIEW.LEFT)
                    frame_bgra = left_image.get_data()
                    frame_bgr = cv2.cvtColor(frame_bgra, cv2.COLOR_BGRA2BGR)
                    try:
                        proc.stdin.write(frame_bgr.tobytes())
                    except BrokenPipeError:
                        break  # ffmpeg died; returncode check below will raise
                    timestamps_s.append(zed.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_nanoseconds() / 1e9)
                    frames_written += 1
                    pbar.update(1)
                elif err == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
                    break
                else:
                    # Corruption mid-stream (e.g. truncated recording): keep the frames
                    # decoded so far rather than discarding the whole video.
                    _log.warning(
                        f"Stopping early at frame {zed.get_svo_position()} of {svo_path.name}: {err}. "
                        f"Saving the {frames_written} frame(s) recovered so far."
                    )
                    truncated = True
                    break
    finally:
        proc.stdin.close()
        stderr = proc.stderr.read()
        proc.wait()
        zed.close()

    if frames_written == 0:
        mp4_path.unlink(missing_ok=True)
        raise CorruptSVOError(f"SVO yielded no decodable frames (corrupted or truncated): {svo_path}")
    if proc.returncode != 0:
        stderr_msg = stderr.decode(errors="replace").strip()
        raise RuntimeError(f"ffmpeg exited with code {proc.returncode}: {stderr_msg}")
    # Sidecar of per-frame capture times, aligned 1:1 with the mp4's frames.
    ts_path = mp4_path.with_suffix(".timestamps.json")
    with open(ts_path, "w") as f:
        json.dump({"unit": "epoch_seconds", "timestamps": timestamps_s}, f)
    if truncated:
        _log.info(f"Partial conversion complete ({frames_written} frames): {mp4_path}")
    else:
        _log.info(f"Conversion complete: {mp4_path}")

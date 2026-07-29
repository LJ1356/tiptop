import json
import logging
import os
import sys
import time
from contextlib import contextmanager
from functools import cache
from pathlib import Path

import numpy as np
from bamboo.client import BambooFrankaClient
from cutamp.robots import (
    RerunRobot,
    load_fr3_franka_rerun,
    load_fr3_robotiq_rerun,
    load_franka_rerun,
    load_panda_robotiq_rerun,
    load_ur5_rerun,
)
from jaxtyping import Bool
from PIL import Image

from tiptop.config import config_dir, tiptop_cfg
from tiptop.ur5.ur5_client import UR5Client

gripper_mask_path = config_dir / "assets" / "gripper_mask.png"

REQUIRED_CUTAMP_VERSION = "0.0.5"


def check_cutamp_version() -> None:
    """Raise RuntimeError if the installed cuTAMP version does not match REQUIRED_CUTAMP_VERSION."""
    try:
        import cutamp

        version = cutamp.__version__
    except AttributeError:
        version = "<0.0.2"
    if version != REQUIRED_CUTAMP_VERSION:
        raise RuntimeError(
            f"cuTAMP version mismatch: required {REQUIRED_CUTAMP_VERSION}, found {version}. "
            "Please run: pixi run install-cutamp"
        )


class ServerHealthCheckError(Exception):
    """Raised when a server health check fails."""


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


@cache
def get_tiptop_cache_dir() -> Path:
    import tiptop

    top_level_dir = Path(tiptop.__file__).parent
    cache_dir = top_level_dir / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _build_bamboo_client() -> BambooFrankaClient:
    """Construct a fresh BambooFrankaClient (new ZMQ connection) from the config."""
    cfg = tiptop_cfg()

    if cfg.robot.type in {"fr3_robotiq", "panda_robotiq"}:
        gripper = "robotiq"
    elif cfg.robot.type in {"panda", "fr3"}:
        gripper = "franka"
    else:
        raise ValueError(f"Unknown robot type {cfg.robot.type}.")

    return BambooFrankaClient(
        control_port=cfg.robot.port, server_ip=cfg.robot.host, gripper_port=cfg.robot.gripper_port, gripper_type=gripper
    )


@cache
def get_bamboo_client() -> BambooFrankaClient:
    """Get the shared BambooFrankaClient instance using the current config."""
    return _build_bamboo_client()


RobotClient = BambooFrankaClient | UR5Client


def get_robot_client() -> RobotClient:
    """Get a RobotClient instance for the robot."""
    cfg = tiptop_cfg()
    if cfg.robot.type in {"fr3_robotiq", "panda_robotiq", "panda", "fr3"}:
        return get_bamboo_client()
    elif cfg.robot.type == "ur5":
        from tiptop.ur5.ur5_client import get_ur5_client

        return get_ur5_client()
    else:
        raise ValueError(f"Unknown robot type: {cfg.robot.type}")


def new_robot_client() -> RobotClient:
    """Build a fresh, uncached RobotClient with its own connection/sockets.

    Unlike get_robot_client(), this does NOT return the shared cached instance.
    Use it when a second thread needs to talk to the robot concurrently: ZMQ REQ
    sockets are not thread-safe, so a background state poller must own a separate
    connection rather than share the main command client's socket.
    """
    cfg = tiptop_cfg()
    if cfg.robot.type in {"fr3_robotiq", "panda_robotiq", "panda", "fr3"}:
        return _build_bamboo_client()
    elif cfg.robot.type == "ur5":
        return UR5Client(cfg.robot.host)
    else:
        raise ValueError(f"Unknown robot type: {cfg.robot.type}")


def release_robot_client(client: RobotClient) -> None:
    """Best-effort graceful release of a RobotClient's connection (ZMQ sockets etc.), e.g. before
    handing the physical robot off to a different controller such as DROID's StableRobotEnv for a
    teleop interlude. See data-collection/ARCHITECTURE.md: bamboo shim and DROID's RobotEnv both
    talk to the same NUC-side polymetis server and cannot hold it at the same time.

    Tries the common close/disconnect method names; a client exposing none of them just gets GC'd
    once every reference (including the get_robot_client cache) is dropped, which is not guaranteed
    to happen immediately -- reconnect_robot_client() at least drops the cache reference.
    """
    log = logging.getLogger(__name__)
    for name in ("close", "disconnect", "shutdown"):
        fn = getattr(client, name, None)
        if callable(fn):
            try:
                fn()
                log.info(f"Released {type(client).__name__} via .{name}()")
            except Exception:
                log.exception(f"{type(client).__name__}.{name}() raised while releasing")
            return
    log.warning(
        f"{type(client).__name__} exposes no close/disconnect/shutdown method; relying on GC to "
        "drop its connection. If a subsequent controller (e.g. teleop) fails to take over the arm, "
        "this is the first place to look."
    )


def wait_for_robot_stationary(
    *,
    velocity_threshold: float = 0.02,  # rad/s, per joint
    settle_seconds: float = 0.4,  # must stay below the threshold this long, continuously
    timeout_seconds: float = 15.0,
    poll_hz: float = 30.0,
) -> bool:
    """Block until the measured joint velocities confirm the arm has actually stopped, or give up.

    Closes a real timing gap in the teleop hand-off: release_robot_client() dropping tiptop's own
    connection does NOT mean the arm has stopped -- the shim is still finishing whatever trajectory
    segment it was mid-way through (terminate_current_policy() handing control back to polymetis's
    default hold controller happens some milliseconds to ~1s later, depending on how much of the
    segment was left). Telling the operator "safe to go stop the shim now" before that has actually
    happened risks killing the shim out from under an arm that is still moving.

    Uses its own short-lived ZMQ REQ connection to the shim's STATE port -- same wire protocol as
    lerobot_capture.JointSampler (msgpack ``{"command": "get_robot_state"}`` -> ``{"success",
    "data": {"q", "dq"}}``), deliberately NOT BambooFrankaClient, so this keeps working even after
    that connection has already been torn down (or never existed, e.g. a UR5 robot).

    Returns True once every joint's |velocity| has stayed under `velocity_threshold` for
    `settle_seconds` straight. Returns False on timeout -- meaning "could not confirm": either the
    state port itself is unreachable, or the arm is genuinely still moving. The caller must treat
    False as a loud warning, never as "probably fine".
    """
    import msgpack
    import zmq

    log = logging.getLogger(__name__)
    host = tiptop_cfg().robot.host
    port = int(os.environ.get("TIPTOP_STATE_PORT", 5557))

    ctx = zmq.Context()
    sock: "zmq.Socket | None" = None

    def _connect() -> None:
        nonlocal sock
        if sock is not None:
            sock.close(linger=0)
        sock = ctx.socket(zmq.REQ)
        sock.setsockopt(zmq.RCVTIMEO, 300)  # ms; a timed-out REQ socket is unusable, so we reconnect
        sock.setsockopt(zmq.LINGER, 0)
        sock.connect(f"tcp://{host}:{port}")

    _connect()
    req = msgpack.packb({"command": "get_robot_state"})
    deadline = time.monotonic() + timeout_seconds
    stationary_since: float | None = None
    try:
        while time.monotonic() < deadline:
            now = time.monotonic()
            dq = None
            try:
                sock.send(req)
                reply = msgpack.unpackb(sock.recv(), raw=False)
                data = reply.get("data") if isinstance(reply, dict) and reply.get("success") else None
                if data:
                    dq = np.asarray(data.get("dq", []), dtype=np.float32).reshape(-1)
            except zmq.Again:
                log.warning(f"wait_for_robot_stationary: state port tcp://{host}:{port} timed out; reconnecting")
                _connect()
            except Exception:
                log.exception("wait_for_robot_stationary: state port read failed; reconnecting")
                _connect()

            if dq is not None and dq.shape == (7,) and float(np.max(np.abs(dq))) < velocity_threshold:
                if stationary_since is None:
                    stationary_since = now
                elif now - stationary_since >= settle_seconds:
                    return True
            else:
                stationary_since = None  # no reading, or still moving: restart the settle window

            time.sleep(max(0.0, 1.0 / poll_hz))
        return False
    finally:
        if sock is not None:
            sock.close(linger=0)
        ctx.term()


def reconnect_robot_client() -> RobotClient:
    """Drop any cached RobotClient and build a fresh one. Use after release_robot_client() to hand
    the robot back from another controller (e.g. after a teleop interlude)."""
    get_bamboo_client.cache_clear()
    try:
        from tiptop.ur5.ur5_client import get_ur5_client

        get_ur5_client.cache_clear()
    except ImportError:
        pass
    return get_robot_client()


def get_robot_rerun(robot_type: str | None = None) -> RerunRobot:
    """Get a RerunRobot instance for the robot."""
    if robot_type is None:
        robot_type = tiptop_cfg().robot.type
    if robot_type == "fr3_robotiq":
        return load_fr3_robotiq_rerun()
    elif robot_type == "panda_robotiq":
        return load_panda_robotiq_rerun()
    elif robot_type == "panda":
        return load_franka_rerun()
    elif robot_type == "fr3":
        return load_fr3_franka_rerun()
    elif robot_type == "ur5":
        return load_ur5_rerun()
    else:
        raise ValueError(f"Unknown robot type: {robot_type}")


def load_gripper_mask() -> Bool[np.ndarray, "h w"]:
    """Load the binary mask for the gripper."""
    gripper_mask_img = Image.open(gripper_mask_path)
    gripper_mask = np.array(gripper_mask_img).astype(bool)
    return gripper_mask


def setup_logging(level: int = logging.INFO):
    # Ensure stdout and stderr use UTF-8 encoding to handle Unicode characters
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

    # Define a custom formatter with ANSI escape codes for colors
    class CustomFormatter(logging.Formatter):
        level_colors = {
            "DEBUG": "\033[34m",  # Blue
            "INFO": "\033[32m",  # Green
            "WARNING": "\033[33m",  # Yellow
            "ERROR": "\033[31m",  # Red
            "CRITICAL": "\033[35m",  # Magenta
        }
        reset = "\033[0m"

        def format(self, record):
            # Format without modifying the record - build colored levelname locally
            log_color = self.level_colors.get(record.levelname, "")
            colored_levelname = f"{log_color}{record.levelname}{self.reset}"

            # Manually build the formatted string with colors
            log_time = self.formatTime(record, self.datefmt)
            message = f"{log_time} - {record.name} - {colored_levelname} - {record.getMessage()}"
            if record.exc_info:
                message = message + "\n" + self.formatException(record.exc_info)
            return message

    # Define the log format
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    formatter = CustomFormatter(log_format, datefmt="%Y-%m-%d %H:%M:%S")

    # Configure the root logger (force reconfiguration)
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Remove only default StreamHandlers (stdout/stderr), keep other handlers (files, etc.)
    for handler in root_logger.handlers[:]:
        if isinstance(handler, logging.StreamHandler) and handler.stream in (sys.stdout, sys.stderr):
            root_logger.removeHandler(handler)

    # Add our custom handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)

    # Bamboo can be INFO level
    logging.getLogger("bamboo").setLevel(logging.INFO)

    # Silence noisy third-party library loggers
    noisy_modules = (
        "asyncio",
        "charset_normalizer",
        "curobo",
        "google_genai",
        "hpack",
        "httpcore",
        "httpx",
        "hydra",
        "matplotlib",
        "PIL",
        "trimesh",
        "urllib3",
        "yourdfpy",
    )
    for module in noisy_modules:
        logging.getLogger(module).setLevel(logging.WARNING)


def add_file_handler(log_file: Path, level: int = logging.DEBUG) -> logging.FileHandler:
    """Add a file handler to the root logger.

    Args:
        log_file: Path to the log file to create
        level: Logging level for the file handler

    Returns:
        The FileHandler instance so it can be removed later
    """
    log_file.parent.mkdir(parents=True, exist_ok=True)

    # Create file handler with plain formatting (no colors) and UTF-8 encoding
    file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    file_handler.setLevel(level)

    # Use same format as console but without colors
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    formatter = logging.Formatter(log_format, datefmt="%Y-%m-%d %H:%M:%S")
    file_handler.setFormatter(formatter)

    # Add to root logger
    root_logger = logging.getLogger()
    root_logger.addHandler(file_handler)
    return file_handler


def remove_file_handler(handler: logging.FileHandler):
    """Remove a file handler from the root logger and close it.

    Args:
        handler: The FileHandler to remove
    """
    root_logger = logging.getLogger()
    root_logger.removeHandler(handler)
    handler.close()


def print_tiptop_banner():
    """Print TiPToP ASCII art banner with ice cream."""
    banner = """
    ████████╗ ██╗ ██████╗ ████████╗  ██████╗  ██████╗
    ╚══██╔══╝ ██║ ██╔══██╗╚══██╔══╝ ██╔═══██╗ ██╔══██╗
       ██║    ██║ ██████╔╝   ██║    ██║   ██║ ██████╔╝
       ██║    ██║ ██╔═══╝    ██║    ██║   ██║ ██╔═══╝
       ██║    ██║ ██║        ██║    ╚██████╔╝ ██║
       ╚═╝    ╚═╝ ╚═╝        ╚═╝     ╚═════╝  ╚═╝
    ═══════════════════════════════════════════════════
       TiPToP is a Planner That just works on Pixels
    ═══════════════════════════════════════════════════
    """
    print(banner)


@contextmanager
def patch_log_level(logger_name: str, log_level: int):
    """Context manager to temporarily override the log level."""
    logger = logging.getLogger(logger_name)
    og_level = logger.getEffectiveLevel()
    logger.setLevel(log_level)
    yield
    logger.setLevel(og_level)

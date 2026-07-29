import asyncio
import ctypes
import json
import os
import logging
import shutil
import signal
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import aiohttp
import numpy as np
import open3d as o3d
import rerun as rr
import tyro
from curobo.geom.types import Cuboid, Mesh
from curobo.types.base import TensorDeviceType
from curobo.wrap.reacher.ik_solver import IKSolver
from curobo.wrap.reacher.motion_gen import MotionGen
from cutamp.config import TAMPConfiguration
from cutamp.envs import TAMPEnvironment
from cutamp.tamp_domain import HandEmpty, Holding, On
from cutamp.utils.rerun_utils import log_curobo_mesh_to_rerun
from jaxtyping import Bool, Float
from scipy.spatial import KDTree

from tiptop.config import load_calibration, tiptop_cfg
from tiptop.execute_plan import execute_cutamp_plan
from tiptop.motion_planning import (
    build_curobo_solvers,
    go_to_capture,
    go_to_home,
    resolve_grasp_orientation_cost,
    resolve_time_dilation_factor,
    resolve_traj_length_norm,
    resolve_trace_cfg,
    summarize_curobo_config,
)
from tiptop.perception.cameras import (
    Camera,
    DepthEstimator,
    Frame,
    ZedCamera,
    get_depth_estimator,
    get_external_camera,
    get_external_camera_2,
    get_hand_camera,
)
from tiptop.perception.m2t2 import m2t2_to_tiptop_transform
from tiptop.perception.sam2 import sam2_client
from tiptop.perception.segmentation import segment_pointcloud_by_masks, segment_table_with_ransac
from tiptop.perception.utils import convert_trimesh_box_to_curobo_cuboid, convert_trimesh_to_curobo_mesh
from tiptop.perception_wrapper import detect_and_segment, predict_depth_and_grasps
from tiptop.planning import build_tamp_config, run_planning, save_tiptop_plan, serialize_plan
from tiptop.recording import (
    record_cameras,
    save_perception_outputs,
    save_run_metadata,
    save_run_outputs,
)
from tiptop.utils import (
    RobotClient,
    add_file_handler,
    check_cutamp_version,
    get_robot_client,
    get_robot_rerun,
    load_gripper_mask,
    print_tiptop_banner,
    reconnect_robot_client,
    release_robot_client,
    remove_file_handler,
    setup_logging,
    wait_for_robot_stationary,
)
from tiptop.lerobot_capture import GRIPPER_MAX_WIDTH, GripperSampler, JointSampler, _read_gripper_width, dump_raw_episode
from tiptop.viz_utils import get_gripper_mesh, get_heatmap
from tiptop.workspace import workspace_cuboids

_log = logging.getLogger(__name__)
tensor_args = TensorDeviceType()

# Sampling rate for the LeRobot DROID-format capture during plan execution (matches DROID).
LEROBOT_FPS = 15

# A measured gripper width (metres) at or above this counts as "already open", so the
# per-episode reset skips re-issuing an open. 90% of the Robotiq 2F-85 full span.
GRIPPER_OPEN_WIDTH = 0.9 * GRIPPER_MAX_WIDTH

_executor_pool = None


def _init_pool_worker() -> None:
    """Set up a save-worker process.

    Ignores SIGINT so only the main process handles Ctrl+C, and asks the kernel to SIGTERM this
    worker if its parent dies. Without the death signal, force-killing a run (SIGKILL, so no atexit
    hook runs) strands the workers: they are reparented to init and survive indefinitely.
    """
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        PR_SET_PDEATHSIG = 1
        ctypes.CDLL("libc.so.6", use_errno=True).prctl(PR_SET_PDEATHSIG, signal.SIGTERM)
    except (OSError, AttributeError):  # non-Linux or no libc: best effort, the pool still works
        pass


_preempting = False  # a rollout abort is unwinding; extra SIGINTs are absorbed until it finishes


def _sigint_preempt(_signum, _frame) -> None:
    """SIGINT preempts the CURRENT ROLLOUT; it must never kill the warmed session.

    The first Ctrl-C (or the data-collection UI's Preempt button) raises KeyboardInterrupt, which
    the rollout loop catches and turns into "abort this rollout, go back to the task prompt".

    Unwinding is not instant -- the cameras stop and the SVO is converted to MP4, which takes
    several seconds. A second Ctrl-C landing in that window used to be raised inside the loop's own
    KeyboardInterrupt handler (or a finally block), escaping to the top-level handler and killing
    the session. So while a preempt is already unwinding, further SIGINTs are absorbed.

    This only softens SIGINT. SIGTERM/SIGKILL -- what the UI's Stop button escalates to, and what
    `q` at the prompt does gracefully -- still end the session.
    """
    global _preempting
    if _preempting:
        _log.warning(
            "Preempt already in progress (closing out the recording) -- ignoring extra Ctrl-C. "
            "The session stays warm; use Stop/Finish to end it."
        )
        return
    _preempting = True
    raise KeyboardInterrupt


def _clear_preempt() -> None:
    """Called once a rollout abort has fully unwound, so the next Ctrl-C preempts again."""
    global _preempting
    _preempting = False


# Set by the "switch to teleop" trigger (SIGUSR1) so the KeyboardInterrupt handler in the rollout
# loop knows to run the teleop hand-off (release the robot, wait for the human to hand it back, then
# auto-replan the same task) instead of just landing at the task prompt like a plain preempt.
_teleop_requested = False

# Set by _run_teleop_handoff once teleop hands the robot back; consumed by the NEXT call to
# _get_task_instruction() so the loop immediately replans/executes the SAME task from the human's
# hand-off pose instead of blocking on a typed task.
_pending_instruction: str | None = None

# Set alongside _pending_instruction; consumed by the rollout loop's per-episode reset so that the
# post-teleop iteration does NOT home the arm or open the gripper. Both would destroy exactly what
# the hand-off exists to preserve: the operator's final pose, and whatever they left in the gripper.
_resumed_from_teleop = False


def _sigusr1_teleop_switch(_signum, _frame) -> None:
    """"Switch to teleop" trigger (SIGUSR1) from the data-collection UI's button.

    Identical unwind to Ctrl-C/Preempt (see _sigint_preempt): the in-flight motion segment finishes
    (bamboo has no abort), no further plan steps are sent, and the loop lands back at the task
    prompt. The only difference is _teleop_requested, which routes the KeyboardInterrupt handler into
    _run_teleop_handoff instead of just waiting for a new typed task.
    """
    global _teleop_requested
    _teleop_requested = True
    _sigint_preempt(_signum, _frame)


def _release_cameras(container: "_DemoContainer") -> None:
    """Close this process's ZED cameras so the teleop driver can open them.

    The warm session holds every configured camera open from get_demo_container until teardown, but
    a ZED can only be opened by one process at a time and DROID's StableRobotEnv opens the same
    serials in its own background capture processes. Without this the handed-off teleop session dies
    on camera open. _reacquire_cameras puts them back.
    """
    for attr in ("cam", "external_cam", "external_cam_2"):
        camera = getattr(container, attr, None)
        if camera is None:
            continue
        try:
            camera.close()
        except Exception:
            _log.exception(f"Failed to close {attr} for the teleop hand-off")
        object.__setattr__(container, attr, None)
    object.__setattr__(container, "depth_estimator", None)


def _reacquire_cameras(container: "_DemoContainer") -> None:
    """Re-open the cameras released by _release_cameras and rebuild what depends on them.

    The getters are not cached, so each call opens the device afresh. The depth estimator closes over
    camera intrinsics, so it is rebuilt against the NEW hand-camera object rather than reused.
    Re-opening a ZED takes a few seconds and can fail (LOW USB BANDWIDTH) -- that raises here, which
    aborts the rollout back to the task prompt rather than silently running with a dead camera.
    """
    cam = get_hand_camera()
    object.__setattr__(container, "cam", cam)
    object.__setattr__(container, "external_cam", get_external_camera())
    object.__setattr__(container, "external_cam_2", get_external_camera_2())
    object.__setattr__(container, "depth_estimator", get_depth_estimator(cam))


def _run_teleop_handoff(container: "_DemoContainer") -> None:
    """Hand the physical arm off to a human teleop session, then block until it's handed back.

    Called from the rollout loop's KeyboardInterrupt handler once the in-flight rollout has already
    unwound. From here:
      1. release this process's RobotClient connection so a separate teleop process (DROID's
         StableRobotEnv) can take over the arm -- they talk to the same NUC-side polymetis server and
         cannot hold it at once (see data-collection/ARCHITECTURE.md).
      2. confirm the arm has ACTUALLY stopped moving (wait_for_robot_stationary) before the shim is
         stopped -- releasing our connection does not mean the shim is done: it still has to finish
         the in-flight trajectory segment and hand control back to polymetis's default hold
         controller, which takes real time. Skipping this check would let the server kill the shim
         (and start teleop) while the arm is still moving. This runs BEFORE the shim is stopped,
         since it reads the shim's own state port.
      3. release the ZED cameras, which the teleop driver needs to open itself.
      4. emit "awaiting_teleop_resume": the server then stops the bamboo shim on the NUC and starts
         the teleop session, and sends "resume" back once that session ends and the shim is back up.
      5. block on stdin for "resume"
      6. re-open the cameras, reconnect the RobotClient, and queue the SAME task instruction so the
         next loop iteration re-runs perception + cuTAMP planning + execution from wherever the human
         left the arm. _resumed_from_teleop makes that iteration skip its usual pre-episode motion
         entirely (home, gripper open, and the move to the capture pose), so nothing disturbs the
         operator's final pose or grasp before the new plan is built.
    """
    global _pending_instruction, _resumed_from_teleop
    _log.info("Teleop switch requested: releasing the robot connection for hand-off")
    _emit_event({"event": "teleop_handoff_start"})
    release_robot_client(container.robot)

    _log.info("Confirming the arm has come to a full stop before handing off to teleop...")
    if wait_for_robot_stationary():
        _log.info("Confirmed: the arm is stationary")
    else:
        _log.warning(
            "Could not confirm the arm has stopped moving (state port unreachable, or it is "
            "genuinely still moving after the timeout) -- warning instead of silently proceeding"
        )
        _emit_event(
            {
                "event": "teleop_handoff_warning",
                "message": (
                    "Could not confirm the arm has stopped moving. The shim will be stopped and "
                    "teleop started anyway — stay clear of the robot and be ready to e-stop."
                ),
            }
        )

    _log.info("Releasing the ZED cameras so the teleop driver can open them")
    _release_cameras(container)

    _emit_event({"event": "awaiting_teleop_resume"})
    _log.info(
        "Robot and cameras released; waiting for 'resume' on stdin (sent once the teleop session "
        "has ended, the arm is free and the shim is back up) before reacquiring and replanning..."
    )
    try:
        while True:
            raw = input().strip()
            cmd = raw
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, dict) and "cmd" in parsed:
                    cmd = parsed["cmd"]
            except ValueError:
                pass
            if cmd.lower() == "resume":
                break
            _log.warning(f"Ignoring unexpected input while awaiting teleop resume: {raw!r}")
    except EOFError:
        raise UserExitException("EOF while awaiting teleop resume")

    _log.info("Resuming: re-opening the cameras and reconnecting to the robot")
    _reacquire_cameras(container)
    object.__setattr__(container, "robot", reconnect_robot_client())
    _emit_event({"event": "teleop_handoff_done"})

    if _LAST_TASK:
        _pending_instruction = _LAST_TASK
        _resumed_from_teleop = True


class UserExitException(Exception):
    """Raised when user explicitly requests to exit."""


def _emit_event(payload: dict) -> None:
    """Append one JSON event line to ``$TIPTOP_EVENTS_FILE`` (the data-collection server's rollout
    state feed). No-op if the env var is unset; never raises, so it can wrap any control-flow point."""
    path = os.environ.get("TIPTOP_EVENTS_FILE")
    if not path:
        return
    try:
        with open(path, "a") as f:
            f.write(json.dumps({"ts": time.time(), **payload}) + "\n")
            f.flush()
    except Exception:
        pass


@dataclass(frozen=True)
class Observation:
    """Snapshot of sensor data and robot state needed for one perception+planning run."""

    frame: Frame
    world_from_cam: Float[np.ndarray, "4 4"]
    q_init: Float[np.ndarray | list, "n"]
    # Additional stereo frames captured back-to-back at the same (static) pose, used for
    # temporal depth smoothing. Empty for replay/websocket paths, which fuse nothing.
    depth_frames: tuple[Frame, ...] = ()


@dataclass(frozen=True)
class _DemoContainer:
    """Container for storing things needed for the live robot demo."""

    robot: RobotClient
    cam: Camera
    external_cam: Camera | None
    external_cam_2: Camera | None
    enable_recording: bool
    ee_from_cam: Float[np.ndarray, "4 4"]
    depth_estimator: DepthEstimator

    gripper_mask: Bool[np.ndarray, "h w"]

    ik_solver: IKSolver
    motion_gen: MotionGen

    # Resolved cuRobo cost/tamp-parameter config the solvers were built with (summarize_curobo_config).
    # Logged and saved per rollout so "did my cfg/tamp/*.yml override apply" is auditable after the
    # fact, not just live in the warmup console (see async_entrypoint).
    curobo_config_summary: dict

    # Raw cfg/tamp/*.yml tamp_overrides, threaded to run_planning for the plan-time knobs it resolves
    # itself (currently trajectory blending -- `blend_trajectory` etc.).
    cost_overrides: dict


@dataclass
class ProcessedScene:
    """Processed 3D scene ready for TAMP."""

    table_cuboid: Cuboid
    object_meshes: dict[str, Mesh]
    object_pcds: dict[str, o3d.geometry.PointCloud]
    grasps: dict[str, dict]  # Label -> grasp data with tensor versions


def capture_live_observation(container: _DemoContainer) -> Observation:
    """Read robot joint positions and compute world_from_cam via forward kinematics."""
    q_curr = container.robot.get_joint_positions()
    q_curr_pt = tensor_args.to_device(q_curr)
    world_from_ee = container.motion_gen.kinematics.get_state(q_curr_pt).ee_pose.get_numpy_matrix()[0]
    world_from_cam = world_from_ee @ container.ee_from_cam

    # Grab a short burst of frames at this static pose for temporal depth smoothing. The first
    # frame is the representative one (used for rgb/intrinsics); the rest feed the median fusion.
    num_frames = max(1, int(tiptop_cfg().perception.depth_smoothing.num_frames))
    frames = [container.cam.read_camera() for _ in range(num_frames)]
    return Observation(
        frame=frames[0],
        world_from_cam=world_from_cam,
        q_init=q_curr,
        depth_frames=tuple(frames),
    )


def get_demo_container(
    num_particles: int,
    num_spheres: int,
    collision_activation_distance: float,
    enable_recording: bool = False,
    cost_overrides: dict | None = None,
    curobo_config_summary: dict | None = None,
) -> _DemoContainer:
    """Cache and warm-up everything needed for the live demo."""
    _log.info("Starting demo warmup...")
    client = get_robot_client()

    # Setup cameras
    cam = get_hand_camera()
    external_cam = get_external_camera()
    # Second exterior camera (DROID exterior_2). None if its config is commented out
    # (deliberate 2-camera setup) or if a configured camera failed to open.
    external_cam_2 = get_external_camera_2()
    ee_from_cam = load_calibration(cam.serial)

    # Recording needs every camera that is configured (uncommented) in tiptop.yml. Fail fast
    # here, before any rollout, so we never silently collect data missing a configured camera.
    if enable_recording:
        if not isinstance(cam, ZedCamera):
            raise NotImplementedError(f"Recording requires a ZED hand camera, got {type(cam).__name__}")
        if not isinstance(external_cam, ZedCamera):
            raise NotImplementedError(f"Recording requires a ZED external camera, got {type(external_cam).__name__}")
        # external_2 is only required when it's uncommented in tiptop.yml. If it's configured but
        # failed to open, abort; if it's commented out, record with the two remaining cameras.
        external_2_configured = tiptop_cfg().cameras.get("external_2") is not None
        if external_2_configured and not isinstance(external_cam_2, ZedCamera):
            raise RuntimeError(
                "Recording requires the configured second external ZED "
                "(cameras.external_2, s/n 31425515), but it is unavailable "
                f"(got {type(external_cam_2).__name__}). It most likely failed to open "
                "(e.g. LOW USB BANDWIDTH) — lower the camera fps/resolution in tiptop.yml or move it "
                "to another USB3 controller; check it is connected. Aborting before the run so no "
                "rollout is collected with a missing camera. To intentionally record with two cameras, "
                "comment out cameras.external_2 in tiptop.yml."
            )

    # Create depth estimator once — closed over camera intrinsics
    # Cache the SAM2 client
    sam2_client()

    # Warm-up IK solver and motion generator (cost_overrides applies the cfg/tamp/*.yml cost knobs).
    ik_solver, motion_gen, _ = build_curobo_solvers(
        num_particles, num_spheres, collision_activation_distance, cost_overrides=cost_overrides
    )
    return _DemoContainer(
        robot=client,
        cam=cam,
        external_cam=external_cam,
        external_cam_2=external_cam_2,
        enable_recording=enable_recording,
        ee_from_cam=ee_from_cam,
        depth_estimator=get_depth_estimator(cam),
        gripper_mask=load_gripper_mask(),
        ik_solver=ik_solver,
        motion_gen=motion_gen,
        curobo_config_summary=curobo_config_summary or {},
        cost_overrides=cost_overrides or {},
    )


async def check_server_health(session: aiohttp.ClientSession):
    """Check health of FoundationStereo and M2T2 server."""
    from tiptop.perception.foundation_stereo import check_health_status as fs_check_health_status
    from tiptop.perception.m2t2 import check_health_status as m2t2_check_health_status

    cfg = tiptop_cfg()
    await asyncio.gather(
        fs_check_health_status(session, cfg.perception.foundation_stereo.url),
        m2t2_check_health_status(session, cfg.perception.m2t2.url),
    )
    _log.info("Server health checks successful!")


def _label_rollout(save_dir: Path, output_dir: str, timestamp: str) -> Path:
    """Prompt user to label rollout as success/failure, moving it out of eval/ to
    <success|failure>/<timestamp>/. Loops on invalid input. Returns the final rollout
    directory (or the unchanged eval dir if skipped) so it can be post-processed."""
    _emit_event({"event": "awaiting_label", "dir": str(save_dir)})
    try:
        while True:
            user_input = (
                input(
                    "\nWas the execution successful? Enter 'y' for success, 'n' for failure, or leave empty to skip: "
                )
                .strip()
                .lower()
            )
            if user_input in ("y", "n"):
                cls = "success" if user_input == "y" else "failure"
                dest = Path(output_dir) / cls / timestamp
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(save_dir, dest)
                _log.info(f"Moved rollout to {cls} directory: {dest}")
                _emit_event({"event": "labeled", "dir": str(dest), "success": user_input == "y"})
                return dest
            elif user_input == "":
                _log.info(f"Keeping rollout in eval directory: {save_dir}")
                return save_dir
            else:
                print("Invalid input. Please enter 'y', 'n', or leave empty to skip.")
    except EOFError:
        _log.info("No input received, keeping rollout in eval directory")
        return save_dir


_LAST_TASK: str | None = None
_postprocess_procs: list[subprocess.Popen] = []

# Manual robot commands accepted at the task prompt, in place of a task instruction. The
# data-collection UI's top-bar buttons drive these over stdin; a terminal user can just type them.
# They run BETWEEN rollouts (the prompt is the one point where the arm is idle and stdin is being
# read), reusing the warmed container -- so no cuRobo re-warm and no second robot connection.
ROBOT_COMMANDS = ("home", "open")


def _open_gripper_if_needed(container) -> float | None:
    """Open the gripper unless the measured width already reads open. Returns the measured width."""
    width = _read_gripper_width(container.robot)
    if width is not None and width >= GRIPPER_OPEN_WIDTH:
        _log.info(f"Gripper already open (width={width:.3f} m >= {GRIPPER_OPEN_WIDTH:.3f} m); skipping open")
        return width
    _log.info(f"Opening gripper (measured width={width})")
    container.robot.open_gripper()
    return width


def _run_robot_command(container, cfg, cmd: str) -> None:
    """Run a manual robot command typed at the task prompt.

    Never raises: a failed nudge (controller hiccup, gripper unreadable) must not tear down the
    warmed session -- the user should just land back at the prompt and be able to retry.
    """
    try:
        if cmd == "home":
            _log.info("Manual command: returning the arm home")
            go_to_home(time_dilation_factor=cfg.robot.time_dilation_factor, motion_gen=container.motion_gen)
        elif cmd == "open":
            _log.info("Manual command: opening the gripper")
            _open_gripper_if_needed(container)
        else:
            raise ValueError(f"Unknown robot command: {cmd}")
        _emit_event({"event": "robot_command", "command": cmd, "ok": True})
    except Exception as e:
        _log.exception(f"Manual robot command '{cmd}' failed: {e}")
        _emit_event({"event": "robot_command", "command": cmd, "ok": False, "error": str(e)})


def _get_task_instruction() -> str:
    """Task for the next rollout. The first comes from ``TIPTOP_TASK`` (non-interactive
    launch); subsequent ones are prompted interactively so the warmed container is reused
    across rollouts. Enter repeats the last task, typing a new one changes it, and
    'q'/'exit'/Ctrl-D ends the session (raising UserExitException).

    A ROBOT_COMMANDS word ('home'/'open') is returned as-is instead of a task; the caller runs it
    and re-prompts. It is deliberately NOT remembered as the last task, so a later bare Enter still
    repeats the real instruction rather than nudging the robot again.

    If a teleop hand-off just finished (_run_teleop_handoff queued _pending_instruction), that is
    consumed here first, WITHOUT blocking on stdin -- the loop replans/executes the same task
    immediately from wherever the human left the arm."""
    global _LAST_TASK, _pending_instruction
    if _pending_instruction is not None:
        instr = _pending_instruction
        _pending_instruction = None
        return instr
    env_task = os.environ.get('TIPTOP_TASK', '')
    if env_task:
        os.environ['TIPTOP_TASK'] = ''  # consume the launch task
        instr = env_task.strip()
        if not instr or instr.lower() in ('exit', 'q', 'quit'):
            raise UserExitException('TIPTOP_TASK empty/exit')
        _LAST_TASK = instr
        return instr
    # Interactive: keep reusing the warm container for back-to-back rollouts.
    suffix = f" [{_LAST_TASK}]" if _LAST_TASK else ""
    _emit_event({"event": "awaiting_task"})
    try:
        raw = input(f"\nNext task (Enter = repeat{suffix}, 'home'/'open' to nudge the robot, 'q' to quit): ").strip()
    except EOFError:
        raise UserExitException('EOF; ending session')
    if raw.lower() in ('q', 'exit', 'quit'):
        raise UserExitException('user quit')
    if raw.lower() in ROBOT_COMMANDS:
        return raw.lower()  # a robot nudge, not a task -- leave _LAST_TASK alone
    if not raw:
        if _LAST_TASK:
            return _LAST_TASK
        raise UserExitException('no task entered; ending session')
    _LAST_TASK = raw
    return raw


def _spawn_postprocess(rollout_dir: Path) -> None:
    """Fire-and-forget background post-processing (gifs + LeRobot export) for one finished
    rollout, so the next rollout can start immediately. No-op if the launcher didn't set
    TIPTOP_POSTPROCESS_SCRIPT (e.g. tiptop-run was started directly, not via run-tiptop.sh)."""
    script = os.environ.get("TIPTOP_POSTPROCESS_SCRIPT")
    if not script:
        return
    try:
        logf = open(rollout_dir / "postprocess.log", "ab")
        proc = subprocess.Popen(
            ["bash", script, str(rollout_dir)],
            stdout=logf, stderr=logf, stdin=subprocess.DEVNULL,
            start_new_session=True,  # detach so it survives and never blocks the run loop
        )
        _postprocess_procs.append(proc)
        _log.info(f"Post-processing {rollout_dir.name} in background (pid {proc.pid}) -> postprocess.log")
    except Exception:
        _log.exception("Failed to launch background post-processing")


def create_tamp_environment(
    object_meshes: dict[str, Mesh], table_cuboid: Cuboid, grounded_atoms: list[dict], include_workspace: bool
) -> tuple[TAMPEnvironment, list[Cuboid | Mesh]]:
    # Reject goals that reference objects not present in the perceived scene.
    # Without this, cuTAMP's BFS runs without stopping, expanding the move-chain on an unreachable goal.
    known_labels = set(object_meshes.keys()) | {table_cuboid.name}
    for atom in grounded_atoms:
        for arg in atom.get("args", []):
            if arg not in known_labels:
                raise ValueError(
                    f"Goal predicate {atom['predicate']}({', '.join(atom['args'])}) "
                    f"references unknown object '{arg}'. Known objects: {sorted(known_labels)}"
                )

    # Identify which objects are used as surfaces (second arg in on(x, y))
    surface_labels = set()
    for atom in grounded_atoms:
        if atom["predicate"] == "on" and len(atom["args"]) == 2:
            surface_labels.add(atom["args"][1])

    # Separate movables and surfaces
    movables = []
    surfaces = []
    for label, mesh in object_meshes.items():
        if label in surface_labels:
            surfaces.append(mesh)
        else:
            movables.append(mesh)
    _log.info(f"Movables: {[m.name for m in movables]}")
    _log.info(f"Surfaces: {[s.name for s in surfaces]}")

    # Create goal state from grounded atoms
    goal_state: set = set()
    has_holding = False
    for atom in grounded_atoms:
        if atom["predicate"] == "on" and len(atom["args"]) == 2:
            movable_label, surface_label = atom["args"]
            goal_state.add(On.ground(movable_label, surface_label))
            _log.info(f"Goal: {movable_label} on {surface_label}")
        elif atom["predicate"] == "holding" and len(atom["args"]) == 1:
            has_holding = True
            movable_label = atom["args"][0]
            goal_state.add(Holding.ground(movable_label))
            _log.info(f"Goal: holding {movable_label}")
    if not has_holding:
        goal_state.add(HandEmpty.ground())

    # All surfaces include table and detected surface objects
    all_surfaces = [table_cuboid, *surfaces]
    statics = list(workspace_cuboids()) if include_workspace else []
    for surface in all_surfaces:
        statics.append(surface)

    # Create TAMP environment
    env = TAMPEnvironment(
        name="tiptop_cutamp",
        movables=movables,
        statics=statics,
        type_to_objects={"Movable": movables, "Surface": all_surfaces},
        goal_state=frozenset(goal_state),
    )
    _log.info(f"Created TAMP environment with {len(movables)} movables, {len(all_surfaces)} surfaces")
    return env, all_surfaces


def process_scene_geometry(
    xyz_map: np.ndarray,
    rgb_map: np.ndarray,
    masks: np.ndarray,
    bboxes: list,
    grasps: dict,
    object_pcds: dict[str, o3d.geometry.PointCloud] | None = None,
) -> ProcessedScene:
    """Process perception results into 3D scene geometry for TAMP.

    Args:
        xyz_map: World-space XYZ coordinates (H, W, 3)
        rgb_map: RGB image (H, W, 3) in 0-255 range
        masks: Segmentation masks from SAM2
        bboxes: Bounding boxes from Gemini
        grasps: Grasp predictions from M2T2
        object_pcds: Optional pre-computed object point clouds

    Returns:
        ProcessedScene with table cuboid, object meshes, pcds, and filtered grasps
    """
    # Segment table with RANSAC (returns trimesh Box)
    table_trimesh = segment_table_with_ransac(xyz_map, rgb_map, masks)
    table_cuboid = convert_trimesh_box_to_curobo_cuboid(table_trimesh, name="table")
    log_curobo_mesh_to_rerun("world/table", table_cuboid.get_mesh(), static_transform=True)

    # For filtering to table plane height
    config = TAMPConfiguration()
    table_top_z = table_trimesh.bounds[1, 2] + config.world_activation_distance + config.coll_sphere_radius * 2
    object_trimeshes, object_pcds_computed = segment_pointcloud_by_masks(
        xyz_map,
        rgb_map,
        masks,
        bboxes,
        table_top_z,
        return_pcd=True,
        erode_pixels=tiptop_cfg().perception.mask_erosion_pixels,
    )

    # Use provided point clouds if available, otherwise use computed ones
    if object_pcds is None:
        object_pcds = object_pcds_computed

    # Associate grasps with objects by checking contact point proximity
    # Build a single KDTree from all object points with label tracking
    obj_labels = list(object_pcds.keys())
    all_points = []
    point_to_label = []  # Maps each point index to its object label
    for label, pcd in object_pcds.items():
        obj_points = np.asarray(pcd.points)
        all_points.append(obj_points)
        point_to_label.extend([label] * len(obj_points))

    all_points = np.vstack(all_points)
    point_to_label = np.array(point_to_label)
    combined_kdtree = KDTree(all_points)

    # Re-associate grasps to objects based on contact point proximity
    # Collect all valid grasps in flat arrays first
    all_poses, all_confs, all_contacts, all_labels = [], [], [], []
    for _, grasp_dict in grasps.items():
        poses, confs, contacts = grasp_dict["poses"], grasp_dict["confidences"], grasp_dict["contacts"]
        if len(contacts) == 0:
            continue

        dists, nearest_idxs = combined_kdtree.query(contacts)
        nearest_labels = point_to_label[nearest_idxs]
        within_thresh = dists < tiptop_cfg().perception.contact_threshold_m
        all_poses.append(poses[within_thresh])
        all_confs.append(confs[within_thresh])
        all_contacts.append(contacts[within_thresh])
        all_labels.append(nearest_labels[within_thresh])

    # Group by object label using boolean masks
    filtered_grasps = {}
    if all_poses:
        all_poses = np.concatenate(all_poses)
        all_confs = np.concatenate(all_confs)
        all_contacts = np.concatenate(all_contacts)
        all_labels = np.concatenate(all_labels)

        for label in obj_labels:
            mask = all_labels == label
            filtered_grasps[label] = {
                "poses": all_poses[mask],
                "confidences": all_confs[mask],
                "contacts": all_contacts[mask],
            }
            count = mask.sum()
            if count > 0:
                _log.info(
                    f"Object {label}: Associated {count} grasps (within {tiptop_cfg().perception.contact_threshold_m * 100:.1f}cm)"
                )
            else:
                _log.warning(f"Object {label}: No grasps within threshold")
    else:
        for label in obj_labels:
            filtered_grasps[label] = {
                "poses": np.array([]).reshape(0, 4, 4),
                "confidences": np.array([]),
                "contacts": np.array([]).reshape(0, 0, 3),
            }
            _log.warning(f"Object {label}: No grasps within threshold")

    gripper_mesh = get_gripper_mesh()
    vertices = np.asarray(gripper_mesh.vertices)
    vertices_hom = np.c_[vertices, np.ones(len(vertices))]  # Add homogeneous coordinate
    faces = np.asarray(gripper_mesh.triangles)
    viz_grasp_dur = 0.0

    # Convert trimesh objects to cuRobo meshes and log to Rerun
    object_meshes = {}
    for label, trimesh_obj in object_trimeshes.items():
        curobo_mesh = convert_trimesh_to_curobo_mesh(trimesh_obj, label)
        object_meshes[label] = curobo_mesh
        label_clean = label.replace(" ", "-")
        log_curobo_mesh_to_rerun(f"world/objects/{label_clean}", curobo_mesh.get_mesh(), static_transform=True)

        # Log the point cloud
        pcd = object_pcds[label]
        rr.log(f"obj_pcd/{label_clean}", rr.Points3D(positions=pcd.points, colors=pcd.colors))

        # Transform grasps to tcp frame
        grasp_dict = filtered_grasps[label]
        world_from_obj = np.eye(4)
        curobo_pose = np.array(curobo_mesh.pose)
        assert np.allclose(curobo_pose[3:], np.array([1.0, 0.0, 0.0, 0.0]))
        world_from_obj[:3, 3] = curobo_pose[:3]
        obj_from_world = np.linalg.inv(world_from_obj)

        world_from_grasp = grasp_dict["poses"] @ m2t2_to_tiptop_transform()
        obj_from_grasp = obj_from_world @ world_from_grasp
        filtered_grasps[label]["grasps_obj"] = tensor_args.to_device(obj_from_grasp)
        filtered_grasps[label]["confidences_pt"] = tensor_args.to_device(filtered_grasps[label]["confidences"])

        if len(world_from_grasp) == 0:
            continue

        # Visualize the resulting grasps
        viz_start = time.perf_counter()
        my_vertices_hom = vertices_hom.copy()

        # Convert to tiptop convention and select top grasps
        grasp_poses = world_from_grasp[:30]
        confidences = filtered_grasps[label]["confidences"][:30]
        transformed_verts = np.einsum("nij,mj->nmi", grasp_poses, my_vertices_hom)[..., :3]
        colors = get_heatmap(confidences)

        for grasp_idx, (verts, color) in enumerate(zip(transformed_verts, colors)):
            rr.log(
                f"grasps/{label}/{grasp_idx:04d}",
                rr.Mesh3D(
                    vertex_positions=verts, triangle_indices=faces, vertex_colors=np.tile(color, (len(verts), 1))
                ),
                static=True,
            )
        viz_grasp_dur += time.perf_counter() - viz_start

    _log.info(f"Visualizing grasps took: {viz_grasp_dur:.2f}s")
    return ProcessedScene(
        table_cuboid=table_cuboid,
        object_meshes=object_meshes,
        object_pcds=object_pcds,
        grasps=filtered_grasps,
    )


async def run_perception(
    session: aiohttp.ClientSession,
    observation: Observation,
    task_instruction: str,
    save_dir: Path,
    depth_estimator: DepthEstimator | None = None,
    gripper_mask: Bool[np.ndarray, "h w"] | None = None,
    include_workspace: bool = True,
    log_to_rerun: bool = True,
) -> tuple[TAMPEnvironment, list, ProcessedScene, list[dict]]:
    start_time = time.perf_counter()

    frame = observation.frame
    rgb = frame.rgb
    if log_to_rerun:
        rr.log("rgb", rr.Image(rgb))

    # Run depth+grasps and detection concurrently
    depth_results, detection_results = await asyncio.gather(
        predict_depth_and_grasps(
            session,
            frame,
            observation.world_from_cam,
            tiptop_cfg().perception.voxel_downsample_size,
            depth_estimator=depth_estimator,
            gripper_mask=gripper_mask,
            depth_frames=observation.depth_frames,
        ),
        detect_and_segment(rgb, task_instruction),
    )
    _log.info(f"Capturing observation and running perception APIs took {time.perf_counter() - start_time:.2f}s")

    # Save results (ProcessPoolExecutor for live mode, default thread pool for h5 mode)
    loop = asyncio.get_running_loop()
    save_future = loop.run_in_executor(
        _executor_pool,
        save_perception_outputs,
        rgb,
        frame.intrinsics,
        depth_results["depth_map"],
        depth_results["xyz_map"],
        depth_results["rgb_map"],
        detection_results["bboxes"],
        detection_results["masks"],
        save_dir,
        gripper_mask,
    )

    if log_to_rerun:
        rr.log(
            "pcd",
            rr.Points3D(
                positions=depth_results["xyz_map"].reshape(-1, 3), colors=depth_results["rgb_map"].reshape(-1, 3)
            ),
        )

    # Run scene geometry processing while saving
    proc_st = time.perf_counter()
    process_coroutine = asyncio.to_thread(
        process_scene_geometry,
        depth_results["xyz_map"],
        depth_results["rgb_map"],
        detection_results["masks"],
        detection_results["bboxes"],
        depth_results["grasps"],
    )
    processed_scene, save_result = await asyncio.gather(process_coroutine, save_future)

    if log_to_rerun:
        bbox_viz, masks_viz = save_result
        rr.log("bboxes", rr.Image(bbox_viz))
        rr.log("masks", rr.Image(masks_viz))

    # PATCH: dump scene_objects.json {label: {centroid, extents}} for /drop_above fallback in cortex_tamp_server
    try:
        import json as _json
        import numpy as _np
        _scene_objs = {}
        for _name, _m in processed_scene.object_meshes.items():
            if getattr(_m, "pose", None) is None or len(_m.pose) < 3:
                continue
            _centroid = [float(x) for x in _m.pose[:3]]
            _extents = None
            try:
                _v = _np.array(_m.vertices)
                if _v.size:
                    _extents = [
                        float(_v[:, 0].max() - _v[:, 0].min()),
                        float(_v[:, 1].max() - _v[:, 1].min()),
                        float(_v[:, 2].max() - _v[:, 2].min()),
                    ]
            except Exception:
                pass
            _scene_objs[_name] = {"centroid": _centroid, "extents": _extents}
        # PATCH 2026-06-02: also serialize M2T2 grasp candidates per object (top-K by
        # confidence) so cortex /pick_cached can pick a real rim/handle grasp without
        # re-running Gemini/SAM2/M2T2. processed_scene.grasps[label] has the raw M2T2
        # output; we transform to TCP frame (m2t2_to_tiptop_transform) so the saved
        # poses are world_from_TCP — directly usable by cuRobo IK in pick_cached.
        try:
            from tiptop.perception.m2t2 import m2t2_to_tiptop_transform as _m2t2_xf
            _xf = _m2t2_xf()
            _TOP_K = 30
            for _gname, _gdict in (processed_scene.grasps or {}).items():
                if _gname not in _scene_objs:
                    continue
                _poses = _gdict.get("poses") if isinstance(_gdict, dict) else None
                _confs = _gdict.get("confidences") if isinstance(_gdict, dict) else None
                if _poses is None or _confs is None or len(_poses) == 0:
                    _scene_objs[_gname]["grasps_world_from_tcp"] = []
                    _scene_objs[_gname]["grasp_confidences"] = []
                    continue
                _wfg = _np.asarray(_poses) @ _np.asarray(_xf)
                _confs = _np.asarray(_confs)
                _order = _np.argsort(-_confs)[:_TOP_K]
                _scene_objs[_gname]["grasps_world_from_tcp"] = _wfg[_order].tolist()
                _scene_objs[_gname]["grasp_confidences"] = _confs[_order].tolist()
        except Exception as _ge:
            _log.warning(f"PATCH grasps: failed to serialize M2T2 grasps: {_ge}")
        (save_dir / "scene_objects.json").write_text(_json.dumps(_scene_objs, indent=2))
        _log.info(f"PATCH: wrote scene_objects.json with {len(_scene_objs)} entries")
    except Exception as _e:
        _log.warning(f"PATCH: failed to dump scene_objects.json: {_e}")
    # PATCH: detect-only mode for cortex /perceive. scene_objects.json is already
    # written above; bail out before any motion planning / grasp execution.
    import os as _os_detect
    if _os_detect.environ.get("TIPTOP_DETECT_ONLY"):
        raise UserExitException("TIPTOP_DETECT_ONLY: perception complete; skipping planning/motion")

    env, all_surfaces = create_tamp_environment(
        processed_scene.object_meshes,
        processed_scene.table_cuboid,
        detection_results["grounded_atoms"],
        include_workspace,
    )
    _log.info(f"Processing scene and perception results took {time.perf_counter() - proc_st:.2f}s")
    _log.info(f"Perception pipeline completed, took {time.perf_counter() - start_time:.2f}s")
    return env, all_surfaces, processed_scene, detection_results["grounded_atoms"]


async def async_entrypoint(container: _DemoContainer, config: TAMPConfiguration, output_dir: str, execute_plan: bool):
    """Main async entrypoint for the live robot demo."""
    global _teleop_requested, _resumed_from_teleop
    cfg = tiptop_cfg()

    # Force TCP handshake for every request
    connector = aiohttp.TCPConnector(limit=10, force_close=True)
    async with aiohttp.ClientSession(connector=connector) as session:
        while True:
            try:
                _log.debug("Preparing TiPToP for next run...")
                await check_server_health(session)

                # Get the task BEFORE any pre-trial robot motion so that quitting (or an empty
                # prompt) ends the session without moving to capture + opening the gripper --
                # which would drop whatever is currently held. Reuses the warmed container.
                task_instruction = _get_task_instruction()  # Let UserExitException propagate
                # A robot nudge ('home'/'open') from the UI's top bar or the prompt: run it against
                # the warm container and go straight back to the prompt -- no rollout, no episode.
                if task_instruction in ROBOT_COMMANDS:
                    _run_robot_command(container, cfg, task_instruction)
                    continue
                _log.info(f"User entered instruction: {task_instruction}")

                # Reset to a clean starting state for the new episode: return the arm home
                # and open the gripper -- but only when they aren't already so. go_to_home
                # no-ops when the arm is already at q_home (go_to_q's distance check), and the
                # gripper open is skipped when the measured width already reads open. This
                # matters most right after a force-stop abort, where the arm may be left
                # mid-motion still gripping an object.
                #
                # ALL OF IT is skipped when this iteration continues a teleop hand-off, including the
                # move to the capture pose below: the point of the hand-off is to replan from exactly
                # the state the operator left behind, so the arm must not move at all before
                # perception. Homing would undo their positioning, opening the gripper would DROP
                # whatever they just picked up, and going to capture would move off their pose.
                # The trade-off is that perception then runs from wherever the arm happens to be --
                # if it occludes the scene, cuTAMP plans against a worse point cloud than usual.
                if _resumed_from_teleop:
                    _resumed_from_teleop = False
                    _log.info(
                        "Resuming after teleop: skipping the home + gripper-open reset and the move "
                        "to the capture pose, so the arm keeps the exact pose (and grasp) the "
                        "operator left it in. Perceiving from that pose"
                    )
                else:
                    _log.info("Resetting robot for new episode: return home + open gripper (if not already)")
                    go_to_home(time_dilation_factor=cfg.robot.time_dilation_factor, motion_gen=container.motion_gen)
                    try:
                        _open_gripper_if_needed(container)
                    except Exception as _e:
                        _log.exception('Gripper open/check failed: ' + str(_e))

                    _log.debug("Moving robot to capture joint positions")
                    go_to_capture(time_dilation_factor=cfg.robot.time_dilation_factor, motion_gen=container.motion_gen)

                now = datetime.now()
                timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
                iso_timestamp = now.isoformat(timespec="seconds")
                rr.init("tiptop_run", recording_id=timestamp, spawn=False)  # PATCH: no DISPLAY in headless subprocess
                # Log workspace for visualization purposes
                robot_rr = get_robot_rerun()
                for obj in workspace_cuboids():
                    log_curobo_mesh_to_rerun(f"world/workspace/{obj.name}", obj.get_mesh(), static_transform=True)

                save_dir = Path(output_dir) / "eval" / timestamp
                _log.info(f"Saving logs, results, and visualizations to {save_dir}")
                _emit_event({"event": "rollout_start", "dir": str(save_dir)})

                # Add log file handler for this run
                file_handler = add_file_handler(save_dir / "tiptop_run.log")
                # Record the resolved cuRobo override config INTO this rollout's log (the warmup-time
                # "RESOLVED cuRobo cost" line predates this handler, so it never lands on disk). Also
                # drop a curobo_config.json alongside it so applied overrides are auditable per episode.
                _resolved = container.curobo_config_summary or {}
                _log.info(f"cuRobo config for this rollout: {json.dumps(_resolved)}")
                (save_dir / "curobo_config.json").write_text(json.dumps(_resolved, indent=2))
                try:
                    # Capture robot state and compute camera pose
                    observation = capture_live_observation(container)
                    robot_rr.set_joint_positions(observation.q_init)

                    # Now we're ready! Start timing
                    _log.info("Running Perception...")
                    perception_start = time.perf_counter()
                    env, all_surfaces, processed_scene, grounded_atoms = await run_perception(
                        session,
                        observation,
                        task_instruction,
                        save_dir,
                        depth_estimator=container.depth_estimator,
                        gripper_mask=container.gripper_mask,
                    )
                    perception_duration = time.perf_counter() - perception_start

                    cutamp_plan = None
                    planning_duration = None
                    failure_reason = None
                    if os.environ.get("TIPTOP_DRY_RUN"):
                        _log.info("PATCH: TIPTOP_DRY_RUN=1 -> skipping planning/execute (perception-only)")
                        failure_reason = "dry_run"
                    else:
                        pass
                    try:
                        if os.environ.get("TIPTOP_DRY_RUN"):
                            raise RuntimeError("dry_run skip")
                        _log.info("Running Planning...")
                        cutamp_plan, planning_duration, failure_reason = run_planning(
                            env,
                            config,
                            q_init=observation.q_init,
                            ik_solver=container.ik_solver,
                            grasps=processed_scene.grasps,
                            motion_gen=container.motion_gen,
                            all_surfaces=all_surfaces,
                            experiment_dir=save_dir / "cutamp",
                            cost_overrides=container.cost_overrides,
                        )
                        _log.info(f"Perception and cuTAMP planning took: {perception_duration + planning_duration:.2f}s")
                        if cutamp_plan is not None:
                            plan_path = save_dir / "tiptop_plan.json"
                            trace_cfg = resolve_trace_cfg(container.cost_overrides)
                            save_tiptop_plan(
                                serialize_plan(cutamp_plan, observation.q_init, trace_cfg=trace_cfg), plan_path
                            )
                            _log.info(f"Saved TiPToP plan to {plan_path}")

                        if cutamp_plan is not None and execute_plan:
                            _log.info("Executing plan...")
                            # Execute with optional recording
                            if container.enable_recording:
                                # Convert SVO -> MP4 after execution. Depth is disabled during
                                # conversion (see convert_svo_to_mp4) so it won't OOM the GPU.
                                cameras_to_record = [
                                    (
                                        container.external_cam,
                                        save_dir / "external_cam.svo",
                                        save_dir / "external_cam.mp4",
                                    ),
                                ]
                                if container.external_cam_2 is not None:
                                    cameras_to_record.append(
                                        (
                                            container.external_cam_2,
                                            save_dir / "external_cam_2.svo",
                                            save_dir / "external_cam_2.mp4",
                                        ),
                                    )
                                if isinstance(container.cam, ZedCamera):
                                    cameras_to_record.append(
                                        (container.cam, save_dir / "hand_cam.svo", save_dir / "hand_cam.mp4"),
                                    )
                                # Sample the measured arm + gripper state over their own sockets while
                                # the cameras record and the plan executes; capture per-step wall-clock
                                # times so the export can align camera frames to the control timeline.
                                # The samplers are OUTER and record_cameras INNER so the cameras stop the
                                # instant execution returns: were the cameras outer, their exit would run
                                # while the ~2 s sampler-thread joins finished, padding the video tail
                                # with stationary frames past the last state frame.
                                exec_timeline: list[dict] = []
                                with (
                                    GripperSampler(container.robot) as gripper_sampler,
                                    JointSampler() as joint_sampler,
                                ):
                                    with record_cameras(cameras_to_record) as rec_window:
                                        execute_cutamp_plan(
                                            cutamp_plan, client=container.robot, timeline=exec_timeline
                                        )
                                # Save the raw measured gripper trace (wall_seconds, width_m) so the
                                # open<->close shape can be inspected directly (snap vs ramp).
                                try:
                                    (save_dir / "_gripper_trace.json").write_text(
                                        json.dumps({"width_samples": gripper_sampler.width_samples})
                                    )
                                except Exception:
                                    _log.exception("Failed to write gripper trace")
                                # mp4s are written on record_cameras exit; map them to DROID image keys.
                                lerobot_cameras = {"observation.images.exterior_1_left": "external_cam.mp4"}
                                if container.external_cam_2 is not None:
                                    lerobot_cameras["observation.images.exterior_2_left"] = "external_cam_2.mp4"
                                if isinstance(container.cam, ZedCamera):
                                    lerobot_cameras["observation.images.wrist_left"] = "hand_cam.mp4"
                                # Data-collection raw episode (robot_state.npz + _meta.json, ARCHITECTURE §3):
                                # MEASURED proprioception from the samplers + COMMANDED plan actions, decoupled.
                                n_frames = 0
                                try:
                                    raw_path = dump_raw_episode(
                                        save_dir,
                                        plan_path,
                                        timeline=exec_timeline,
                                        joint_samples=joint_sampler.samples,
                                        gripper_samples=gripper_sampler.samples,
                                        instruction=task_instruction,
                                        cameras=lerobot_cameras,
                                        fps=LEROBOT_FPS,
                                        config_id=os.environ.get("TIPTOP_CONFIG_ID"),
                                        record_start=rec_window.get("t_start"),
                                        record_stop=rec_window.get("t_stop"),
                                    )
                                    if raw_path is not None:
                                        n_frames = json.loads((save_dir / "_meta.json").read_text()).get("n_frames", 0)
                                except Exception:
                                    _log.exception("Failed to dump raw episode")
                                _emit_event({"event": "rollout_saved", "dir": str(save_dir), "n_frames": n_frames})
                            else:
                                execute_cutamp_plan(cutamp_plan, client=container.robot)
                            _log.info("Finished executing plan!")
                        elif cutamp_plan is not None:
                            _log.info("Skipping cuTAMP plan execution on real robot")
                        else:
                            _log.warning(f"No plan found: {failure_reason}")

                        _log.debug(f"Finished run for instruction: {task_instruction}")
                    finally:
                        # Always save env, grasps, metadata, and artifacts regardless of success
                        save_run_outputs(save_dir, env, processed_scene.grasps)
                        save_run_metadata(
                            save_dir=save_dir,
                            timestamp=iso_timestamp,
                            task_instruction=task_instruction,
                            q_at_capture=observation.q_init,
                            world_from_cam=observation.world_from_cam,
                            perception_duration=perception_duration,
                            grounded_atoms=grounded_atoms,
                            planning_success=cutamp_plan is not None,
                            planning_failure_reason=failure_reason,
                            planning_duration=planning_duration,
                        )
                        _log.info(f"Logs, results, and visualizations saved to {save_dir}")

                    if execute_plan:
                        final_dir = _label_rollout(save_dir, output_dir, timestamp)
                        # Post-process this rollout (gifs + LeRobot export) in the background so
                        # the next rollout can start immediately instead of blocking on it.
                        _spawn_postprocess(final_dir)
                        # PATCH (cortex v3): DO NOT auto-open the gripper after Pick.
                        # The original tiptop demo opened the gripper post-pick for
                        # standalone "did the grasp work?" tests. For cortex we WANT
                        # to keep the object held so Haiku can decide whether to Place
                        # next. Removing the open_gripper() call here.
                except Exception:
                    _log.exception("TiPToP run failed")
                    raise
                finally:
                    # Always remove the file handler after the run
                    remove_file_handler(file_handler)
            except UserExitException:
                _log.info("User requested exit")
                break
            except KeyboardInterrupt:
                # Preempt from the data-collection UI (SIGINT), or a terminal Ctrl-C. Treat it
                # as "abort THIS rollout" rather than "end the session": unwind the in-flight
                # rollout (its finally-blocks have already run during propagation) and loop back
                # to the task prompt so another episode can be collected without a full re-warm.
                # The graceful stop path ("q\n" -> UserExitException) is what ends the session.
                #
                # NOTE: this stops us sending any further plan steps, but it cannot stop a
                # trajectory the controller is already executing -- bamboo hands the whole segment
                # over in one execute_trajectory request and has no abort command, so the arm runs
                # to the end of the current segment regardless. The hardware E-stop is the only
                # instant stop. See the Preempt copy in the data-collection UI.
                _log.info(
                    "Rollout aborted (Ctrl-C / preempt); no further plan steps will be sent. "
                    "Keeping session warm, returning to task prompt"
                )
                _emit_event({"event": "rollout_aborted"})
                # Unwind is done (the finally-blocks above ran as the exception propagated), so a
                # new Ctrl-C should preempt the next rollout rather than be swallowed.
                _clear_preempt()
                if _teleop_requested:
                    _teleop_requested = False
                    _run_teleop_handoff(container)
                continue
            except Exception as e:
                # A single rollout failing (a transient Gemini/perception 503, a planning
                # error, a health-check blip, ...) must NOT tear down the warmed session --
                # otherwise "collect another" would lose the whole warmed container and force
                # a full re-warm. Log it (the traceback streams to the data-collection UI),
                # then loop back to the task prompt so the user can just retry.
                _log.exception(f"Rollout failed ({type(e).__name__}: {e}); keeping session warm, returning to task prompt")
                continue


def _sync_entrypoint(
    output_dir: str = "tiptop_outputs",
    max_planning_time: float = 60.0,
    opt_steps_per_skeleton: int = 500,
    execute_plan: bool = True,
    cutamp_visualize: bool = False,
    num_particles: int = 256,
    enable_recording: bool = False,
    curobo_overrides: str | None = None,
):
    """
    TiPToP live robot runner. Runs continuously on the real robot.

    Args:
        output_dir: Top-level directory to save outputs to; a timestamped subdirectory is created per run.
        max_planning_time: Maximum time to spend planning with cuTAMP across all skeletons (approximate).
        opt_steps_per_skeleton: Number of optimization steps per skeleton in cuTAMP.
        execute_plan: Whether to execute the plan on the real robot.
        cutamp_visualize: Whether to visualize cuTAMP optimization.
        num_particles: Number of particles for cuTAMP; decrease if running out of GPU memory.
        enable_recording: Whether to record external camera video during execution.
        curobo_overrides: cuRobo cost overrides as a JSON file path OR inline JSON (the cfg/tamp/*.yml
            cost knobs, e.g. vae_manifold_weight); applied at solver build time so every plan uses them.
    """
    assert max_planning_time > 0
    assert opt_steps_per_skeleton > 0
    assert num_particles > 0

    print_tiptop_banner()
    check_cutamp_version()
    # The pid is how the data-collection server reaches THIS process directly. It launches us behind
    # a `pixi run` wrapper, so the wrapper -- not us -- is its direct child; signalling that child
    # would never arrive here, and signalling the whole process group would take out the wrapper and
    # the save-pool workers too. Publishing the real pid lets it target just us (see
    # Session.switchToTeleop, which sends SIGUSR1 for the teleop hand-off).
    _emit_event({"event": "session_start", "pid": os.getpid()})

    # Lazy import breaks the tiptop_run <-> tiptop_websocket_server import cycle.
    from tiptop.tiptop_websocket_server import _load_curobo_overrides

    cost_overrides = _load_curobo_overrides(curobo_overrides)
    # num_particles / opt_steps_per_skeleton may be set from the cfg/tamp yml (tamp_overrides) so a
    # data-gen config controls solver effort without CLI flags; an override wins over the CLI default.
    # (These key names are also echoed by summarize_curobo_config.)
    if cost_overrides.get("num_particles") is not None:
        num_particles = int(cost_overrides["num_particles"])
    if cost_overrides.get("opt_steps_per_skeleton") is not None:
        opt_steps_per_skeleton = int(cost_overrides["opt_steps_per_skeleton"])
    if num_particles <= 0 or opt_steps_per_skeleton <= 0:
        raise ValueError(
            f"num_particles and opt_steps_per_skeleton must be positive, got "
            f"{num_particles=}, {opt_steps_per_skeleton=}"
        )
    _log.info(f"Solver effort: num_particles={num_particles}, opt_steps_per_skeleton={opt_steps_per_skeleton}")
    cfg = tiptop_cfg()
    # time_dilation_factor[_literal] is a plan-time knob (not a cuRobo cost weight), so it is NOT
    # handled by build_curobo_solvers/apply_cost_overrides — resolve it here and thread it into the
    # TAMP config, mirroring tiptop_websocket_server. Without this, cfg/tamp/{tdf,vae_tdf}.yml's
    # time_dilation_factor_literal would be silently dropped.
    time_dilation_factor = resolve_time_dilation_factor(cost_overrides, cfg.robot.time_dilation_factor)
    # Resolved cost/tamp-param config the solvers get built with; stashed on the container so each
    # rollout can record it (async_entrypoint), making override application auditable per episode.
    curobo_config_summary = summarize_curobo_config(cost_overrides, time_dilation_factor)
    if cost_overrides:
        _log.info(f"cuRobo cost overrides active: {cost_overrides}")
        _log.info(f"Resolved time_dilation_factor={time_dilation_factor}")

    config = build_tamp_config(
        num_particles=num_particles,
        max_planning_time=max_planning_time,
        opt_steps=opt_steps_per_skeleton,
        robot_type=cfg.robot.type,
        time_dilation_factor=time_dilation_factor,
        collision_activation_distance=0.0,
        enable_visualizer=cutamp_visualize,
        # move-cost norm for cuTAMP (Euclidean unless a cfg/tamp yml opts into "inf"), same as
        # time_dilation_factor this is a TAMP-config knob, not a cuRobo cost weight.
        traj_length_norm=resolve_traj_length_norm(cost_overrides),
        grasp_orientation_cost=resolve_grasp_orientation_cost(cost_overrides),
    )

    global _executor_pool
    setup_logging(level=logging.DEBUG)

    container = get_demo_container(
        num_particles, config.coll_n_spheres, 0.0, enable_recording, cost_overrides, curobo_config_summary
    )
    # Workers fork from a process that has already initialised CUDA (curobo + the ZED cameras), so
    # they inherit its CUDA context. That costs no extra VRAM while we are alive, but the driver
    # cannot reclaim the context until every process holding it exits -- so a worker that outlives a
    # force-killed run pins ~3.8GB of VRAM until reboot, and the next run OOMs (including inside
    # zed.open(), which needs GPU memory to decode an SVO). _init_pool_worker's death signal is what
    # guarantees they never outlive us. Do NOT switch this to forkserver/spawn to dodge the
    # inheritance: those re-import this module in each worker, and importing it initialises CUDA,
    # giving every worker its own ~600MB context -- strictly worse.
    _executor_pool = ProcessPoolExecutor(max_workers=4, initializer=_init_pool_worker)

    # SIGINT preempts the current rollout instead of ending the session -- and stays safe when it is
    # pressed repeatedly, which is exactly what a user does when the arm keeps moving through the
    # tail of its current trajectory segment. Installed after the pool so its workers (which set
    # SIG_IGN in their own initializer) are unaffected.
    signal.signal(signal.SIGINT, _sigint_preempt)
    # SIGUSR1 is the "switch to teleop" trigger from the data-collection UI's button: same in-flight
    # abort as SIGINT, plus _teleop_requested routes the KeyboardInterrupt handler into the teleop
    # hand-off (release the robot, wait for it back, auto-replan the same task). See
    # _sigusr1_teleop_switch / _run_teleop_handoff.
    signal.signal(signal.SIGUSR1, _sigusr1_teleop_switch)

    exit_code = 1
    try:
        asyncio.run(async_entrypoint(container, config, output_dir, execute_plan))
        exit_code = 0
    except (UserExitException, KeyboardInterrupt) as e:
        if isinstance(e, KeyboardInterrupt):
            _log.info("Interrupted during startup/shutdown (Ctrl+C)")
        else:
            _log.debug("Exit detected")
        exit_code = 0
    finally:
        if container is not None:
            _log.debug("Tearing down cameras and robot...")
            # Every handle is optional here: a teleop hand-off releases the cameras and the robot
            # client for the duration of the teleop session, so ending the session in that window
            # (Stop, or the driver dying while parked) finds them already gone.
            for attr in ("cam", "external_cam", "external_cam_2", "robot"):
                handle = getattr(container, attr, None)
                if handle is None:
                    continue
                try:
                    handle.close()
                except Exception:
                    _log.exception(f"Failed to close {attr} during teardown")
        if _executor_pool is not None:
            # Reap the workers rather than just detaching from them: cancel what has not started,
            # then give a save in flight a moment to finish before terminating the stragglers.
            # shutdown() drops the executor's handles on its workers, so grab them first. The 5s is a
            # budget shared across all of them, not per worker, so a pool of stragglers cannot add
            # 5s each to shutdown.
            workers = list((getattr(_executor_pool, "_processes", None) or {}).values())
            _executor_pool.shutdown(wait=False, cancel_futures=True)
            deadline = time.monotonic() + 5.0
            for proc in workers:
                proc.join(timeout=max(0.0, deadline - time.monotonic()))
                if proc.is_alive():
                    _log.warning(f"Save worker {proc.pid} still alive after shutdown; terminating")
                    proc.terminate()
        # Wait for any background per-rollout post-processing (gifs + LeRobot export) to finish
        # so the session doesn't exit mid-export. Ctrl-C here leaves them running detached.
        pending = [p for p in _postprocess_procs if p.poll() is None]
        if pending:
            _log.info(f"Waiting for {len(pending)} background post-processing job(s) to finish...")
            try:
                for p in pending:
                    p.wait()
            except KeyboardInterrupt:
                _log.info("Leaving post-processing running in the background; exiting now.")
        _emit_event({"event": "session_end"})
        sys.exit(exit_code)


def entrypoint():
    tyro.cli(_sync_entrypoint)


if __name__ == "__main__":
    entrypoint()

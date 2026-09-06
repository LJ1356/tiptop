import asyncio
import ctypes
import json
import logging
import os
import shutil
import signal
import subprocess
import sys
import time
import uuid
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Awaitable, Callable, Sequence

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

from tiptop.config import as_robot_type as _as_robot_type
from tiptop.config import load_calibration, tiptop_cfg
from tiptop.execute_plan import execute_cutamp_dual_plan, execute_cutamp_plan
from tiptop.goal_clearing import (
    build_clearing_goal,
    drop_return_to_initial,
    final_configuration,
    move_meshes,
    placed_poses,
    resolve_clear_goal_surfaces,
)
from tiptop.lerobot_capture import (
    GRIPPER_MAX_WIDTH,
    GripperSampler,
    JointSampler,
    _read_gripper_width,
    dump_raw_episode,
)
from tiptop.motion_planning import (
    build_curobo_solvers,
    go_to_capture,
    go_to_dual_home,
    go_to_home,
    apply_perception_overrides,
    resolve_grasp_center_cost,
    resolve_grasp_orientation_cost,
    resolve_max_motion_refine_attempts,
    resolve_posture_selection,
    resolve_require_m2t2_grasps,
    resolve_time_dilation_factor,
    resolve_trace_cfg,
    resolve_traj_length_norm,
    resolve_transit_apex,
    summarize_curobo_config,
)
from cutamp.posture_prior import posture_ref_summary
from tiptop.perception.cameras import (
    Camera,
    DepthEstimator,
    Frame,
    ZedCamera,
    camera_mount,
    get_depth_estimator,
    get_external_camera,
    get_external_camera_2,
    get_hand_camera,
)
from tiptop.perception.m2t2 import augment_flipped_grasps, m2t2_to_tiptop_transform
from tiptop.perception.sam2 import sam2_client
from tiptop.perception.segmentation import TABLE_BOX_CLEARANCE, segment_pointcloud_by_masks, segment_table_with_ransac
from tiptop.perception.utils import (
    convert_trimesh_box_to_curobo_cuboid,
    convert_trimesh_to_curobo_mesh,
    project_spheres_to_mask,
)
from tiptop.perception_wrapper import detect_and_segment, predict_depth_and_grasps
from tiptop.planning import build_tamp_config, run_planning, save_tiptop_plan, serialize_plan
from tiptop.recording import (
    record_cameras,
    save_perception_outputs,
    save_run_metadata,
    save_run_outputs,
)
from tiptop.scene_reset import build_reset_goal, reset_goal_builder
from tiptop.utils import (
    NumpyEncoder,
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
from tiptop.viz_utils import get_gripper_mesh, get_heatmap
from tiptop.workspace import workspace_cuboids
from tiptop.yam import IDLE_ARM_TOLERANCE_RAD, NEUTRAL_Q, active_arm, arm_of, is_yam
from tiptop.yam.capture import (
    BimanualJointSampler,
    dump_bimanual_episode,
    segment_from_motion,
    segments_from_dual_plan,
    segments_from_plan,
)
from tiptop.yam.task_split import arm_goal_builder, handover_goal_builder
from tiptop.yam.yam_client import GRIPPER_STROKE_M as YAM_GRIPPER_STROKE_M

_log = logging.getLogger(__name__)
tensor_args = TensorDeviceType()

# Sampling rate for the LeRobot DROID-format capture during plan execution (matches DROID).
LEROBOT_FPS = 15

# A measured gripper width (metres) at or above this counts as "already open", so the
# per-episode reset skips re-issuing an open. 90% of the Robotiq 2F-85 full span.
GRIPPER_OPEN_WIDTH = 0.9 * GRIPPER_MAX_WIDTH

# How far to raise the table collision box when a goal places ONTO the table (create_tamp_environment).
# The full clearance puts its top exactly on the detected plane. Lower it (e.g. 0.015) if a scene
# reset starts coming back "no reset plan": the last few mm of the sink are what let the fingertips
# get under a flat object, and landing placements a little above the plane still avoids the press.
TABLE_PLACEMENT_RAISE = TABLE_BOX_CLEARANCE

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


# Set by the "switch to teleop" trigger (SIGUSR1). Unlike a preempt this is COOPERATIVE: nothing is
# aborted. The flag is polled at the safe boundaries of a rollout (between plan steps, and again
# before execution starts) and the hand-off runs there, so the arm is left at the end of a plan step
# rather than mid-task. Cleared by whoever runs the hand-off.
_teleop_requested = False

# Set by _run_teleop_handoff once teleop hands the robot back; consumed by the NEXT call to
# _get_task_instruction() so the loop immediately replans/executes the SAME task from the human's
# hand-off pose instead of blocking on a typed task.
_pending_instruction: str | None = None

# Set alongside _pending_instruction: the next rollout must NOT run its usual return-home + open-
# gripper reset. Homing would undo the hand-off (the whole point is to replan from where the human
# left the arm) and opening the gripper would drop whatever the operator is holding.
_skip_episode_reset = False

# The narrower half of that skip, for a leg that follows a HUMAN-IN-THE-LOOP phase rather than an
# operator-initiated hand-off: suppress the arm MOTION (return home, move to the capture pose) while
# still running the ordinary gripper-open check. A HITL human phase leaves the arm wherever the
# person put it and the robot holding nothing, so homing is pure wasted motion -- the next phase is a
# continuation of the same trajectory and should plan from the pose the human left, not from q_home.
# The gripper check is kept because nothing during a human phase guarantees the fingers are open, and
# cuTAMP plans every phase from a HandEmpty initial state regardless (see planning.environment_initial_state).
#
# The cost, accepted deliberately: with `cameras.perception: external` the arm no longer clears the
# third-person camera's view, so it can land in the point cloud, the RANSAC table fit and the grasps.
# That is the same bargain the operator-initiated hand-off has always made (_skip_episode_reset).
_skip_episode_arm_motion = False

# Set alongside it, but a SEPARATE thing: the next rollout is another leg of the SAME trajectory, so
# it keeps _trajectory_id and its episode merges with the legs before it. The two used to be one flag,
# which was wrong as soon as a leg wanted the arm reset without starting a new trajectory -- a
# human-in-the-loop phase does exactly that (the human opened a box; the robot holds nothing, and the
# next phase needs the capture pose for perception). Clearing the reset flag then silently minted a
# new trajectory id, which both lost the HITL plan mid-task and split the episode's legs apart.
_continue_trajectory = False

# The task plan (cuTAMP plan skeleton) behind THIS rollout's motion plan, and the one a rollout
# resuming from a hand-off should reuse instead of searching for its own.
#
# Only the symbolic search is skipped: the resumed rollout re-perceives the scene and re-solves the
# grasps, placements and trajectories, which it must, since the human moved things. What it does not
# do is reconsider WHICH task plan to follow -- the operator handed the arm back mid-attempt at a
# plan that was already chosen, so the arm carries on with that one rather than possibly switching
# to a different skeleton for the same goal. Rejected automatically (and a full search run instead)
# if the plan no longer applies to the newly perceived scene -- see planning.skeleton_reuse_rejection.
_last_plan_skeleton = None
_reuse_plan_skeleton = None

# Hand-off lineage. One task attempt can span many legs -- tamp, teleop, tamp, ... -- and they are
# ONE trajectory, not N episodes. Every leg is stamped with this id; collect/merge_trajectory.py
# joins them into a single episode afterwards. A fresh rollout mints a new id; a rollout resuming
# from a hand-off keeps the current one (this process lives across the whole hand-off, so a module
# global is all the continuity that is needed). Segment ORDER is never derived from a counter --
# the merge sorts legs by their camera record_start, which needs no agreement between processes.
_trajectory_id: str | None = None

# True once the CURRENT trajectory is known to span several legs. Such a trajectory is post-processed
# by collect/merge_trajectory.py after the legs are joined; exporting the final leg on its own here
# would export a fragment of the episode.
#
# Set from TWO places, despite the name: _run_teleop_handoff, and the robot->robot continuation at
# the end of the rollout loop -- a human-in-the-loop plan whose next phase is another ROBOT phase
# spans several legs without teleop ever being involved. Grepping for _run_teleop_handoff alone will
# tell you this cannot be true on a robot-only trajectory, and that is wrong.
_trajectory_handed_off = False

# True only while the driver is blocked in input() at a stdin prompt. There, no rollout checkpoint
# will ever be reached, so SIGUSR1 has to raise out of the prompt instead of setting a flag; see
# _sigusr1_teleop_switch. Each prompt site clears it as its first statement after input() returns,
# keeping the window where a raise could land in unrelated bookkeeping down to a few bytecodes.
_at_prompt = False


class TeleopHandoffRequested(Exception):
    """SIGUSR1 arrived while the driver sat at a stdin prompt: hand the arm off from there."""


def _sigusr1_teleop_switch(_signum, _frame) -> None:
    """"Switch to teleop" trigger (SIGUSR1) from the data-collection UI's button.

    Deliberately NOT a preempt: it does not abort anything. The current plan step runs to completion
    (execute_cutamp_plan polls _teleop_requested between steps), this rollout's partial episode is
    saved unlabeled, and only then does the hand-off run -- so the arm ends up parked at a plan
    boundary instead of wherever an abort happened to catch it.

    At a stdin prompt there is no such checkpoint to reach, so raise out of input() instead.
    """
    global _teleop_requested
    _teleop_requested = True
    _emit_event({"event": "teleop_switch_pending"})
    if _at_prompt:
        raise TeleopHandoffRequested


def _consume_teleop_request() -> bool:
    """True (once) if a teleop hand-off has been requested; clears the flag so it can be re-armed."""
    global _teleop_requested
    if not _teleop_requested:
        return False
    _teleop_requested = False
    return True


# Margin between releasing a ZED and telling another process it may open it. ZedCamera.close()
# already blocks for the SDK's teardown (~14s for two cameras, measured) and the device is claimable
# about a second later, so this is slack rather than a readiness check.
CAMERA_RELEASE_SETTLE_S = 2.0


def _release_cameras(container: "_DemoContainer") -> list[str]:
    """Close every ZED this process holds so the teleop driver can open them. Returns their serials.

    A ZED is exclusive to one process: DROID's StableRobotEnv opens the same serials we do (it
    discovers them from the SDK's device list), so it cannot start while our handles are alive.
    Releasing only the robot connection is not enough.

    Closing our handles is only half of it: the forked save workers hold the same devices, so the
    caller must also _shutdown_save_pool() before anything else can open them.
    """
    serials: list[str] = []
    for attr in ("cam", "external_cam", "external_cam_2"):
        cam = getattr(container, attr, None)
        if cam is None:
            continue
        serial = str(getattr(cam, "serial", "") or "")
        try:
            cam.close()
            _log.info(f"Released camera {attr} (s/n {serial or '?'}) for teleop")
            if serial:
                serials.append(serial)
        except Exception:
            _log.exception(f"Failed to close camera {attr}; teleop may not be able to open it")
        object.__setattr__(container, attr, None)
    return serials


def _shutdown_save_pool() -> None:
    """Reap the save workers so they release the camera fds they inherited from us.

    The pool is forked AFTER the cameras and CUDA are up (see _sync_entrypoint, which explains why
    that ordering is deliberate), so every worker inherits the parent's CUDA context -- and, with
    it, the parent's open /dev/video handles. Closing our own ZedCamera objects therefore does NOT
    free the devices: four workers still hold them, the SDK reports the cameras as serial 0 /
    NOT AVAILABLE, and the teleop driver's open fails with CAMERA NOT DETECTED. Killing the workers
    is what actually hands the cameras over; _restart_save_pool re-forks them afterwards.

    Deliberately does not block on a save in flight: the operator is waiting for the arm, and the
    worst case is losing one rollout's perception debug images, never episode data.
    """
    global _executor_pool
    if _executor_pool is None:
        return
    _log.info("Stopping the save workers: they hold the cameras until they exit")
    # shutdown() drops the executor's handles on its workers, so grab them first. The budget is
    # shared across all of them, not per worker.
    workers = list((getattr(_executor_pool, "_processes", None) or {}).values())
    _executor_pool.shutdown(wait=False)
    _executor_pool = None
    deadline = time.monotonic() + 5.0
    for proc in workers:
        proc.join(timeout=max(0.0, deadline - time.monotonic()))
        if proc.is_alive():
            _log.warning(f"Save worker {proc.pid} still holding the cameras after shutdown; terminating")
            proc.terminate()
            proc.join(timeout=2.0)


def _restart_save_pool() -> None:
    """Re-fork the save pool once we have the cameras and the robot back.

    Same invariant as _sync_entrypoint: fork from a process whose CUDA context and cameras are
    already up, so the workers share the parent's context instead of each building their own. They
    inherit the new camera handles too -- which is exactly why the next hand-off reaps them again.
    """
    global _executor_pool
    if _executor_pool is None:
        _executor_pool = ProcessPoolExecutor(max_workers=4, initializer=_init_pool_worker)


def _reacquire_cameras(container: "_DemoContainer", *, had_external_cam_2: bool) -> None:
    """Re-open the cameras released by _release_cameras, once teleop has handed them back.

    container.depth_estimator is left alone on purpose: get_depth_estimator closes over the
    intrinsics VALUE, not the camera object, and the same serial at the same resolution reports the
    same intrinsics, so it stays valid across the swap.
    """
    _log.info("Re-opening the cameras teleop was using")
    object.__setattr__(container, "cam", get_hand_camera())
    object.__setattr__(container, "external_cam", get_external_camera())
    # get_external_camera_2 returns None both when it isn't configured and when it fails to open, so
    # a camera that was recording before the hand-off and is None now would otherwise just vanish
    # from the next episode's videos.
    external_cam_2 = get_external_camera_2()
    if external_cam_2 is None and had_external_cam_2:
        _log.warning(
            "The second external camera did not come back after the teleop hand-off; the next "
            "rollouts will record without it"
        )
    object.__setattr__(container, "external_cam_2", external_cam_2)


def _run_teleop_handoff(
    container: "_DemoContainer", can_finish: bool = False, plan_next: str | None = None
) -> bool:
    """Hand the physical arm off to a human teleop session, then block until it's handed back.

    Returns True when the operator ended the trajectory from teleop ("resume_finish") rather than
    handing it back to be replanned: their demonstration finished the task, so the caller closes the
    trajectory out instead of queueing another rollout. Returns False for a plain "resume", which is
    the unchanged behaviour -- replan the same task from the hand-off pose.

    `can_finish` says whether the caller HAS a partial rollout to close out. Only the mid-rollout
    hand-off does; the ones taken at a prompt or after a preempt have no leg to label and no
    trajectory to end, so they advertise `can_finish: false` and downgrade a "resume_finish" to a
    plain resume rather than stranding the operator.

    `plan_next` says what a PLAIN "resume" will do, when something already knows -- "robot" (a phase
    follows this one, so the arm carries on) or "end" (this was the plan's last phase, so handing
    back closes the trajectory out and lands at the label prompt). A HITL human phase always knows;
    a hand-off the operator took mid-rollout does not, and passes None. It is advisory, for the UI's
    single "next" control: nothing here branches on it, because a plain "resume" already does the
    right thing in both cases -- _hitl_human_phase advances the plan and labels when it is finished.

    Called at a rollout checkpoint (see _sigusr1_teleop_switch), so the current plan step has already
    finished and this rollout's partial episode is already on disk. From here:
      1. release this process's RobotClient connection so a separate teleop process (DROID's
         StableRobotEnv) can take over the arm -- they talk to the same NUC-side polymetis server and
         cannot hold it at once (see data-collection/ARCHITECTURE.md).
      2. confirm the arm has ACTUALLY stopped moving (wait_for_robot_stationary) before anyone is
         told it's safe to touch the shim -- releasing our connection does not mean the shim is
         done: it still has to finish the in-flight trajectory segment and hand control back to
         polymetis's default hold controller, which takes real time. Skipping this check would let
         an operator kill the shim (or start teleop) while the arm is still moving.
      3. release the ZED cameras too (_release_cameras) -- teleop opens the same serials. This runs
         AFTER the stationary check so a "could not confirm the arm stopped" warning reaches the
         operator immediately, rather than behind several seconds of camera teardown.
      4. emit "awaiting_teleop_resume" so the data-collection server knows it's safe to start the
         teleop session, then block on stdin for "resume" (written once the operator hands control
         back and the teleop process has exited, releasing the arm and the cameras).
      5. re-open the cameras, reconnect the RobotClient, and queue the SAME task instruction with
         _skip_episode_reset so the next loop iteration replans from the hand-off pose without
         moving the arm first -- no return to home, no gripper open, no move to the capture pose.
         The task plan this rollout was following is queued with it (_reuse_plan_skeleton), so the
         resumed rollout re-solves the motion for the SAME plan rather than searching for another.
         Skipped entirely when the operator chose to finish: nothing is queued, and the caller
         labels the partial leg instead.
    """
    global _pending_instruction, _skip_episode_reset, _trajectory_handed_off, _reuse_plan_skeleton
    global _continue_trajectory
    _log.info("Teleop switch: releasing the robot connection for hand-off")
    # This trajectory now spans several legs, so its export waits for the merge (see _spawn_postprocess).
    _trajectory_handed_off = True
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
                    "Could not confirm the arm has stopped moving. Visually confirm it is "
                    "stationary before touching the shim or starting teleop."
                ),
            }
        )

    had_external_cam_2 = container.external_cam_2 is not None
    released_any = _release_cameras(container)
    # Our own handles are only half of it -- the forked save workers hold the same devices.
    _shutdown_save_pool()
    if released_any:
        # close() blocks until the SDK has torn the camera down; with our handles gone AND the
        # save workers reaped, another process can claim the device about a second later
        # (measured). Leave a little slack and let the opener, which retries, confirm the rest.
        time.sleep(CAMERA_RELEASE_SETTLE_S)

    # trajectory_id rides along so the server can stamp the teleop leg it is about to spawn with the
    # same id, making the human's demonstration a segment of this trajectory rather than its own episode.
    # can_finish tells the server whether to offer "hand back and finish" alongside the plain
    # hand-back, so the UI never shows a button this hand-off cannot honour. plan_next goes further:
    # where the plan already settles what comes after the hand-back, the UI shows ONE button and says
    # what it will do, instead of making the operator re-decide something the plan has decided.
    _emit_event({
        "event": "awaiting_teleop_resume",
        "trajectory_id": _trajectory_id,
        "can_finish": can_finish,
        "plan_next": plan_next,
    })
    _log.info(
        "Robot and cameras released; waiting for 'resume' on stdin (sent once the operator hands "
        "control back and the teleop process has exited) before reconnecting and replanning..."
    )
    finish = False
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
            if cmd.lower() == "resume_finish":
                # Downgraded rather than refused: the arm and cameras have already been released and
                # the operator is waiting, so an unhonourable finish must still hand control back.
                finish = can_finish
                if not finish:
                    _log.warning(
                        "'resume_finish' received for a hand-off with no rollout to close out "
                        "(taken at a prompt, or after a preempt); resuming normally instead"
                    )
                break
            _log.warning(f"Ignoring unexpected input while awaiting teleop resume: {raw!r}")
    except EOFError:
        raise UserExitException("EOF while awaiting teleop resume")

    _log.info("Resuming: taking the robot and cameras back")
    # Any switch-to-teleop pressed WHILE we were handed off is already satisfied by this hand-off;
    # leaving it armed would bounce the resumed rollout straight back out at its first step.
    _consume_teleop_request()
    # Same slack in the other direction: teleop's capture processes have just died (possibly by
    # SIGKILL, without closing their cameras, so the SDK teardown never ran). ZedCamera retries on
    # its own, but each retry costs a USB reboot -- a moment here is cheaper than a failed attempt.
    time.sleep(CAMERA_RELEASE_SETTLE_S)
    _reacquire_cameras(container, had_external_cam_2=had_external_cam_2)
    object.__setattr__(container, "robot", reconnect_robot_client())
    _restart_save_pool()
    _emit_event({"event": "teleop_handoff_done"})

    if finish:
        # The operator's demonstration finished the task, so there is nothing to replan. Queue
        # nothing: the caller labels the partial leg, and that `labeled` event is what ends the
        # trajectory and has the server merge the tamp and teleop legs into one episode.
        _log.info("Teleop finished the task: closing the trajectory out instead of replanning")
        return True

    if _LAST_TASK:
        _pending_instruction = _LAST_TASK
        _skip_episode_reset = True
        _continue_trajectory = True
        # Whatever task plan was last found, the resumed rollout carries on with. Handing off at the
        # prompt or between rollouts reuses the PREVIOUS rollout's plan, which is the right one:
        # _pending_instruction is that same task. None only when no plan has been found to reuse --
        # a hand-off during this rollout's perception/planning (cleared at the top of every rollout),
        # or before any rollout has planned -- and then the resumed rollout plans the task normally.
        _reuse_plan_skeleton = _last_plan_skeleton
        if _reuse_plan_skeleton is not None:
            _log.info(f"Resumed rollout will reuse this task plan: {[op.name for op in _reuse_plan_skeleton]}")
    return False


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
    # Image-space mask of the robot's own geometry, dropped from the point cloud: the static
    # gripper mask in the wrist camera's view, the projected collision spheres in a third-person
    # camera's (where the arm is in frame from wherever it happens to be standing).
    robot_mask: Bool[np.ndarray, "h w"] | None = None


@dataclass(frozen=True)
class _DemoContainer:
    """Container for storing things needed for the live robot demo."""

    robot: RobotClient
    cam: Camera
    external_cam: Camera | None
    external_cam_2: Camera | None
    enable_recording: bool

    # Which camera slot perception reads: "hand" or "external" (cameras.perception). Stored as a key
    # rather than a handle for two reasons: the recording roles stay untouched — `cam` is always the
    # hand slot and `external_cam` always exterior_1, whichever of them TAMP happens to perceive
    # through — and the teleop hand-off closes and re-opens every camera, so a cached handle would go
    # stale. Resolve it with perception_camera().
    perception_cam_key: str
    # Calibration entry of the PERCEPTION camera, read per `cam_mount` below: ee_from_cam for a
    # camera on the wrist, world_from_cam for one bolted to the room.
    ee_from_cam: Float[np.ndarray, "4 4"]
    # Built from the PERCEPTION camera's intrinsics — FoundationStereo is given fx/fy/cx/cy and the
    # baseline of the camera whose stereo pair it is fed.
    depth_estimator: DepthEstimator

    # Wrist-view image-space mask of the gripper, dropped from the point cloud. None when a
    # world-mounted camera does perception: the mask is only meaningful in the wrist's view.
    gripper_mask: Bool[np.ndarray, "h w"] | None

    # How the perception camera's calibration entry is read: "ee" (world_from_cam = FK(q) @
    # ee_from_cam, the Franka wrist camera) or "world" (the entry IS world_from_cam, a camera fixed
    # in the scene — the bimanual YAM's top camera, or the third-person camera when
    # cameras.perception is "external"). See perception.cameras.camera_mount.
    cam_mount: str

    # cuRobo solvers keyed by ROBOT TYPE, because a bimanual YAM episode plans with two embodiments:
    # each arm has its own ee_link and locks the other arm at the neutral posture, so each needs its
    # own IKSolver and MotionGen. Single-embodiment robots have exactly one entry and the properties
    # below resolve to it unconditionally, leaving the Franka path unchanged.
    solvers: dict[str, tuple[IKSolver, MotionGen]]

    # cuTAMP configuration per robot type — same reason: TAMPConfiguration carries `robot`.
    tamp_configs: dict[str, TAMPConfiguration]

    @property
    def ik_solver(self) -> IKSolver:
        return self.solvers[tiptop_cfg().robot.type][0]

    @property
    def motion_gen(self) -> MotionGen:
        return self.solvers[tiptop_cfg().robot.type][1]

    @property
    def tamp_config(self) -> TAMPConfiguration:
        return self.tamp_configs[tiptop_cfg().robot.type]

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


def perception_camera(container: _DemoContainer) -> Camera:
    """The camera perception reads — the hand slot, or the exterior one when ``cameras.perception``
    says so.

    Resolved per call rather than held, for two reasons. The container keeps one handle per RECORDING
    slot, so the perception role never renames a camera in the episode (``hand_cam.svo`` /
    ``external_cam.svo``); and the teleop hand-off closes every camera and re-opens it afterwards
    (_release_cameras / _reacquire_cameras), so a cached handle would go stale.
    """
    cam = container.cam if container.perception_cam_key == "hand" else container.external_cam
    if cam is None:
        raise RuntimeError(
            f"The {container.perception_cam_key} camera does perception (cameras.perception) but is not open"
        )
    return cam


def capture_live_observation(container: _DemoContainer) -> Observation:
    """Read robot joint positions and resolve the camera's world pose.

    For an END-EFFECTOR camera that is forward kinematics: ``FK(q) @ ee_from_cam``, so the pose is
    only as good as the joint reading it was taken with. For a camera FIXED in the scene — the
    bimanual YAM's third-person D435, or the side-view ZED when ``cameras.perception: external`` —
    the calibration entry already IS ``world_from_cam`` and no kinematics are involved; the arm's
    configuration cannot move it. ``q_init`` is read either way, because cuTAMP plans from the
    measured state regardless of what is holding the camera.

    A third-person camera also sees the arm itself, so the robot's collision spheres are projected
    into the frame and dropped from the point cloud (the wrist view uses its painted gripper mask
    instead).
    """
    # bimanual_yam_dual has no single active arm (see YamClient.arm) -- q_init has to be all 12
    # numbers, both because cuTAMP's dual chain plans over all of them and because the FK branch
    # below (unused by the YAM's fixed third-person camera, but generic) would need the full state
    # too. get_joint_positions() would raise here; it reads whichever ONE arm is "active".
    if tiptop_cfg().robot.type == "bimanual_yam_dual":
        q_curr = container.robot.get_dual_joint_positions()
    else:
        q_curr = container.robot.get_joint_positions()
    cfg = tiptop_cfg()
    kin_state = None
    if container.cam_mount == "world":
        world_from_cam = container.ee_from_cam
    else:
        q_curr_pt = tensor_args.to_device(q_curr)
        kin_state = container.motion_gen.kinematics.get_state(q_curr_pt)
        world_from_cam = kin_state.ee_pose.get_numpy_matrix()[0] @ container.ee_from_cam

    # Grab a short burst of frames at this static pose for temporal depth smoothing. The first
    # frame is the representative one (used for rgb/intrinsics); the rest feed the median fusion.
    num_frames = max(1, int(cfg.perception.depth_smoothing.num_frames))
    frames = [perception_camera(container).read_camera() for _ in range(num_frames)]

    if container.perception_cam_key == "external":
        # The side-view camera sees the arm itself, from wherever it is standing, and no fixed
        # image-space mask can cover that. Project the same collision spheres cuRobo plans against
        # into this frame instead -- they follow the joints, so the arm is dropped from the point
        # cloud whether it is at home, holding something, or mid-hand-off. (Gemini is already told
        # not to report the robot, so only the geometry needs handling.) The YAM's fixed top camera
        # keeps its current behaviour -- no self-mask at all -- since nothing in that setup has been
        # calibrated against a projected-sphere mask.
        if kin_state is None:
            kin_state = container.motion_gen.kinematics.get_state(tensor_args.to_device(q_curr))
        if kin_state.link_spheres_tensor is None:
            raise RuntimeError(
                "cuRobo returned no collision spheres for the current joint state, so the robot "
                "cannot be masked out of the third-person view"
            )
        robot_mask = project_spheres_to_mask(
            kin_state.link_spheres_tensor[0].cpu().numpy(),
            world_from_cam,
            frames[0].intrinsics,
            frames[0].rgb.shape[:2],
            # Defaulted rather than required: TIPTOP_CONFIG replaces tiptop.yml wholesale, so an
            # embodiment config that opts into external perception need not repeat this knob.
            margin_m=float(cfg.perception.get("robot_mask_margin_m", 0.02)),
        )
        _log.debug(f"Robot self-mask covers {robot_mask.mean():.1%} of the third-person view")
    else:
        robot_mask = container.gripper_mask

    return Observation(
        frame=frames[0],
        world_from_cam=world_from_cam,
        q_init=q_curr,
        depth_frames=tuple(frames),
        robot_mask=robot_mask,
    )


def configured_arms() -> list[str]:
    """Arms a bimanual YAM rollout uses, in execution order. Empty for any other embodiment.

    ``robot.arms: [left, right]`` runs both in sequence (the sim's ``--bimanual``); a single-element
    list collects with one arm and only parks the other.
    """
    cfg = tiptop_cfg()
    # bimanual_yam_dual plans both arms as ONE 12-DOF chain (see tiptop.yam.task_split /
    # _run_yam_dual_rollout) -- it is a yam type but has no per-arm sequential loop, so it must
    # return [] here too. Without this check, arm_of(cfg.robot.type) below would raise: "dual" is
    # not one of ARMS, even on a config that never sets `robot.arms` at all -- `dict.get`'s default
    # argument is evaluated eagerly, so that call happens on every invocation, not just when the
    # `arms` key is missing.
    if not is_yam(cfg.robot.type) or cfg.robot.type == "bimanual_yam_dual":
        return []
    arms = list(cfg.robot.get("arms", [arm_of(cfg.robot.type)]))
    if not arms:
        raise ValueError("robot.arms is empty; name at least one arm")
    for arm in arms:
        arm_of(f"bimanual_yam_{arm}")  # validates, raises with a useful message
    return arms


def _planning_robot_types() -> list[str]:
    """Every robot type this session needs solvers for — one per arm on a bimanual YAM."""
    arms = configured_arms()
    return [f"bimanual_yam_{arm}" for arm in arms] if arms else [tiptop_cfg().robot.type]


def _check_recording_cameras(cam, external_cam, external_cam_2) -> None:
    """Fail before any rollout if a configured camera cannot be recorded.

    A ZED records through the SDK to an SVO; any other camera is encoded to MP4 directly
    (``recording.record_cameras``). Either is fine — what is not fine is a camera that is configured
    but did not open, because the run would then silently collect episodes missing a view.
    """
    for slot, camera in (("hand", cam), ("external", external_cam), ("external_2", external_cam_2)):
        configured = tiptop_cfg().cameras.get(slot) is not None
        if configured and camera is None:
            raise RuntimeError(
                f"Recording requires the configured camera cameras.{slot} "
                f"(s/n {tiptop_cfg().cameras[slot].get('serial')}), but it is unavailable — it most "
                "likely failed to open (e.g. LOW USB BANDWIDTH). Lower its fps/resolution, move it to "
                "another USB3 controller, or check it is connected. Aborting before the run so no "
                f"rollout is collected with a missing camera; to record without it, comment out "
                f"cameras.{slot} in the config."
            )


def get_demo_container(
    num_particles: int,
    num_spheres: int,
    collision_activation_distance: float,
    enable_recording: bool = False,
    cost_overrides: dict | None = None,
    curobo_config_summary: dict | None = None,
    tamp_configs: dict[str, TAMPConfiguration] | None = None,
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

    # Which camera perception reads, and how its calibration entry is interpreted. A third-person
    # camera is bolted to the room: its entry IS world_from_cam (droid stores third-person
    # extrinsics base-relative, wrist extrinsics gripper-relative), so nothing about it depends on
    # where the arm is. The hand slot follows its own `mount` — "ee" on the Franka wrist, "world"
    # for the YAM's top camera.
    perception_cam_key = str(tiptop_cfg().cameras.get("perception", "hand"))
    if perception_cam_key == "hand":
        perception_cam = cam
        mount = camera_mount("hand")
    elif perception_cam_key == "external":
        # get_external_camera() raises if the camera cannot be opened, so reaching here means it is
        # live — unlike external_2, a missing exterior camera is never shrugged off.
        perception_cam = external_cam
        mount = "world"
    else:
        raise ValueError(f"cameras.perception must be 'hand' or 'external', got {perception_cam_key!r}")
    # For an ee-mounted camera this is ee_from_cam; for a world-mounted one the same entry is read
    # as world_from_cam. capture_live_observation branches on `cam_mount`.
    ee_from_cam = load_calibration(perception_cam.serial)
    _log.info(f"Perception reads the {perception_cam_key} camera (s/n {perception_cam.serial}, {mount}-mounted)")

    if enable_recording:
        _check_recording_cameras(cam, external_cam, external_cam_2)

    # Create depth estimator once — closed over camera intrinsics
    # Cache the SAM2 client
    sam2_client()

    # Warm-up IK solver and motion generator, once per planning embodiment (cost_overrides applies
    # the cfg/tamp/*.yml cost knobs). A sequential-bimanual session warms BOTH arms up front rather
    # than lazily: the second arm's first plan would otherwise pay a full cuRobo warmup mid-episode,
    # with the cameras rolling and the first arm holding whatever it just placed.
    solvers: dict[str, tuple[IKSolver, MotionGen]] = {}
    for robot_type in _planning_robot_types():
        with _as_robot_type(robot_type):
            _log.info(f"Warming up cuRobo solvers for {robot_type}...")
            ik_solver, motion_gen, _ = build_curobo_solvers(
                num_particles, num_spheres, collision_activation_distance, cost_overrides=cost_overrides
            )
            solvers[robot_type] = (ik_solver, motion_gen)

    # The gripper mask erases a WRIST camera's own fingers from the point cloud — it is painted in
    # that camera's image. A camera fixed in the scene never sees the gripper in a fixed place, so
    # the same mask there would delete an arbitrary patch of the scene; the third-person path masks
    # the arm by projecting cuRobo's collision spheres instead (capture_live_observation).
    gripper_mask = None if mount == "world" else load_gripper_mask()

    return _DemoContainer(
        robot=client,
        cam=cam,
        external_cam=external_cam,
        external_cam_2=external_cam_2,
        enable_recording=enable_recording,
        perception_cam_key=perception_cam_key,
        ee_from_cam=ee_from_cam,
        depth_estimator=get_depth_estimator(perception_cam),
        gripper_mask=gripper_mask,
        cam_mount=mount,
        solvers=solvers,
        tamp_configs=tamp_configs or {},
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


def _label_rollout(save_dir: Path, output_dir: str, timestamp: str, recording_enabled: bool) -> Path | None:
    """Prompt user to label rollout as success/failure, moving it out of eval/ to
    <success|failure>/<timestamp>/. Loops on invalid input. Returns the final rollout
    directory (or the unchanged eval dir if skipped) so it can be post-processed.

    A rollout that executed nothing -- planning failed, so no `_meta.json` was written -- holds no
    episode to file. It is DELETED rather than filed, and None is returned; the label is still
    reported first, because it is what ends the trajectory and the server merges a hand-off's legs
    on it. Only checked when recording is on, since otherwise no rollout ever writes `_meta.json`.

    A "switch to teleop" (SIGUSR1) landing here raises TeleopHandoffRequested out of the prompt (see
    _at_prompt) -- the rollout stays unlabeled in eval/ and the caller hands the arm off."""
    global _at_prompt
    nothing_recorded = recording_enabled and not (save_dir / "_meta.json").exists()

    def discard() -> None:
        shutil.rmtree(save_dir, ignore_errors=True)
        _log.info(f"Nothing was executed in this rollout, so there is no episode to keep: removed {save_dir}")

    _emit_event({"event": "awaiting_label", "dir": str(save_dir)})
    try:
        while True:
            _at_prompt = True
            user_input = input(
                "\nWas the execution successful? Enter 'y' for success, 'n' for failure, or leave empty to skip: "
            )
            _at_prompt = False
            user_input = user_input.strip().lower()
            if user_input in ("y", "n"):
                dest = save_dir
                if not nothing_recorded:
                    cls = "success" if user_input == "y" else "failure"
                    dest = Path(output_dir) / cls / timestamp
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(save_dir, dest)
                    _log.info(f"Moved rollout to {cls} directory: {dest}")
                # The label rates the whole task attempt, so it is also the one unambiguous signal
                # that the trajectory has ENDED (the operator could always have handed off again).
                # The server merges the trajectory's legs on this event, and files the merged
                # episode under this success flag -- which is the only place the label survives
                # when this rollout itself is discarded just below.
                _emit_event({
                    "event": "labeled",
                    "dir": str(dest),
                    "success": user_input == "y",
                    "trajectory_id": _trajectory_id,
                })
                if nothing_recorded:
                    discard()
                    return None
                return dest
            elif user_input == "":
                if nothing_recorded:
                    discard()
                    return None
                _log.info(f"Keeping rollout in eval directory: {save_dir}")
                return save_dir
            else:
                print("Invalid input. Please enter 'y', 'n', or leave empty to skip.")
    except EOFError:
        _log.info("No input received, keeping rollout in eval directory")
        return save_dir
    finally:
        _at_prompt = False


_LAST_TASK: str | None = None
_postprocess_procs: list[subprocess.Popen] = []

# Manual robot commands accepted at the task prompt, in place of a task instruction. The
# data-collection UI's buttons drive these over stdin; a terminal user can just type them.
# They run BETWEEN rollouts (the prompt is the one point where the arm is idle and stdin is being
# read), reusing the warmed container -- so no cuRobo re-warm and no second robot connection.
# 'reset' is the odd one out: it runs a whole (unrecorded) perceive-plan-execute cycle rather than a
# single nudge, so async_entrypoint dispatches it instead of _run_robot_command.
ROBOT_COMMANDS = ("home", "open", "reset")


def _gripper_open_threshold_m() -> float:
    """Measured width at or above which the gripper counts as already open, in metres.

    Per-embodiment, because it is 90% of that gripper's own full span and the two grippers do not
    have the same one: the Robotiq 2F-85 opens to 0.085 m, the YAM's linear jaws to 0.096. Using the
    Robotiq figure on a YAM would read a gripper still 15% closed on an object as "already open" and
    skip the open that was supposed to release it.
    """
    if is_yam(tiptop_cfg().robot.type):
        return 0.9 * YAM_GRIPPER_STROKE_M
    return GRIPPER_OPEN_WIDTH


def _open_gripper_if_needed(container, arm: str | None = None) -> float | None:
    """Open the gripper unless the measured width already reads open. Returns the measured width.

    ``arm`` addresses a specific hand -- only meaningful in dual mode (see ``YamClient.arm``); every
    other call site leaves it None and this behaves exactly as before the parameter existed.
    """
    width = _read_gripper_width(container.robot, arm)
    threshold = _gripper_open_threshold_m()
    if width is not None and width >= threshold:
        _log.info(f"Gripper already open (width={width:.3f} m >= {threshold:.3f} m); skipping open")
        return width
    _log.info(f"Opening gripper (measured width={width})")
    if arm is not None:
        container.robot.open_gripper(arm=arm)
    else:
        container.robot.open_gripper()
    return width


def _run_robot_command(container, cfg, cmd: str) -> None:
    """Run a manual robot command typed at the task prompt.

    Never raises: a failed nudge (controller hiccup, gripper unreadable) must not tear down the
    warmed session -- the user should just land back at the prompt and be able to retry.
    """
    try:
        arms = configured_arms()
        dual_handover = cfg.robot.type == "bimanual_yam_dual"
        if cmd == "home":
            if arms:
                # Every configured arm, through cuRobo. This is the documented recovery after an
                # aborted rollout left an arm away from the neutral posture, which is the one state
                # that makes the other arm's plans untrustworthy (see _assert_idle_arm_parked).
                _log.info(f"Manual command: returning arms {arms} home")
                home_all_arms(container)
            elif dual_handover:
                _log.info("Manual command: returning both arms home (dual)")
                go_to_dual_home(time_dilation_factor=cfg.robot.time_dilation_factor, motion_gen=container.motion_gen)
            else:
                _log.info("Manual command: returning the arm home")
                go_to_home(time_dilation_factor=cfg.robot.time_dilation_factor, motion_gen=container.motion_gen)
        elif cmd == "open":
            _log.info("Manual command: opening the gripper")
            if arms:
                for arm in arms:
                    with active_arm(arm):
                        _open_gripper_if_needed(container)
            elif dual_handover:
                for arm in ("left", "right"):
                    _open_gripper_if_needed(container, arm=arm)
            else:
                _open_gripper_if_needed(container)
        else:
            raise ValueError(f"Unknown robot command: {cmd}")
        _emit_event({"event": "robot_command", "command": cmd, "ok": True})
    except Exception as e:
        _log.exception(f"Manual robot command '{cmd}' failed: {e}")
        _emit_event({"event": "robot_command", "command": cmd, "ok": False, "error": str(e)})


def _disarm_rollout_carryover() -> None:
    """Drop every flag armed to carry ONE rollout into the next.

    They exist to make a continuation -- a teleop hand-off resume, a HITL phase, a robot->robot leg --
    re-enter the rollout loop without resetting the arm or minting a new trajectory. Each is consumed
    at the top of the next rollout, so the only way one survives is a continuation that stopped being
    one: the plan finished, or was abandoned. See _get_task_instruction for where that is known.
    """
    global _skip_episode_reset, _skip_episode_arm_motion, _continue_trajectory, _reuse_plan_skeleton
    armed = [
        name
        for name, value in (
            ("_skip_episode_reset", _skip_episode_reset),
            ("_skip_episode_arm_motion", _skip_episode_arm_motion),
            ("_continue_trajectory", _continue_trajectory),
            ("_reuse_plan_skeleton", _reuse_plan_skeleton is not None),
        )
        if value
    ]
    if armed:
        # Worth a line: this is the moment a finished trajectory stops being one, and if the next
        # episode ever misbehaves again this says exactly what it inherited.
        _log.info(f"Back at the task prompt: disarming {', '.join(armed)} -- the next episode is a fresh one")
    _skip_episode_reset = False
    _skip_episode_arm_motion = False
    _continue_trajectory = False
    _reuse_plan_skeleton = None


def _get_task_instruction() -> str:
    """Task for the next rollout. The first comes from ``TIPTOP_TASK`` (non-interactive
    launch); subsequent ones are prompted interactively so the warmed container is reused
    across rollouts. Enter repeats the last task, typing a new one changes it, and
    'q'/'exit'/Ctrl-D ends the session (raising UserExitException).

    A ROBOT_COMMANDS word ('home'/'open'/'reset') is returned as-is instead of a task; the caller
    runs it and re-prompts. It is deliberately NOT remembered as the last task, so a later bare Enter
    still repeats the real instruction rather than nudging the robot (or resetting the scene) again.

    If a teleop hand-off just finished (_run_teleop_handoff queued _pending_instruction), that is
    consumed here first, WITHOUT blocking on stdin -- the loop replans/executes the same task
    immediately from wherever the human left the arm."""
    global _LAST_TASK, _pending_instruction
    if _pending_instruction is not None:
        instr = _pending_instruction
        _pending_instruction = None
        return instr
    # Nothing queued, so the loop is about to block on the operator -- and REACHING this prompt is
    # itself the statement that no trajectory is still in flight. A continuation re-enters the loop
    # through _pending_instruction above precisely so it never emits `awaiting_task` (the one state
    # in which the data-collection server lets someone home the arm or start another rollout midway
    # through a trajectory). So every carry-over flag is stale here by construction.
    #
    # Concretely, this is the "Collect another didn't send the arm home" bug. A finished HITL task
    # left `_continue_trajectory` set: the hand-off's resume armed it, the human-phase branch keeps it
    # deliberately (clearing it mid-plan minted a new trajectory id and split the episode), and
    # nothing cleared it when the plan turned out to be OVER. The next episode then read itself as
    # another leg of the trajectory just labeled -- no return to home, and through the same flag the
    # old `_trajectory_id` and `_trajectory_handed_off` carried into it, which also cost that episode
    # its post-processing. Disarmed HERE rather than at each arming site: there are four of those and
    # this is the one place all of them are provably stale.
    _disarm_rollout_carryover()
    env_task = os.environ.get("TIPTOP_TASK", "")
    if env_task:
        os.environ["TIPTOP_TASK"] = ""  # consume the launch task
        instr = env_task.strip()
        if not instr or instr.lower() in ("exit", "q", "quit"):
            raise UserExitException("TIPTOP_TASK empty/exit")
        _LAST_TASK = instr
        return instr
    # Interactive: keep reusing the warm container for back-to-back rollouts.
    suffix = f" [{_LAST_TASK}]" if _LAST_TASK else ""
    _emit_event({"event": "awaiting_task"})
    global _at_prompt
    try:
        _at_prompt = True
        raw = input(
            f"\nNext task (Enter = repeat{suffix}, 'home'/'open' to nudge the robot, "
            f"'reset' to put the scene back, 'q' to quit): "
        )
        _at_prompt = False
        raw = raw.strip()
    except EOFError:
        raise UserExitException("EOF; ending session")
    finally:
        _at_prompt = False
    if raw.lower() in ("q", "exit", "quit"):
        raise UserExitException("user quit")
    if raw.lower() == "resume":
        # A hand-off resume that arrived late (e.g. the server wrote it while we were already back at
        # the prompt, because the hand-off never armed). Planning a task called "resume" would be
        # nonsense, so re-prompt instead.
        _log.warning("Ignoring a 'resume' line at the task prompt: no teleop hand-off is in progress")
        return _get_task_instruction()
    if raw.lower() in ROBOT_COMMANDS:
        return raw.lower()  # a robot nudge, not a task -- leave _LAST_TASK alone
    if not raw:
        if _LAST_TASK:
            return _LAST_TASK
        raise UserExitException("no task entered; ending session")
    _LAST_TASK = raw
    return raw


# --- Human-in-the-loop planning -------------------------------------------------------------------
#
# Opt-in (`hitl.enabled` in the task's cfg/tamp config, passed as --hitl-config). With it off, none of
# this runs and the `tiptop.hitl` package is never even imported, so a problem in it cannot affect an
# ordinary session.
#
# A HITL task spans SEVERAL rollouts of the loop below: the VLM breaks the instruction into an
# ordered list of PHASES, each robot phase is one rollout aimed at that phase's sub-goal, and each
# human phase -- something only a person can do, like opening a box -- is one teleop leg. The plan
# therefore has to outlive a hand-off, which is what _hitl_session is for. Legs share a trajectory_id exactly as an ordinary hand-off's do, so the
# episode the operator ends up with is the whole task, merged by collect/merge_trajectory.py.
_hitl_cfg = None  # HITLConfig, resolved once at startup
_hitl_session = None  # tiptop.hitl.session.HITLSession for the task in progress


def _hitl_enabled() -> bool:
    return _hitl_cfg is not None and _hitl_cfg.enabled


def _hitl_reset_session(reason: str = "") -> None:
    """Drop the session, so the next rollout re-proposes and re-plans from scratch."""
    global _hitl_session
    if _hitl_session is not None and reason:
        _log.info(f"HITL: ending the session ({reason})")
    _hitl_session = None


def _hitl_arm_recorder(save_dir: Path) -> None:
    """Point the VLM audit trail at this rollout's directory (no-op when HITL is off)."""
    if not _hitl_enabled():
        return
    from tiptop.hitl.record import set_recorder

    set_recorder(save_dir / "vlm" if _hitl_cfg.save_vlm_io else None)


def _hitl_verification_image(container: "_DemoContainer"):
    """A frame to check the human's work in, preferring a camera that sees the whole workspace.

    After a hand-off the arm is wherever the operator left it, and the episode reset that would
    normally return it to the capture pose is deliberately skipped. A wrist camera therefore points
    somewhere arbitrary, so a third-person view is used when the rig has one.
    """
    from tiptop.hitl.grounding import to_pil

    cam = container.external_cam if container.external_cam is not None else perception_camera(container)
    return to_pil(cam.read_camera().rgb)


def _episode_reset_actions(
    *,
    resuming_from_handoff: bool,
    resuming_after_human_phase: bool,
    continuing_trajectory: bool,
    perception_cam_key: str,
    cam_mount: str = "ee",
) -> tuple[bool, bool, bool]:
    """What the pre-rollout reset should do: ``(go_home, open_gripper, go_to_capture)``.

    Pulled out of the rollout loop because it is the one decision here that has been got wrong more
    than once (see the flag definitions and the HITL hand-off branch), and it is pure -- four
    booleans and two camera facts in, three out -- so it can be tested without a robot.

    * An ordinary rollout resets fully: home, open the gripper, and (wrist perception only) drive to
      the capture pose. A rollout left mid-motion still gripping something starts clean.
    * A **teleop hand-off resume** moves the arm not at all and touches nothing: homing would undo
      the hand-off, and opening the gripper would drop whatever the operator is holding.
    * A leg resuming after a **HITL human phase** skips only the MOTION. The next phase continues the
      same trajectory and should plan from the pose the human left, but nothing about a person
      handling the arm guarantees the fingers are open -- and cuTAMP plans every phase from a
      HandEmpty initial state regardless -- so the gripper check stays.
    * A leg **continuing the same trajectory** without either of those -- the HITL robot->robot
      continuation -- does not home either, for the same reason: home is the one pose the task never
      asked the arm to be in, and driving there between two legs of one episode records a return to
      home into the middle of the demonstration. It DOES keep the capture pose when perception reads
      the wrist camera, since that move buys a guaranteed view of the table rather than a reset.

    The capture pose is only ever worth driving to for a camera the ARM CARRIES, which is both of
    ``perception_cam_key == "hand"`` and ``cam_mount == "ee"``: the hand SLOT is world-mounted on the
    bimanual YAM (its top D435 is bolted to the base column), and moving the arm in front of a fixed
    camera puts it in the point cloud rather than pointing it anywhere. ``cam_mount`` defaults to
    "ee" so the Franka arrangement -- the only one that existed when this was written -- reads
    exactly as before.

    Skipping the capture pose costs a guaranteed view: perception is geometrically correct from any
    pose (capture_live_observation recomputes a wrist camera's world_from_cam by FK, and a camera
    fixed in the scene does not move with the arm), but the arm no longer clears the shot.
    """
    wrist_perception = perception_cam_key == "hand" and cam_mount == "ee"
    if resuming_from_handoff:
        return False, False, False
    if resuming_after_human_phase:
        return False, True, False
    if continuing_trajectory:
        return False, True, wrist_perception
    return True, True, wrist_perception


def _hitl_detect_hint(task_instruction: str) -> list[str]:
    """Objects a plan already in progress is waiting for, to be named to the detector.

    Only the ones a human phase was supposed to have created and that nothing has bound yet. They are
    the detections most easily missed -- a block prised out of a tower is scenery to a detector
    working from the original instruction -- and the pass that misses one is the pass that needs it.
    Empty for a fresh task, so the first plan is still made against whatever the instruction alone
    turns up.
    """
    session = _hitl_session
    if session is None or not session.matches(task_instruction, _trajectory_id):
        return []
    return [
        f"{d.name}: {d.description} (should be on the {d.anchor})" if d.description else d.name
        for d in session.spec.unbound_deferred
    ]


def _hitl_goal_resolver(task_instruction: str, rgb, state: dict):
    """Build the callback run_perception uses to aim this rollout at one segment of the outer plan.

    ``state`` is filled in for the rollout body: ``human`` is the phase to hand over when the plan's
    next phase is a person's, and ``failure`` is why there is no goal to plan for.

    Nothing here raises. An exception out of run_perception would be swallowed by the rollout's own
    ``finally``, which saves ``env`` and ``processed_scene`` -- names that a failure here leaves
    unbound, so the real reason would be replaced by a NameError. Reporting a reason instead also
    keeps the rollout on the ordinary "no plan found" path, which still reaches the label prompt --
    and that label is what ends a handed-off trajectory and merges its legs.
    """
    from tiptop.hitl.grounding import to_pil
    from tiptop.hitl.planning import match_drifted_names
    from tiptop.hitl.session import build_session

    async def resolve(
        object_names: list[str], table_name: str, geometry: dict | None = None
    ) -> tuple[list[dict], set[str] | None, set[str] | None]:
        global _hitl_session
        session = _hitl_session
        if session is not None and not session.matches(task_instruction, _trajectory_id):
            _hitl_reset_session("a new task or trajectory started")
            session = None
        if session is not None:
            # An object a human phase was supposed to create may exist now. Bind it BEFORE the drift
            # check, so a name that has just become real is checked like any other from here on.
            session.bind_new_objects(geometry or {}, object_names)
            # Gemini names objects afresh every pass and the names drift ("toy" and "box" one pass,
            # "blue_toy" and "cardboard_box" the next). A plan naming an object this pass did not
            # produce would die in create_tamp_environment's unknown-object check, so re-bind it to
            # this pass's labels where that can be done unambiguously. Throwing the plan away instead
            # -- which is what used to happen -- re-planned the whole task from a scene already half
            # rearranged, and asked the human to redo the phase they had just finished.
            missing = sorted(session.objects_named() - set(object_names) - {table_name})
            if missing:
                # Candidates are this pass's labels that the plan does not ALREADY own. Subtracting
                # all_names rather than objects_named(): a name only the COMPLETED phases used is
                # still a plan object, and offering it as a re-binding target lets one spec object
                # be folded into another -- SceneTypes.rebind merges them inside a frozenset, and
                # the leg then aims at the thing the robot has already put away. Nothing downstream
                # re-validates, so this is the only place to refuse it.
                pool = sorted(set(object_names) - set(session.spec.scene_types.all_names))
                mapping = match_drifted_names(missing, pool)
                if mapping is None:
                    # Not every drifted name has to be re-bindable for THIS leg to run. `missing`
                    # spans every phase still to come, so a plan ending in "cover the bowls with the
                    # cloth" gates the robot's next pick-and-place on the cloth -- an object no robot
                    # phase names, and one a single missed detection is enough to lose. Try again
                    # over just what this leg needs; anything else can drift until the phase that
                    # actually names it comes round.
                    needed = sorted(set(missing) & session.objects_needed_now())
                    if needed != missing:
                        mapping = match_drifted_names(needed, pool)
                        if mapping is not None:
                            unbound = ", ".join(sorted(set(missing) - set(mapping)))
                            _log.warning(
                                f"HITL: carrying on without re-binding {unbound} -- not needed by the "
                                f"phase about to run. A later phase naming them will fail if they are "
                                f"still undetected then"
                            )
                if mapping is None:
                    _hitl_reset_session(f"perception no longer detects {', '.join(missing)}")
                    session = None
                elif mapping:
                    session.rebind(mapping)
        if session is None:
            try:
                session, reason = await build_session(
                    to_pil(rgb), task_instruction, object_names, table_name, _hitl_cfg, _trajectory_id
                )
            except Exception as exc:
                _log.exception("HITL: could not propose a plan for this task")
                state["failure"] = f"HITL proposal failed ({type(exc).__name__}: {exc})"
                return [], None, None
            if session is None:
                state["failure"] = reason
                return [], None, None
            if session.spec.unrepresented:
                # The plan covers less than the instruction asked for. Surfaced as an event as well
                # as printed, so it is visible in the data-collection UI rather than only in the log.
                _emit_event({
                    "event": "hitl_instruction_not_fully_represented",
                    "instruction": task_instruction,
                    "unrepresented": [dict(u) for u in session.spec.unrepresented],
                    "detected_objects": sorted(object_names),
                })
            _hitl_session = session

        state["session"] = session
        phase = session.current
        if phase is None:
            # Every phase is done, so there is nothing left to aim this rollout at. Only reachable if
            # a finished session outlived the rollout that finished it.
            state["failure"] = "the human-in-the-loop plan is already complete"
            return [], set(session.spec.scene_types.surfaces), session.robot_movables()
        if phase.is_human:
            # Nothing for the robot to do before the human acts. An empty goal keeps cuTAMP out of it
            # entirely (see the `human` branch before run_planning).
            state["human"] = phase
            return [], set(session.spec.scene_types.surfaces), session.robot_movables()
        unbound = session.unbound_needed_now()
        if unbound:
            # The human phase ran and was verified, but nothing in this pass looks like the object it
            # was supposed to leave behind. Refusing here is the point: the goal would otherwise name
            # an object create_tamp_environment has never heard of, and the alternative -- binding to
            # whatever carries a similar name -- is how the robot ends up moving the wrong block.
            where = {d.name: d.anchor for d in session.spec.deferred}
            state["failure"] = (
                "; ".join(
                    f"the human step was to leave {name} on {where.get(name, 'the anchor')}, but "
                    f"nothing detected there matches it"
                    for name in unbound
                )
                # The session is NOT reset here: the human's phase is done and verified, and its
                # progress is worth keeping. Re-running the task re-perceives and tries to bind again,
                # so say so -- otherwise the operator's reasonable reading is that the task is lost.
                + ". The human step is still recorded as done, so re-run this task to try again; "
                "moving it clear of anything else on that surface helps it be seen."
            )
            return [], set(session.spec.scene_types.surfaces), session.robot_movables()
        run = session.robot_run()
        if len(run) > 1:
            # Worth saying out loud: the operator sees ONE plan cover several of the phases the
            # proposal stage just listed, and the arm runs them without stopping in between.
            _log.info(
                f"HITL: planning {len(run)} consecutive robot phases as a single cuTAMP goal -- "
                + "; ".join(p.description for p in run)
            )
        return session.goal_dicts(), set(session.spec.scene_types.surfaces), session.robot_movables()

    return resolve


def _await_human_phase(message: str) -> str:
    """Show the human what to do and block until they say how they did it.

    Returns 'done' (they did it by hand, without the teleop rig) or 'abort'. Pressing "Switch to
    teleop" in the data-collection UI raises TeleopHandoffRequested out of the input() below --
    _at_prompt is what makes SIGUSR1 do that rather than wait for a rollout checkpoint that will
    never come here -- and the caller runs the hand-off from there.
    """
    global _at_prompt
    print(message, flush=True)
    while True:
        try:
            _at_prompt = True
            raw = input("Waiting for the human phase ('done' when finished by hand, 'abort' to give up): ")
            _at_prompt = False
        except EOFError:
            raise UserExitException("EOF while awaiting a human phase")
        finally:
            _at_prompt = False
        answer = raw.strip().lower()
        if answer in ("done", "y", "yes"):
            return "done"
        if answer in ("abort", "skip", "n", "no"):
            return "abort"
        if answer in ("q", "exit", "quit"):
            raise UserExitException("user quit at a human phase")
        if answer in ("resume", "resume_finish"):
            # A hand-off resume that arrived after the hand-off had already finished -- the server
            # writes one when the teleop child exits, and it lands here if we are back at a prompt.
            # Same treatment as at the task prompt: say so plainly and keep waiting.
            _log.warning(
                f"Ignoring a {raw.strip()!r} line at the human-phase prompt: the hand-off it belongs "
                "to has already finished. Press 'Switch to teleop' again for THIS phase, or type 'done'"
            )
            continue
        _log.warning(f"Ignoring {raw!r}: type 'done' or 'abort', or press 'Switch to teleop' in the UI")


async def _hitl_human_phase(container: "_DemoContainer", phase, can_finish: bool) -> bool:
    """Hand one human phase over, then check from a fresh image that it happened.

    Returns True when this rollout should be labeled and the trajectory closed out. Three ways there,
    all the same thing to the caller -- a `labeled` event, which ends the trajectory and has the
    server merge its legs: the operator ended it from teleop ("return & finish"); this was the task's
    LAST phase, so there is nothing left to plan; or the check gave up on a middle phase, which stops
    the plan but must NOT cost the operator the demonstration. Returning False leaves the trajectory
    open, which is what lets a robot phase AFTER a human one still happen.

    **The last phase is not checked at all.** Verification exists because every LATER phase is planned
    against the belief that this one happened -- and after the final phase there is no later phase.
    What there is instead is the operator, labeling the whole trajectory seconds later: a human
    verdict on the finished episode, which strictly supersedes a per-phase classifier. So the plan
    ends and the label prompt decides whether the episode is kept.

    Elsewhere a failed check is not silently accepted: the operator is told what is still missing and
    given another go (``verify_retries``). Continuing on a false belief is the one outcome worth
    avoiding -- but so is throwing away a demonstration a human just gave because one classifier call
    went the wrong way, which is why a spent retry budget still ends at the label prompt rather than
    dropping the legs unlabeled.
    """
    from tiptop.hitl.grounding import missing_statements, verify_phase
    from tiptop.hitl.session import handoff_message, phase_summary, retry_message

    session = _hitl_session
    assert session is not None
    attempts_left = _hitl_cfg.verify_retries
    finished_from_teleop = False

    while True:
        _emit_event({
            "event": "awaiting_human_phase",
            "instructions": phase.instructions,
            **phase_summary(session, phase),
        })
        try:
            answer = _await_human_phase(handoff_message(session, phase))
        except TeleopHandoffRequested:
            # "Switch to teleop" was pressed. Consume the request here so the hand-off we are about
            # to run satisfies it, rather than leaving it armed to fire again in the next rollout.
            _consume_teleop_request()
            _log.info("HITL: handing the arm over for the human phase")
            # The plan, not the operator, decides what a hand-back does here: the last phase closes
            # the trajectory out (the advance() below finishes the session and returns True), any
            # earlier one carries on into the next phase. Reported so the UI can say which.
            finished_from_teleop = _run_teleop_handoff(
                container,
                can_finish=can_finish,
                plan_next="end" if session.is_final_phase() else "robot",
            )
            answer = "done"
        if answer == "abort":
            _hitl_reset_session("the operator abandoned the human phase")
            return finished_from_teleop

        # The plan's LAST phase is not verified. Nothing downstream is planned against it -- there is
        # no later phase to plan -- and the operator labels the whole trajectory seconds later, which
        # is a human verdict on the finished episode and strictly supersedes a per-phase classifier.
        # Checking here did real damage rather than none: a verdict that went the wrong way with the
        # retries spent returned False, so the caller never labeled, and a demonstration the human had
        # just finished was dropped unlabeled and never merged (data-collection ARCHITECTURE.md §6c).
        # Emitted with skipped:true so the audit trail says it was NOT checked rather than that it
        # passed.
        if session.is_final_phase():
            _log.info("HITL: last phase of the task -- not verifying; the operator labels the episode")
            _emit_event({
                "event": "human_phase_verified",
                "description": phase.description,
                "ok": True,
                "skipped": True,
                "verdicts": [],
            })
            session.advance()
            _hitl_reset_session("every phase of the task is done")
            return True

        try:
            ok, verdicts = await verify_phase(
                _hitl_verification_image(container), phase, session.spec.invented, _hitl_cfg
            )
        except Exception:
            # A classifier that cannot be reached must not cost the operator their demonstration.
            _log.exception("HITL: could not verify the human step; accepting it unchecked")
            ok, verdicts = True, []
        session.verdicts.extend(verdicts)
        _emit_event({
            "event": "human_phase_verified",
            "description": phase.description,
            "ok": ok,
            "verdicts": [v.summary() for v in verdicts],
        })
        if ok or not _hitl_cfg.verify_enforced:
            if not ok:
                _log.warning("HITL: the human step did not verify, but verify_enforced is off; continuing")
            session.advance()
            if session.finished:
                # Unreachable as things stand -- the final phase returned above without ever being
                # verified, and advance() moves a human phase by one. Kept as the safe fallback if
                # that ever stops holding: a finished plan must reach the label prompt, never fall
                # through to another rollout.
                _log.info("HITL: every phase of the task is done")
                _hitl_reset_session("every phase of the task is done")
                return True
            _log.info(
                f"HITL: {len(session.phases) - session.index} phase(s) still to go; the robot "
                "carries on once the arm is handed back"
            )
            if finished_from_teleop:
                # Phases remain, but the operator chose "return & finish": their demonstration
                # ended the task their own way, so the trajectory closes out here. The session has
                # to go with it -- leaving it live would have the next attempt resume midway through
                # a plan for a scene the operator has since rearranged.
                _hitl_reset_session("the operator ended the trajectory from teleop")
                return True
            return False

        missing = missing_statements(verdicts)
        _log.warning(f"HITL: the human phase did not verify. Still expected: {missing}")
        if attempts_left <= 0:
            # Out of retries, so the plan stops here -- but the demonstration is still HANDED TO THE
            # OPERATOR rather than thrown away. True, not finished_from_teleop: returning False left
            # the caller with nothing to label, so the legs recorded so far were dropped unlabeled and
            # never merged. What the config promises ("the operator still labels the episode, so a
            # false negative is recoverable by answering the label prompt") is this: the label prompt
            # appears and they keep it, mark it suboptimal, or call it a failure.
            print(retry_message(missing, 0), flush=True)
            _hitl_reset_session("the human phase could not be verified")
            return True
        attempts_left -= 1
        print(retry_message(missing, attempts_left + 1), flush=True)
        if finished_from_teleop:
            # They already ended the trajectory; there is no leg left to hand off again.
            _hitl_reset_session("the human phase could not be verified")
            return True


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
            stdout=logf,
            stderr=logf,
            stdin=subprocess.DEVNULL,
            start_new_session=True,  # detach so it survives and never blocks the run loop
        )
        _postprocess_procs.append(proc)
        _log.info(f"Post-processing {rollout_dir.name} in background (pid {proc.pid}) -> postprocess.log")
    except Exception:
        _log.exception("Failed to launch background post-processing")


# Franka's ~0.855m nominal reach plus margin for the gripper's TCP, used by unreachable_goal_movable.
MAX_MOVABLE_REACH_M = 0.9


def unreachable_goal_movable(env: TAMPEnvironment, grounded_atoms: list[dict]) -> str | None:
    """Reason a movable the goal names sits outside the arm's reach, or None if they are all fine.

    Perception occasionally grounds a label to an object across the room; no Pick for it can ever be
    solved, and cuTAMP would spend a whole optimization pass before reporting only "no satisfying
    particles found". Returned as a reason rather than raised so the caller can treat it as an
    ordinary planning failure -- the rollout then still reaches the label prompt, and that label is
    what merges the legs of a handed-off trajectory.

    Only goal-named movables are checked -- distant clutter is left alone, and so are surfaces (an
    out-of-reach surface fails in Place instead, which this doesn't cover).
    """
    goal_labels = {arg for atom in grounded_atoms for arg in atom.get("args", [])}
    for mesh in env.movables:
        if mesh.name not in goal_labels or getattr(mesh, "pose", None) is None or len(mesh.pose) < 3:
            continue
        reach = float(np.linalg.norm(np.asarray(mesh.pose[:3], dtype=float)))
        if reach > MAX_MOVABLE_REACH_M:
            return (
                f"Goal object '{mesh.name}' is {reach:.2f}m from the robot base, outside its "
                f"{MAX_MOVABLE_REACH_M}m reach -- perception has most likely grounded the label to "
                f"the wrong object (centroid {[round(float(c), 3) for c in mesh.pose[:3]]})"
            )
    return None


# Words that name the table. Gemini is told not to detect the table (perception/prompts/
# detect_and_translate.txt), so it has no detected label to build a predicate from and reaches for a
# synonym instead -- 'desktop', 'wooden_table' and 'table_surface' have all been observed on the same
# scene. The fitted plane is always called 'table', so those are folded onto it rather than failing a
# goal that is otherwise perfectly well formed.
_TABLE_WORDS = frozenset(
    {"table", "tabletop", "desk", "desktop", "counter", "countertop", "worktop", "workbench", "surface"}
)


def resolve_table_aliases(grounded_atoms: list[dict], known_labels: set[str], table_name: str) -> None:
    """Rewrite table synonyms in the surface slot of on(...) goals to the fitted plane's name, in place.

    An arg is treated as a synonym only when it is NOT itself a detected object and one of its
    underscore-separated words names a table. That keeps 'wooden_table' and 'table_surface' while
    leaving a genuinely wrong surface such as 'snack_box' to fail the unknown-object check -- and
    leaving a table Gemini did detect (against instructions) bound to its own mesh, as before.

    Only the surface slot is rewritten: the table is never a movable, so a table word in the first
    arg is a mistranslation, not a naming slip, and should still be rejected.
    """
    for atom in grounded_atoms:
        args = atom.get("args", [])
        if atom.get("predicate") != "on" or len(args) != 2 or args[1] in known_labels:
            continue
        if _TABLE_WORDS & set(args[1].lower().replace(" ", "_").split("_")):
            _log.warning(f"Goal names the table '{args[1]}'; resolving it to the fitted plane '{table_name}'")
            args[1] = table_name


def create_tamp_environment(
    object_meshes: dict[str, Mesh],
    table_cuboid: Cuboid,
    grounded_atoms: list[dict],
    include_workspace: bool,
    surface_labels: set[str] | None = None,
    extra_surface_labels: set[str] | None = None,
    movable_labels: set[str] | None = None,
) -> tuple[TAMPEnvironment, list[Cuboid | Mesh]]:
    """Build the cuTAMP environment and goal for one rollout.

    Three independent ways to steer the typing, all of them None/absent for an ordinary rollout,
    which infers everything from its own goal:

    ``surface_labels`` REPLACES the inferred surface set. Left None it is inferred from this
    rollout's own goal, which is the only behaviour an ordinary rollout ever sees. A human-in-the-loop
    task passes it because its goal changes between segments: inferring per segment would make the
    cloth a Surface (and therefore a static obstacle) while the toy is placed on it, then a Movable
    once the goal no longer mentions it -- the world geometry would change mid-task.

    ``extra_surface_labels`` ADDS to whichever of the two the line above produced: labels that must
    be treated as surfaces even though no goal atom places anything on them. A scene reset needs this
    because its goal names only the table, so the plate the objects are coming off would be
    classified movable and the planner would be free to pick the plate up. See
    ``scene_reset.build_reset_goal``. Additive rather than a second override so the two compose: a
    caller may pin the surface set AND name one more.

    ``movable_labels`` restricts what the robot may PICK. Left None every non-surface detection is a
    Movable, which is right for an ordinary rollout: cuTAMP's search is free to shift whatever is in
    the way. A human-in-the-loop task passes it because a scene shared with a person contains the
    person's things -- the screwdriver they push the jenga block out with is detected, so three of
    the four skeletons cuTAMP enumerated for "put the block back on the tower" began by picking the
    screwdriver up and one of them placed it on the tower. Anything excluded is still perceived and
    still collision-checked; it just moves from Movable into ``statics``, so the arm routes around it
    instead of grasping it.
    """
    # Reject goals that reference objects not present in the perceived scene.
    # Without this, cuTAMP's BFS runs without stopping, expanding the move-chain on an unreachable goal.
    known_labels = set(object_meshes.keys()) | {table_cuboid.name}
    resolve_table_aliases(grounded_atoms, known_labels, table_cuboid.name)
    for atom in grounded_atoms:
        for arg in atom.get("args", []):
            if arg not in known_labels:
                raise ValueError(
                    f"Goal predicate {atom['predicate']}({', '.join(atom['args'])}) "
                    f"references unknown object '{arg}'. Known objects: {sorted(known_labels)}"
                )

    # Identify which objects are used as surfaces (second arg in on(x, y)) unless the caller pinned
    # the set outright, plus any the caller designated explicitly on top of that.
    if surface_labels is None:
        surface_labels = set()
        for atom in grounded_atoms:
            if atom["predicate"] == "on" and len(atom["args"]) == 2:
                surface_labels.add(atom["args"][1])
    else:
        surface_labels = set(surface_labels)  # copied: extra_surface_labels must not leak to the caller
    if extra_surface_labels:
        surface_labels |= {label for label in extra_surface_labels if label in object_meshes}

    # Whatever the caller excluded, the object this rollout's goal is ABOUT stays a Movable: cuTAMP
    # validates goal literals against the parameter types, so a goal over something typed out of
    # Movable is not a worse plan, it is an unplannable one.
    if movable_labels is not None:
        movable_labels = set(movable_labels)
        for atom in grounded_atoms:
            if atom["predicate"] in ("on", "holding") and atom.get("args"):
                movable_labels.add(atom["args"][0])

    # Separate movables, surfaces, and -- when the caller named the movables -- the rest, which stay
    # in the scene as obstacles rather than as things the search may pick up.
    movables = []
    surfaces = []
    obstacles = []
    for label, mesh in object_meshes.items():
        if label in surface_labels:
            surfaces.append(mesh)
        elif movable_labels is not None and label not in movable_labels:
            obstacles.append(mesh)
        else:
            movables.append(mesh)
    _log.info(f"Movables: {[m.name for m in movables]}")
    _log.info(f"Surfaces: {[s.name for s in surfaces]}")
    if obstacles:
        _log.info(f"Obstacles (perceived and avoided, never picked): {[o.name for o in obstacles]}")

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

    # All surfaces include table and detected surface objects.
    #
    # A goal that places ONTO the table needs the table's true top face, which the collision cuboid
    # is not: segment_table_with_ransac sinks it TABLE_BOX_CLEARANCE below the RANSAC plane so flat
    # objects lying on the table stay graspable. Placement height comes straight off that face
    # (cuTAMP's place_4dof_sampler adds only 1-10 mm), so placing on the unmodified cuboid commands
    # the object ~1-2 cm INTO the tabletop, with no collision to reject it -- the obstacle really is
    # that low. Grow the box upwards to the true plane, leaving its underside where it was.
    # Conditional, so every goal that does not name the table keeps the existing geometry exactly.
    if table_cuboid.name in surface_labels:
        _log.info(
            f"Goal places on '{table_cuboid.name}': raising its box top by {TABLE_PLACEMENT_RAISE} m toward the "
            "detected surface, so placements land ON the table rather than inside it"
        )
        table_cuboid = replace(
            table_cuboid,
            dims=[table_cuboid.dims[0], table_cuboid.dims[1], table_cuboid.dims[2] + TABLE_PLACEMENT_RAISE],
            pose=[*table_cuboid.pose[:2], table_cuboid.pose[2] + TABLE_PLACEMENT_RAISE / 2, *table_cuboid.pose[3:]],
        )
    all_surfaces = [table_cuboid, *surfaces]
    statics = list(workspace_cuboids()) if include_workspace else []
    for surface in all_surfaces:
        statics.append(surface)
    statics.extend(obstacles)

    # Create TAMP environment.
    #
    # Detected surfaces are `pick_transparent`: an open container (a plate, a bowl) reconstructs as
    # a mesh whose cuRobo collision proxy is a filled OBB spanning its full height, so every object
    # resting in it is embedded in an obstacle and cannot be picked -- measured on a scene reset,
    # grasp IK for the three toys was 0/25, 0/1 and 13/30 with the plate in the world and 25/25,
    # 1/1 and 30/30 with it out. cuTAMP hides them from the IK solvers and from the reach-in motion
    # segments only; they stay full obstacles for transit, keep their true top face for placement
    # height, and are still screened against a PLACED object (movable_to_world and the candidate-
    # placement filter keep the full checker), so placing ONTO the plate is unaffected. The table
    # is excluded because it is not
    # a reconstruction: segment_table_with_ransac already sinks it TABLE_BOX_CLEARANCE below the
    # detected plane for exactly this reason.
    env = TAMPEnvironment(
        name="tiptop_cutamp",
        movables=movables,
        statics=statics,
        pick_transparent=[surface.name for surface in surfaces],
        type_to_objects={"Movable": movables, "Surface": all_surfaces},
        goal_state=frozenset(goal_state),
    )
    _log.info(
        f"Created TAMP environment with {len(movables)} movables, {len(all_surfaces)} surfaces"
        + (f", {len(obstacles)} unpickable obstacles" if obstacles else "")
    )
    return env, all_surfaces


def process_scene_geometry(
    xyz_map: np.ndarray,
    rgb_map: np.ndarray,
    masks: np.ndarray,
    bboxes: list,
    grasps: dict,
    valid_mask: np.ndarray | None = None,
    object_pcds: dict[str, o3d.geometry.PointCloud] | None = None,
) -> ProcessedScene:
    """Process perception results into 3D scene geometry for TAMP.

    Args:
        xyz_map: World-space XYZ coordinates (H, W, 3)
        rgb_map: RGB image (H, W, 3) in 0-255 range
        masks: Segmentation masks from SAM2
        bboxes: Bounding boxes from Gemini
        grasps: Grasp predictions from M2T2
        valid_mask: Optional (H, W) mask of usable points (see predict_depth_and_grasps): the robot's
            own geometry and invalid depth are excluded from the table fit and the object meshes
        object_pcds: Optional pre-computed object point clouds

    Returns:
        ProcessedScene with table cuboid, object meshes, pcds, and filtered grasps
    """
    # Segment table with RANSAC (returns trimesh Box)
    table_trimesh = segment_table_with_ransac(xyz_map, rgb_map, masks, valid_mask=valid_mask)
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
        valid_mask=valid_mask,
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
        if tiptop_cfg().perception.get("augment_flipped_grasps", False):
            world_from_grasp, grasp_dict = augment_flipped_grasps(world_from_grasp, grasp_dict)
            filtered_grasps[label] = grasp_dict
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
    include_workspace: bool = True,
    log_to_rerun: bool = True,
    resolve_goal: (
        Callable[[list[str], str, dict], Awaitable[tuple[list[dict], set[str] | None, set[str] | None]]] | None
    ) = None,
    detect_hint: Sequence[str] = (),
    goal_builder=None,
) -> tuple[TAMPEnvironment, list, ProcessedScene, list[dict]]:
    """Perceive the scene and build the cuTAMP environment and goal for it.

    Two independent ways to replace the goal Gemini translated the instruction into; both are None
    for an ordinary rollout, and no caller passes both (they belong to different features).

    ``resolve_goal`` is the human-in-the-loop one. Given the object labels this pass detected, the
    fitted table's name, and where each detection is, it returns
    (grounded_atoms, surface_labels, movable_labels), aiming this rollout at one segment of a longer
    plan. It is awaited, because building that segment can mean a VLM round-trip.

    ``goal_builder`` is the scene-reset one. It is called with ``(processed_scene, detected_atoms)``
    and returns ``(grounded_atoms, extra_surface_labels)``, so the reset plans against a goal built
    from geometry instead of language (``scene_reset.reset_goal_builder``). Synchronous, because it
    is pure geometry.

    Either way the atoms actually planned for are what is returned, so a caller's run metadata
    records the effective goal rather than the discarded one.

    Object DETECTION is driven by the full instruction, which is what makes Gemini name things in
    terms relevant to the task; ``detect_hint`` only adds objects a plan already in progress is
    waiting for, and never takes any away.
    """
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
            robot_mask=observation.robot_mask,
            depth_frames=observation.depth_frames,
        ),
        detect_and_segment(rgb, task_instruction, extra_objects=detect_hint),
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
        observation.robot_mask,
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
        depth_results["valid_mask"],
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

    grounded_atoms = detection_results["grounded_atoms"]
    surface_labels = None
    movable_labels = None
    extra_surface_labels = None
    if resolve_goal is not None:
        # The labels only exist now, which is why this is a callback rather than a value the caller
        # could have passed in: on the first rollout of a HITL task the whole proposal stage runs
        # here, against the objects this pass actually detected.
        # The geometry goes with the labels: a HITL plan can name an object that did not exist when
        # it was made, and where a detection IS decides which one that name refers to. Imported here
        # rather than at module scope for the reason every other tiptop.hitl import is: with the
        # feature off, the package is never loaded, so a problem in it cannot affect an ordinary run.
        from tiptop.hitl.planning import scene_geometry

        grounded_atoms, surface_labels, movable_labels = await resolve_goal(
            sorted(processed_scene.object_meshes),
            processed_scene.table_cuboid.name,
            scene_geometry(processed_scene.object_meshes),
        )
    if goal_builder is not None:
        # Applied after resolve_goal, so a caller that somehow set both gets the geometric goal --
        # the reset's whole point is to ignore what the language said. No caller does today: HITL
        # drives resolve_goal from the rollout body, the reset drives goal_builder from
        # _run_scene_reset, and neither path reaches the other.
        grounded_atoms, extra_surface_labels = goal_builder(processed_scene, grounded_atoms)

    env, all_surfaces = create_tamp_environment(
        processed_scene.object_meshes,
        processed_scene.table_cuboid,
        grounded_atoms,
        include_workspace,
        surface_labels=surface_labels,
        extra_surface_labels=extra_surface_labels,
        movable_labels=movable_labels,
    )
    _log.info(f"Processing scene and perception results took {time.perf_counter() - proc_st:.2f}s")
    _log.info(f"Perception pipeline completed, took {time.perf_counter() - start_time:.2f}s")
    return env, all_surfaces, processed_scene, grounded_atoms


def plan_clear_then_task(
    config: TAMPConfiguration,
    processed_scene: ProcessedScene,
    q_init,
    detected_atoms: list[dict],
    save_dir: Path,
    *,
    ik_solver: IKSolver,
    motion_gen: MotionGen,
    cost_overrides: dict | None = None,
    return_home: bool = True,
) -> tuple[list | None, float, str | None, list[str]]:
    """Plan the blockers off the goal surfaces, then the task, and CONCATENATE the two plans.

    Returns ``(plan, planning_seconds, failure_reason, cleared_labels)``. With nothing in the way
    this is exactly ``run_planning`` on the instruction's own goal, so a scene that needs no clearing
    plans as it always did.

    **Why two plans and not one goal with the clearing atoms appended.** cuTAMP would then be free to
    order the 5 objects however BFS enumerates them, and nothing in the symbolic layer knows the
    blocker has to go first -- so most orderings place onto a still-occupied surface and fail. That
    enumeration order also depends on PYTHONHASHSEED, which is unset, so it is drawn afresh per
    PROCESS: measured on ``failure/2026-08-16_23-48-17``, the appended atom was grounded correctly
    and all five skeletons still came back with the banana 4th or 5th. Planning the clearing on its
    own removes the ordering question instead of gambling on it (``tiptop.goal_clearing``).

    **The second plan is not re-perceived**, deliberately. Both halves execute inside ONE recorded
    episode, and a perception pass between them would park the arm at the capture pose for several
    seconds mid-episode -- a stationary stretch in the middle of the demonstration, with the same
    command/observation mismatch the parking segments elsewhere in this file exist to avoid. Instead
    the blocker's new pose is read off the first plan itself (:func:`placed_poses`): cuTAMP stamps
    each Place step with the pose it updated the collision world to, so no reconstruction is involved
    and the two phases agree on where the object is by construction.

    **Nor does the arm go home in between.** cuTAMP ends every plan with a ``GoToInitial``, so
    concatenating two of them verbatim drives the arm back to the home pose after the clearing and
    straight back out again for the first pick. The clearing's copy is dropped
    (:func:`drop_return_to_initial`) and the task plans from wherever the clearing actually left the
    arm, so its opening ``MoveFree`` is one motion between the two configurations instead of two
    through home. The task's own ``GoToInitial`` stays -- that one ends the episode.

    ``return_home=False`` suppresses that last one, for the same reason ``run_planning`` takes the
    flag: this whole two-phase plan may itself be one LEG of a longer HITL episode, and a leg with
    phases still to come stops at the retract so the next one continues from there. It is applied
    only to the TASK plan -- the clearing's own ``GoToInitial`` is dropped outright either way, so
    asking cuTAMP not to emit it would change nothing but would make ``drop_return_to_initial``'s
    contract depend on a flag.

    Skeleton reuse (``run_planning``'s ``reuse_plan_skeleton``/``plan_out``) is deliberately NOT
    plumbed through: two cuTAMP calls produce no single skeleton to hand back, so a hand-off out of a
    clearing episode replans from scratch. Same degradation as a rejected stale skeleton.
    """
    meshes, table = processed_scene.object_meshes, processed_scene.table_cuboid
    clearing_atoms, goal_surfaces = build_clearing_goal(meshes, table, detected_atoms)
    blockers = [atom["args"][0] for atom in clearing_atoms]

    def _plan(env, all_surfaces, q, subdir, grasps, q_return=None, plan_return_home=True):
        return run_planning(
            env,
            config,
            q_init=q,
            ik_solver=ik_solver,
            grasps=grasps,
            motion_gen=motion_gen,
            all_surfaces=all_surfaces,
            experiment_dir=save_dir / subdir,
            cost_overrides=cost_overrides,
            q_return=q_return,
            return_home=plan_return_home,
        )

    if not blockers:
        _log.info("No goal surface is blocked; planning the task directly")
        env, all_surfaces = create_tamp_environment(meshes, table, detected_atoms, include_workspace=True)
        plan, duration, failure = _plan(
            env, all_surfaces, q_init, "cutamp", processed_scene.grasps, plan_return_home=return_home
        )
        return plan, duration, failure, []

    _log.info(f"Goal surfaces are blocked by {blockers}; planning the clearing first, then the task")
    env, all_surfaces = create_tamp_environment(
        meshes, table, clearing_atoms, include_workspace=True, extra_surface_labels=goal_surfaces
    )
    clear_plan, clear_duration, failure = _plan(
        env, all_surfaces, q_init, "cutamp_clearing", processed_scene.grasps
    )
    if clear_plan is None:
        return None, clear_duration, f"clearing {blockers} off the goal surface(s): {failure}", []

    # Where the clearing plan leaves the arm and the blockers, so the task plans against that world
    # rather than the one perception saw.
    placed = placed_poses(clear_plan)
    moved = move_meshes(meshes, placed)
    # Drop the clearing plan's trailing GoToInitial: the task plan's own opening MoveFree will take
    # the arm from wherever the clearing left it straight to the first grasp, instead of driving home
    # in between and setting straight back out. Done AFTER placed_poses, which reads the Place
    # steps earlier in the plan and is unaffected either way.
    clear_plan = drop_return_to_initial(clear_plan)
    for label in placed:
        _log.info(
            f"'{label}' cleared to {[round(float(x), 3) for x in moved[label].pose[:3]]} "
            f"(was {[round(float(x), 3) for x in meshes[label].pose[:3]]})"
        )
    q_after = final_configuration(clear_plan)
    if q_after is None:
        return None, clear_duration, "clearing plan contained no trajectory to continue from", []

    # The blockers' M2T2 grasps were harvested at their OLD poses, so they no longer describe the
    # object. Empty them rather than dropping the key: cuTAMP indexes `self.grasps[obj]` without a
    # membership check (particle_initialization._sample_grasps), so a missing entry is a KeyError the
    # moment any skeleton picks that object -- and skeletons that do reach phase two. An empty
    # `grasps_obj` takes the same branch a never-detected object takes, falling back to heuristic
    # grasps computed at the pose the object is actually in.
    task_grasps = dict(processed_scene.grasps)
    for label in placed:
        if label in task_grasps:
            task_grasps[label] = {**task_grasps[label], "grasps_obj": [], "confidences_pt": []}
    env, all_surfaces = create_tamp_environment(moved, table, detected_atoms, include_workspace=True)
    # q_return: the task plan STARTS from where the clearing handed over, so its own notion of
    # "initial" is that mid-episode pose -- without this the episode would end by driving back to
    # where the blocker was set down instead of to the pose it began at.
    task_plan, task_duration, failure = _plan(
        env, all_surfaces, q_after, "cutamp", task_grasps, q_return=q_init, plan_return_home=return_home
    )
    duration = clear_duration + task_duration
    if task_plan is None:
        return None, duration, f"task after clearing {blockers}: {failure}", []

    _log.info(
        f"Two-phase plan: {len(clear_plan)} step(s) to clear {blockers}, "
        f"{len(task_plan)} for the task, executed as one episode"
    )
    return clear_plan + task_plan, duration, None, blockers


def _plan_largest_solvable_reset(
    container: _DemoContainer,
    config: TAMPConfiguration,
    processed_scene: ProcessedScene,
    q_init,
    goal_atoms: list[dict],
    env: TAMPEnvironment,
    all_surfaces: list,
    save_dir: Path,
) -> tuple[list | None, list[str], list[str]]:
    """Plan the biggest subset of the reset goal cuTAMP can actually solve.

    Returns ``(plan, moved_labels, skipped_labels)``; ``plan`` is None only if no single object could
    be planned.

    A reset goal is one plan over N objects, and cuTAMP must satisfy every object's Pick and Place
    at once -- so ONE unreachable object sinks the whole reset and nothing moves. That is not a
    corner case: an object nestled on the surface it has to come off gets few clean M2T2 grasps, and
    with none of them IK-reachable its Pick constraint reports 0/256 satisfying while every other
    object's is fine. (Measured: pink_toy 4 grasps -> 0/256, against blue_toy's 73 -> 129/256. All
    three toys stayed on the plate.)

    So on failure we drop the object with the FEWEST grasp candidates -- the signal that actually
    predicts the blocker -- and re-plan the rest, down to a single object. Moving two of three toys
    and naming the third beats moving none. Each attempt is a complete co-planned multi-object plan,
    so placements are still solved together and cannot collide; only the goal shrinks.
    """
    n_grasps = {
        label: len(processed_scene.grasps.get(label, {}).get("poses", ())) for label in processed_scene.object_meshes
    }
    # Hardest (fewest grasps) last, so pop() drops the most likely blocker first.
    labels = sorted((a["args"][0] for a in goal_atoms), key=lambda label: -n_grasps.get(label, 0))
    table_name = processed_scene.table_cuboid.name
    _, surfaces = build_reset_goal(processed_scene.object_meshes, processed_scene.table_cuboid)
    _log.info(f"Reset planning order (most graspable first): {[(label, n_grasps.get(label)) for label in labels]}")

    skipped: list[str] = []
    attempt = 0
    while labels:
        attempt += 1
        if attempt > 1:
            # Rebuild the environment for the shrunken goal. Perception is NOT re-run -- the scene
            # has not changed, only what we are asking for.
            atoms = [{"predicate": "on", "args": [label, table_name]} for label in labels]
            env, all_surfaces = create_tamp_environment(
                processed_scene.object_meshes,
                processed_scene.table_cuboid,
                atoms,
                include_workspace=True,
                extra_surface_labels=surfaces,
            )
        cutamp_plan, _, failure_reason = run_planning(
            env,
            config,
            q_init=q_init,
            ik_solver=container.ik_solver,
            grasps=processed_scene.grasps,
            motion_gen=container.motion_gen,
            all_surfaces=all_surfaces,
            experiment_dir=save_dir / ("cutamp" if attempt == 1 else f"cutamp_retry{attempt - 1}"),
            cost_overrides=container.cost_overrides,
        )
        if cutamp_plan is not None:
            return cutamp_plan, labels, skipped
        dropped = labels.pop()
        skipped.append(dropped)
        _log.warning(
            f"No reset plan for {len(labels) + 1} object(s) ({failure_reason}); dropping "
            f"'{dropped}' ({n_grasps.get(dropped)} grasp candidates) and retrying with {labels or 'nothing left'}"
        )
    return None, [], skipped


# Camera slot -> (episode filename, LeRobot image key) for the bimanual YAM. Deliberately NOT the
# DROID names: this rig's three views are a fixed third-person camera and two wrist cameras, which
# do not map onto exterior_1 / exterior_2 / wrist. data-collection's CAMERA_FILES allowlists
# (collect/episodes.py and server/lib/episodes.js) carry both sets.
YAM_CAMERA_FILES = {
    "hand": ("top_cam.mp4", "observation.images.top"),
    "external": ("left_wrist_cam.mp4", "observation.images.left_wrist"),
    "external_2": ("right_wrist_cam.mp4", "observation.images.right_wrist"),
}


def _yam_recording_targets(container: _DemoContainer, save_dir: Path) -> tuple[list, dict[str, str]]:
    """``(record_cameras arg, {lerobot key: filename})`` for whichever YAM cameras are present."""
    targets, keys = [], {}
    for slot, camera in (
        ("hand", container.cam),
        ("external", container.external_cam),
        ("external_2", container.external_cam_2),
    ):
        if camera is None:
            continue
        filename, lerobot_key = YAM_CAMERA_FILES[slot]
        # raw_path is only used by the ZED SDK recorder; a RealSense encodes straight to the mp4.
        targets.append((camera, save_dir / f"{Path(filename).stem}.svo", save_dir / filename))
        keys[lerobot_key] = filename
    return targets, keys


def _binary_gripper(state: dict) -> float:
    """The arm's last gripper COMMAND, on the binary convention (0 = open, 1 = closed).

    Reads ``gripper_target`` — what the gripper was last told — in preference to ``gripper``, where
    it actually ended up. The two differ exactly when the jaws stop on an object: a gripper
    commanded shut but held half open by a wide toy is commanded CLOSED, and thresholding the
    measurement would record it as open. The measurement is only a fallback for a server too old to
    report the target.

    The command channel must be exactly 0.0 or 1.0 — a continuous value there is the feedback trap
    the capture rewrite exists to prevent, and both ``dump_bimanual_episode`` and ``build_lerobot``
    reject an episode containing one. Proprioception keeps the continuous measurement; only the
    command is binarised.
    """
    opening = state.get("gripper_target")
    if opening is None:
        opening = state.get("gripper", 1.0)
    return float(1.0 - float(opening) >= 0.5)


def _read_bimanual_start(client) -> tuple[np.ndarray, np.ndarray]:
    """``(q[12], binary gripper command[2])`` at the start of an episode.

    Seeds the commanded timeline: before the first segment runs, "what was commanded" is "hold where
    the arms are, with the grippers as they are".
    """
    from tiptop.yam import ARMS
    from tiptop.yam.capture import ARM_SLICES, GRIPPER_INDEX

    q = np.zeros(len(ARMS) * 6, dtype=np.float32)
    gripper = np.zeros(len(ARMS), dtype=np.float32)
    for arm in ARMS:
        state = client.get_arm_state(arm)
        q[ARM_SLICES[arm]] = np.asarray(state["q"], dtype=np.float32)[:6]
        gripper[GRIPPER_INDEX[arm]] = _binary_gripper(state)
    return q, gripper


def _current_gripper_command(client, arm: str) -> float:
    """The arm's gripper command (binary; 0 open, 1 closed) to hold across a non-plan motion."""
    try:
        return _binary_gripper(client.get_arm_state(arm))
    except Exception:
        _log.warning(f"Could not read the {arm} gripper; assuming open for the hold segment")
        return 0.0


def _assert_idle_arm_parked(container: _DemoContainer) -> None:
    """Refuse to plan while the other arm is somewhere cuRobo does not think it is.

    This is the one invariant sequential-bimanual TAMP rests on. The planning arm's cuRobo config
    locks the other arm's joints at NEUTRAL_Q and collision-checks it there, so if the parked arm
    has been left mid-motion — after an aborted rollout, say — every plan is validated against a
    robot that does not exist, and the planner will happily route straight through it.

    Recovering automatically is not safe: getting there is a long joint-space move the server would
    have to interpolate blind, with no collision checking. Homing through cuRobo is what the `home`
    robot command does, so say so and stop.
    """
    from tiptop.yam import other_arm

    idle = other_arm(arm_of(tiptop_cfg().robot.type))
    measured = np.asarray(container.robot.get_arm_state(idle)["q"], dtype=np.float64)[:6]
    error = float(np.abs(measured - np.asarray(NEUTRAL_Q)).max())
    if error > IDLE_ARM_TOLERANCE_RAD:
        raise RuntimeError(
            f"the {idle} arm is {error:.3f} rad from the neutral posture (limit "
            f"{IDLE_ARM_TOLERANCE_RAD}), but the planner collision-checks it there. Run the 'home' "
            "robot command to bring both arms back through cuRobo before collecting."
        )
    _log.debug(f"{idle} arm parked at neutral (max error {error:.3f} rad)")


def home_all_arms(container: _DemoContainer) -> None:
    """Return every configured arm to the neutral posture, through cuRobo, one at a time.

    Order matters. Each arm's plan assumes the other is already at NEUTRAL_Q, so homing an arm while
    the other is displaced is exactly the situation :func:`_assert_idle_arm_parked` refuses. Nothing
    can make that first move fully safe when both arms are out of place, so it is done with the
    workspace-clear assumption a manual `home` already carries, and warns when it applies.
    """
    from tiptop.yam import ARMS, other_arm

    tdf = tiptop_cfg().robot.time_dilation_factor
    for arm in ARMS:
        # Only arms this session warmed solvers for. A single-arm config still has the other arm
        # physically present, but it is never planned for, so it cannot be homed through cuRobo.
        if f"bimanual_yam_{arm}" not in container.solvers:
            continue
        with active_arm(arm):
            idle = other_arm(arm)
            measured = np.asarray(container.robot.get_arm_state(idle)["q"], dtype=np.float64)[:6]
            if float(np.abs(measured - np.asarray(NEUTRAL_Q)).max()) > IDLE_ARM_TOLERANCE_RAD:
                _log.warning(
                    f"Homing the {arm} arm while the {idle} arm is away from the neutral posture — "
                    "the collision model does not describe the parked arm for this move. Keep the "
                    "workspace clear and watch the arms."
                )
            go_to_home(time_dilation_factor=tdf, motion_gen=container.motion_gen)


async def _run_one_arm(
    session: aiohttp.ClientSession,
    container: _DemoContainer,
    save_dir: Path,
    task_instruction: str,
    arm: str,
    base_y: float,
) -> dict:
    """Perceive, plan and execute one arm's share of the goal. Returns segments + metadata.

    Perception runs fresh for each arm — the same choice the sim harness makes — because by the time
    the second arm plans, the first has already moved its objects and the world model from before is
    stale. The goal is not re-grounded though: ``arm_goal_builder`` filters the atoms Gemini already
    produced for the whole instruction, so the two arms cannot disagree about the task and neither
    depends on the VLM naming an object the same way twice.
    """
    cfg = tiptop_cfg()
    tdf = cfg.robot.time_dilation_factor
    arm_dir = save_dir / arm
    segments, meta = [], {"arm": arm}

    _assert_idle_arm_parked(container)

    # Getting to the capture posture is a real commanded motion inside the recorded window, so it is
    # recorded as a segment. Without that the video would show the arm moving while the action
    # channel said "hold" — the proprioception/action mismatch the capture rewrite exists to avoid.
    gripper_cmd = _current_gripper_command(container.robot, arm)
    homing = go_to_home(time_dilation_factor=tdf, motion_gen=container.motion_gen)
    if homing is not None:
        segments.append(
            segment_from_motion(
                arm, homing["positions"], homing["velocities"], homing["t_start"], homing["t_end"], gripper_cmd
            )
        )
    # Opening the gripper is a COMMAND, inside the recorded window, that no plan step covers. Left
    # out, the action channel would keep saying "closed" for the whole perception pause while the
    # video shows the jaws opening -- the same command/observation mismatch the parking segments
    # above exist to avoid. Recorded as a stationary segment at the pose the arm is already holding.
    gripper_open_start = time.time()
    try:
        _open_gripper_if_needed(container)
    except Exception as e:
        _log.exception(f"Gripper open/check failed for the {arm} arm: {e}")
    gripper_after = _current_gripper_command(container.robot, arm)
    if gripper_after != gripper_cmd:
        # Two rows so the change spans the measured actuation window rather than landing on an
        # instant; the arm is stationary throughout, so both carry the same pose at zero velocity.
        held = np.tile(np.asarray(container.robot.get_joint_positions(), dtype=np.float32), (2, 1))
        segments.append(segment_from_motion(arm, held, None, gripper_open_start, time.time(), gripper_after))
    gripper_cmd = gripper_after

    observation = capture_live_observation(container)
    perception_start = time.perf_counter()
    env, all_surfaces, processed_scene, grounded_atoms = await run_perception(
        session,
        observation,
        task_instruction,
        arm_dir,
        depth_estimator=container.depth_estimator,
        goal_builder=arm_goal_builder(arm, base_y),
    )
    meta["perception_duration"] = time.perf_counter() - perception_start
    meta["grounded_atoms"] = grounded_atoms
    meta["q_at_capture"] = observation.q_init
    meta["world_from_cam"] = observation.world_from_cam
    save_run_outputs(arm_dir, env, processed_scene.grasps)

    if not grounded_atoms:
        _log.info(f"{arm} arm: nothing on its side of the midline to do; skipping")
        meta["status"] = "nothing_to_do"
        return {"segments": [s for s in segments if s is not None], "meta": meta, "plan_path": None}

    _log.info(f"{arm} arm: planning for {[a['args'] for a in grounded_atoms]}")
    cutamp_plan, planning_duration, failure_reason = run_planning(
        env,
        container.tamp_config,
        q_init=observation.q_init,
        ik_solver=container.ik_solver,
        grasps=processed_scene.grasps,
        motion_gen=container.motion_gen,
        all_surfaces=all_surfaces,
        experiment_dir=arm_dir / "cutamp",
        cost_overrides=container.cost_overrides,
    )
    meta["planning_duration"] = planning_duration
    meta["planning_failure_reason"] = failure_reason
    meta["planning_success"] = cutamp_plan is not None
    if cutamp_plan is None:
        _log.warning(f"{arm} arm: no plan found ({failure_reason})")
        meta["status"] = "no_plan"
        return {"segments": [s for s in segments if s is not None], "meta": meta, "plan_path": None}

    plan_path = save_dir / f"tiptop_plan_{arm}.json"
    trace_cfg = resolve_trace_cfg(container.cost_overrides)
    save_tiptop_plan(serialize_plan(cutamp_plan, observation.q_init, trace_cfg=trace_cfg), plan_path)

    _log.info(f"{arm} arm: executing plan ({len(cutamp_plan)} steps)")
    timeline: list[dict] = []
    execute_cutamp_plan(cutamp_plan, client=container.robot, timeline=timeline)
    segments.extend(segments_from_plan(arm, plan_path, timeline))

    # Park this arm again so the OTHER arm's collision model is true before it plans.
    gripper_cmd = _current_gripper_command(container.robot, arm)
    parking = go_to_home(time_dilation_factor=tdf, motion_gen=container.motion_gen)
    if parking is not None:
        segments.append(
            segment_from_motion(
                arm, parking["positions"], parking["velocities"], parking["t_start"], parking["t_end"], gripper_cmd
            )
        )

    meta["status"] = "executed"
    return {"segments": [s for s in segments if s is not None], "meta": meta, "plan_path": plan_path}


async def _run_yam_rollout(
    session: aiohttp.ClientSession,
    container: _DemoContainer,
    save_dir: Path,
    task_instruction: str,
) -> dict:
    """One sequential-bimanual episode: each arm plans and executes its own share, in turn.

    This is the shape ``droid-sim-evals/eval/yam_tiptop_eval.py --bimanual`` runs in simulation —
    split the objects at the robot's midline, give each arm the part of the goal on its side, and
    run the two plans back to back inside one rollout. It is sequential bimanual, not simultaneous
    dual-arm TAMP: cuTAMP plans one kinematic chain, and the arm that is not planning is a locked
    obstacle in the other's collision model.

    Cameras and the state sampler run across the WHOLE episode, including the perception and
    planning pauses between the two arms. Those pauses are genuinely stationary and are recorded as
    explicit hold rows, so the frames are honest — they are simply not very interesting, and they
    are what the non-idle training filter is for.
    """
    cfg = tiptop_cfg()
    arms = configured_arms()
    base_y = float(cfg.robot.get("base_y", 0.0))
    _log.info(f"Bimanual rollout: arms {arms}, splitting objects at y = {base_y}")

    cameras_to_record, lerobot_cameras = _yam_recording_targets(container, save_dir)
    if not cameras_to_record:
        raise RuntimeError("no cameras configured to record; check the `cameras` block of the config")

    segments, per_arm, plan_paths = [], [], {}
    with BimanualJointSampler(
        host=cfg.robot.get("host", "127.0.0.1"),
        port=int(os.environ.get("TIPTOP_STATE_PORT", cfg.robot.get("state_port", 5557))),
    ) as sampler:
        with record_cameras(cameras_to_record) as rec_window:
            q_start, gripper_start = _read_bimanual_start(container.robot)
            for arm in arms:
                with active_arm(arm):
                    result = await _run_one_arm(session, container, save_dir, task_instruction, arm, base_y)
                segments.extend(result["segments"])
                per_arm.append(result["meta"])
                if result["plan_path"] is not None:
                    plan_paths[arm] = result["plan_path"].name

    # The combined plan index. data-collection counts an episode as collected only when
    # `tiptop_plan.json` exists next to `robot_state.npz` (collect/config.py::_is_complete), and a
    # bimanual episode has one plan per arm — so this names them rather than being a plan itself.
    (save_dir / "tiptop_plan.json").write_text(
        json.dumps(
            {
                "version": "2.0.0-bimanual",
                "embodiment": "bimanual_yam",
                "arms": arms,
                "plans": plan_paths,
                "instruction": task_instruction,
            },
            indent=2,
            cls=NumpyEncoder,
        )
    )

    n_frames = 0
    npz_path = dump_bimanual_episode(
        save_dir,
        segments=segments,
        joint_samples=sampler.samples,
        instruction=task_instruction,
        cameras=lerobot_cameras,
        q_start=q_start,
        gripper_start=gripper_start,
        fps=LEROBOT_FPS,
        config_id=os.environ.get("TIPTOP_CONFIG_ID"),
        record_start=rec_window.get("t_start"),
        record_stop=rec_window.get("t_stop"),
        arms_used=[m["arm"] for m in per_arm if m.get("status") == "executed"],
    )
    if npz_path is not None:
        n_frames = json.loads((save_dir / "_meta.json").read_text()).get("n_frames", 0)

    return {"n_frames": n_frames, "per_arm": per_arm, "planned": bool(plan_paths)}


async def _run_yam_dual_rollout(
    session: aiohttp.ClientSession,
    container: _DemoContainer,
    save_dir: Path,
    task_instruction: str,
) -> dict:
    """One SIMULTANEOUS dual-arm handover episode: one perceive-plan-execute cycle against the
    12-DOF ``bimanual_yam_dual`` embodiment.

    Unlike :func:`_run_yam_rollout`, this is NOT a per-arm loop: cuTAMP plans and collision-checks
    both hands inside one 12-DOF configuration directly (see ``bimanual_yam_dual.yml``'s
    self-collision spheres and ``DualKinematicConstraint``), so there is no "arm that isn't planning
    is a locked obstacle" step and no need to re-perceive between arms — one perception pass and one
    cuTAMP call cover the whole episode. :func:`tiptop.yam.task_split.handover_goal_builder` narrows
    the grounded goal to the single atom that actually needs a handover (v1: at most one per
    episode, a domain constraint of ``handover_tamp_operators``, not a simplification).
    """
    cfg = tiptop_cfg()
    tdf = cfg.robot.time_dilation_factor
    base_y = float(cfg.robot.get("base_y", 0.0))
    _log.info(f"Dual-arm rollout: handover objects split at y = {base_y}")

    cameras_to_record, lerobot_cameras = _yam_recording_targets(container, save_dir)
    if not cameras_to_record:
        raise RuntimeError("no cameras configured to record; check the `cameras` block of the config")

    segments: list = []
    meta: dict = {"arm": "dual"}
    plan_path = None

    with BimanualJointSampler(
        host=cfg.robot.get("host", "127.0.0.1"),
        port=int(os.environ.get("TIPTOP_STATE_PORT", cfg.robot.get("state_port", 5557))),
    ) as sampler:
        with record_cameras(cameras_to_record) as rec_window:
            q_start, gripper_start = _read_bimanual_start(container.robot)

            # Getting to the capture posture is a real commanded motion inside the recorded window
            # (same reasoning as _run_one_arm's homing segment).
            gripper_cmd = (
                _current_gripper_command(container.robot, "left"),
                _current_gripper_command(container.robot, "right"),
            )
            homing = go_to_dual_home(time_dilation_factor=tdf, motion_gen=container.motion_gen)
            if homing is not None:
                segments.append(
                    segment_from_motion(
                        "dual", homing["positions"], homing["velocities"],
                        homing["t_start"], homing["t_end"], gripper_cmd,
                    )
                )
            gripper_open_start = time.time()
            for arm in ("left", "right"):
                try:
                    _open_gripper_if_needed(container, arm=arm)
                except Exception as e:
                    _log.exception(f"Gripper open/check failed for the {arm} arm: {e}")
            gripper_after = (
                _current_gripper_command(container.robot, "left"),
                _current_gripper_command(container.robot, "right"),
            )
            if gripper_after != gripper_cmd:
                held = np.tile(np.asarray(container.robot.get_dual_joint_positions(), dtype=np.float32), (2, 1))
                segments.append(
                    segment_from_motion("dual", held, None, gripper_open_start, time.time(), gripper_after)
                )
            gripper_cmd = gripper_after

            observation = capture_live_observation(container)
            perception_start = time.perf_counter()
            env, all_surfaces, processed_scene, grounded_atoms = await run_perception(
                session,
                observation,
                task_instruction,
                save_dir,
                depth_estimator=container.depth_estimator,
                goal_builder=handover_goal_builder(base_y),
            )
            meta["perception_duration"] = time.perf_counter() - perception_start
            meta["grounded_atoms"] = grounded_atoms
            meta["q_at_capture"] = observation.q_init
            meta["world_from_cam"] = observation.world_from_cam
            save_run_outputs(save_dir, env, processed_scene.grasps)

            if grounded_atoms:
                _log.info(f"dual: planning handover for {[a['args'] for a in grounded_atoms]}")
                cutamp_plan, planning_duration, failure_reason = run_planning(
                    env,
                    container.tamp_config,
                    q_init=observation.q_init,
                    ik_solver=container.ik_solver,
                    grasps=processed_scene.grasps,
                    motion_gen=container.motion_gen,
                    all_surfaces=all_surfaces,
                    experiment_dir=save_dir / "cutamp",
                    cost_overrides=container.cost_overrides,
                )
                meta["planning_duration"] = planning_duration
                meta["planning_failure_reason"] = failure_reason
                meta["planning_success"] = cutamp_plan is not None

                if cutamp_plan is None:
                    _log.warning(f"dual: no plan found ({failure_reason})")
                    meta["status"] = "no_plan"
                else:
                    plan_path = save_dir / "tiptop_plan.json"
                    trace_cfg = resolve_trace_cfg(container.cost_overrides)
                    save_tiptop_plan(serialize_plan(cutamp_plan, observation.q_init, trace_cfg=trace_cfg), plan_path)

                    _log.info(f"dual: executing plan ({len(cutamp_plan)} steps)")
                    timeline: list[dict] = []
                    execute_cutamp_dual_plan(cutamp_plan, client=container.robot, timeline=timeline)
                    segments.extend(segments_from_dual_plan(plan_path, timeline))

                    gripper_cmd = (
                        _current_gripper_command(container.robot, "left"),
                        _current_gripper_command(container.robot, "right"),
                    )
                    parking = go_to_dual_home(time_dilation_factor=tdf, motion_gen=container.motion_gen)
                    if parking is not None:
                        segments.append(
                            segment_from_motion(
                                "dual", parking["positions"], parking["velocities"],
                                parking["t_start"], parking["t_end"], gripper_cmd,
                            )
                        )
                    meta["status"] = "executed"
            else:
                _log.info("dual: nothing needs a handover this episode; skipping")
                meta["status"] = "nothing_to_do"

    n_frames = 0
    npz_path = dump_bimanual_episode(
        save_dir,
        segments=[s for s in segments if s is not None],
        joint_samples=sampler.samples,
        instruction=task_instruction,
        cameras=lerobot_cameras,
        q_start=q_start,
        gripper_start=gripper_start,
        fps=LEROBOT_FPS,
        config_id=os.environ.get("TIPTOP_CONFIG_ID"),
        record_start=rec_window.get("t_start"),
        record_stop=rec_window.get("t_stop"),
        arms_used=["dual"] if meta.get("status") == "executed" else [],
    )
    if npz_path is not None:
        n_frames = json.loads((save_dir / "_meta.json").read_text()).get("n_frames", 0)

    return {"n_frames": n_frames, "per_arm": [meta], "planned": plan_path is not None}


def _execute_plan_recorded(
    container: _DemoContainer,
    cutamp_plan: list,
    plan_path: Path,
    save_dir: Path,
    instruction: str,
    *,
    should_stop: Callable[[], bool] | None = None,
    trajectory_id: str | None = None,
) -> tuple[int, bool]:
    """Execute a single-arm cuTAMP plan with the cameras and state samplers running.

    Writes the camera mp4s, ``_gripper_trace.json`` and the data-collection raw episode
    (``robot_state.npz`` + ``_meta.json``, ARCHITECTURE.md §3) into ``save_dir``, and returns
    ``(frame_count, stopped_early)`` -- 0 frames when the dump was skipped (see
    :func:`dump_raw_episode`).

    ``should_stop`` is polled at each step boundary; when it fires the plan stops there and
    ``stopped_early`` comes back True, which is how a "switch to teleop" (SIGUSR1) lands mid-plan
    without tearing the recording down -- the cameras close out around whatever ran. The scene reset
    passes neither this nor ``trajectory_id``: a reset is not an episode, so there is no trajectory
    to join and nothing to hand over.

    ``trajectory_id`` stamps ``_meta.json`` so the legs of one hand-off trajectory can be merged
    afterwards (collect/merge_trajectory.py).

    Shared by the rollout body and the scene reset, which record identically; only where the
    artifacts land differs. Single-arm only: the bimanual episode format is assembled from the
    per-arm segments :func:`_run_one_arm` produces, which neither caller has here.
    """
    # Convert SVO -> MP4 after execution. Depth is disabled during conversion
    # (see convert_svo_to_mp4) so it won't OOM the GPU.
    cameras_to_record = [
        (container.external_cam, save_dir / "external_cam.svo", save_dir / "external_cam.mp4"),
    ]
    if container.external_cam_2 is not None:
        cameras_to_record.append(
            (container.external_cam_2, save_dir / "external_cam_2.svo", save_dir / "external_cam_2.mp4"),
        )
    if isinstance(container.cam, ZedCamera):
        cameras_to_record.append((container.cam, save_dir / "hand_cam.svo", save_dir / "hand_cam.mp4"))

    # Sample the measured arm + gripper state over their own sockets while the cameras record and
    # the plan executes; capture per-step wall-clock times so the export can align camera frames to
    # the control timeline. The samplers are OUTER and record_cameras INNER so the cameras stop the
    # instant execution returns: were the cameras outer, their exit would run while the ~2 s
    # sampler-thread joins finished, padding the video tail with stationary frames past the last
    # state frame.
    exec_timeline: list[dict] = []
    with GripperSampler(container.robot) as gripper_sampler, JointSampler() as joint_sampler:
        with record_cameras(cameras_to_record) as rec_window:
            stopped = bool(
                execute_cutamp_plan(
                    cutamp_plan, client=container.robot, timeline=exec_timeline, should_stop=should_stop
                )
            )

    # Save the raw measured gripper trace (wall_seconds, width_m) so the open<->close shape can be
    # inspected directly (snap vs ramp).
    try:
        (save_dir / "_gripper_trace.json").write_text(json.dumps({"width_samples": gripper_sampler.width_samples}))
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
            instruction=instruction,
            cameras=lerobot_cameras,
            fps=LEROBOT_FPS,
            config_id=os.environ.get("TIPTOP_CONFIG_ID"),
            record_start=rec_window.get("t_start"),
            record_stop=rec_window.get("t_stop"),
            trajectory_id=trajectory_id,
        )
        if raw_path is not None:
            n_frames = json.loads((save_dir / "_meta.json").read_text()).get("n_frames", 0)
    except Exception:
        _log.exception("Failed to dump raw episode")
    return n_frames, stopped


async def _run_scene_reset(
    session: aiohttp.ClientSession, container: _DemoContainer, config: TAMPConfiguration, output_dir: str
) -> None:
    """Put the scene back the way the last rollout found it, reusing the warmed container.

    A finished rollout leaves the scene in its goal state, so the next episode cannot start until the
    toys are off the plate. This runs one extra perceive-plan-execute cycle whose goal is
    ``on(obj, table)`` for every object something else is holding up (``scene_reset``).

    It is deliberately NOT an episode: no cameras are recorded, no state is sampled, nothing is
    labelled and nothing is written under ``eval/``. The artifacts land in
    ``<output_dir>/resets/<ts>/`` instead, which neither the collected-episode count nor the LeRobot
    dataset build looks at, so a reset can never leak into the training data.

    Never raises except on KeyboardInterrupt (a preempt), which the rollout loop turns into "back to
    the task prompt" -- a failed reset must leave the warmed session alive so the operator can retry
    or just move the objects by hand.
    """
    cfg = tiptop_cfg()
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_dir = Path(output_dir) / "resets" / timestamp
    _emit_event({"event": "reset_start", "dir": str(save_dir)})
    file_handler = None
    try:
        file_handler = add_file_handler(save_dir / "scene_reset.log")
        # Its own rerun recording, so the reset's perception does not land on top of the last
        # rollout's; the next rollout's rr.init takes the global recording back.
        rr.init("tiptop_run", recording_id=f"reset_{timestamp}", spawn=False)

        # Whatever the rollout ended holding has to go before we look at the scene, and the arm has
        # to be out of the cameras' way. Both no-op when already so, as at the start of an episode.
        _log.info("Scene reset: returning home and opening the gripper")
        if configured_arms():
            # A reset is a normal single-arm plan, run with whichever arm the config starts on, so
            # the OTHER arm has to be where that arm's collision model believes it is. Homing both
            # first is also what makes the check below pass in the ordinary case.
            home_all_arms(container)
            _assert_idle_arm_parked(container)
        else:
            go_to_home(time_dilation_factor=cfg.robot.time_dilation_factor, motion_gen=container.motion_gen)
        _open_gripper_if_needed(container)
        if container.perception_cam_key == "hand":
            go_to_capture(time_dilation_factor=cfg.robot.time_dilation_factor, motion_gen=container.motion_gen)
        else:
            # The third-person camera already sees the whole table, and q_capture would only put the
            # arm in front of it. Perceive from home instead (see the rollout path for why).
            _log.debug("Scene reset: perception reads the exterior camera; staying at home")

        # The instruction only steers DETECTION -- the goal comes from geometry (scene_reset), so
        # nothing here depends on the VLM naming objects the same way twice. Reusing the last task
        # verbatim is simply what makes it detect the same set of objects.
        instruction = _LAST_TASK
        if not instruction:
            raise RuntimeError("no task has run yet, so there is nothing to reset")
        _log.info(f"Scene reset: perceiving with the last task's instruction {instruction!r}")

        observation = capture_live_observation(container)
        env, all_surfaces, processed_scene, goal_atoms = await run_perception(
            session,
            observation,
            instruction,
            save_dir,
            depth_estimator=container.depth_estimator,
            goal_builder=reset_goal_builder(),
        )
        (save_dir / "reset.json").write_text(
            json.dumps({"instruction": instruction, "goal_atoms": goal_atoms}, indent=2)
        )
        # Saved BEFORE planning: cutamp_env.pkl (world AABBs + grasps) is exactly what you need to
        # tell a genuinely infeasible reset -- an object embedded in a container it has to come out
        # of -- from a tuning problem, and planning is the step that fails in that case.
        save_run_outputs(save_dir, env, processed_scene.grasps)
        if not goal_atoms:
            _log.info("Scene reset: nothing to move, every object is already on the table")
            _emit_event({"event": "reset_done", "dir": str(save_dir), "moved": 0})
            return

        cutamp_plan, moved, skipped = _plan_largest_solvable_reset(
            container, config, processed_scene, observation.q_init, goal_atoms, env, all_surfaces, save_dir
        )
        if cutamp_plan is None:
            raise RuntimeError(f"cuTAMP found no reset plan for any of {sorted(skipped)}")
        plan_path = save_dir / "tiptop_plan.json"
        save_tiptop_plan(serialize_plan(cutamp_plan, observation.q_init), plan_path)

        _log.info(f"Executing scene reset ({len(moved)} object(s) back to the table: {moved})")
        # A reset executes like a rollout, so it records like one: cameras + measured state land in
        # this reset's own directory. That does NOT make it an episode -- resets/ is not one of
        # data-collection's status buckets (server/lib/episodes.js STATUS_DIRS) and neither the
        # collected-episode count nor the LeRobot build looks outside success/ (collect/config.py
        # _count_collected, collect/build_lerobot.py), so a reset still cannot reach training data.
        # Recording failures propagate to the handler below rather than falling back to a bare
        # execute: the plan may already have run, and re-running it would move the objects twice.
        n_frames = 0
        if container.enable_recording and not configured_arms():
            n_frames, _ = _execute_plan_recorded(
                container, cutamp_plan, plan_path, save_dir, f"scene reset: {instruction}"
            )
            _log.info(f"Recorded the reset: {n_frames} state frames + camera mp4s under {save_dir}")
        else:
            if container.enable_recording:
                _log.info("Reset recording is single-arm only (see _execute_plan_recorded); executing unrecorded")
            execute_cutamp_plan(cutamp_plan, client=container.robot)
        go_to_home(time_dilation_factor=cfg.robot.time_dilation_factor, motion_gen=container.motion_gen)
        if skipped:
            _log.warning(f"Scene reset complete, but {skipped} could not be planned -- move those by hand")
        else:
            _log.info("Scene reset complete")
        _emit_event(
            {
                "event": "reset_done",
                "dir": str(save_dir),
                "moved": len(moved),
                "skipped": skipped,
                "n_frames": n_frames,
            }
        )
    except KeyboardInterrupt:
        _log.info("Scene reset preempted (Ctrl-C)")
        _emit_event({"event": "reset_failed", "dir": str(save_dir), "error": "aborted"})
        raise  # the loop's handler clears the preempt latch and returns us to the task prompt
    except UserExitException:
        # run_perception raises this to end the whole session (TIPTOP_DETECT_ONLY). Catching it below
        # as a plain reset failure would keep a session alive that asked to exit.
        _emit_event({"event": "reset_failed", "dir": str(save_dir), "error": "session exiting"})
        raise
    except Exception as e:
        _log.exception(f"Scene reset failed ({type(e).__name__}: {e}); reset the scene by hand and carry on")
        _emit_event({"event": "reset_failed", "dir": str(save_dir), "error": f"{type(e).__name__}: {e}"})
    finally:
        if file_handler is not None:
            remove_file_handler(file_handler)


async def async_entrypoint(container: _DemoContainer, config: TAMPConfiguration, output_dir: str, execute_plan: bool):
    """Main async entrypoint for the live robot demo."""
    cfg = tiptop_cfg()
    arms = configured_arms()
    bimanual = bool(arms)
    dual_handover = cfg.robot.type == "bimanual_yam_dual"
    if bimanual:
        _log.info(f"Bimanual YAM session: arms {arms} (sequential — one cuTAMP plan per arm per episode)")
    elif dual_handover:
        _log.info("Dual-arm YAM session: bimanual_yam_dual (simultaneous — one cuTAMP plan per episode)")

    # Force TCP handshake for every request
    connector = aiohttp.TCPConnector(limit=10, force_close=True)
    async with aiohttp.ClientSession(connector=connector) as session:
        while True:
            try:
                _log.debug("Preparing TiPToP for next run...")
                await check_server_health(session)

                # A teleop switch that was armed somewhere without a checkpoint -- during labeling,
                # post-processing, warmup -- would otherwise sit until the NEXT rollout's first plan
                # step, letting the arm home, re-perceive and execute before handing over. Catch it
                # here, where nothing is in flight, so the hand-off happens when it was asked for.
                if _consume_teleop_request():
                    _log.info("Teleop switch requested between rollouts: handing the arm over now")
                    _run_teleop_handoff(container)
                    continue

                # Get the task BEFORE any pre-trial robot motion so that quitting (or an empty
                # prompt) ends the session without moving to capture + opening the gripper --
                # which would drop whatever is currently held. Reuses the warmed container.
                task_instruction = _get_task_instruction()  # Let UserExitException propagate
                # A robot command from the UI or the prompt: run it against the warm container and go
                # straight back to the prompt -- no rollout, no episode. 'home'/'open' are single
                # nudges; 'reset' is a whole unrecorded perceive-plan-execute cycle, hence async.
                if task_instruction in ROBOT_COMMANDS:
                    if task_instruction == "reset":
                        if dual_handover:
                            # build_reset_goal/_plan_largest_solvable_reset assume a single-arm
                            # pick-and-place domain; handover_tamp_operators has no general
                            # Pick/Place to reset an arbitrary scene with. Refuse loudly rather than
                            # let this hit arm_of("dual") deep inside the single-arm reset path.
                            _log.error(
                                "'reset' is not supported for bimanual_yam_dual yet -- move the "
                                "object(s) back by hand between handover episodes."
                            )
                            _emit_event({
                                "event": "reset_failed", "dir": "", "error": "unsupported in dual mode",
                            })
                        else:
                            await _run_scene_reset(session, container, config, output_dir)
                    else:
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
                # Skipped entirely when resuming from a teleop hand-off: this run is a CONTINUATION
                # of the same task from wherever the operator left the arm, so homing would throw
                # that away and opening the gripper would drop whatever they are holding. Skipped in
                # PART -- the motion but not the gripper -- for a leg following a HITL human phase.
                # _episode_reset_actions decides WHETHER each of the three moves happens; the
                # branches below decide WHICH arms make it, since a bimanual or dual YAM session
                # homes and opens both.
                global _skip_episode_reset, _trajectory_id, _trajectory_handed_off
                global _last_plan_skeleton, _reuse_plan_skeleton, _continue_trajectory
                global _skip_episode_arm_motion
                resuming_from_handoff = _skip_episode_reset
                _skip_episode_reset = False
                # Consumed exactly like the flag above -- one rollout only, so a motion skip armed by
                # a human phase can never leak into a later task typed at the prompt.
                resuming_after_human_phase = _skip_episode_arm_motion
                _skip_episode_arm_motion = False
                continuing_trajectory = _continue_trajectory
                _continue_trajectory = False
                # The task plan to reuse is armed by the hand-off and consumed here, exactly like
                # the reset skip -- one rollout only, so it can never leak into a later task.
                reuse_skeleton = _reuse_plan_skeleton if resuming_from_handoff else None
                _reuse_plan_skeleton = None
                _last_plan_skeleton = None
                # A resumed rollout continues the SAME trajectory as the leg that handed off; any
                # other rollout starts a fresh one. Deliberately keyed on continuing_trajectory rather
                # than on either reset-skip flag: a HITL robot->robot continuation resumes the
                # trajectory while still wanting its arm reset, so the two are not the same question.
                if not continuing_trajectory:
                    _trajectory_id = uuid.uuid4().hex[:16]
                    _trajectory_handed_off = False
                do_home, do_gripper, do_capture = _episode_reset_actions(
                    resuming_from_handoff=resuming_from_handoff,
                    resuming_after_human_phase=resuming_after_human_phase,
                    continuing_trajectory=continuing_trajectory,
                    perception_cam_key=container.perception_cam_key,
                    cam_mount=container.cam_mount,
                )
                if resuming_from_handoff:
                    _log.info(
                        "Resuming after a teleop hand-off: not moving the arm at all -- no return "
                        "home, no gripper open, no move to the capture pose. Perception and planning "
                        "run from exactly where the operator left it"
                    )
                elif do_home:
                    _log.info("Resetting robot for new episode: return home + open gripper (if not already)")
                    if bimanual:
                        # Both arms, because each one's plans collision-check the other at the neutral
                        # posture. Doing this BEFORE the recording starts keeps the episode's first
                        # frames at a settled pose rather than mid-recovery.
                        home_all_arms(container)
                    elif dual_handover:
                        # Both arms move together for a dual session -- there is no "other arm locked as
                        # an obstacle" step to protect (see _run_yam_dual_rollout's docstring).
                        go_to_dual_home(
                            time_dilation_factor=cfg.robot.time_dilation_factor, motion_gen=container.motion_gen
                        )
                    else:
                        go_to_home(time_dilation_factor=cfg.robot.time_dilation_factor, motion_gen=container.motion_gen)
                elif resuming_after_human_phase:
                    # A leg resuming after a HITL human phase: the arm stays put like a hand-off
                    # resume, but the gripper check below still runs. See _episode_reset_actions.
                    _log.info(
                        "Resuming after a HITL human phase: not homing the arm and not moving to the "
                        "capture pose -- this phase is a continuation of the same trajectory and "
                        "plans from the pose the human left. Only the gripper check runs"
                    )
                else:
                    # A robot->robot continuation of the same trajectory. Same reasoning: the arm is
                    # already parked at the previous leg's retract, and homing between two legs of one
                    # episode is a return to home recorded into the middle of the demonstration.
                    _log.info(
                        "Continuing the same trajectory: not homing the arm -- this leg plans from "
                        "where the previous one stopped. The gripper check still runs"
                    )
                if do_gripper:
                    if bimanual:
                        for _arm in arms:
                            with active_arm(_arm):
                                try:
                                    _open_gripper_if_needed(container)
                                except Exception as _e:
                                    _log.exception(f"Gripper open/check failed for the {_arm} arm: {_e}")
                    elif dual_handover:
                        for _arm in ("left", "right"):
                            try:
                                _open_gripper_if_needed(container, arm=_arm)
                            except Exception as _e:
                                _log.exception(f"Gripper open/check failed for the {_arm} arm: {_e}")
                    else:
                        try:
                            _open_gripper_if_needed(container)
                        except Exception as _e:
                            _log.exception('Gripper open/check failed: ' + str(_e))

                if do_capture:
                    # Perception reads a camera ON THE WRIST, so the arm goes to q_capture to point
                    # it at the scene.
                    _log.debug("Moving robot to capture joint positions")
                    go_to_capture(time_dilation_factor=cfg.robot.time_dilation_factor, motion_gen=container.motion_gen)
                elif container.perception_cam_key == "hand" and container.cam_mount == "ee":
                    # Wrist perception, but the arm is deliberately left where the last leg or the
                    # human put it. What q_capture buys is a guaranteed view of the whole table; from
                    # a hand-off pose the camera sees only what it happens to point at, so perception
                    # can come back partial (or fail outright, in which case the rollout is abandoned
                    # and the session drops back to the prompt).
                    _log.debug("Continuing a trajectory; leaving the wrist camera where it is")
                else:
                    # A camera fixed in the scene (the side-view ZED, the YAM's top D435) already sees
                    # the whole table, and q_capture would only put the arm in front of it -- an arm in
                    # frame ends up in the point cloud, the RANSAC table fit and the grasps. It is
                    # masked out by its collision spheres in the exterior case, but a mask is not free:
                    # it erases whatever it occludes. Perceive from wherever the arm already is, which
                    # after a full reset is home -- further out of the shot. When it is a hand-off pose
                    # instead, the arm may be in frame regardless; that is the cost of continuing.
                    _log.debug("Perception reads a camera fixed in the scene; not moving to q_capture")

                # Set once a "switch to teleop" has been observed at one of this rollout's
                # checkpoints (see _sigusr1_teleop_switch); drives the hand-off after the rollout
                # has closed itself out.
                handoff_pending = False

                # Set when this rollout finished a HITL robot leg and the plan's NEXT phase is STILL
                # the robot's, so the loop carries straight on with it instead of closing the
                # trajectory out. Consecutive robot phases are normally coalesced into this leg's
                # single goal (HITLSession.robot_run), so what reaches here is the case that cannot
                # be: a phase moving an object this leg already moved, which one cuTAMP plan cannot
                # express (Pick deletes HasNotPickedUp).
                #
                # Deliberately NOT handoff_pending: that flag means "a human is taking the arm", and
                # the branch it drives releases the robot connection and every camera and then blocks
                # on stdin for a 'resume' nobody was asked to send. This continuation hands nothing
                # over.
                next_robot_phase = False

                # Filled in by the HITL goal resolver during perception (empty when HITL is off):
                # "session" is the task's plan, "human" the phase to hand over when the plan's next
                # phase is a person's rather than robot work.
                hitl_state: dict = {}

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

                # Every image sent to the VLM this rollout, and what it answered, land in
                # <save_dir>/vlm. Armed here rather than inside the HITL code so the verification
                # queries -- which run after the rollout has closed itself out -- are still filed
                # against the rollout that asked for them.
                _hitl_arm_recorder(save_dir)

                # Add log file handler for this run
                file_handler = add_file_handler(save_dir / "tiptop_run.log")
                # Record the resolved cuRobo override config INTO this rollout's log (the warmup-time
                # "RESOLVED cuRobo cost" line predates this handler, so it never lands on disk). Also
                # drop a curobo_config.json alongside it so applied overrides are auditable per episode.
                _resolved = container.curobo_config_summary or {}
                _log.info(f"cuRobo config for this rollout: {json.dumps(_resolved)}")
                (save_dir / "curobo_config.json").write_text(json.dumps(_resolved, indent=2))
                if bimanual:
                    # Sequential-bimanual rollout: each arm perceives, plans and executes its own
                    # share of the goal in turn. It owns the whole recorded window (cameras + the
                    # 14-D state sampler), so it stands apart from the single-arm body below rather
                    # than being threaded through it with per-arm conditionals.
                    try:
                        result = await _run_yam_rollout(session, container, save_dir, task_instruction)
                        _emit_event({"event": "rollout_saved", "dir": str(save_dir), "n_frames": result["n_frames"]})
                        save_run_metadata(
                            save_dir=save_dir,
                            timestamp=iso_timestamp,
                            task_instruction=task_instruction,
                            q_at_capture=result["per_arm"][0].get("q_at_capture") if result["per_arm"] else None,
                            world_from_cam=result["per_arm"][0].get("world_from_cam") if result["per_arm"] else None,
                            perception_duration=sum(m.get("perception_duration") or 0.0 for m in result["per_arm"]),
                            grounded_atoms=[a for m in result["per_arm"] for a in (m.get("grounded_atoms") or [])],
                            planning_success=result["planned"],
                            planning_failure_reason="; ".join(
                                f"{m['arm']}: {m['planning_failure_reason']}"
                                for m in result["per_arm"]
                                if m.get("planning_failure_reason")
                            )
                            or None,
                            planning_duration=sum(m.get("planning_duration") or 0.0 for m in result["per_arm"]),
                        )
                        (save_dir / "bimanual_summary.json").write_text(
                            json.dumps({"arms": arms, "per_arm": result["per_arm"]}, indent=2, cls=NumpyEncoder)
                        )
                        _log.info(
                            f"Bimanual rollout finished: {[(m['arm'], m.get('status')) for m in result['per_arm']]}"
                        )
                    except Exception:
                        _log.exception("Bimanual rollout failed")
                        raise
                    finally:
                        remove_file_handler(file_handler)
                    if execute_plan:
                        final_dir = _label_rollout(save_dir, output_dir, timestamp, container.enable_recording)
                        # None when the rollout executed nothing and was discarded (_label_rollout).
                        if final_dir is not None:
                            _spawn_postprocess(final_dir)
                    continue

                if dual_handover:
                    # Simultaneous dual-arm/handover rollout: one perceive-plan-execute cycle
                    # against the 12-DOF chain, not a per-arm loop. Owns the whole recorded window
                    # the same way the sequential-bimanual branch above does.
                    try:
                        result = await _run_yam_dual_rollout(session, container, save_dir, task_instruction)
                        _emit_event({"event": "rollout_saved", "dir": str(save_dir), "n_frames": result["n_frames"]})
                        meta0 = result["per_arm"][0] if result["per_arm"] else {}
                        save_run_metadata(
                            save_dir=save_dir,
                            timestamp=iso_timestamp,
                            task_instruction=task_instruction,
                            q_at_capture=meta0.get("q_at_capture"),
                            world_from_cam=meta0.get("world_from_cam"),
                            perception_duration=meta0.get("perception_duration") or 0.0,
                            grounded_atoms=meta0.get("grounded_atoms") or [],
                            planning_success=result["planned"],
                            planning_failure_reason=meta0.get("planning_failure_reason"),
                            planning_duration=meta0.get("planning_duration") or 0.0,
                        )
                        (save_dir / "dual_summary.json").write_text(
                            json.dumps({"per_arm": result["per_arm"]}, indent=2, cls=NumpyEncoder)
                        )
                        _log.info(f"Dual-arm rollout finished: status={meta0.get('status')}")
                    except Exception:
                        _log.exception("Dual-arm rollout failed")
                        raise
                    finally:
                        remove_file_handler(file_handler)
                    if execute_plan:
                        final_dir = _label_rollout(save_dir, output_dir, timestamp, container.enable_recording)
                        # None when the rollout executed nothing and was discarded (_label_rollout).
                        if final_dir is not None:
                            _spawn_postprocess(final_dir)
                    continue

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
                        resolve_goal=(
                            _hitl_goal_resolver(task_instruction, observation.frame.rgb, hitl_state)
                            if _hitl_enabled()
                            else None
                        ),
                        detect_hint=_hitl_detect_hint(task_instruction),
                    )
                    perception_duration = time.perf_counter() - perception_start

                    cutamp_plan = None
                    planning_duration = None
                    failure_reason = None
                    cleared: list[str] = []
                    if os.environ.get("TIPTOP_DRY_RUN"):
                        _log.info("PATCH: TIPTOP_DRY_RUN=1 -> skipping planning/execute (perception-only)")
                        failure_reason = "dry_run"
                    else:
                        pass
                    try:
                        if os.environ.get("TIPTOP_DRY_RUN"):
                            raise RuntimeError("dry_run skip")
                        _log.info("Running Planning...")
                        plan_out: dict = {}
                        # An out-of-reach goal object is unplannable, so don't spend a cuTAMP pass
                        # discovering that. Reported as an ordinary planning failure rather than
                        # raised: the rollout keeps going down the "no plan found" path and still
                        # reaches the label prompt, which is what merges a handed-off trajectory.
                        unreachable = unreachable_goal_movable(env, grounded_atoms)
                        planned = False
                        if hitl_state.get("failure"):
                            # The proposal or the outer plan did not come off. Reported as an
                            # ordinary planning failure, so the rollout still reaches the label
                            # prompt rather than unwinding past it.
                            _log.error(f"Not planning: {hitl_state['failure']}")
                            cutamp_plan, planning_duration = None, 0.0
                            failure_reason = hitl_state["failure"]
                            _hitl_reset_session("the plan could not be built")
                        elif hitl_state.get("human") is not None:
                            # The plan's next phase is the human's, not the robot's. There is no goal
                            # for cuTAMP to solve, so skip it -- reported the same way an unplannable
                            # goal is, so this rollout still walks the usual path to the hand-off.
                            _log.info("HITL: this phase is the human's; not running cuTAMP")
                            cutamp_plan, planning_duration, failure_reason = None, 0.0, "awaiting a human step"
                        elif unreachable:
                            _log.error(f"Not planning: {unreachable}")
                            cutamp_plan, planning_duration, failure_reason = None, 0.0, unreachable
                        else:
                            planned = True
                            # cuTAMP ends a plan by driving the arm back to q_home. That is right for
                            # a plan that IS the episode, and wrong for one that is a leg of a longer
                            # one: a HITL task with phases still to come continues from where this leg
                            # stops, so a return home in between is motion the task never asked for,
                            # recorded into the middle of the demonstration -- and the next leg then
                            # plans from home instead of from the pose it was handed. Ask for it only
                            # on the last leg; a non-HITL rollout has no session and always gets it.
                            hitl_session_now = hitl_state.get("session")
                            plan_return_home = hitl_session_now is None or hitl_session_now.is_last_leg()
                            if not plan_return_home:
                                _log.info(
                                    "HITL: more phases follow this one, so the plan will stop at the "
                                    "retract instead of driving the arm home"
                                )
                            # `clear_goal_surfaces` in cfg/tamp tamp_overrides opts into planning the
                            # blockers off the goal surfaces first and concatenating that plan with the
                            # task's, so both run inside this one recorded episode. Off -> exactly the
                            # single run_planning call this has always made. See tiptop.goal_clearing.
                            #
                            # The clearing path carries `return_home` (a non-final HITL leg must still
                            # stop at the retract) but NOT the skeleton reuse: it rebuilds its own
                            # environment from geometry and issues two cuTAMP calls, so there is no one
                            # skeleton to hand back. `plan_out` therefore stays empty and a later
                            # hand-off falls back to a fresh search -- the same graceful degradation
                            # run_planning already applies when it rejects a stale skeleton. It also
                            # does not carry HITL's surface/movable restrictions. No cfg/tamp config
                            # sets clear_goal_surfaces and hitl.enabled together today; this is the
                            # place to wire the rest through if one ever does.
                            if resolve_clear_goal_surfaces(container.cost_overrides):
                                cutamp_plan, planning_duration, failure_reason, cleared = plan_clear_then_task(
                                    config,
                                    processed_scene,
                                    observation.q_init,
                                    grounded_atoms,
                                    save_dir,
                                    ik_solver=container.ik_solver,
                                    motion_gen=container.motion_gen,
                                    cost_overrides=container.cost_overrides,
                                    return_home=plan_return_home,
                                )
                            else:
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
                                    reuse_plan_skeleton=reuse_skeleton,
                                    plan_out=plan_out,
                                    return_home=plan_return_home,
                                )
                        # Remember this rollout's task plan in case it hands off to teleop, and tell
                        # the UI whether the one we were given was actually reused (run_planning
                        # rejects a stale plan, and falls back to a full search if it yields nothing).
                        # Skipped planning above -> nothing to remember and nothing to report: the
                        # task plan we were handed was neither reused nor replaced, so leave it in
                        # place for the next attempt rather than clearing it.
                        if planned:
                            _last_plan_skeleton = plan_out.get("plan_skeleton")
                        if reuse_skeleton is not None and planned:
                            reused = bool(plan_out.get("reused"))
                            plan_str = ", ".join(op.name for op in reuse_skeleton)
                            # `message` is what puts this in the operator's session log: the server
                            # mirrors any event carrying one, and has no case for this event itself.
                            _emit_event(
                                {
                                    "event": "task_plan_reuse",
                                    "reused": reused,
                                    "plan": [op.name for op in reuse_skeleton],
                                    "message": (
                                        f"replanned the motion for the task plan from before the hand-off ({plan_str})"
                                        if reused
                                        else f"could not keep the task plan from before the hand-off ({plan_str}); "
                                        "planned the task from scratch — see the log above for why"
                                    ),
                                }
                            )
                        _log.info(f"Perception and cuTAMP planning took: {perception_duration + planning_duration:.2f}s")
                        if cutamp_plan is not None:
                            plan_path = save_dir / "tiptop_plan.json"
                            trace_cfg = resolve_trace_cfg(container.cost_overrides)
                            save_tiptop_plan(
                                serialize_plan(cutamp_plan, observation.q_init, trace_cfg=trace_cfg), plan_path
                            )
                            _log.info(f"Saved TiPToP plan to {plan_path}")

                        # A "switch to teleop" that arrived during perception/planning: hand over
                        # here rather than starting a plan we are about to interrupt anyway. The arm
                        # has not moved since perception, so there is nothing to finish.
                        if _consume_teleop_request():
                            _log.info("Teleop switch requested during perception/planning: not starting this plan")
                            handoff_pending = True

                        if cutamp_plan is not None and execute_plan and not handoff_pending:
                            _log.info("Executing plan...")
                            # Execute with optional recording
                            if container.enable_recording:
                                # Recorded execution is shared with the scene reset
                                # (_execute_plan_recorded); only the artifacts' destination differs.
                                # should_stop is what lets a "switch to teleop" land mid-plan: the
                                # plan stops at the first step boundary after the button is pressed,
                                # and the recording closes out around whatever ran.
                                n_frames, handoff_pending = _execute_plan_recorded(
                                    container,
                                    cutamp_plan,
                                    plan_path,
                                    save_dir,
                                    # _meta.json's instruction is the LANGUAGE LABEL the dataset
                                    # trains on, which is not always the goal we planned: a task
                                    # whose teleop leg does something the on()/holding() goal
                                    # language cannot express is launched with a reduced
                                    # TIPTOP_TASK while TIPTOP_INSTRUCTION still describes the
                                    # whole demonstration. Absent (tiptop-run started directly),
                                    # they are the same thing.
                                    os.environ.get("TIPTOP_INSTRUCTION") or task_instruction,
                                    should_stop=_consume_teleop_request,
                                    trajectory_id=_trajectory_id,
                                )
                                _emit_event({"event": "rollout_saved", "dir": str(save_dir), "n_frames": n_frames})
                            else:
                                handoff_pending = execute_cutamp_plan(
                                    cutamp_plan, client=container.robot, should_stop=_consume_teleop_request
                                )
                            _log.info(
                                "Stopped the plan for a teleop hand-off" if handoff_pending
                                else "Finished executing plan!"
                            )
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
                            cleared=cleared,
                        )
                        _log.info(f"Logs, results, and visualizations saved to {save_dir}")

                    # HITL bookkeeping, outside the rollout's own try/finally but before the label
                    # decision: a task whose next step is a human's must NOT be labeled or exported
                    # here, because this rollout is one leg of it rather than a finished episode.
                    # Setting handoff_pending is what routes it down the branch that already knows
                    # that (the same one an operator-initiated hand-off takes).
                    hitl_session = hitl_state.get("session")
                    if hitl_session is not None:
                        human_phase = hitl_state.get("human")
                        if human_phase is None and execute_plan and cutamp_plan is not None and not handoff_pending:
                            # This rollout executed a robot phase to completion. Record the task plan
                            # cuTAMP found for it BEFORE advancing: the phase only said what had to
                            # be established, cuTAMP decided how.
                            hitl_session.record_tamp_plan(hitl_session.index, plan_out)
                            hitl_session.advance()
                            nxt = hitl_session.current
                            human_phase = nxt if (nxt is not None and nxt.is_human) else None
                            # ... and if the next phase is STILL the robot's -- a phase advance()
                            # could not fold into this leg's goal -- this leg is not the end of
                            # anything either: the loop carries on with it below. False when nxt is
                            # None, so the last phase of an all-robot plan still falls through to
                            # the label prompt and still ends the session.
                            next_robot_phase = nxt is not None and not nxt.is_human
                        hitl_state["human"] = human_phase
                        if human_phase is not None:
                            handoff_pending = True
                        (save_dir / "hitl.json").write_text(json.dumps(hitl_session.to_json(), indent=2))
                        if human_phase is None and hitl_session.finished:
                            _hitl_reset_session("every phase of the task is done")

                    if execute_plan and (handoff_pending or next_robot_phase):
                        # Neither of these is a finished task attempt, so neither is labeled here:
                        # the label is what ENDS a trajectory and merges its legs, and rating a leg
                        # that is only part of the work would end it early. Post-processing is
                        # skipped for the same reason -- this directory is one leg, and exporting it
                        # alone would produce a fragment.
                        #
                        #   handoff_pending  -- the arm is about to go to a human, and the label
                        #                       prompt would block the hand-off they are waiting on.
                        #   next_robot_phase -- the HITL plan's next phase is the robot's too, so
                        #                       the loop replans and executes it below.
                        _log.info(
                            f"HITL: robot phase done and another robot phase follows; leaving this "
                            f"leg unlabeled in {save_dir}"
                            if next_robot_phase
                            else f"Teleop hand-off: leaving the partial rollout unlabeled in {save_dir}"
                        )
                    elif execute_plan:
                        final_dir = _label_rollout(save_dir, output_dir, timestamp, container.enable_recording)
                        # Post-process this rollout (gifs + LeRobot export) in the background so
                        # the next rollout can start immediately instead of blocking on it. Skipped
                        # for a multi-leg trajectory: this dir is only its LAST leg, and exporting
                        # it would produce a fragment. merge_trajectory.py joins the legs first.
                        # final_dir is None when the rollout executed nothing and was discarded.
                        if final_dir is not None and not _trajectory_handed_off:
                            _spawn_postprocess(final_dir)
                        # PATCH (cortex v3): DO NOT auto-open the gripper after Pick.
                        # The original tiptop demo opened the gripper post-pick for
                        # standalone "did the grasp work?" tests. For cortex we WANT
                        # to keep the object held so Haiku can decide whether to Place
                        # next. Removing the open_gripper() call here.
                except TeleopHandoffRequested:
                    raise  # a hand-off from the label prompt, not a failure
                except Exception:
                    _log.exception("TiPToP run failed")
                    raise
                finally:
                    # Always remove the file handler after the run
                    remove_file_handler(file_handler)

                # Outside the rollout's try/finally: everything this rollout owned (recording,
                # log handler) is closed, so the arm can be handed to the operator. Blocks until
                # they hand it back, then loops round and replans the same task from there.
                #
                # Unless they hand it back with "finish": the demonstration completed the task, so
                # this leg is labeled here instead. That is the ONLY thing the trajectory still
                # needs -- `labeled` is what ends it, and the server merges the tamp and teleop legs
                # on that event (see sessions.js _trackTrajectory). No _spawn_postprocess for the
                # same reason the normal path skips it on a handed-off trajectory: this dir is one
                # leg, and the merge post-processes the joined episode.
                if handoff_pending:
                    human_phase = hitl_state.get("human")
                    if human_phase is not None:
                        # A planner-initiated hand-off: the human is told WHAT to do and what will be
                        # checked afterwards, instead of being handed a bare arm. The hand-off itself
                        # is still the operator's to start (the same "Switch to teleop" button), so
                        # nothing takes the arm out from under them.
                        global _pending_instruction
                        finished = await _hitl_human_phase(container, human_phase, can_finish=True)
                        if finished:
                            # The task is over. _run_teleop_handoff queues the same instruction for
                            # the next iteration on a plain "resume", which would start the WHOLE
                            # task again from a fresh proposal; drop it and go back to the prompt.
                            _pending_instruction = None
                        # The next phase has a different goal, so the plan this rollout followed is
                        # the wrong one to resume with -- and _run_teleop_handoff queues it
                        # unconditionally. Left set, skeleton_reuse_rejection accepts it whenever the
                        # next segment's goal is a subset of what it already achieves (HandEmpty
                        # alone, say), and the arm silently repeats the chunk it has just done.
                        _reuse_plan_skeleton = None
                        # The reset, however, is only HALF dropped. _run_teleop_handoff armed the
                        # full skip (no homing, no gripper open); a human phase wants the gripper
                        # check back -- nothing about a person driving the arm guarantees the fingers
                        # are open, and cuTAMP plans the next phase from a HandEmpty initial state
                        # either way -- but it does NOT want the arm driven home first. Homing is a
                        # continuation of the same trajectory thrown away: the next phase should plan
                        # from the pose the human left, exactly as an operator hand-off does. (This
                        # once set the full reset, on the reasoning that the capture pose was needed
                        # "without which perception looks wherever the operator happened to leave the
                        # wrist" -- which does not apply under `cameras.perception: external`, the
                        # deployed setting, where the arm never drives to q_capture at all.)
                        # _continue_trajectory is deliberately LEFT set: this is still the same
                        # trajectory, and clearing both (as one flag once did) minted a new
                        # trajectory id, which dropped the plan mid-task and split the episode.
                        # When the plan is FINISHED it is stale rather than wrong to leave -- the
                        # task prompt disarms it (_disarm_rollout_carryover), which is what stops the
                        # next episode inheriting this one's trajectory and skipping its homing.
                        _skip_episode_reset = False
                        # Armed only when the loop goes STRAIGHT back into this task, which is
                        # exactly what a queued _pending_instruction means (_get_task_instruction
                        # consumes it without blocking on stdin). The narrow condition matters: three
                        # ways out of _hitl_human_phase reach here without one -- the task finished,
                        # the operator abandoned the phase, or they did it by hand and never pressed
                        # "Switch to teleop", so _run_teleop_handoff never queued anything. All three
                        # drop to the bare prompt, and a motion skip left standing there would
                        # suppress the homing of the NEXT, unrelated task typed into it. This mirrors
                        # _skip_episode_reset's own arming, which is likewise inside the branch that
                        # queues the instruction.
                        _skip_episode_arm_motion = not finished and _pending_instruction is not None
                        # Re-write the audit record: the verification verdicts only exist now, and
                        # for the LAST human step of a task there is no later rollout to record them.
                        # Read from hitl_state, NOT the module global: completing the task drops the
                        # session, and guarding on the global here meant the one rollout whose record
                        # mattered most -- the finished task, with its verdicts -- kept the stale copy
                        # written before the human acted.
                        record_session = hitl_state.get("session")
                        if record_session is not None:
                            (save_dir / "hitl.json").write_text(json.dumps(record_session.to_json(), indent=2))
                    else:
                        finished = _run_teleop_handoff(container, can_finish=True)
                    if finished:
                        _label_rollout(save_dir, output_dir, timestamp, container.enable_recording)
                    continue

                # The HITL plan's next phase is the robot's too. A sibling `if`, not an `elif`: the
                # branch above always `continue`s, and the two are different kinds of continuation --
                # that one waits on a person, this one just goes round again. Nothing is handed over,
                # so _run_teleop_handoff is deliberately NOT called: it would release the robot
                # connection and every camera and then block on stdin for a resume nobody sent.
                #
                # Placed AFTER the rollout's own try/finally so a leg that threw arms nothing: the
                # next rollout then mints a fresh trajectory id, HITLSession.matches() fails, and the
                # half-walked plan is correctly dropped rather than resumed against a stale scene.
                if next_robot_phase:
                    # Re-enter the loop without prompting the operator -- and so without emitting
                    # `awaiting_task`, the one state in which the data-collection server would let
                    # someone home the arm or start another rollout midway through this trajectory.
                    # task_instruction, not _LAST_TASK: this is the exact string HITLSession.matches()
                    # compares against, and a nudge ('home'/'open') never reaches here anyway.
                    _pending_instruction = task_instruction
                    # Same trajectory, so the id survives (see the guard at the top of the loop) and
                    # the session is still recognised as the plan we are midway through.
                    _continue_trajectory = True
                    # Several legs, so the export waits for collect/merge_trajectory.py rather than
                    # exporting this one on its own -- which would be a fragment of the attempt.
                    _trajectory_handed_off = True
                    # _skip_episode_reset is deliberately NOT set: the robot holds nothing after a
                    # Place, so the next phase still runs its gripper check (and, under wrist
                    # perception, still drives to the capture pose). What it does NOT do any more is
                    # home first -- _continue_trajectory above is what tells _episode_reset_actions
                    # this leg continues an episode already in progress, and homing between two legs
                    # of one episode records a return to home into the middle of the demonstration.
                    # Leaving _skip_episode_reset clear also leaves reuse_skeleton unset for the next
                    # rollout, which is right -- the next phase has a different goal, and a stale
                    # skeleton is accepted whenever the new goal is a subset of what it already
                    # achieves, so the arm would silently repeat the chunk it has just done.
                    _emit_event({
                        "event": "hitl_phase_complete",
                        "trajectory_id": _trajectory_id,
                        "dir": str(save_dir),
                        "phase_index": hitl_session.index,
                        "n_phases": len(hitl_session.phases),
                        "message": (
                            f"HITL phase {hitl_session.index} of {len(hitl_session.phases)} done; "
                            f"the robot carries on with the next one in the same trajectory"
                        ),
                    })
                    continue
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
                # A half-walked HITL plan must not survive a preempt: the operator's next Enter
                # repeats the task, and resuming at segment 2 of a plan whose segment 1 was aborted
                # would run the wrong work against the wrong scene.
                _hitl_reset_session("the rollout was preempted")
                # Unwind is done (the finally-blocks above ran as the exception propagated), so a
                # new Ctrl-C should preempt the next rollout rather than be swallowed.
                _clear_preempt()
                # A hand-off had been requested but was preempted before it reached a checkpoint
                # (operator pressed Switch to teleop, then Preempt). The arm is stopped and the
                # rollout is unwound, which is exactly the hand-off precondition -- honour it here
                # rather than leaving the request armed to fire mid-way through the NEXT rollout.
                if _consume_teleop_request():
                    _run_teleop_handoff(container)
                continue
            except TeleopHandoffRequested:
                # SIGUSR1 raised out of a stdin prompt (task or label): nothing is in flight, the
                # arm is idle between rollouts, so hand it over directly.
                _consume_teleop_request()
                _log.info("Teleop switch requested at the prompt: handing the arm over from here")
                _run_teleop_handoff(container)
                continue
            except Exception as e:
                # A single rollout failing (a transient Gemini/perception 503, a planning
                # error, a health-check blip, ...) must NOT tear down the warmed session --
                # otherwise "collect another" would lose the whole warmed container and force
                # a full re-warm. Log it (the traceback streams to the data-collection UI),
                # then loop back to the task prompt so the user can just retry.
                _log.exception(
                    f"Rollout failed ({type(e).__name__}: {e}); keeping session warm, returning to task prompt"
                )
                # Same reasoning as the preempt path: a plan cannot be resumed halfway through when
                # the rollout that was meant to advance it unwound instead.
                _hitl_reset_session("the rollout failed")
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
    hitl_config: str | None = None,
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
        hitl_config: the cfg/tamp/*.yml `hitl` block as a JSON file path OR inline JSON. With
            `enabled: true`, a goal TiPToP cannot express (folding a cloth) is planned as robot work
            plus human steps, and each human step is handed over as a teleop leg -- see tiptop.hitl.
            Absent or disabled, the run behaves exactly as it always has.
    """
    assert max_planning_time > 0
    assert opt_steps_per_skeleton > 0
    assert num_particles > 0

    print_tiptop_banner()
    check_cutamp_version()
    _emit_event({"event": "session_start"})

    # Lazy import breaks the tiptop_run <-> tiptop_websocket_server import cycle.
    from tiptop.tiptop_websocket_server import _load_curobo_overrides

    cost_overrides = _load_curobo_overrides(curobo_overrides)
    # Same file-or-inline-JSON spelling as curobo_overrides. Imported here rather than at module
    # scope so that an ordinary session never loads the package at all.
    global _hitl_cfg
    if hitl_config:
        from tiptop.hitl.config import load_hitl_config

        _hitl_cfg = load_hitl_config(hitl_config)
        if _hitl_cfg.enabled:
            _log.info(
                f"Human-in-the-loop planning is ON (proposal: {_hitl_cfg.proposal_model}, "
                f"grounding: {_hitl_cfg.vlm_model})"
            )
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
    # Perception knobs from the same tamp_overrides dict. Applied HERE -- before the entrypoint runs
    # any perception -- because they change the grasp candidates cuTAMP is given, and no downstream
    # cost can select a grasp perception never handed over. See apply_perception_overrides.
    for _key, (_old, _new) in apply_perception_overrides(cfg, cost_overrides).items():
        _log.info(f"Perception override: {_key} {_old} -> {_new}")
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

    # One TAMPConfiguration per planning embodiment. A sequential-bimanual YAM episode plans with
    # both `bimanual_yam_left` and `bimanual_yam_right`, and TAMPConfiguration carries `robot`;
    # every other robot yields a single entry, identical to before. `arm_mode`/`dual_task` only
    # apply to the `bimanual_yam_dual` embodiment (_planning_robot_types() yields exactly one entry
    # for it), and default to cuTAMP's own single-arm defaults everywhere else.
    apex_height, apex_min_dist = resolve_transit_apex(cost_overrides)
    if apex_height > 0:
        _log.info(f"Transit apex active: {apex_height}m (min transit distance {apex_min_dist}m)")
    posture_selection = resolve_posture_selection(cost_overrides)
    if posture_selection.get("posture_selection_seeds", 0) > 1:
        _log.info(
            "Teleop-posture IK branch selection active: %s seeds | prior: %s",
            posture_selection["posture_selection_seeds"],
            posture_ref_summary(posture_selection.get("posture_ref")),
        )
    tamp_configs = {
        robot_type: build_tamp_config(
            num_particles=num_particles,
            max_planning_time=max_planning_time,
            opt_steps=opt_steps_per_skeleton,
            robot_type=robot_type,
            time_dilation_factor=time_dilation_factor,
            collision_activation_distance=0.0,
            enable_visualizer=cutamp_visualize,
            # move-cost norm for cuTAMP (Euclidean unless a cfg/tamp yml opts into "inf"), same as
            # time_dilation_factor this is a TAMP-config knob, not a cuRobo cost weight.
            traj_length_norm=resolve_traj_length_norm(cost_overrides),
            grasp_orientation_cost=resolve_grasp_orientation_cost(cost_overrides),
            grasp_center_cost=resolve_grasp_center_cost(cost_overrides),
            arm_mode=cfg.robot.get("arm_mode", "single"),
            dual_task=cfg.robot.get("dual_task", "parallel"),
            max_motion_refine_attempts=resolve_max_motion_refine_attempts(cost_overrides),
            # Apex waypoint in each Pick/Place transit (0 = off); a TAMP-config knob, not a
            # cuRobo cost weight. See resolve_transit_apex.
            transit_apex_height=apex_height,
            transit_apex_min_dist=apex_min_dist,
            # IK branch selection by teleop posture (off unless the cfg sets
            # posture_selection_seeds). See resolve_posture_selection.
            posture_selection=posture_selection,
            # Where a plan's closing GoToInitial drives to, per RUN. A per-CALL q_return
            # (plan_clear_then_task's second half) overrides it; run_planning's return_home=False
            # suppresses the drive altogether for a non-final HITL leg. All three coexist in cuTAMP.
            q_home=cfg.robot.q_home,
            # Fail instead of silently substituting collision-sphere heuristic grasps for an object
            # perception proposed nothing for (off unless the cfg sets it). See
            # resolve_require_m2t2_grasps.
            require_m2t2_grasps=resolve_require_m2t2_grasps(cost_overrides),
        )
        for robot_type in _planning_robot_types()
    }
    # The scene-reset path takes a single config; it runs under whichever arm is active, and the
    # first entry is the arm the config starts on.
    config = tamp_configs[_planning_robot_types()[0]]

    global _executor_pool
    setup_logging(level=logging.DEBUG)

    container = get_demo_container(
        num_particles,
        config.coll_n_spheres,
        0.0,
        enable_recording,
        cost_overrides,
        curobo_config_summary,
        tamp_configs=tamp_configs,
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
    # SIGUSR1 is the "switch to teleop" trigger from the data-collection UI's button. Unlike SIGINT
    # it aborts nothing: it arms a request that the rollout honours at its next step boundary, then
    # hands the robot + cameras to a teleop process and waits for them back. See
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
            container.cam.close()
            if container.external_cam is not None:
                container.external_cam.close()
            if container.external_cam_2 is not None:
                container.external_cam_2.close()
            container.robot.close()
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

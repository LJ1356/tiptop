import logging
import os
import threading
import time
from collections.abc import Callable

from tiptop.utils import RobotClient, get_robot_client

_log = logging.getLogger(__name__)

# --- Gripper/arm overlap ---------------------------------------------------------------------
# A TAMP plan issues each gripper open/close as its own step, sandwiched between two arm
# trajectories (arrive -> actuate -> depart). Executing that step and waiting for the gripper to
# finish parks the arm for the full ~0.5-1 s actuation, which the LeRobot export records as a run
# of zero-velocity frames. The DROID-style non-idle training filter then drops any idle
# joint-velocity run of >= 7 frames -- and the gripper open/close timestep sits right inside it,
# so it is filtered out.
#
# We overlap the actuation with the FOLLOWING trajectory (the lift after a close, the retreat
# after an open) so the arm keeps moving across the transition. On a Robotiq the gripper talks to
# the gripper server on its OWN socket, separate from the arm control socket, so we run the command
# on a background thread and start the next trajectory concurrently on the control socket; the
# transition then lands inside a moving (non-idle) segment and survives the filter.
#
# The threaded command MUST be issued blocking=False + polled client-side (see _spawn_gripper_command
# and _wait_for_gripper_settled), NOT as a server-side blocking command. The gripper server handles
# requests on a single thread: a blocking command occupies it for the whole actuation, starving the
# LeRobot gripper sampler's state polls so the recorded open<->close ramp collapses to an instant
# jump. A background thread frees only the *client*, not the server; the client-side poll frees the
# server socket between reads. (This was the "gripper obs jumps" regression -- see git history.)
#
# Only overlapped when the gripper has a separate socket (Robotiq). A Franka hand routes gripper
# commands through the control socket, so a thread would race the arm trajectory -- there we fall
# back to the blocking path. Set TIPTOP_GRIPPER_OVERLAP=0 to force the blocking hold everywhere.
GRIPPER_OVERLAP = os.environ.get("TIPTOP_GRIPPER_OVERLAP", "1") != "0"

# Seconds to stay parked after firing a CLOSE before the arm departs, so the fingers seat on the
# object before the lift begins. This stationary settle is the only idle window left, so keep it
# comfortably below min_idle_len / fps (~7/15 = 0.47 s) or it becomes a filtered idle run itself:
# ~0.2 s is ~3 export frames, well under the 7-frame threshold, while the rest of the close
# overlaps the (moving) lift. Main grasp-reliability knob -- raise toward ~0.3 s if objects slip,
# lower toward 0 for a pure swoop grasp that relies on gripper force control during the lift.
CLOSE_CONTACT_DELAY_S = float(os.environ.get("TIPTOP_CLOSE_CONTACT_DELAY_S", "0.0"))

# An OPEN releases an already-placed object, so the arm can start retreating immediately.
OPEN_CONTACT_DELAY_S = float(os.environ.get("TIPTOP_OPEN_CONTACT_DELAY_S", "0.0"))


class ExecutionFailure(Exception):
    """Failure in executing plan on robot."""


def _gripper_state(client) -> dict | None:
    """Best-effort read of the gripper state dict ({width, is_grasped, is_moving}). None if unavailable."""
    try:
        res = client.get_gripper_state()
    except Exception:
        return None
    if not isinstance(res, dict):
        return None
    state = res.get("state", res)
    return state if isinstance(state, dict) else None


def _wait_for_gripper_settled(client, *, timeout: float = 5.0, poll: float = 0.02) -> None:
    """Block until the gripper stops moving, polling client-side.

    This mirrors the bamboo gripper server's own ``_spin_until_done`` exit condition,
    but runs here so the (single-threaded) server socket stays free *between* polls --
    letting the LeRobot gripper sampler read the position register during the motion and
    capture the open<->close ramp, which a server-side blocking command hides entirely.

    Falls back to a short fixed wait if the gripper state can't be read, so the arm never
    starts the next trajectory before the gripper has had time to actuate.
    """
    start = time.time()
    saw_moving = False
    while time.time() - start < timeout:
        time.sleep(poll)
        state = _gripper_state(client)
        if state is None or "is_moving" not in state:
            time.sleep(1.0)  # unreadable state: wait out a conservative actuation time
            return
        if state["is_moving"]:
            saw_moving = True
        elif saw_moving or (time.time() - start) > 0.25:
            # Stopped after moving, or never moved and already settled (e.g. already open).
            return
    _log.warning("Gripper did not report settled within %.1fs; continuing", timeout)


def _command_gripper(client, action: str):
    """Issue an open/close gripper command and block until it settles (the non-overlapped path).

    Used when overlap is off or unavailable (a trailing gripper step, a Franka hand, or
    GRIPPER_OVERLAP disabled). Issued blocking=False then polled client-side so the gripper state
    socket stays free for the sampler; on grippers whose server ignores blocking=False the send
    itself blocks, which is fine here -- we mean to wait.
    """
    fn = client.open_gripper if action == "open" else client.close_gripper
    try:
        result = fn(speed=1.0, blocking=False)
    except TypeError:
        # Client without a blocking flag (e.g. UR5) -- it blocks until done itself.
        return fn(speed=1.0)
    _wait_for_gripper_settled(client)
    return result


def _can_overlap_gripper(client) -> bool:
    """True if the gripper has its own socket (Robotiq), so its (blocking) command can run on a
    background thread without racing the arm control socket. A Franka hand shares the control
    socket, so it must stay on the blocking path."""
    return getattr(client, "gripper_socket", None) is not None


def _spawn_gripper_command(client, action: str) -> tuple[threading.Thread, dict]:
    """Fire an open/close gripper command on a BACKGROUND thread; return ``(thread, box)``.

    Running it on a thread lets the caller start the next arm trajectory (on the control socket)
    concurrently -- keeping the arm moving across the gripper transition. Only safe when the gripper
    uses its own socket (see :func:`_can_overlap_gripper`). ``box`` receives ``{"result": ...}`` or
    ``{"exc": ...}`` once the thread finishes; :func:`_join_gripper` surfaces either to the caller.

    The worker uses the same ``blocking=False`` + client-side poll path as :func:`_command_gripper`,
    NOT a server-side blocking command. A blocking command would occupy the gripper server's single
    request thread for the whole actuation, starving the LeRobot gripper sampler's state polls and
    collapsing the recorded open<->close ramp into an instantaneous jump. Client-side polling frees
    the gripper socket between reads so the sampler captures the ramp while the arm still overlaps.
    """
    box: dict = {}

    def _run() -> None:
        try:
            box["result"] = _command_gripper(client, action)
        except BaseException as exc:  # noqa: BLE001 - re-raised on the main thread at join time
            box["exc"] = exc

    thread = threading.Thread(target=_run, name=f"gripper-{action}", daemon=True)
    thread.start()
    return thread, box


def _join_gripper(pending: "tuple[threading.Thread, dict] | None") -> None:
    """Wait for a spawned gripper command to finish and re-raise any error it captured."""
    if pending is None:
        return
    thread, box = pending
    thread.join()
    if "exc" in box:
        raise box["exc"]


def execute_cutamp_plan(
    cutamp_plan: list[dict],
    client: RobotClient | None = None,
    timeline: list | None = None,
    should_stop: "Callable[[], bool] | None" = None,
) -> bool:
    """Execute the plan from cuTAMP on the real robot. Returns True if it stopped early.

    If ``timeline`` is provided, one entry per plan step is appended to it (in order),
    each ``{"type", "label", "t_start", "t_end"}`` with wall-clock (epoch) seconds.
    The LeRobot export uses this to place each control frame on the real execution
    clock -- including the gripper-actuation pauses that the plan timeline omits -- so
    camera frames can be aligned to states by hardware timestamp.

    ``should_stop`` is polled once per completed step (after any overlapped gripper command has been
    joined, so nothing is left actuating) and stops the plan there. That is the safe place to hand
    the arm to a human: it is parked at a plan boundary rather than mid-trajectory, which is the one
    thing an abort cannot give you -- bamboo has no abort, so a preempt still runs out the current
    segment and just skips the rest of the plan.
    """
    if client is None:
        client = get_robot_client()

    start_time = time.perf_counter()
    # An overlapped gripper command keeps actuating (on the gripper socket) while the next
    # trajectory runs; we hold its handle here and join it before the next gripper / at the end.
    pending_gripper: "tuple[threading.Thread, dict] | None" = None
    for step, action_dict in enumerate(cutamp_plan):
        action_start_time = time.perf_counter()
        step_t_start = time.time()
        action_type = action_dict["type"]
        action_label = action_dict["label"]

        # Form log message
        msg = f"Executing step {step + 1}/{len(cutamp_plan)}: {action_label}. Action type: {action_dict['type']}"
        if action_type == "gripper":
            msg += f" ({action_dict['action']})"
        elif action_type == "trajectory":
            msg += f" ({len(action_dict['plan'].position)} waypoints)"
        else:
            raise ValueError(f"Unknown action type in cuTAMP plan: {action_dict['type']}")
        _log.info(msg)

        # Now execute the actions
        if action_type == "gripper":
            action = action_dict["action"]
            if action not in ("open", "close"):
                raise ValueError(f"Unknown gripper action: {action}")
            # A previously overlapped gripper command may still own the gripper socket; finish it
            # (surfacing any error) before issuing another.
            _join_gripper(pending_gripper)
            pending_gripper = None
            # Overlap the actuation with the next trajectory (if any) so the arm doesn't sit idle
            # across the transition -- otherwise the training filter drops the gripper open/close
            # frames (see GRIPPER_OVERLAP). Needs a following trajectory and a separate gripper
            # socket; otherwise (e.g. last step, Franka hand, overlap disabled) block as before.
            next_is_trajectory = (
                step + 1 < len(cutamp_plan) and cutamp_plan[step + 1].get("type") == "trajectory"
            )
            if GRIPPER_OVERLAP and next_is_trajectory and _can_overlap_gripper(client):
                # Fire the gripper command on a thread (non-blocking + client-side poll, so the
                # sampler still sees the ramp); the NEXT loop iteration runs the trajectory
                # concurrently on the control socket. A brief stationary settle first lets the
                # fingers seat before the arm departs.
                pending_gripper = _spawn_gripper_command(client, action)
                settle_delay = CLOSE_CONTACT_DELAY_S if action == "close" else OPEN_CONTACT_DELAY_S
                if settle_delay > 0:
                    time.sleep(settle_delay)
                result = {"success": True}  # real result/exception surfaced when the thread is joined
            else:
                result = _command_gripper(client, action)

        elif action_type == "trajectory":
            # Extract joint position and velocity waypoints for the trajectory
            waypoints = action_dict["plan"].position.cpu().numpy()
            velocities = action_dict["plan"].velocity.cpu().numpy()
            timings = [action_dict["dt"]] * len(waypoints)
            result = client.execute_joint_impedance_path(
                joint_confs=waypoints, joint_vels=velocities, durations=timings
            )

        else:
            raise ValueError(f"Unexpected action type in cuTAMP plan: {action_dict['type']}")

        # Raise error if execution failed
        if result is None:
            raise RuntimeError("Fatal error: result should not be None")
        # if not result["success"]:
        #     raise ExecutionFailure(result["error"])

        if timeline is not None:
            timeline.append(
                {"type": action_type, "label": action_label, "t_start": step_t_start, "t_end": time.time()}
            )

        action_duration = time.perf_counter() - action_start_time
        _log.debug(f"Executing {action_type} action took {action_duration:.2f}s")

        if should_stop is not None and should_stop():
            # Finish any overlapped gripper first: the arm must be fully idle before another
            # controller can take it.
            _join_gripper(pending_gripper)
            _log.info(
                f"Stopping after step {step + 1}/{len(cutamp_plan)} ({action_label}) at the caller's "
                f"request; the remaining {len(cutamp_plan) - step - 1} step(s) will not be sent"
            )
            return True

    # A trailing overlapped gripper (rare -- plans usually end on a trajectory) must finish here.
    _join_gripper(pending_gripper)

    # Now we're done executing plan open-loop without any failures on the controller side
    duration = time.perf_counter() - start_time
    _log.info(f"Real robot execution took {duration:.2f}s")
    return False

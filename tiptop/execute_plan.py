import logging
import os
import threading
import time
from collections.abc import Callable

from tiptop.utils import RobotClient, get_robot_client
from tiptop.yam import step_arms as _step_arms
from tiptop.yam.capture import ARM_SLICES

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


# Hand each trajectory segment to the shim's queue instead of waiting for it, so the arm flows
# through gripper events instead of stopping at every one (see _QueuedArm). Falls back automatically
# when the shim predates the queue. TIPTOP_QUEUED_TRAJ=0 forces the legacy per-segment blocking path.
QUEUED_TRAJ = os.environ.get("TIPTOP_QUEUED_TRAJ", "1") != "0"

# How many trajectory segments to keep queued AHEAD of the one the arm is running. 1 is what makes
# the arm flow through a gripper event: when segment N finishes, N+1 is already in the shim's queue,
# so there is no round trip to stall on. Larger buys nothing (the arm can only be in one place) and
# costs abort latency, so this is a constant rather than a knob.
_TRAJ_LOOKAHEAD = 1

# Seconds BEFORE the arm reaches a gripper event at which the gripper command is fired. The gripper
# takes ~0.5-1 s to actuate, and with queueing the arm no longer parks at the event -- it flows
# straight into the next segment -- so firing exactly on arrival means the fingers close during the
# first part of the lift. Raise this toward ~0.4 if grasps are missed; lower to 0 if the fingers
# knock the object on approach. 0 reproduces the old firing instant, minus the stop.
GRIPPER_LEAD_S = float(os.environ.get("TIPTOP_GRIPPER_LEAD_S", "0.0"))


class ExecutionFailure(Exception):
    """Failure in executing plan on robot."""


class _QueuedArm:
    """Submit trajectory segments to the shim's queue over a DEDICATED control socket.

    Why its own socket: the shared client's control socket is a ZMQ REQ, which is neither
    thread-safe nor tolerant of interleaved traffic, and during execution the LeRobot state sampler
    and the gripper thread are both live on the cached client. Raw sends there would wedge it in
    EFSM. ``new_robot_client`` exists for exactly this (see its docstring).

    ``available`` is False against a shim without the queue commands, and the caller then uses the
    old blocking path -- so this is safe to leave on with an un-upgraded NUC.
    """

    def __init__(self):
        self._client = None
        self.available = False
        if not QUEUED_TRAJ:
            return
        try:
            import msgpack  # noqa: F401  (present wherever the bamboo client is)

            from tiptop.utils import new_robot_client

            self._client = new_robot_client()
            self._sock = self._client.control_socket
            # Probe BOTH commands. An intermediate shim has the queue but not the arrival wait, and
            # queueing without that wait fires every gripper at submit time -- the whole plan
            # actuating in seconds while the arm is still on its first segment. Discovering that
            # mid-episode leaves the arm moving with a raised exception, so refuse up front and take
            # the blocking path instead. seq=0 is always already "done", so the probe returns at once.
            self.available = bool(
                self._rpc({"command": "trajectory_queue_status"}).get("success")
                and self._rpc({"command": "wait_trajectory_arrival", "seq": 0,
                               "timeout": 1.0}).get("success")
            )
        except Exception:
            _log.exception("Queued-trajectory probe failed; using the per-segment blocking path")
            self.available = False
        if self.available:
            _log.info("Trajectory queueing enabled: segments hand over without stopping the arm.")
        else:
            _log.info("Trajectory queueing unavailable (shim too old or disabled); blocking path.")

    def _rpc(self, request: dict, timeout_ms: int = 15000) -> dict:
        import msgpack
        import zmq

        old = self._sock.getsockopt(zmq.RCVTIMEO)
        self._sock.setsockopt(zmq.RCVTIMEO, timeout_ms)
        try:
            self._sock.send(msgpack.packb(request))
            return msgpack.unpackb(self._sock.recv(), raw=False) or {}
        finally:
            self._sock.setsockopt(zmq.RCVTIMEO, old)

    def submit(self, waypoints, velocities, dt: float) -> dict:
        """Queue one segment and return as soon as the shim has accepted it."""
        wps = [{"q_goal": q.tolist(), "velocity": v.tolist(), "duration": dt}
               for q, v in zip(waypoints, velocities)]
        return self._rpc({"command": "execute_trajectory",
                          "data": {"waypoints": wps, "default_duration": dt,
                                   "default_velocity": [], "queue": True}})

    def times(self) -> dict:
        """{seq: [t_start, t_end]} epoch seconds per finished batch, from the shim."""
        try:
            return self._rpc({"command": "trajectory_times"}).get("times") or {}
        except Exception:
            _log.exception("trajectory_times failed; timeline keeps the submit instants")
            return {}

    def wait_arrival(self, seq: int, lead: float = 0.0, timeout: float = 180.0) -> dict:
        """Block until the arm reaches the end of segment ``seq`` (or is ``lead`` s short of it)."""
        return self._rpc({"command": "wait_trajectory_arrival", "seq": seq, "lead": lead,
                          "timeout": timeout}, timeout_ms=int(timeout * 1000) + 5000)

    def wait_done(self, timeout: float = 180.0) -> dict:
        return self._rpc({"command": "wait_trajectory_done", "timeout": timeout},
                         timeout_ms=int(timeout * 1000) + 5000)

    def abort(self) -> None:
        try:
            self._rpc({"command": "abort_trajectory"})
        except Exception:
            _log.exception("abort_trajectory failed")

    def close(self) -> None:
        try:
            if self._client is not None:
                self._client.close()
        except Exception:
            pass


class _DualQueuedArm:
    """Submit BOTH arms' trajectory segments to the shim's queue, over one dedicated control socket.

    Sibling to :class:`_QueuedArm`, same shape (its own control socket, bypassing the shared
    client's single-active-arm abstraction -- see YamClient.arm) applied to two arms. There is no
    combined-request wire message for a 12-DOF trajectory: each segment is split into its two 6-DOF
    column slices (:data:`ARM_SLICES`) and submitted as two separate ``execute_trajectory`` requests,
    each naming its arm explicitly in ``data.arm`` -- the shim already routes on that field per
    request (``yam_arm_server.py::_arm``), so no new server protocol is needed for this.

    **Synchronization is NOT validated here.** Each ``queue: True`` submission returns as soon as
    it is enqueued, not once executed, so two back-to-back submissions on this one socket cost only
    an RPC round trip (sub-ms to a few ms over loopback) before both arms' independent per-arm
    streamer threads start moving. That is the expected mechanism, but it has not been measured
    against real hardware timing. Per the project plan: before trusting this in a real handover, log
    both arms' measured joint trajectories (``BimanualJointSampler``/``StateCache``, already
    100 Hz) during a slow non-object dual approach and confirm motion-start skew is under 50 ms. If
    it is not, the documented fallback is an optional ``start_at: <epoch float>`` field on
    ``execute_trajectory``, computed once client-side and passed identically to both submissions,
    consumed by ``TrajectoryStreamer._stream`` in place of ``now`` -- not built here because it
    should only be built if the measurement demands it.

    Requires the server's trajectory queue (the same capability :class:`_QueuedArm` probes for);
    there is no non-queued fallback for dual execution, since the blocking single-segment path has
    no way to move both arms at once.
    """

    def __init__(self):
        self._client = None
        self.available = False
        try:
            import msgpack  # noqa: F401  (present wherever the bamboo client is)

            from tiptop.utils import new_robot_client

            self._client = new_robot_client()
            self._sock = self._client.control_socket
            self.available = bool(
                self._rpc({"command": "trajectory_queue_status"}).get("success")
                and self._rpc({"command": "wait_trajectory_arrival", "seq": 0,
                               "timeout": 1.0}).get("success")
            )
        except Exception:
            _log.exception("Dual queued-trajectory probe failed")
            self.available = False

    def _rpc(self, request: dict, timeout_ms: int = 15000) -> dict:
        import msgpack
        import zmq

        old = self._sock.getsockopt(zmq.RCVTIMEO)
        self._sock.setsockopt(zmq.RCVTIMEO, timeout_ms)
        try:
            self._sock.send(msgpack.packb(request))
            return msgpack.unpackb(self._sock.recv(), raw=False) or {}
        finally:
            self._sock.setsockopt(zmq.RCVTIMEO, old)

    def submit(self, waypoints12, velocities12, dt: float) -> dict[str, int]:
        """Split a 12-wide segment into both arms' slices and queue each, back to back.

        Returns ``{"left": seq, "right": seq}``. Both RPCs are issued before either is waited on --
        the second submission's round trip is the only source of motion-start skew between the arms
        under this mechanism (see class docstring).
        """
        seqs: dict[str, int] = {}
        for arm, sl in ARM_SLICES.items():
            wps = [
                {"q_goal": q.tolist(), "velocity": v.tolist(), "duration": dt}
                for q, v in zip(waypoints12[:, sl], velocities12[:, sl])
            ]
            res = self._rpc({"command": "execute_trajectory",
                             "data": {"waypoints": wps, "default_duration": dt,
                                      "default_velocity": [], "queue": True, "arm": arm}})
            if not res.get("success"):
                raise ExecutionFailure(f"queueing {arm} arm trajectory failed: {res.get('error')}")
            seqs[arm] = int(res["seq"])
        return seqs

    def wait_arrival(self, seqs: dict[str, int], lead: float = 0.0, timeout: float = 180.0) -> dict[str, dict]:
        """Block until every named arm reaches the end of its segment (or ``lead`` s short of it)."""
        return {
            arm: self._rpc({"command": "wait_trajectory_arrival", "seq": seq, "arm": arm,
                            "lead": lead, "timeout": timeout}, timeout_ms=int(timeout * 1000) + 5000)
            for arm, seq in seqs.items()
        }

    def wait_done(self, timeout: float = 180.0) -> dict:
        """Drain every arm's queue. No ``arm`` field -> the server waits on all of them at once."""
        return self._rpc({"command": "wait_trajectory_done", "timeout": timeout},
                         timeout_ms=int(timeout * 1000) + 5000)

    def abort(self) -> None:
        """Abort every arm's queue. No ``arm`` field -> both streamers stop, which is what any
        interrupt or failure during dual execution should do -- never leave one arm still moving
        toward a shared configuration the other has abandoned."""
        try:
            self._rpc({"command": "abort_trajectory"})
        except Exception:
            _log.exception("abort_trajectory failed")

    def close(self) -> None:
        try:
            if self._client is not None:
                self._client.close()
        except Exception:
            pass


def _gripper_state(client, arm: str | None = None) -> dict | None:
    """Best-effort read of the gripper state dict ({width, is_grasped, is_moving}). None if unavailable.

    ``arm`` is only meaningful for a YamClient in dual mode, where there is no default active arm to
    fall back to (see YamClient.arm); every other client ignores it via ``arm or client.arm``-style
    defaulting inside the client method itself, so passing it is always safe.
    """
    try:
        res = client.get_gripper_state(arm) if arm is not None else client.get_gripper_state()
    except Exception:
        return None
    if not isinstance(res, dict):
        return None
    state = res.get("state", res)
    return state if isinstance(state, dict) else None


def _wait_for_gripper_settled(client, *, arm: str | None = None, timeout: float = 5.0, poll: float = 0.02) -> None:
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
        state = _gripper_state(client, arm)
        if state is None or "is_moving" not in state:
            time.sleep(1.0)  # unreadable state: wait out a conservative actuation time
            return
        if state["is_moving"]:
            saw_moving = True
        elif saw_moving or (time.time() - start) > 0.25:
            # Stopped after moving, or never moved and already settled (e.g. already open).
            return
    _log.warning("Gripper did not report settled within %.1fs; continuing", timeout)


def _command_gripper(client, action: str, arm: str | None = None):
    """Issue an open/close gripper command and block until it settles (the non-overlapped path).

    Used when overlap is off or unavailable (a trailing gripper step, a Franka hand, or
    GRIPPER_OVERLAP disabled). Issued blocking=False then polled client-side so the gripper state
    socket stays free for the sampler; on grippers whose server ignores blocking=False the send
    itself blocks, which is fine here -- we mean to wait.

    ``arm`` addresses a specific hand (only meaningful for the dual-arm YAM, e.g.
    execute_cutamp_dual_plan's hard-sequenced handover close/open); every other caller leaves it
    None and the client falls back to its single active arm, exactly as before this parameter existed.
    """
    fn = client.open_gripper if action == "open" else client.close_gripper
    kwargs = {"speed": 1.0, "blocking": False}
    if arm is not None:
        kwargs["arm"] = arm
    try:
        result = fn(**kwargs)
    except TypeError:
        # Client without a blocking flag (e.g. UR5) -- it blocks until done itself.
        kwargs.pop("blocking")
        return fn(**kwargs)
    _wait_for_gripper_settled(client, arm=arm)
    return result


def _can_overlap_gripper(client) -> bool:
    """True if the gripper has its own socket (Robotiq), so its (blocking) command can run on a
    background thread without racing the arm control socket. A Franka hand shares the control
    socket, so it must stay on the blocking path."""
    return getattr(client, "gripper_socket", None) is not None


def _spawn_gripper_command(client, action: str, arm: str | None = None) -> tuple[threading.Thread, dict]:
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
            box["result"] = _command_gripper(client, action, arm=arm)
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


def _rewrite_timeline(timeline: list | None, seq_of_step: dict, times: dict) -> None:
    """Replace each queued step's submit instant with the span the arm actually moved over.

    Queueing means the executor returns from a segment long before the arm runs it, so the
    ``t_start``/``t_end`` recorded inline are submit instants. ``lerobot_capture._flatten_plan``
    spreads that step's control frames linearly across the span to align camera frames to states by
    hardware timestamp -- so leaving submit instants there would compress every trajectory's frames
    into a few milliseconds. The shim reports the real spans; splice them in.
    """
    if not timeline or not times:
        return
    patched = 0
    # Count only steps that HAVE a timeline entry: an early stop (should_stop) breaks the loop with
    # _TRAJ_LOOKAHEAD segments already submitted, so seq_of_step legitimately runs past the recorded
    # prefix. Those are not spans the shim failed to report, and counting them would fire the
    # mis-alignment error below on every hand-off -- training the operator to ignore the real one.
    in_timeline = [s for s in seq_of_step if 0 <= s < len(timeline)]
    for step, seq in seq_of_step.items():
        span = times.get(str(seq))
        if span and 0 <= step < len(timeline):
            timeline[step]["t_start"], timeline[step]["t_end"] = float(span[0]), float(span[1])
            patched += 1
    missing = len(in_timeline) - patched
    if missing:
        # Not recoverable here, and a silently wrong timeline corrupts the export's alignment --
        # so say so loudly rather than let it pass as a good episode.
        _log.error("Trajectory timeline: %d/%d queued steps got no span back from the shim; those "
                   "steps keep their submit instants and their exported frames will be mis-aligned.",
                   missing, len(in_timeline))
    else:
        _log.info("Trajectory timeline: %d queued steps re-anchored to their real motion spans.", patched)


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
    segment and just skips the rest of the plan. On the queued path the loop runs _TRAJ_LOOKAHEAD
    (= 1) segments ahead of the arm, so the stop drains what is already in the shim's queue: the arm
    ends up at most ONE trajectory segment past the last step that got a ``timeline`` entry. That is
    deliberate -- draining is what keeps the stop at a segment boundary -- and it is why the saved
    episode is a prefix of where the arm physically parked, never the other way round.
    """
    if client is None:
        client = get_robot_client()

    start_time = time.perf_counter()
    queued = _QueuedArm()
    # plan-step index -> queue seq, so the real motion span can replace the submit instant below.
    seq_of_step: dict[int, int] = {}
    # Trajectory segments are submitted AHEAD of where the arm is, so the plan order alone no longer
    # says when anything happens. _submit_through keeps _TRAJ_LOOKAHEAD segments in flight; the
    # gripper branch then waits for the arm to actually arrive before it fires. Without that wait the
    # whole plan is queued in a few seconds and every gripper actuates while the arm is still on the
    # first segment -- which is exactly what happened on 2026-07-28.
    traj_steps = [i for i, s in enumerate(cutamp_plan) if s.get("type") == "trajectory"]
    n_submitted = 0

    def _submit_through(count: int) -> None:
        nonlocal n_submitted
        while n_submitted < min(count, len(traj_steps)):
            i = traj_steps[n_submitted]
            plan_i = cutamp_plan[i]["plan"]
            res = queued.submit(plan_i.position.cpu().numpy(), plan_i.velocity.cpu().numpy(),
                                float(cutamp_plan[i]["dt"]))
            if not res.get("success"):
                raise ExecutionFailure(f"queueing trajectory step {i + 1} failed: {res.get('error')}")
            seq_of_step[i] = int(res["seq"])
            n_submitted += 1

    traj_seen = 0        # trajectory segments passed in plan order
    last_traj_step = None
    stop_requested = False   # set when should_stop fires; returned once the queue has drained
    # An overlapped gripper command keeps actuating (on the gripper socket) while the next
    # trajectory runs; we hold its handle here and join it before the next gripper / at the end.
    pending_gripper: "tuple[threading.Thread, dict] | None" = None
    try:
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
                if queued.available and last_traj_step is not None:
                    # Keep the next segment queued FIRST -- that is what lets the arm roll through this
                    # event instead of stopping -- then block until the arm actually reaches it, so the
                    # gripper fires here and not when the segment was merely accepted.
                    _submit_through(traj_seen + _TRAJ_LOOKAHEAD)
                    arrived = queued.wait_arrival(seq_of_step[last_traj_step], lead=GRIPPER_LEAD_S)
                    if not arrived.get("success"):
                        queued.abort()
                        raise ExecutionFailure(
                            f"arm did not reach the gripper event at step {step + 1}: {arrived.get('error')}"
                        )
                    step_t_start = time.time()   # the wait happened above; time the actuation, not it
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
                    # concurrently on the control socket.
                    pending_gripper = _spawn_gripper_command(client, action)
                    settle_delay = CLOSE_CONTACT_DELAY_S if action == "close" else OPEN_CONTACT_DELAY_S
                    if settle_delay > 0 and not queued.available:
                        # Stationary settle: lets the fingers seat before the arm departs. This only
                        # parks the arm on the BLOCKING path, where the next trajectory cannot start
                        # until this loop issues it. Under queueing the next segment is already in the
                        # shim's queue and streaming, so sleeping here would delay the loop without
                        # slowing the arm -- use TIPTOP_GRIPPER_LEAD_S to give the gripper a head start
                        # instead.
                        time.sleep(settle_delay)
                    result = {"success": True}  # real result/exception surfaced when the thread is joined
                else:
                    result = _command_gripper(client, action)

            elif action_type == "trajectory":
                # Extract joint position and velocity waypoints for the trajectory
                waypoints = action_dict["plan"].position.cpu().numpy()
                velocities = action_dict["plan"].velocity.cpu().numpy()
                if queued.available:
                    # Submit this segment and the lookahead, then move on -- the arm runs them
                    # asynchronously. The timeline entry below is therefore a SUBMIT instant; the real
                    # motion span is spliced in by _rewrite_timeline once the shim reports it.
                    _submit_through(traj_seen + 1 + _TRAJ_LOOKAHEAD)
                    last_traj_step = step
                    traj_seen += 1
                    result = {"success": True}
                else:
                    last_traj_step = step
                    traj_seen += 1
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
                # The teleop hand-off checkpoint, met with the queued streaming path. BREAK rather
                # than return: under queueing this loop is _TRAJ_LOOKAHEAD (= 1) segments ahead of
                # the arm, so what is already in the shim's queue is still moving it. Falling
                # through to the drain below lets exactly that finish -- which is the whole point of
                # the checkpoint, an arm parked at a segment boundary instead of wherever an abort
                # caught it -- and the drain is also where the overlapped gripper is joined, so
                # nothing is left actuating when the human takes over. Aborting instead would be the
                # mid-trajectory stop this exists to avoid. On the blocking path nothing is queued
                # and the break is the immediate stop it has always been.
                #
                # The timeline entry for this step was appended just above, so the recorded episode
                # stays a contiguous PREFIX of the plan (lerobot_capture._flatten_plan depends on
                # that); the arm simply parks up to one segment beyond its last recorded frame.
                stop_requested = True
                _log.info(
                    f"Stopping after step {step + 1}/{len(cutamp_plan)} ({action_label}) at the caller's "
                    f"request; the remaining {len(cutamp_plan) - step - 1} step(s) will not be issued "
                    "(a segment already queued still runs out)"
                )
                break

    except BaseException:
        # A preempt (the UI's Preempt button, or a terminal Ctrl-C) unwinds through here. Segments
        # are submitted AHEAD of where the arm is, so merely leaving the loop would let the shim
        # keep streaming everything already queued -- the arm working on through the plan after the
        # operator asked it to stop. Dropping the queue is the closest thing to "stop" available
        # from here; the segment already streaming still runs to its end, and only the physical
        # E-stop is instant.
        if queued.available:
            _log.warning("Execution interrupted -- aborting the trajectory queue")
            queued.abort()
            queued.close()
        # Join WITHOUT re-raising: _join_gripper surfaces whatever the gripper thread caught, which
        # here would replace the interrupt that actually stopped us and hide why the rollout ended.
        if pending_gripper is not None:
            pending_gripper[0].join(timeout=5.0)
        raise

    # Queued segments are still running: the loop above only handed them over. Wait for the arm to
    # actually finish before the gripper join and before the caller tears the episode down.
    if queued.available:
        drained = queued.wait_done()
        if not drained.get("success"):
            _log.error("Trajectory queue did not drain (%s); aborting the remainder",
                       drained.get("error"))
            queued.abort()
        _rewrite_timeline(timeline, seq_of_step, queued.times())
        queued.close()

    # A trailing overlapped gripper (rare -- plans usually end on a trajectory) must finish here.
    _join_gripper(pending_gripper)

    # Now we're done executing plan open-loop without any failures on the controller side
    duration = time.perf_counter() - start_time
    _log.info(f"Real robot execution took {duration:.2f}s")
    return stop_requested


def execute_cutamp_dual_plan(
    cutamp_plan: list[dict], client: RobotClient | None = None, timeline: list | None = None
) -> None:
    """Execute a SIMULTANEOUS dual-arm cuTAMP plan (``bimanual_yam_dual``, ``arm_mode="dual"``).

    Sibling to :func:`execute_cutamp_plan`, not a branch inside it: a dual/handover plan moves both
    arms' columns on every trajectory segment and names one or two hands per gripper step
    (``arm``/``arms``, see :func:`tiptop.planning.serialize_plan`), different enough in shape from
    the single-arm per-step client-scoping that threading it through the existing loop would
    obscure both paths. Motion goes through :class:`_DualQueuedArm`, the same queued wire protocol
    ``execute_cutamp_plan`` uses, just issued twice per segment (see that class's synchronization
    caveat -- it has not been validated against real hardware timing).

    A handover's gripper ordering -- the taker closes, THEN the giver opens, with no trajectory step
    between them in the raw cuTAMP plan (``motion_solver.py``'s ``Handover`` branch) -- is a physical
    safety requirement: the held object must never be unsupported. It is hard-sequenced here (wait
    for the close to actually settle, via polled gripper state, before the open is even issued) and
    is NOT run through ``execute_cutamp_plan``'s overlap-with-next-motion optimization, which exists
    for data-quality reasons and must not be allowed to reorder or race this pair.

    If ``timeline`` is provided, one entry per plan step is appended (in order), each
    ``{"type", "label", "t_start", "t_end"}`` with wall-clock (epoch) seconds -- same contract as
    :func:`execute_cutamp_plan`.
    """
    if client is None:
        client = get_robot_client()

    start_time = time.perf_counter()
    dual = _DualQueuedArm()
    if not dual.available:
        raise ExecutionFailure(
            "dual-arm execution requires the arm server's trajectory queue (trajectory_queue_status / "
            "wait_trajectory_arrival) -- the blocking single-segment path has no way to move both arms "
            "at once. Update the arm server before running a bimanual_yam_dual session."
        )

    seqs_of_step: dict[int, dict[str, int]] = {}
    last_traj_step: int | None = None

    try:
        for step, action_dict in enumerate(cutamp_plan):
            step_t_start = time.time()
            action_type = action_dict["type"]
            action_label = action_dict["label"]

            msg = f"Executing dual step {step + 1}/{len(cutamp_plan)}: {action_label}. Action type: {action_type}"
            if action_type == "gripper":
                msg += f" ({action_dict['action']}, arms={_step_arms(action_dict)})"
            elif action_type == "trajectory":
                msg += f" ({len(action_dict['plan'].position)} waypoints, both arms)"
            else:
                raise ValueError(f"Unknown action type in cuTAMP plan: {action_type}")
            _log.info(msg)

            if action_type == "trajectory":
                waypoints = action_dict["plan"].position.cpu().numpy()
                velocities = action_dict["plan"].velocity.cpu().numpy()
                seqs_of_step[step] = dual.submit(waypoints, velocities, float(action_dict["dt"]))
                last_traj_step = step

            else:  # gripper
                action = action_dict["action"]
                if action not in ("open", "close"):
                    raise ValueError(f"Unknown gripper action: {action}")
                arms = _step_arms(action_dict)

                # Same reasoning as execute_cutamp_plan: segments are submitted ahead of where the
                # arms physically are, so without waiting for arrival, every gripper would fire while
                # the arms are still on the first segment.
                if last_traj_step is not None:
                    arrived = dual.wait_arrival(
                        {a: seqs_of_step[last_traj_step][a] for a in arms}, lead=GRIPPER_LEAD_S
                    )
                    if not all(r.get("success") for r in arrived.values()):
                        dual.abort()
                        raise ExecutionFailure(
                            f"arm(s) {arms} did not reach the gripper event at step {step + 1}: {arrived}"
                        )
                    step_t_start = time.time()  # the wait happened above; time the actuation, not it

                # A handover close is followed immediately (no trajectory step between) by the
                # giver's open on the OTHER arm -- detect that shape and hard-sequence it: wait for
                # this close to actually settle before returning, so the loop cannot reach the next
                # step's open until the object is confirmed held. Every other gripper event (a
                # simultaneous PickBoth/PlaceBoth pair, or a lone open/close with no such neighbor)
                # fires all its named arms together and waits for all of them.
                is_handover_close = (
                    action == "close"
                    and len(arms) == 1
                    and step + 1 < len(cutamp_plan)
                    and cutamp_plan[step + 1].get("type") == "gripper"
                    and cutamp_plan[step + 1].get("action") == "open"
                    and _step_arms(cutamp_plan[step + 1]) != arms
                )
                if is_handover_close:
                    _command_gripper(client, action, arm=arms[0])
                else:
                    threads = [_spawn_gripper_command(client, action, arm=a) for a in arms]
                    for t in threads:
                        _join_gripper(t)

            if timeline is not None:
                timeline.append(
                    {"type": action_type, "label": action_label, "t_start": step_t_start, "t_end": time.time()}
                )

    except BaseException:
        # Same reasoning as execute_cutamp_plan's interrupt handler, for both arms at once: dropping
        # the queue is the closest thing to "stop" available from here (only the physical E-stop is
        # instant), and _DualQueuedArm.abort() with no arm field already stops both streamers.
        _log.warning("Dual execution interrupted -- aborting both arms' trajectory queues")
        dual.abort()
        dual.close()
        raise

    drained = dual.wait_done()
    if not drained.get("success"):
        _log.error("Dual trajectory queue did not drain (%s); aborting the remainder", drained.get("error"))
        dual.abort()
    dual.close()

    duration = time.perf_counter() - start_time
    _log.info(f"Real robot dual-arm execution took {duration:.2f}s")

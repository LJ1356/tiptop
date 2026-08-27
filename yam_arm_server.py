#!/usr/bin/env python3
"""ZMQ arm server for the bimanual YAM — the i2rt counterpart of ``bamboo_polymetis_shim.py``.

TiPToP's robot client speaks ZMQ + msgpack to three REP sockets (control / gripper / state). On the
Franka those are served by the bamboo shim on the NUC; this file serves the same protocol for two
i2rt YAM arms on CAN, so ``tiptop/yam/yam_client.py`` is a thin client and nothing else in tiptop
has to care which arm vendor is underneath.

**Why a separate process rather than importing i2rt into tiptop.** i2rt pins ``numpy==2.2.6`` and
``rerun-sdk>=0.32.2``; tiptop's pixi env is on numpy 2.4.2 / rerun 0.27.3 and cuRobo is pinned
against those. Installing i2rt there resolves into a fight. Running it out-of-process is also what
the Franka already does, so the client-side code path is identical.

Run it in an environment that has ``i2rt``, ``pyzmq`` and ``msgpack`` — e.g. the openpi YAM venv,
which already carries i2rt::

    /home/prism-yam/openpi/examples/yam/.venv/bin/pip install pyzmq msgpack
    /home/prism-yam/openpi/examples/yam/.venv/bin/python yam_arm_server.py \
        --left-channel can_left --right-channel can_right

CAN must be up first (1 Mbps)::

    sudo ip link set can_left  up type can bitrate 1000000
    sudo ip link set can_right up type can bitrate 1000000

## Protocol

Every request is a msgpack dict with a ``command`` key; arm-scoped commands also carry
``"arm": "left"|"right"``. Replies are ``{"success": bool, ...}`` and never raise past the socket —
a wedged REP socket would hang the whole session.

**control port (default 5555)**

| command | payload | reply |
|---|---|---|
| ``ping`` | — | ``{success}`` |
| ``get_robot_state`` | ``arm`` | ``{success, data: {q[6], dq[6], gripper, time_sec}}`` |
| ``execute_trajectory`` | ``{data: {arm, waypoints[{q_goal[6], velocity[6], duration}], default_duration, queue}}`` | ``{success, seq}`` when queued, else blocks |
| ``wait_trajectory_arrival`` | ``seq, lead, timeout`` | ``{success, arrived}`` |
| ``wait_trajectory_done`` | ``timeout`` | ``{success}`` |
| ``trajectory_times`` | — | ``{success, times: {seq: [t_start, t_end]}}`` |
| ``trajectory_queue_status`` | — | ``{success, queued, submitted, done}`` |
| ``abort_trajectory`` | — | ``{success}`` |
| ``move_to`` | ``arm, q_goal[6 or 7], duration`` | ``{success}`` — interpolated park/home move |
| ``hold`` | ``arm`` | ``{success}`` — latch the current pose as the PD target |
| ``relax`` | ``arm`` | ``{success}`` — back to gravity-comp idle |
| ``free_drive`` | ``arm, hold_gripper`` | ``{success}`` — limp arm, still-clamped gripper, for hand-guiding |

**gripper port (default 5559)** — ``open_gripper`` / ``close_gripper`` / ``get_gripper_state``, each
taking ``arm``. Width is reported in metres so tiptop's existing normalisation
(``lerobot_capture.GRIPPER_MAX_WIDTH``) applies unchanged.

**state port (default 5557)** — ``get_robot_state`` with no ``arm`` returns BOTH arms at once, from
a ~100 Hz background cache. This is what the LeRobot capture samples during execution, and it must
answer while the control socket is parked inside a trajectory, exactly as on the Franka.

Every socket is served by its own thread, so a blocking trajectory on the control socket never
starves a gripper poll or a state read.
"""

import argparse
import logging
import threading
import time

import msgpack
import numpy as np
import zmq

log = logging.getLogger("yam_arm_server")

ARMS = ("left", "right")

# 6 revolute joints + 1 gripper. i2rt exposes the gripper NORMALISED to [0, 1] on both read and
# write (JointMapper.to_command_joint_pos_space / to_robot_joint_pos_space), where the endpoints are
# the limits found by the startup calibration. 1.0 is open — see the `start_joints` of the shipped
# eval configs, e.g. openpi/examples/yam/main.py DEFAULT_START_JOINTS.
ARM_DOF = 6
FULL_DOF = 7
GRIPPER_OPEN = 1.0
GRIPPER_CLOSED = 0.0

# Jaw travel at full open, metres — i2rt's `gripper_stroke` for the LINEAR_4310 gripper
# (i2rt/robots/config/linear_4310.yml). Used only to report a width in metres on the gripper socket,
# because that is the unit tiptop's Robotiq-shaped client contract speaks. cuTAMP's kinematic model
# quotes 0.099 m for the same jaws (YAM_MAX_OPENING); the 3 mm is the pad thickness and does not
# matter for a 0..1 normalisation.
GRIPPER_STROKE_M = 0.096

# Elbow-up rest posture. MUST match cutamp.robots.bimanual_yam.bimanual_yam_neutral_joint_positions
# and the `lock_joints` of the cuRobo configs: when one arm plans, the other is collision-checked at
# exactly this configuration, so the parked arm has to physically be here or the planner is lying.
NEUTRAL_Q = (0.0, np.pi / 4, np.pi / 2, 0.0, 0.0, 0.0)

# Refuse a trajectory whose first waypoint is further than this from where the arm actually is.
# The YAM runs stiff position control (kp 80 on the shoulder joints); stepping it a long way in one
# control period is a violent motion, and every legitimate caller plans FROM the measured state.
MAX_START_JUMP_RAD = 0.25

# Streaming rate ceiling. cuRobo hands us ~50 Hz waypoints and i2rt's own control loop runs ~300 Hz,
# so this only bounds the interpolation we insert between sparse waypoints.
STREAM_HZ = 200.0


def _now() -> float:
    return time.time()


class YamArm:
    """One i2rt YAM arm: 6 joints + a normalised gripper, with a latched command vector.

    i2rt's ``command_joint_pos`` takes all 7 numbers at once and its internal ~300 Hz loop holds the
    last one, so "move the arm" and "move the gripper" are the same call. This class keeps the
    latched 7-vector so the two can be driven independently by different threads: the trajectory
    streamer writes joints 0..5, the gripper handler writes joint 6, and neither clobbers the other.
    """

    def __init__(self, name: str, channel: str, engage_on_connect: bool = True):
        from i2rt.robots.get_robot import get_yam_robot
        from i2rt.robots.utils import GripperType

        self.name = name
        self.channel = channel
        log.info(f"[{name}] connecting on {channel} (gripper calibration runs now — jaws will move)")
        # zero_gravity_mode=True: the arm comes up backdrivable. The first command_joint_pos engages
        # PD at the configured gains, which is why `engage` latches the MEASURED pose first.
        self._robot = get_yam_robot(channel=channel, gripper_type=GripperType.LINEAR_4310)
        self._lock = threading.Lock()
        self._target = self.read_full()
        self._engaged = False
        if engage_on_connect:
            self.engage()
        log.info(f"[{name}] ready; q={np.round(self._target[:ARM_DOF], 3).tolist()} "
                 f"gripper={self._target[ARM_DOF]:.3f}")

    # -- state ---------------------------------------------------------------------------------
    def read_full(self) -> np.ndarray:
        """Measured 7-vector [q1..q6, gripper] — gripper normalised 0..1."""
        q = np.asarray(self._robot.get_joint_pos(), dtype=np.float64).reshape(-1)
        if len(q) < FULL_DOF:  # a NO_GRIPPER build would be 6-wide; keep the shape contract
            q = np.pad(q, (0, FULL_DOF - len(q)))
        return q[:FULL_DOF]

    def observation(self) -> dict:
        """Measured position/velocity/effort, split into arm and gripper."""
        obs = self._robot.get_observations()
        pos = np.asarray(obs.get("joint_pos", []), dtype=np.float64).reshape(-1)[:ARM_DOF]
        vel = np.asarray(obs.get("joint_vel", []), dtype=np.float64).reshape(-1)[:ARM_DOF]
        grip = float(np.asarray(obs.get("gripper_pos", [0.0])).reshape(-1)[0])
        grip_vel = float(np.asarray(obs.get("gripper_vel", [0.0])).reshape(-1)[0])
        grip_eff = float(np.asarray(obs.get("gripper_eff", [0.0])).reshape(-1)[0])
        return {
            "q": pos.tolist(),
            "dq": vel.tolist(),
            "gripper": grip,
            "gripper_vel": grip_vel,
            "gripper_eff": grip_eff,
            # What the gripper was last TOLD, as opposed to where it ended up. The two differ
            # whenever the jaws stop on an object, and the recorded action channel needs the
            # command: a gripper commanded shut but held at 0.5 by a wide object is "closed", and
            # inferring that from the measurement alone would record it as open.
            "gripper_target": self.gripper_target,
            "time_sec": _now(),
        }

    # -- command -------------------------------------------------------------------------------
    def engage(self) -> None:
        """Latch the measured pose as the PD target, taking the arm out of gravity-comp idle.

        Commanding exactly where the arm already is means engaging cannot produce motion; every
        later command moves from a known held pose rather than from a floppy one.
        """
        with self._lock:
            self._target = self.read_full()
            self._robot.command_joint_pos(self._target.copy())
            self._engaged = True

    def relax(self) -> None:
        """Back to gravity-comp idle (backdrivable). Used on shutdown."""
        with self._lock:
            self._engaged = False
        try:
            self._robot.enter_gravity_comp_idle()
        except Exception:
            log.exception(f"[{self.name}] enter_gravity_comp_idle failed")

    def free_drive(self, hold_gripper: bool = True) -> None:
        """Backdrivable ARM, still-clamped GRIPPER — for hand-guiding the arm.

        ``relax`` zeroes every joint's gains including the gripper's, so anything held in the jaws
        is dropped. Hand-guiding with a calibration board clamped in needs the opposite: the six arm
        joints limp, the gripper holding exactly as hard as before. i2rt's ``command_joint_state``
        takes per-joint gains, so this sends kp 0 with the gravity-comp damping on joints 0..5 and
        the configured gains on joint 6 — gravity compensation itself is added in i2rt's control
        loop regardless of the commanded gains, so the arm still floats rather than falling.
        """
        info = self._robot.get_robot_info()
        kp = np.zeros(FULL_DOF, dtype=np.float64)
        kd = np.asarray(info["grav_comp_kd"], dtype=np.float64).reshape(-1)[:FULL_DOF].copy()
        if hold_gripper:
            kp[ARM_DOF] = float(np.asarray(info["kp"]).reshape(-1)[ARM_DOF])
            kd[ARM_DOF] = float(np.asarray(info["kd"]).reshape(-1)[ARM_DOF])
        with self._lock:
            # Latch the measured arm pose as the (unused, kp=0) target so re-engaging from here does
            # not step; the gripper keeps the target it was last COMMANDED, not where the board is
            # holding the jaws open, or letting go of the gains would let the board slip.
            self._target[:ARM_DOF] = self.read_full()[:ARM_DOF]
            self._robot.command_joint_state(
                {"pos": self._target.copy(), "vel": np.zeros(FULL_DOF), "kp": kp, "kd": kd}
            )
            self._engaged = False

    def set_arm(self, q_arm) -> None:
        """Command joints 0..5, holding the latched gripper target."""
        with self._lock:
            self._target[:ARM_DOF] = np.asarray(q_arm, dtype=np.float64).reshape(-1)[:ARM_DOF]
            self._robot.command_joint_pos(self._target.copy())
            self._engaged = True

    def set_gripper(self, value: float) -> None:
        """Command joint 6 (normalised 0..1), holding the latched arm target."""
        with self._lock:
            self._target[ARM_DOF] = float(np.clip(value, 0.0, 1.0))
            self._robot.command_joint_pos(self._target.copy())
            self._engaged = True

    @property
    def gripper_target(self) -> float:
        with self._lock:
            return float(self._target[ARM_DOF])

    def close(self) -> None:
        try:
            self.relax()
            self._robot.close()
        except Exception:
            log.exception(f"[{self.name}] close failed")


class TrajectoryStreamer:
    """Per-arm waypoint streamer with a batch QUEUE, so consecutive plan segments join.

    Modelled on ``bamboo_polymetis_shim._TrajectoryStreamer`` and answering the same commands, so
    tiptop's ``execute_plan._QueuedArm`` drives this unchanged. The reason it exists is data quality,
    not throughput: a cuTAMP plan is one batch per operation group, so without a queue the arm stops
    dead at every gripper event while the reply travels back and the next batch is issued. Those
    zero-velocity frames are exactly what the DROID-style non-idle training filter deletes, taking
    the gripper transition with them.

    One worker owns the queue and streams against a CONTINUOUS deadline clock, so the first setpoint
    of batch N+1 goes out one waypoint-duration after the last of batch N.

    Two details carried over from the Franka streamer because they are not obvious:

    * The deadline is reset to ``now`` whenever it has fallen into the past (the queue genuinely went
      idle). Without that, a batch arriving after a pause is streamed as fast as the loop can issue
      setpoints, trying to catch up to a stale clock — a burst of motion at full rate.
    * Aborts bump an EPOCH rather than clearing the deque. The worker may already have popped a batch
      and be mid-stream, so clearing the queue does not stop the arm; re-checking the epoch between
      setpoints does.
    """

    _MAX_QUEUED = 64  # backstop; a plan is ~7 batches

    def __init__(self, arm: YamArm, seq_allocator):
        self._arm = arm
        # Sequence numbers come from a server-wide counter, NOT a per-streamer one, so they identify
        # a batch uniquely across both arms. tiptop's `_QueuedArm.times()` asks for
        # `trajectory_times` without naming an arm and splices the reply into one timeline keyed by
        # seq; per-arm counters would both start at 1 and silently overwrite each other.
        self._alloc_seq = seq_allocator
        self._cv = threading.Condition()
        self._q: list = []
        self._epoch = 0
        self._seq = 0  # highest seq this streamer has been handed
        self._owned: set[int] = set()
        self._done_seq = 0
        self._errors: dict[int, str] = {}
        # Wall-clock span each batch actually occupied. A queued client only ever sees the submit
        # instant, but the LeRobot export places every control frame on this span to align camera
        # frames to states — so the only party that knows when a batch really ran reports it back.
        self._times: dict[int, tuple[float, float]] = {}
        # Projected monotonic end time per batch, published as soon as the batch starts streaming.
        # This is what lets a caller fire a gripper AT the moment the arm reaches the last waypoint
        # rather than at the moment it queued the batch.
        self._eta: dict[int, float] = {}
        self._t_next = 0.0
        self._stop = False
        self._thread = threading.Thread(target=self._run, name=f"traj-{arm.name}", daemon=True)
        self._thread.start()

    # -- client-facing -------------------------------------------------------------------------
    def submit(self, waypoints: list, default_duration: float) -> int:
        seq = self._alloc_seq()
        with self._cv:
            if len(self._q) >= self._MAX_QUEUED:
                raise RuntimeError(f"trajectory queue full ({self._MAX_QUEUED})")
            self._seq = seq
            self._owned.add(seq)
            self._q.append((self._epoch, seq, waypoints, default_duration))
            self._cv.notify_all()
            return seq

    def owns(self, seq: int) -> bool:
        """True if ``seq`` was issued to this streamer. Routes seq-addressed commands to the right arm."""
        with self._cv:
            return seq in self._owned

    def abort(self) -> None:
        """Drop everything queued AND stop the batch being streamed (see the epoch note above).

        Dropped batches are marked DONE. They never reach the worker, so nothing else would ever
        advance ``_done_seq`` past them — and ``wait_idle`` waits for exactly that, so an abort would
        otherwise leave the streamer permanently un-drainable and every later park or home
        (``_move_to`` waits for idle first) would time out for the rest of the session.
        """
        with self._cv:
            self._epoch += 1
            dropped = [batch[1] for batch in self._q]
            self._q.clear()
            if dropped:
                self._done_seq = max(self._done_seq, max(dropped))
                log.info(f"[{self._arm.name}] aborted {len(dropped)} queued batch(es): {dropped}")
            self._cv.notify_all()

    def wait_for(self, seq: int, timeout: float) -> dict:
        with self._cv:
            ok = self._cv.wait_for(lambda: self._done_seq >= seq, timeout=timeout)
        if not ok:
            return {"success": False, "error": f"timeout waiting for trajectory {seq}"}
        err = self._errors.pop(seq, None)
        return {"success": False, "error": err} if err else {"success": True}

    def _error_for(self, seq: int) -> str | None:
        """Any failure recorded against ``seq``. Peeks — the batch's own waiter still needs it."""
        return self._errors.get(seq)

    def wait_arrival(self, seq: int, lead: float = 0.0, timeout: float = 120.0) -> dict:
        """Block until batch ``seq`` finishes, or is within ``lead`` seconds of finishing.

        A batch that FAILED (a refused first waypoint, say) is "done", so this has to report the
        failure rather than a successful arrival. The queued execution path only ever calls this and
        ``wait_idle`` — never ``wait_for`` — so if they both answered "success" a refused trajectory
        would pass silently and the plan would carry on actuating the gripper at a pose the arm
        never reached.
        """
        deadline = time.monotonic() + timeout
        with self._cv:
            while True:
                if self._done_seq >= seq:
                    err = self._error_for(seq)
                    return {"success": False, "error": err} if err else {"success": True, "arrived": True}
                eta = self._eta.get(seq)
                now = time.monotonic()
                if eta is not None and now >= eta - lead:
                    return {"success": True, "arrived": True, "lead": lead}
                if now >= deadline:
                    return {"success": False, "error": f"timeout waiting for arrival of {seq}"}
                wait = deadline - now
                if eta is not None:
                    wait = min(wait, max(0.001, eta - lead - now))
                self._cv.wait(timeout=wait)

    def wait_idle(self, timeout: float) -> dict:
        """Block until the queue drains. Reports any batch that failed while it was draining."""
        with self._cv:
            ok = self._cv.wait_for(lambda: not self._q and self._done_seq >= self._seq, timeout=timeout)
            errors = dict(self._errors)
        if not ok:
            return {"success": False, "error": "timeout draining queue"}
        if errors:
            detail = "; ".join(f"seq {s}: {e}" for s, e in sorted(errors.items()))
            return {"success": False, "error": f"trajectory failed while draining ({detail})"}
        return {"success": True}

    def times(self) -> dict:
        """``{seq: [t_start, t_end]}`` epoch seconds for finished batches; drained on read."""
        with self._cv:
            out = {str(k): [v[0], v[1]] for k, v in self._times.items()}
            self._times.clear()
            return out

    def status(self) -> dict:
        with self._cv:
            return {"queued": len(self._q), "submitted": self._seq, "done": self._done_seq}

    def shutdown(self) -> None:
        """Stop the worker, cutting short anything it is streaming.

        The abort is what makes the join reliable: ``_stream`` only checks the epoch between
        setpoints, so without bumping it a worker part-way through a long batch would ignore
        ``_stop`` until the batch ended, the join would time out, and the caller would go on to
        relax and close the arm underneath a thread still commanding it.
        """
        self.abort()
        with self._cv:
            self._stop = True
            self._cv.notify_all()
        self._thread.join(timeout=5.0)
        if self._thread.is_alive():
            log.warning(f"[{self._arm.name}] trajectory worker did not stop; not safe to close the arm yet")

    # -- worker --------------------------------------------------------------------------------
    def _finish(self, seq: int, err: str | None, span: tuple[float, float] | None = None) -> None:
        with self._cv:
            if err:
                self._errors[seq] = err
            if span is not None:
                self._times[seq] = span
            self._eta.pop(seq, None)
            self._done_seq = max(self._done_seq, seq)
            self._cv.notify_all()

    def _run(self) -> None:
        while True:
            with self._cv:
                while not self._q and not self._stop:
                    self._cv.wait(timeout=0.1)
                if self._stop:
                    return
                epoch, seq, waypoints, default_duration = self._q.pop(0)
            if epoch != self._epoch:  # aborted before it started
                self._finish(seq, None)
                continue
            try:
                span = self._stream(epoch, seq, waypoints, default_duration)
                self._finish(seq, None, span)
            except Exception as e:  # noqa: BLE001 — a bad batch must not kill the worker
                log.exception(f"[{self._arm.name}] trajectory {seq} failed")
                self._finish(seq, str(e))

    def _stream(self, epoch: int, seq: int, waypoints: list, default_duration: float) -> tuple[float, float]:
        durations = [float(w.get("duration", default_duration) or default_duration) for w in waypoints]
        goals = [np.asarray(w["q_goal"], dtype=np.float64).reshape(-1)[:ARM_DOF] for w in waypoints]

        measured = self._arm.read_full()[:ARM_DOF]
        jump = float(np.abs(goals[0] - measured).max())
        if jump > MAX_START_JUMP_RAD:
            raise RuntimeError(
                f"first waypoint is {jump:.3f} rad from the measured pose (limit {MAX_START_JUMP_RAD}); "
                "refusing to step a stiff arm that far. Plan from the measured state, or `move_to` first."
            )

        now = time.monotonic()
        # Continuous clock across back-to-back batches; reset only when the queue genuinely idled,
        # otherwise a batch arriving after a pause replays at full rate trying to catch up.
        if self._t_next < now:
            self._t_next = now
        total = sum(durations)
        with self._cv:
            self._eta[seq] = self._t_next + total
            self._cv.notify_all()

        t_start = _now()
        period = 1.0 / STREAM_HZ
        for goal, dur in zip(goals, durations):
            if self._epoch != epoch:
                log.info(f"[{self._arm.name}] trajectory {seq} aborted mid-stream")
                break
            self._arm.set_arm(goal)
            self._t_next += max(dur, 0.0)
            # Pace to an ABSOLUTE deadline: command_joint_pos is cheap but not free, and sleeping
            # the full duration on top of it runs every trajectory slow by that overhead.
            while True:
                remaining = self._t_next - time.monotonic()
                if remaining <= 0:
                    break
                time.sleep(min(remaining, period))
        return (t_start, _now())


class StateCache:
    """~100 Hz snapshot of both arms, so the state socket answers instantly and never blocks.

    The control socket is parked inside a blocking trajectory for seconds at a time; the LeRobot
    joint sampler polls at 30 Hz throughout and has to get answers. Same split as the Franka shim's
    background poller.
    """

    def __init__(self, arms: dict[str, YamArm], hz: float = 100.0):
        self._arms = arms
        self._hz = hz
        self._lock = threading.Lock()
        self._data: dict | None = None
        self._stop = threading.Event()
        threading.Thread(target=self._run, name="yam-state-poller", daemon=True).start()

    def _run(self) -> None:
        period = 1.0 / self._hz
        while not self._stop.is_set():
            tick = time.perf_counter()
            try:
                snap = {"time_sec": _now()}
                for name, arm in self._arms.items():
                    snap[name] = arm.observation()
                with self._lock:
                    self._data = snap
            except Exception:
                pass  # keep the last good sample; a transient CAN hiccup is not worth dropping
            time.sleep(max(0.0, period - (time.perf_counter() - tick)))

    def get(self) -> dict | None:
        with self._lock:
            return self._data

    def stop(self) -> None:
        self._stop.set()


class YamServer:
    """Owns the arms, the per-arm streamers and the state cache; serves the three sockets."""

    def __init__(self, channels: dict[str, str]):
        self.arms: dict[str, YamArm] = {}
        try:
            for name, channel in channels.items():
                self.arms[name] = YamArm(name, channel)
        except BaseException:
            # i2rt's control threads outlive a failed __init__ and keep the CAN socket claimed, so
            # release whatever connected before re-raising.
            self.close()
            raise
        self._seq_lock = threading.Lock()
        self._seq = 0
        self.streamers = {name: TrajectoryStreamer(arm, self._next_seq) for name, arm in self.arms.items()}
        self.state = StateCache(self.arms)
        # Which arm an arm-less request means. tiptop drives one arm at a time and its
        # `execute_plan._QueuedArm` is deliberately embodiment-agnostic — it sends bare
        # `execute_trajectory` / `wait_trajectory_*` with no arm field — so the arm is session state
        # here rather than a field on every request. `yam_client.YamClient` sets it from the active
        # cuRobo embodiment (`robot.type: bimanual_yam_<arm>`) before it plans or executes.
        self._active_arm = next(iter(self.arms))

    def _next_seq(self) -> int:
        with self._seq_lock:
            self._seq += 1
            return self._seq

    # -- helpers -------------------------------------------------------------------------------
    def _arm(self, req: dict) -> YamArm:
        name = (req or {}).get("arm") or self._active_arm
        if name not in self.arms:
            raise KeyError(f"unknown arm {name!r}; this server has {sorted(self.arms)}")
        return self.arms[name]

    def _streamer(self, req: dict) -> TrajectoryStreamer:
        return self.streamers[self._arm(req).name]

    def _streamer_for_seq(self, seq: int, req: dict) -> TrajectoryStreamer:
        """Route a seq-addressed command to whichever arm actually issued that seq.

        Seqs are server-wide, so the arm is recoverable from the number alone; falling back to the
        active arm keeps seq 0 (tiptop's capability probe, which no streamer owns) working.
        """
        for streamer in self.streamers.values():
            if streamer.owns(seq):
                return streamer
        return self._streamer(req)

    # -- control socket ------------------------------------------------------------------------
    def handle_control(self, req: dict) -> dict:
        cmd = (req or {}).get("command", "")
        if cmd == "ping":
            return {"success": True}

        if cmd == "get_robot_state":
            obs = self._arm(req).observation()
            return {"success": True, "data": obs}

        if cmd == "execute_trajectory":
            data = req.get("data", {}) or {}
            waypoints = data.get("waypoints", [])
            if not waypoints:
                return {"success": False, "error": "empty waypoints"}
            streamer = self._streamer(data)
            default_duration = float(data.get("default_duration", 0.02) or 0.02)
            seq = streamer.submit(waypoints, default_duration)
            if data.get("queue"):
                return {"success": True, "queued": True, "seq": seq}
            total = sum(float(w.get("duration", default_duration) or default_duration) for w in waypoints)
            return streamer.wait_for(seq, timeout=total * 1.5 + 10.0)

        if cmd == "set_active_arm":
            name = req.get("arm")
            if name not in self.arms:
                return {"success": False, "error": f"unknown arm {name!r}; this server has {sorted(self.arms)}"}
            self._active_arm = name
            log.info(f"active arm -> {name}")
            return {"success": True, "arm": name}

        if cmd == "wait_trajectory_arrival":
            seq = int(req.get("seq", 0))
            return self._streamer_for_seq(seq, req).wait_arrival(
                seq, lead=float(req.get("lead", 0.0)), timeout=float(req.get("timeout", 120.0)),
            )

        if cmd == "wait_trajectory_done":
            timeout = float(req.get("timeout", 120.0))
            # No arm named -> drain every arm, which is what end-of-plan teardown wants.
            names = [self._arm(req).name] if req.get("arm") else list(self.streamers)
            for name in names:
                res = self.streamers[name].wait_idle(timeout)
                if not res.get("success"):
                    return res
            return {"success": True}

        if cmd == "trajectory_times":
            if req.get("arm"):
                return {"success": True, "times": self._streamer(req).times()}
            merged: dict = {}
            for streamer in self.streamers.values():
                merged.update(streamer.times())
            return {"success": True, "times": merged}

        if cmd == "trajectory_queue_status":
            if req.get("arm"):
                return {"success": True, **self._streamer(req).status()}
            return {"success": True, "arms": {n: s.status() for n, s in self.streamers.items()}}

        if cmd == "abort_trajectory":
            names = [self._arm(req).name] if req.get("arm") else list(self.streamers)
            for name in names:
                self.streamers[name].abort()
            return {"success": True}

        if cmd == "move_to":
            arm = self._arm(req)
            q_goal = np.asarray(req.get("q_goal", []), dtype=np.float64).reshape(-1)
            if len(q_goal) not in (ARM_DOF, FULL_DOF):
                return {"success": False, "error": f"q_goal must be {ARM_DOF} or {FULL_DOF} long, got {len(q_goal)}"}
            duration = float(req.get("duration", 3.0))
            if not 0.5 <= duration <= 60.0:
                # This is the one motion path exempt from MAX_START_JUMP_RAD, because parking
                # genuinely has to travel. A short duration would turn that exemption into a
                # full-distance step at the interpolation rate, which is exactly what the guard
                # exists to prevent — so the duration is bounded instead.
                return {"success": False, "error": f"duration must be between 0.5 and 60 s, got {duration}"}
            return self._move_to(arm, q_goal, duration)

        if cmd == "hold":
            self._arm(req).engage()
            return {"success": True}

        if cmd == "free_drive":
            arm = self._arm(req)
            drained = self.streamers[arm.name].wait_idle(timeout=5.0)
            if not drained.get("success"):
                return {"success": False, "error": "arm is still executing a trajectory"}
            arm.free_drive(hold_gripper=bool(req.get("hold_gripper", True)))
            return {"success": True}

        if cmd == "relax":
            self._arm(req).relax()
            return {"success": True}

        return {"success": False, "error": f"Unknown control command: {cmd}"}

    def _move_to(self, arm: YamArm, q_goal: np.ndarray, duration: float) -> dict:
        """Interpolated move for parking/homing — the one path allowed to cross a large distance.

        ``execute_trajectory`` refuses a big first step because a planned trajectory should always
        start where the arm is. Parking an idle arm at the neutral posture genuinely has to travel,
        so it ramps there over ``duration`` instead.
        """
        streamer = self.streamers[arm.name]
        drained = streamer.wait_idle(timeout=max(duration, 5.0))
        if not drained.get("success"):
            return {"success": False, "error": "arm is still executing a trajectory"}
        try:
            start = arm.read_full()
            goal = start.copy()
            goal[:ARM_DOF] = q_goal[:ARM_DOF]
            if len(q_goal) == FULL_DOF:
                goal[ARM_DOF] = float(np.clip(q_goal[ARM_DOF], 0.0, 1.0))
            steps = max(2, int(duration * 100.0))
            for alpha in np.linspace(0.0, 1.0, steps):
                blend = (1.0 - alpha) * start + alpha * goal
                arm.set_arm(blend[:ARM_DOF])
                if len(q_goal) == FULL_DOF:
                    arm.set_gripper(blend[ARM_DOF])
                time.sleep(duration / steps)
            return {"success": True}
        except Exception as e:  # noqa: BLE001
            log.exception(f"[{arm.name}] move_to failed")
            return {"success": False, "error": str(e)}

    # -- gripper socket ------------------------------------------------------------------------
    def handle_gripper(self, req: dict) -> dict:
        cmd = (req or {}).get("command", "")
        if cmd == "ping":
            return {"success": True}
        if cmd in ("get_gripper_state", "get_state"):
            return self._gripper_state(self._arm(req))
        if cmd in ("open_gripper", "close_gripper"):
            arm = self._arm(req)
            target = GRIPPER_OPEN if cmd == "open_gripper" else GRIPPER_CLOSED
            # bamboo takes an absolute `width` in metres on open; honour it so the client contract
            # matches, but the common case is a bare open/close.
            if cmd == "open_gripper" and req.get("width") is not None:
                target = float(np.clip(float(req["width"]) / GRIPPER_STROKE_M, 0.0, 1.0))
            arm.set_gripper(target)
            if bool(req.get("blocking", True)):
                self._wait_gripper_settled(arm, target)
            return {"success": True, "target": target}
        return {"success": False, "error": f"Unknown gripper command: {cmd}"}

    def _gripper_state(self, arm: YamArm) -> dict:
        obs = arm.observation()
        normalized = float(obs["gripper"])
        target = arm.gripper_target
        moving = abs(float(obs["gripper_vel"])) > 0.02
        # "Grasped" = commanded shut but stopped short of it with load on the motor: the jaws are on
        # something. The effort threshold is deliberately loose — this flag is diagnostic only,
        # nothing in tiptop's control flow branches on it.
        grasped = bool(target <= 0.05 and normalized > 0.05 and abs(float(obs["gripper_eff"])) > 0.05)
        return {
            "success": True,
            "state": {
                "width": normalized * GRIPPER_STROKE_M,
                "normalized": normalized,
                "target": target,
                "is_grasped": grasped,
                "is_moving": moving,
                "current": float(obs["gripper_eff"]),
            },
        }

    @staticmethod
    def _wait_gripper_settled(arm: YamArm, target: float, timeout: float = 2.0) -> None:
        """Block until the jaws stop moving (reached the target, or stalled on an object)."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            time.sleep(0.02)
            obs = arm.observation()
            if abs(float(obs["gripper"]) - target) < 0.02:
                return
            if abs(float(obs["gripper_vel"])) < 0.01 and time.monotonic() > deadline - timeout + 0.25:
                return  # stalled: closed onto an object, or already where it was asked to be

    # -- state socket --------------------------------------------------------------------------
    def handle_state(self, req: dict) -> dict:
        cmd = (req or {}).get("command", "")
        if cmd == "ping":
            return {"success": True}
        if cmd != "get_robot_state":
            return {"success": False, "error": f"Unknown state command: {cmd}"}
        data = self.state.get()
        if data is None:
            return {"success": False, "error": "state cache not warmed yet"}
        if req.get("arm"):
            name = req["arm"]
            if name not in data:
                return {"success": False, "error": f"unknown arm {name!r}"}
            return {"success": True, "data": {**data[name], "time_sec": data["time_sec"]}}
        return {"success": True, "data": data}

    def close(self) -> None:
        for streamer in getattr(self, "streamers", {}).values():
            streamer.shutdown()
        state = getattr(self, "state", None)
        if state is not None:
            state.stop()
        for arm in getattr(self, "arms", {}).values():
            arm.close()


def _serve(socket: zmq.Socket, handler, label: str) -> None:
    """REP loop. Always sends exactly one reply per request — a double send wedges the socket.

    Exits when the ZMQ context is destroyed, which is how shutdown reaches a thread parked in
    ``recv()``: closing the socket from another thread is undefined, but terminating the context
    raises ``ContextTerminated`` in the blocked call. Without that this loop would catch the
    resulting error and spin.
    """
    log.info(f"{label} socket listening")
    while True:
        replied = False
        try:
            req = msgpack.unpackb(socket.recv(), raw=False)
            resp = handler(req)
            replied = True
            socket.send(msgpack.packb(resp, use_bin_type=True))
        except (zmq.ContextTerminated, zmq.ZMQError) as e:
            if isinstance(e, zmq.ContextTerminated) or e.errno in (zmq.ETERM, zmq.ENOTSOCK):
                log.info(f"{label} socket closing")
                return
            log.exception(f"{label} socket error")
            return
        except Exception as e:  # noqa: BLE001
            log.exception(f"{label} handler error")
            if not replied:
                try:
                    socket.send(msgpack.packb({"success": False, "error": f"{type(e).__name__}: {e}"}))
                except Exception:
                    return


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bind", default="0.0.0.0")
    p.add_argument("--control-port", type=int, default=5555)
    p.add_argument("--gripper-port", type=int, default=5559)
    p.add_argument("--state-port", type=int, default=5557)
    p.add_argument("--left-channel", default="can_left")
    p.add_argument("--right-channel", default="can_right")
    p.add_argument("--arms", default="left,right",
                   help="comma-separated arms to connect (e.g. 'left' for a single-arm bring-up)")
    p.add_argument("--park", action="store_true",
                   help="move every connected arm to the neutral posture on startup. Only do this "
                        "with the workspace clear — the arms travel to get there.")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    wanted = [a.strip() for a in args.arms.split(",") if a.strip()]
    unknown = [a for a in wanted if a not in ARMS]
    if unknown:
        raise SystemExit(f"--arms must be a subset of {ARMS}, got {unknown}")
    channels = {"left": args.left_channel, "right": args.right_channel}
    server = YamServer({a: channels[a] for a in wanted})

    if args.park:
        for name, arm in server.arms.items():
            log.info(f"[{name}] parking at the neutral posture")
            server._move_to(arm, np.asarray(NEUTRAL_Q), duration=4.0)

    ctx = zmq.Context.instance()
    sockets = []
    try:
        for port, handler, label in (
            (args.control_port, server.handle_control, "control"),
            (args.gripper_port, server.handle_gripper, "gripper"),
            (args.state_port, server.handle_state, "state"),
        ):
            sock = ctx.socket(zmq.REP)
            sock.bind(f"tcp://{args.bind}:{port}")
            sockets.append(sock)
            threading.Thread(target=_serve, args=(sock, handler, label), daemon=True).start()
            log.info(f"{label} bound on tcp://{args.bind}:{port}")

        log.info(f"YAM arm server ready: arms={sorted(server.arms)}")
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        log.info("Shutting down (Ctrl-C)")
    finally:
        # Arms first: stop commanding and release CAN before anything else goes away. Then destroy
        # the context, which unblocks the serve threads parked in recv() (closing their sockets from
        # here instead would be undefined behaviour) and closes every socket for us.
        server.close()
        ctx.destroy(linger=0)


if __name__ == "__main__":
    main()

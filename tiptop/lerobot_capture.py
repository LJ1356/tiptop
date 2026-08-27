"""Build ``robot_state.npz`` + ``_meta.json`` for the data-collection LeRobot export.

tiptop executes a TAMP plan as a sequence of dense joint-impedance trajectories. The recorded
episode keeps the MEASURED robot state and the COMMANDED plan strictly decoupled (ARCHITECTURE.md
§3), which is the whole point of this module: an earlier design wrote plan positions as
"proprioception" and a lead-shifted copy of the measured gripper as the gripper action -- a
feedback trap that is deliberately gone.

:class:`JointSampler` and :class:`GripperSampler` are background threads that sample the measured
arm state (over the shim's dedicated state port) and gripper width (over the gripper port) during
execution, each with wall-clock timestamps. :func:`dump_raw_episode` resamples everything onto a
uniform 15 Hz wall-clock grid: the COMMANDED arrays come from the plan (``tiptop_plan.json``,
spread across each step's measured ``[t_start, t_end]``), the MEASURED arrays come from the samplers
by nearest timestamp, and ``frame_time`` (float64 epoch seconds) is the master timeline the build
uses to align camera frames.
"""

import json
import logging
import os
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np

_log = logging.getLogger(__name__)

# State-port default; the shim's background-poller REP socket that JointSampler reads from.
DEFAULT_STATE_PORT = 5557

# Robot control / DROID target rate. The plan is stored at 50 Hz; we resample to this.
DEFAULT_TARGET_FPS = 15

# Robotiq 2F-85 fully-open width in metres; used to normalise the measured width to the
# DROID gripper convention (0 = open, 1 = closed).
GRIPPER_MAX_WIDTH = 0.085

# ---- the DEPLOYABLE joint-velocity action (``action_joint_velocity``) --------------------------- #
#
# The trained action channel is DROID's normalized joint velocity, and the deploy executor reads it
# back in that space: ``RobotEnv(action_space="joint_velocity")`` runs each command through
# ``droid/robot_ik/robot_ik_solver.py`` as ``joint_delta = jv * max_joint_delta`` once per control
# step. Inverting that is the whole definition -- ``jv = (commanded - measured) / max_joint_delta``:
#
#     action_joint_velocity = DROID_JV_GAIN * (cmd_joint_position - joint_position)
#
# So it is a TRACKING ERROR, not a speed. It grows when the arm lags, which is what makes a policy
# trained on it push harder under load, and it is the quantity DROID itself stores (fitted R^2 0.923
# against 200 lerobot/droid_1.0.1 episodes).
#
# ``cmd_joint_velocity`` is NOT this and is deliberately left alone: it is the cuTAMP plan's own
# feedforward rad/s, which is what oopsie_export publishes (its JV_SCALE table) and what the plan
# analyses read. The two are different physical quantities, they differ per frame in SIGN as well as
# magnitude, and the historical bug was exporting the plan velocity into the trained action channel:
# every TAMP dataset built before this landed under-commands by ~1.27x against the executor, worst on
# the elbow. Writing BOTH arrays under names that say what they are is the fix -- a single key whose
# meaning depended on who captured the episode is what allowed the confusion.
DROID_JV_GAIN = 5.0  # 1 / max_joint_delta (0.2 rad), from droid/robot_ik/robot_ik_solver.py

# Sanity band for the per-episode scale statistic (see _check_jv_ratio). This check exists because
# the gain above is DROID's, while the lag it acts on is tiptop's: the blocking
# ``move_to_joint_positions`` executor happens to track closely enough that the product lands where
# DROID's does. That is an empirical property of the CURRENT executor, not an identity -- retune its
# gains, its blending or its step timing and the ratio moves, and without a warning the next
# collection run would ship a silently mis-scaled dataset.
#
# Bounds are the measured per-episode range of cohorts this capture is meant to resemble: this lab's
# teleop spans 1.33-1.77 (n=20) and the four TAMP datasets recomputed this way span 1.15-2.20 (n=80),
# so the band is their union plus ~10% margin. Pooled COHORT means are much tighter (teleop 1.48,
# TAMP 1.44, lerobot/droid_1.0.1 1.44) -- it is the per-episode spread that forces a band this wide.
#
# What it therefore catches: a GROSS scale error, which is the realistic executor-drift failure --
# a missing gain lands at 0.29, a stray /3 at 0.50, a doubled tracking lag at ~3.0.
# What it does NOT catch: the historical bug (exporting the plan's feedforward rad/s), whose
# per-episode ratios run 1.05-1.33 and therefore OVERLAP healthy episodes -- no per-episode
# threshold can separate those two populations, and pretending otherwise would be a false comfort.
# That bug is prevented structurally instead: ``action_joint_velocity`` is derived in one place from
# the two arrays it must relate, so there is no longer a second channel for it to be confused with.
#
# Warn, never refuse: a ratio outside the band is a real rollout with a suspect action channel, and
# the operator needs to see it rather than lose the episode.
DROID_JV_RATIO_BAND = (1.05, 2.50)

# A joint counts toward the statistic only if it actually moved: mean |velocity| over the episode
# above this, in rad/s. Near-static joints put a ~0 denominator under the ratio -- on DROID that
# alone produces per-episode values from 0.0 to 73.5, which is noise, not signal.
_JV_RATIO_MOVING_FLOOR = 0.05


def _read_gripper_width(robot, arm: str | None = None) -> float | None:
    """Best-effort read of the measured gripper opening width in metres. None if unavailable.

    The bamboo client returns ``{"success": ..., "state": {"width": <m>, ...}}``; older
    code read ``["width"]`` directly and always missed, defaulting the gripper to a
    constant. Navigate the real payload, tolerating the flatter shapes too.

    ``arm`` addresses a specific hand -- only meaningful for a YamClient in dual mode, where there
    is no default active arm to fall back to (see ``YamClient.arm``); every other client/embodiment
    leaves it None and nothing changes.
    """
    try:
        if not hasattr(robot, "get_gripper_state"):
            return None
        res = robot.get_gripper_state(arm) if arm is not None else robot.get_gripper_state()
        if not isinstance(res, dict):
            return float(res)
        if res.get("success") is False:
            return None
        state = res.get("state", res)
        width = state.get("width") if isinstance(state, dict) else state
        return None if width is None else float(width)
    except Exception:
        return None


class GripperSampler:
    """Background thread sampling the *measured* gripper width during execution.

    Reads go over the gripper server's own ZMQ socket (separate from the joint-impedance
    control socket), so they are not starved by the blocking trajectory execution that
    previously throttled joint-state polling to ~20 samples. A ZMQ REQ socket is not
    thread-safe, so this acquires its OWN client connection rather than sharing the
    command client's; it falls back to the shared client (and logs) if that fails.

    ``samples`` holds ``(wall_clock_seconds, closedness)`` pairs, closedness on the DROID
    convention 0 = open, 1 = closed. Use as a context manager around plan execution::

        with GripperSampler(robot) as g:
            execute_cutamp_plan(plan, client=robot)
        # g.samples now holds the measured gripper trace
    """

    def __init__(self, robot, fps: int = 20):
        from tiptop.utils import new_robot_client

        self._owns_client = False
        try:
            self.robot = new_robot_client()
            self._owns_client = True
        except Exception as e:
            _log.warning(
                f"Could not create a dedicated gripper-state client ({e}); falling back to the shared client."
            )
            self.robot = robot
        self.fps = int(fps)
        self.samples: list[tuple[float, float]] = []  # (wall_seconds, closedness)
        self.width_samples: list[tuple[float, float]] = []  # (wall_seconds, raw width m) for diagnostics
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._unavailable_logged = False

    def _loop(self):
        period = 1.0 / self.fps
        while not self._stop.is_set():
            tick = time.perf_counter()
            width = _read_gripper_width(self.robot)
            if width is not None:
                now = time.time()
                closed = float(np.clip(1.0 - width / GRIPPER_MAX_WIDTH, 0.0, 1.0))
                self.samples.append((now, closed))
                self.width_samples.append((now, width))
            elif not self._unavailable_logged:
                _log.warning("Measured gripper width unavailable; export will fall back to plan gripper events")
                self._unavailable_logged = True
            time.sleep(max(0.0, period - (time.perf_counter() - tick)))

    def __enter__(self) -> "GripperSampler":
        self._thread = threading.Thread(target=self._loop, name="lerobot-gripper-sampler", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self.width_samples:
            w = np.asarray([x[1] for x in self.width_samples])
            # If this range is tiny or not within ~[0, 0.085] m, the width units are off
            # and the [0,1] normalisation will saturate -- worth seeing in the log.
            _log.info(
                "Gripper sampler: %d reads, raw width range [%.4f, %.4f] (expect ~0..0.085 m)",
                len(w), float(w.min()), float(w.max()),
            )
        if self._owns_client:
            try:
                self.robot.close()
            except Exception as e:
                _log.warning(f"Failed to close dedicated gripper-state client: {e}")
        return False


class JointSampler:
    """Background thread sampling the *measured* arm state during execution, over the STATE port.

    Talks to the shim's dedicated state socket (``bamboo_polymetis_shim._state_handler``) with its
    own ZMQ REQ + msgpack connection -- NOT a :class:`BambooFrankaClient`, whose get_robot_state
    goes to the control port that is blocked inside the trajectory execution. The state port serves
    a cache filled by a background poller, so it answers even mid-motion.

    ``samples`` holds ``(wall_seconds, q[7], dq[7])`` tuples (measured joint positions/velocities).
    A dead/absent state server degrades to a warning + empty ``samples`` (RCVTIMEO), never a hang.
    Use as a context manager around plan execution::

        with JointSampler() as j:
            execute_cutamp_plan(plan, client=robot)
        # j.samples now holds the measured joint trace
    """

    def __init__(self, fps: int = 30):
        import msgpack
        import zmq

        from tiptop.config import tiptop_cfg

        self._msgpack = msgpack
        self._zmq = zmq
        self.host = tiptop_cfg().robot.host
        self.port = int(os.environ.get("TIPTOP_STATE_PORT", DEFAULT_STATE_PORT))
        self.fps = int(fps)
        self.samples: list[tuple[float, np.ndarray, np.ndarray]] = []  # (wall_seconds, q[7], dq[7])
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._unavailable_logged = False
        self._ctx = zmq.Context()
        self._sock: "zmq.Socket | None" = None
        self._connect()

    def _connect(self) -> None:
        """(Re)create the REQ socket. A timed-out recv leaves REQ unusable, so recover by reconnecting."""
        if self._sock is not None:
            self._sock.close(linger=0)
        self._sock = self._ctx.socket(self._zmq.REQ)
        self._sock.setsockopt(self._zmq.RCVTIMEO, 300)  # ms; a dead server degrades to empty samples
        self._sock.setsockopt(self._zmq.LINGER, 0)
        self._sock.connect(f"tcp://{self.host}:{self.port}")

    def _warn_unavailable(self, detail: str) -> None:
        if not self._unavailable_logged:
            _log.warning(
                "Joint state server tcp://%s:%d unavailable (%s); measured joint samples will be empty "
                "-- the raw episode dump will be skipped. Is the shim's --state-port running?",
                self.host, self.port, detail,
            )
            self._unavailable_logged = True

    def _loop(self) -> None:
        period = 1.0 / self.fps
        req = self._msgpack.packb({"command": "get_robot_state"})
        while not self._stop.is_set():
            tick = time.perf_counter()
            try:
                self._sock.send(req)
                reply = self._msgpack.unpackb(self._sock.recv(), raw=False)
                data = reply.get("data") if isinstance(reply, dict) and reply.get("success") else None
                if data:
                    q = np.asarray(data.get("q", []), dtype=np.float32).reshape(-1)
                    dq = np.asarray(data.get("dq", []), dtype=np.float32).reshape(-1)
                    if q.shape == (7,) and dq.shape == (7,):
                        self.samples.append((time.time(), q, dq))
            except self._zmq.Again:
                self._warn_unavailable("recv timeout")
                self._connect()
            except Exception as e:  # noqa: BLE001 - degrade to empty samples, never crash the rollout
                self._warn_unavailable(str(e))
                self._connect()
            time.sleep(max(0.0, period - (time.perf_counter() - tick)))

    def __enter__(self) -> "JointSampler":
        self._thread = threading.Thread(target=self._loop, name="lerobot-joint-sampler", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        _log.info("Joint sampler: %d measured samples", len(self.samples))
        if self._sock is not None:
            self._sock.close(linger=0)
        self._ctx.term()
        return False


def _load_plan(plan_path: Path) -> dict:
    """Load a serialized tiptop plan, converting trajectory arrays to float32 ndarrays."""
    with open(plan_path) as f:
        plan = json.load(f)
    for step in plan["steps"]:
        if step["type"] == "trajectory":
            step["positions"] = np.asarray(step["positions"], dtype=np.float32)
            step["velocities"] = np.asarray(step["velocities"], dtype=np.float32)
    return plan


def _flatten_plan(plan: dict, timeline: list | None = None, dof: int = 7) -> dict:
    """Flatten plan steps into dense 50 Hz arrays.

    ``dof`` is the arm's joint count — 7 for the Franka, 6 for one YAM arm. It only sizes the
    zero-velocity hold rows and the ``q_init`` fallback; every other array comes from the plan.

    Returns a dict with, for the M dense rows:
      positions[M,dof], velocities[M,dof], gripper[M], dt[M] (per-row duration),
      t_plan[M] (start time of each row on the plan clock), and
      t_wall[M] (wall-clock time of each row, NaN where no execution timeline).

    The gripper channel is held across trajectory rows: it starts at 0.0 (open) and
    flips to 1.0 (closed) / 0.0 (open) at each ``gripper`` event, taking effect on the
    rows that follow. Wall-clock times come from ``timeline`` (one entry per plan step,
    in order, each ``{"t_start", "t_end"}``): a trajectory step's rows are spread
    linearly between its measured start and end.

    A gripper step is instantaneous in the plan but takes real time on the robot; the timeline
    reports that measured duration. We insert stationary "hold" rows (arm frozen at its last pose,
    zero velocity) spanning it, so the export emits frames covering the actuation instead of
    jumping straight to the fully open/closed state.

    With overlapped execution (see ``execute_plan.GRIPPER_OVERLAP``) this measured duration is
    only the brief post-fire contact settle -- the rest of the actuation happens while the NEXT
    trajectory runs -- so these hold rows are just a few frames and the open<->close ramp is
    captured across the following (moving) trajectory. Keeping that stationary run short is what
    lets the gripper transition survive the non-idle training filter. Without overlap the hold
    spans the whole ~0.5-1 s actuation, as before.
    """
    HOLD_DT = 0.02  # 50 Hz, matching the plan's trajectory rate, for inserted hold rows
    pos_chunks, vel_chunks, grip_chunks, dt_chunks, twall_chunks = [], [], [], [], []
    q_init = np.asarray(plan.get("q_init", np.zeros(dof)), dtype=np.float32).reshape(-1)
    last_pos = q_init  # arm pose to freeze at during a gripper pause
    g = 0.0  # DROID convention: 0 = open, 1 = closed. Episodes start open.
    for i, step in enumerate(plan["steps"]):
        entry = timeline[i] if (timeline is not None and i < len(timeline)) else None
        has_wall = entry is not None and entry.get("t_start") is not None and entry.get("t_end") is not None

        if step["type"] == "trajectory":
            pos = np.asarray(step["positions"], dtype=np.float32)
            vel = np.asarray(step["velocities"], dtype=np.float32)
            n = len(pos)
            if n == 0:
                continue
            dt = float(step["dt"])
            pos_chunks.append(pos)
            vel_chunks.append(vel)
            grip_chunks.append(np.full(n, g, dtype=np.float32))
            dt_chunks.append(np.full(n, dt, dtype=np.float64))
            if has_wall:
                ts, te = float(entry["t_start"]), float(entry["t_end"])
                twall = np.full(n, ts, dtype=np.float64) if n == 1 else np.linspace(ts, te, n)
            else:
                twall = np.full(n, np.nan, dtype=np.float64)
            twall_chunks.append(twall)
            last_pos = pos[-1]
        elif step["type"] == "gripper":
            g = 1.0 if step["action"] == "close" else 0.0
            # Insert stationary hold rows spanning the measured actuation pause so the
            # gripper ramp is captured. Skipped without a timeline (no known duration).
            if has_wall:
                ts, te = float(entry["t_start"]), float(entry["t_end"])
                n_hold = max(1, round((te - ts) / HOLD_DT))
                pos_chunks.append(np.tile(last_pos, (n_hold, 1)))
                vel_chunks.append(np.zeros((n_hold, dof), dtype=np.float32))
                grip_chunks.append(np.full(n_hold, g, dtype=np.float32))
                dt_chunks.append(np.full(n_hold, HOLD_DT, dtype=np.float64))
                twall_chunks.append(np.linspace(ts, te, n_hold))

    if not pos_chunks:
        return {k: np.empty((0,)) for k in ("positions", "velocities", "gripper", "dt", "t_plan", "t_wall")}

    positions = np.concatenate(pos_chunks, axis=0)
    velocities = np.concatenate(vel_chunks, axis=0)
    gripper = np.concatenate(grip_chunks, axis=0)
    dt = np.concatenate(dt_chunks, axis=0)
    t_wall = np.concatenate(twall_chunks, axis=0)
    # Start time of each row on the plan clock: 0, dt0, dt0+dt1, ...
    t_plan = np.concatenate([[0.0], np.cumsum(dt)[:-1]])
    return {
        "positions": positions,
        "velocities": velocities,
        "gripper": gripper,
        "dt": dt,
        "t_plan": t_plan,
        "t_wall": t_wall,
    }


def _gripper_from_measurements(frame_wall: np.ndarray, gripper_samples: list | None) -> np.ndarray | None:
    """Per-frame gripper closedness from the measured trace, aligned by wall-clock time.

    Returns None (caller falls back to plan events) if there is no usable measured trace
    or the frames have no wall-clock times to align against.
    """
    if not gripper_samples or not np.all(np.isfinite(frame_wall)):
        return None
    gs = np.asarray(gripper_samples, dtype=np.float64)  # [K, 2]: (wall_seconds, closedness)
    if gs.ndim != 2 or len(gs) == 0:
        return None
    gt, gv = gs[:, 0], gs[:, 1]
    nearest = np.abs(gt[None, :] - frame_wall[:, None]).argmin(axis=1)
    return gv[nearest].astype(np.float32)


def _nearest_by_wall(sample_t: np.ndarray, sample_v: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Nearest ``sample_v`` row for each grid time by wall clock (sample_t need not be uniform)."""
    nearest = np.abs(sample_t[None, :] - grid[:, None]).argmin(axis=1)
    return sample_v[nearest]


def _check_jv_ratio(action_jv: np.ndarray, joint_position: np.ndarray, *, fps: int, where: str) -> float | None:
    """Warn when the deployable action's scale leaves DROID's band (see DROID_JV_RATIO_BAND).

    The statistic is per-joint ``mean|action| / mean|measured velocity|``, averaged over the joints
    that actually moved (``_JV_RATIO_MOVING_FLOOR``) -- the same one the DROID/teleop/TAMP cohorts
    were compared on. Returns the ratio, or None when it is not computable (a too-short episode, or
    one where fewer than two joints moved), and never raises: this is a data-quality signal, not a
    gate.
    """
    if len(joint_position) < 2:
        return None
    realized = np.abs(np.diff(joint_position, axis=0) * float(fps)).mean(axis=0)
    commanded = np.abs(action_jv[:-1]).mean(axis=0)
    moving = realized > _JV_RATIO_MOVING_FLOOR
    if int(moving.sum()) < 2:
        return None
    ratio = float((commanded[moving] / realized[moving]).mean())
    lo, hi = DROID_JV_RATIO_BAND
    if not (lo <= ratio <= hi):
        _log.warning(
            "action_joint_velocity scale %.2f is OUTSIDE DROID's band [%.2f, %.2f] for %s. The "
            "episode is still usable, but a policy trained on it will %s the arm at deploy. This "
            "usually means the executor's tracking lag changed (gains, blending or step timing) -- "
            "re-measure against lerobot/droid_1.0.1 before collecting a full dataset.",
            ratio, lo, hi, where, "under-drive" if ratio < lo else "over-drive",
        )
    else:
        _log.debug("action_joint_velocity scale %.2f (DROID band [%.2f, %.2f])", ratio, lo, hi)
    return ratio


def dump_raw_episode(
    save_dir: Path,
    plan_path: Path,
    *,
    timeline: list,
    joint_samples: list,
    gripper_samples: list,
    instruction: str,
    cameras: dict[str, str],
    fps: int = DEFAULT_TARGET_FPS,
    config_id: str | None = None,
    record_start: float | None = None,
    record_stop: float | None = None,
    trajectory_id: str | None = None,
) -> Path | None:
    """Write ``robot_state.npz`` + ``_meta.json`` (ARCHITECTURE.md §3) for one executed rollout.

    Resamples everything onto a uniform ``fps`` wall-clock grid over the execution timeline:
      * COMMANDED arrays (cmd_joint_position/velocity, binary cmd_gripper) come from the tiptop
        plan, spread across each step's measured [t_start, t_end] by :func:`_flatten_plan`, taken at
        the nearest-preceding dense row per grid time.
      * MEASURED arrays (joint_position, gripper_position) come from the samplers by nearest
        wall-clock sample -- proprioception is the true measured state, decoupled from the command.
      * DERIVED: ``action_joint_velocity``, the deployable DROID-convention action, is the tracking
        error between the two -- ``DROID_JV_GAIN * (cmd_joint_position - joint_position)``. This is
        the channel a policy is trained on; ``cmd_joint_velocity`` (the plan's own feedforward
        rad/s) is a DIFFERENT quantity and is kept alongside it, not replaced.

    ``record_start`` / ``record_stop`` (epoch seconds bracketing the camera recording window from
    :func:`recording.record_cameras`) are written into ``_meta.json`` so the build can align each
    state frame to a camera frame by wall clock (ARCHITECTURE.md "Camera <-> state alignment"); each
    is written as a float, or ``None`` when unavailable.

    A plan that stopped early (a teleop hand-off) is dumped as the partial rollout it is: only the
    executed prefix of the plan has wall times, and only that prefix is kept. ``trajectory_id`` ties
    such a leg to the other legs of the same task attempt (see ``collect/merge_trajectory.py``).

    Returns the npz path, or None if fewer than two rows executed or the measured joint trace is
    missing (in which case we REFUSE to fall back to plan positions -- that silent fallback is the
    exact proprioception bug this rewrite fixes).
    """
    save_dir = Path(save_dir)
    plan_path = Path(plan_path)
    if not plan_path.is_file():
        _log.warning("No plan at %s; skipping raw episode dump", plan_path)
        return None

    dense = _flatten_plan(_load_plan(plan_path), timeline=timeline)
    t_wall = dense["t_wall"]
    # A teleop hand-off (and any other early stop) leaves the tail of the plan unexecuted. Those
    # steps get no timeline entry, so _flatten_plan gives them NaN wall times -- and since
    # execute_cutamp_plan appends one entry per step, in order, immediately before honouring
    # should_stop, the executed rows are always a contiguous PREFIX. Keep that prefix and dump the
    # partial rollout: it is a real leg of a hand-off trajectory, and dropping it threw away every
    # state/action the human-in-the-loop run produced.
    executed = int(np.count_nonzero(np.isfinite(t_wall)))
    if executed and not np.all(np.isfinite(t_wall[:executed])):
        _log.error("Execution timeline has non-finite wall times inside the executed prefix "
                   "(%d finite of %d rows); refusing to guess where the rollout stopped. save_dir=%s",
                   executed, len(t_wall), save_dir)
        return None
    if executed < len(t_wall):
        _log.info("Partial rollout: keeping the %d executed rows of %d (the plan stopped early, "
                  "e.g. a teleop hand-off)", executed, len(t_wall))
        dense = {k: v[:executed] for k, v in dense.items()}
        t_wall = dense["t_wall"]
    m = len(t_wall)
    if m < 2:
        _log.warning("Plan has no usable execution timeline (%d executed rows); skipping raw episode dump", m)
        return None

    # Uniform fps grid over the measured wall-clock span; the last frame clamps to t_wall[-1].
    t0, t1 = float(t_wall[0]), float(t_wall[-1])
    n = max(2, int(round((t1 - t0) * fps)) + 1)
    grid = np.minimum(t0 + np.arange(n) / float(fps), t1)

    # COMMANDED: nearest-preceding dense row (t_wall is non-decreasing across steps).
    idx = np.clip(np.searchsorted(t_wall, grid, side="right") - 1, 0, m - 1)
    cmd_joint_position = dense["positions"][idx].astype(np.float32)  # [n, 7]
    cmd_joint_velocity = dense["velocities"][idx].astype(np.float32)  # [n, 7]
    cmd_gripper = dense["gripper"][idx].astype(np.float32)  # [n], plan command
    assert np.all((cmd_gripper == 0.0) | (cmd_gripper == 1.0)), "plan gripper command is not binary 0/1"

    # MEASURED joints: refuse to fabricate proprioception from the plan if the trace is missing.
    js = list(joint_samples or [])
    if not js:
        _log.error("MEASURED joint trace is EMPTY; refusing to dump raw episode (would falsely record "
                   "plan positions as proprioception -- the bug this rewrite fixes). save_dir=%s", save_dir)
        return None
    js_t = np.asarray([s[0] for s in js], dtype=np.float64)
    js_q = np.stack([np.asarray(s[1], dtype=np.float32).reshape(-1) for s in js])  # [K, 7]
    if js_q.ndim != 2 or js_q.shape[1] != 7:
        _log.error("MEASURED joint trace is unusable (shape %s); refusing to dump raw episode. save_dir=%s",
                   js_q.shape, save_dir)
        return None
    joint_position = _nearest_by_wall(js_t, js_q, grid).astype(np.float32)  # [n, 7]

    # MEASURED gripper: nearest closedness in [0, 1] from the gripper sampler.
    gripper_position = _gripper_from_measurements(grid, gripper_samples)
    if gripper_position is None:
        _log.error("MEASURED gripper trace is unavailable; refusing to dump raw episode (proprioception "
                   "must be measured, not a plan copy). save_dir=%s", save_dir)
        return None
    gripper_position = np.clip(gripper_position, 0.0, 1.0).astype(np.float32)  # [n]

    # DEPLOYABLE action: DROID's normalized joint velocity = the tracking error between the plan's
    # commanded target and the measured arm, in the executor's own units (see DROID_JV_GAIN). This
    # is what build_lerobot exports as action.joint_velocity and what a policy is trained to emit.
    # Both operands are already on the same 15 Hz grid, so no realignment is needed.
    action_joint_velocity = (
        DROID_JV_GAIN * (cmd_joint_position - joint_position)
    ).astype(np.float32)  # [n, 7]
    _check_jv_ratio(action_joint_velocity, joint_position, fps=fps, where=str(save_dir))

    save_dir.mkdir(parents=True, exist_ok=True)
    npz_path = save_dir / "robot_state.npz"
    np.savez(
        npz_path,
        joint_position=joint_position,
        gripper_position=gripper_position,
        cmd_joint_position=cmd_joint_position,
        cmd_joint_velocity=cmd_joint_velocity,
        action_joint_velocity=action_joint_velocity,
        cmd_gripper=cmd_gripper,
        # float64: epoch seconds (~1.78e9) in float32 have 128 s resolution, collapsing every
        # frame to one timestamp. frame_time is the master timeline, so it must stay float64.
        frame_time=grid.astype(np.float64),
    )
    meta = {
        "instruction": instruction,
        "fps": int(fps),
        "n_frames": int(n),
        "config_id": config_id,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "source": "tiptop",
        # Which convention `action_joint_velocity` is in, so a consumer never has to infer it from
        # the source. Absent = an episode captured before that array existed, whose only velocity
        # array is the plan's feedforward rad/s (see build_lerobot's action-selection note).
        "action_convention": "droid_joint_velocity",
        "cameras": cameras,
        "record_start": float(record_start) if record_start is not None else None,
        "record_stop": float(record_stop) if record_stop is not None else None,
        # Hand-off lineage: every leg of one task attempt (tamp -> teleop -> tamp -> ...) shares a
        # trajectory_id, and collect/merge_trajectory.py joins them into a single episode. None for
        # a rollout that was never handed off.
        "trajectory_id": trajectory_id,
        "segment_source": "tamp",
    }
    (save_dir / "_meta.json").write_text(json.dumps(meta, indent=2))
    _log.info("Wrote raw episode (%d frames @ %d Hz, %.1fs) to %s", n, fps, t1 - t0, npz_path)
    return npz_path

"""Bimanual ``robot_state.npz`` + ``_meta.json`` for a sequential-bimanual YAM rollout.

The Franka path (:mod:`tiptop.lerobot_capture`) records one arm executing one cuTAMP plan. A YAM
episode is different in two ways, and both are why this is a separate module rather than a widened
``dump_raw_episode`` — the DROID capture is load-bearing for every dataset already collected and is
left byte-identical:

* **Two plans, one episode.** cuTAMP plans a single kinematic chain, so a bimanual rollout is the
  left arm's plan and then the right arm's, with the arm that just finished parked back at the
  neutral posture in between (the other arm's cuRobo config locks it there — see
  :mod:`tiptop.yam`). The interstitial park is a real commanded motion, so it is recorded as a
  segment too; otherwise the video would show the arm moving while the action channel said "hold",
  which is exactly the proprioception/action mismatch the DROID rewrite existed to remove.

* **14 numbers per frame, always.** State is
  ``[L joints 1-6, L gripper, R joints 1-6, R gripper]`` — the layout openpi's
  ``pi05_molmoact2_bimanual`` and the MolmoAct2 dataset use. Both arms are recorded whichever one is
  moving; the idle arm contributes its held command (velocity 0) to the action channel and its
  measured pose to proprioception.

The measured/commanded split is the same contract as ARCHITECTURE.md §3: proprioception is what the
encoders read, the action is what the plan asked for, and the two are never copies of each other.
"""

import json
import logging
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from tiptop.lerobot_capture import _flatten_plan, _load_plan, _nearest_by_wall
from tiptop.yam import ARM_DOF, ARMS, BIMANUAL_ARM_DOF, step_arms

_log = logging.getLogger(__name__)

# Column ranges of each arm inside the 12-wide joint arrays. Left first, matching the state layout
# every YAM consumer on this workstation uses (openpi examples/yam, molmoact2 examples/yam).
ARM_SLICES = {"left": slice(0, ARM_DOF), "right": slice(ARM_DOF, BIMANUAL_ARM_DOF)}
GRIPPER_INDEX = {"left": 0, "right": 1}

# Row spacing for synthesised hold rows (perception pauses, gripper actuations). 50 Hz, matching the
# plan's own trajectory rate so the dense timeline has one consistent resolution.
HOLD_DT = 0.02

# DROID gripper convention for the COMMANDED channel: 0 = open, 1 = closed. The YAM's own hardware
# normalisation is the opposite way round (i2rt reports 1.0 = open), so measured gripper readings are
# inverted into this convention on the way in — see BimanualJointSampler.
CMD_GRIPPER_OPEN = 0.0


SEGMENT_ARMS = (*ARMS, "dual")


@dataclass
class Segment:
    """One commanded arm motion, on the wall clock.

    Produced either by a cuTAMP plan (:func:`segments_from_plan` / :func:`segments_from_dual_plan`)
    or by an interstitial cuRobo move such as parking an arm (:func:`segment_from_motion`). ``arm``
    is ``"left"``/``"right"`` for a SEQUENTIAL-bimanual segment (says which 6 columns of the
    bimanual arrays these numbers belong in; the other arm holds whatever it was last commanded), or
    ``"dual"`` for a genuinely SIMULTANEOUS dual-arm/handover segment, whose ``positions``/
    ``velocities`` are already 12-wide (both arms, not sliced) and ``gripper`` is ``[N, 2]`` (both
    hands' commanded state at once) rather than one scalar channel.
    """

    arm: str
    positions: np.ndarray  # [N, 6] (single arm) or [N, 12] (arm="dual") commanded joint positions
    velocities: np.ndarray  # same shape as positions
    gripper: np.ndarray  # [N] (single arm) or [N, 2] (arm="dual") commanded, binary 0=open / 1=closed
    t_wall: np.ndarray  # [N] float64 epoch seconds

    def __post_init__(self) -> None:
        if self.arm not in SEGMENT_ARMS:
            raise ValueError(f"segment arm must be one of {SEGMENT_ARMS}, got {self.arm!r}")
        n = len(self.positions)
        if not (len(self.velocities) == len(self.gripper) == len(self.t_wall) == n):
            raise ValueError(
                f"segment arrays disagree: pos={n} vel={len(self.velocities)} "
                f"grip={len(self.gripper)} t={len(self.t_wall)}"
            )
        if self.arm == "dual" and n:
            pos_dof = np.asarray(self.positions).shape[-1] if np.ndim(self.positions) > 1 else 1
            grip_dof = np.asarray(self.gripper).shape[-1] if np.ndim(self.gripper) > 1 else 1
            if pos_dof != BIMANUAL_ARM_DOF or grip_dof != len(ARMS):
                raise ValueError(
                    f"a 'dual' segment must carry positions/velocities of width {BIMANUAL_ARM_DOF} "
                    f"and gripper of width {len(ARMS)} (both hands); got pos width {pos_dof}, "
                    f"gripper width {grip_dof}"
                )


def segments_from_plan(arm: str, plan_path: Path, timeline: list) -> list[Segment]:
    """One Segment covering an executed cuTAMP plan, on its measured wall-clock span.

    Reuses the Franka flattener so gripper events, their measured actuation pauses and the
    trajectory rows are laid out identically; only the DOF differs.
    """
    dense = _flatten_plan(_load_plan(Path(plan_path)), timeline=timeline, dof=ARM_DOF)
    t_wall = np.asarray(dense["t_wall"], dtype=np.float64)
    if len(t_wall) < 2 or not np.all(np.isfinite(t_wall)):
        _log.warning("Plan at %s has no usable execution timeline; contributing no segment", plan_path)
        return []
    return [
        Segment(
            arm=arm,
            positions=np.asarray(dense["positions"], dtype=np.float32),
            velocities=np.asarray(dense["velocities"], dtype=np.float32),
            gripper=np.asarray(dense["gripper"], dtype=np.float32),
            t_wall=t_wall,
        )
    ]


def _flatten_dual_plan(plan: dict, timeline: list | None = None) -> dict:
    """Flatten a DUAL-mode (``bimanual_yam_dual``) plan's steps into dense rows.

    Mirrors :func:`tiptop.lerobot_capture._flatten_plan`, with the two differences a genuinely
    simultaneous dual-arm/handover plan forces: every trajectory row is already
    :data:`BIMANUAL_ARM_DOF` wide (one 12-DOF chain, not one arm's slice needing a hold-column), and
    the gripper channel is ``[M, 2]`` -- a gripper step only flips the hand(s) it names (see
    :func:`tiptop.yam.step_arms`), not a single shared scalar, or a handover's taker-close /
    giver-open pair would each stomp the other's state.
    """
    HOLD_DT = 0.02  # 50 Hz, matching the plan's trajectory rate, for inserted hold rows
    pos_chunks, vel_chunks, grip_chunks, twall_chunks = [], [], [], []
    q_init = np.asarray(plan.get("q_init", np.zeros(BIMANUAL_ARM_DOF)), dtype=np.float32).reshape(-1)
    last_pos = q_init  # both arms' pose to freeze at during a gripper pause
    g = np.zeros(len(ARMS), dtype=np.float32)  # DROID convention: 0=open, 1=closed. Episodes start open.
    for i, step in enumerate(plan["steps"]):
        entry = timeline[i] if (timeline is not None and i < len(timeline)) else None
        has_wall = entry is not None and entry.get("t_start") is not None and entry.get("t_end") is not None

        if step["type"] == "trajectory":
            pos = np.asarray(step["positions"], dtype=np.float32)
            vel = np.asarray(step["velocities"], dtype=np.float32)
            n = len(pos)
            if n == 0:
                continue
            pos_chunks.append(pos)
            vel_chunks.append(vel)
            grip_chunks.append(np.tile(g, (n, 1)))
            if has_wall:
                ts, te = float(entry["t_start"]), float(entry["t_end"])
                twall = np.full(n, ts, dtype=np.float64) if n == 1 else np.linspace(ts, te, n)
            else:
                twall = np.full(n, np.nan, dtype=np.float64)
            twall_chunks.append(twall)
            last_pos = pos[-1]
        elif step["type"] == "gripper":
            g = g.copy()
            for arm in step_arms(step):
                g[GRIPPER_INDEX[arm]] = 1.0 if step["action"] == "close" else 0.0
            # Insert stationary hold rows spanning the measured actuation pause, same reasoning as
            # _flatten_plan: skipped without a timeline (no known duration).
            if has_wall:
                ts, te = float(entry["t_start"]), float(entry["t_end"])
                n_hold = max(1, round((te - ts) / HOLD_DT))
                pos_chunks.append(np.tile(last_pos, (n_hold, 1)))
                vel_chunks.append(np.zeros((n_hold, BIMANUAL_ARM_DOF), dtype=np.float32))
                grip_chunks.append(np.tile(g, (n_hold, 1)))
                twall_chunks.append(np.linspace(ts, te, n_hold))

    if not pos_chunks:
        return {
            "positions": np.empty((0, BIMANUAL_ARM_DOF)),
            "velocities": np.empty((0, BIMANUAL_ARM_DOF)),
            "gripper": np.empty((0, len(ARMS))),
            "t_wall": np.empty((0,)),
        }
    return {
        "positions": np.concatenate(pos_chunks, axis=0),
        "velocities": np.concatenate(vel_chunks, axis=0),
        "gripper": np.concatenate(grip_chunks, axis=0),
        "t_wall": np.concatenate(twall_chunks, axis=0),
    }


def segments_from_dual_plan(plan_path: Path, timeline: list) -> list[Segment]:
    """One ``arm="dual"`` Segment covering an executed SIMULTANEOUS dual-arm/handover plan.

    Sibling to :func:`segments_from_plan`: same "one Segment per executed plan" shape, but the plan
    is one 12-DOF chain (``bimanual_yam_dual``) rather than one arm's 6-DOF slice, so it needs its
    own flattener (:func:`_flatten_dual_plan`) rather than the Franka/single-YAM-arm one.
    """
    dense = _flatten_dual_plan(_load_plan(Path(plan_path)), timeline=timeline)
    t_wall = np.asarray(dense["t_wall"], dtype=np.float64)
    if len(t_wall) < 2 or not np.all(np.isfinite(t_wall)):
        _log.warning("Plan at %s has no usable execution timeline; contributing no segment", plan_path)
        return []
    return [
        Segment(
            arm="dual",
            positions=np.asarray(dense["positions"], dtype=np.float32),
            velocities=np.asarray(dense["velocities"], dtype=np.float32),
            gripper=np.asarray(dense["gripper"], dtype=np.float32),
            t_wall=t_wall,
        )
    ]


def segment_from_motion(
    arm: str, positions, velocities, t_start: float, t_end: float, gripper: float | tuple[float, float]
) -> Segment | None:
    """A Segment for a cuRobo move that is not part of a plan — parking or homing an arm.

    These motions happen inside the recorded window (the arm has to be back at the neutral posture
    before the other arm's plan is valid), so leaving them out would put unexplained motion in the
    video against a flat action channel.

    ``gripper`` is a single float for ``arm in ARMS`` (that one hand's held closedness) or a
    ``(left, right)`` pair for ``arm="dual"`` (both hands' held closedness) -- a dual park/home move
    commands both arms at once, so it needs both grippers' state, same as a "dual" Segment from a
    plan (see :func:`segments_from_dual_plan`).
    """
    if positions is None or len(positions) == 0 or not (np.isfinite(t_start) and np.isfinite(t_end)):
        return None
    dof = BIMANUAL_ARM_DOF if arm == "dual" else ARM_DOF
    pos = np.asarray(positions, dtype=np.float32).reshape(len(positions), -1)[:, :dof]
    vel = (
        np.asarray(velocities, dtype=np.float32).reshape(len(velocities), -1)[:, :dof]
        if velocities is not None
        else np.zeros_like(pos)
    )
    n = len(pos)
    t_wall = np.full(n, float(t_start)) if n == 1 else np.linspace(float(t_start), float(t_end), n)
    if arm == "dual":
        grip = np.tile(np.asarray(gripper, dtype=np.float32).reshape(1, 2), (n, 1))
    else:
        grip = np.full(n, float(gripper), dtype=np.float32)
    return Segment(
        arm=arm,
        positions=pos,
        velocities=vel,
        gripper=grip,
        t_wall=t_wall.astype(np.float64),
    )


class BimanualJointSampler:
    """Background thread sampling BOTH arms' measured state during execution.

    Reads the arm server's dedicated STATE port, which serves a background cache and therefore
    answers while the control socket is parked inside a trajectory — the same split as the Franka's
    shim. ``samples`` holds ``(wall_seconds, q[12], dq[12], gripper_closedness[2])``.

    The gripper is converted here, once: i2rt reports a normalised opening where 1.0 is fully open,
    and every downstream consumer (the npz, ``build_lerobot``, the DROID action convention) wants
    closedness where 1.0 is shut. Converting at the boundary keeps the rest of the pipeline on one
    convention.

    Use as a context manager around execution::

        with BimanualJointSampler(host, port) as sampler:
            ...execute both arms' plans...
        # sampler.samples now holds the measured bimanual trace
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 5557, fps: int = 30):
        from tiptop.yam.yam_client import YamStateReader

        self._reader = YamStateReader(host=host, port=port)
        self.fps = int(fps)
        self.samples: list[tuple[float, np.ndarray, np.ndarray, np.ndarray]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._unavailable_logged = False

    def _loop(self) -> None:
        period = 1.0 / self.fps
        while not self._stop.is_set():
            tick = time.perf_counter()
            try:
                self._sample_once()
            except Exception:
                # Never let one bad reply end the thread. A dead sampler stops collecting silently,
                # and every later grid time then resolves to the last sample it did get — an episode
                # written with proprioception frozen part-way through, which is worse than no
                # episode. dump_bimanual_episode's coverage check is the backstop; this is the fix.
                if not self._unavailable_logged:
                    _log.exception("Bimanual state sample failed; continuing to poll")
                    self._unavailable_logged = True
            time.sleep(max(0.0, period - (time.perf_counter() - tick)))

    def _sample_once(self) -> None:
        """One poll of both arms, appended to `samples`. Raises on a malformed reply."""
        data = self._reader.read()
        if data is None:
            if not self._unavailable_logged:
                _log.warning(
                    "YAM state server unavailable; measured joint samples will be empty and the "
                    "raw episode dump will be skipped. Is yam_arm_server.py running?"
                )
                self._unavailable_logged = True
        else:
            q = np.zeros(BIMANUAL_ARM_DOF, dtype=np.float32)
            dq = np.zeros(BIMANUAL_ARM_DOF, dtype=np.float32)
            grip = np.zeros(len(ARMS), dtype=np.float32)
            ok = True
            for arm in ARMS:
                arm_state = data.get(arm)
                if not arm_state:
                    if not self._unavailable_logged:
                        _log.error(
                            "The YAM arm server is not reporting the %s arm, so no bimanual state "
                            "can be recorded and this episode will be discarded. Start the server "
                            "with both arms (it defaults to --arms left,right): even when only one "
                            "arm plans, the other has to be commanded to the neutral posture the "
                            "planner collision-checks it at.",
                            arm,
                        )
                        self._unavailable_logged = True
                    ok = False
                    break
                q[ARM_SLICES[arm]] = np.asarray(arm_state["q"], dtype=np.float32)[:ARM_DOF]
                dq[ARM_SLICES[arm]] = np.asarray(arm_state["dq"], dtype=np.float32)[:ARM_DOF]
                # i2rt: 1.0 = open. Downstream: 1.0 = closed.
                grip[GRIPPER_INDEX[arm]] = 1.0 - float(arm_state["gripper"])
            if ok:
                self.samples.append((time.time(), q, dq, np.clip(grip, 0.0, 1.0)))

    def __enter__(self) -> "BimanualJointSampler":
        self._thread = threading.Thread(target=self._loop, name="yam-bimanual-sampler", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        _log.info("Bimanual joint sampler: %d measured samples", len(self.samples))
        self._reader.close()
        return False


def _dense_bimanual(segments: list[Segment], q_start: np.ndarray, gripper_start: np.ndarray) -> dict:
    """Lay the segments out as one dense 14-wide commanded timeline.

    Segments are placed on the wall clock in order; the arm a segment does not drive holds its last
    commanded pose at zero velocity, and the gaps between segments (perception, planning) become
    explicit hold rows so every frame in the recorded window has a command behind it.
    """
    ordered = sorted((s for s in segments if len(s.positions)), key=lambda s: float(s.t_wall[0]))
    if not ordered:
        return {"positions": np.empty((0, BIMANUAL_ARM_DOF)), "velocities": np.empty((0, BIMANUAL_ARM_DOF)),
                "gripper": np.empty((0, len(ARMS))), "t_wall": np.empty((0,))}

    held_q = np.asarray(q_start, dtype=np.float32).reshape(BIMANUAL_ARM_DOF).copy()
    held_g = np.asarray(gripper_start, dtype=np.float32).reshape(len(ARMS)).copy()

    pos_chunks, vel_chunks, grip_chunks, t_chunks = [], [], [], []
    previous_end: float | None = None

    for segment in ordered:
        start = float(segment.t_wall[0])
        # Gap since the last segment: both arms genuinely stationary (perceiving, planning, or
        # waiting on the operator). Emit hold rows rather than letting the resampler interpolate
        # across the gap, which would smear the previous motion over a static scene.
        if previous_end is not None and start - previous_end > HOLD_DT:
            n_hold = max(1, int(round((start - previous_end) / HOLD_DT)))
            pos_chunks.append(np.tile(held_q, (n_hold, 1)))
            vel_chunks.append(np.zeros((n_hold, BIMANUAL_ARM_DOF), dtype=np.float32))
            grip_chunks.append(np.tile(held_g, (n_hold, 1)))
            t_chunks.append(np.linspace(previous_end, start, n_hold))

        n = len(segment.positions)
        positions = np.tile(held_q, (n, 1))
        velocities = np.zeros((n, BIMANUAL_ARM_DOF), dtype=np.float32)
        grippers = np.tile(held_g, (n, 1))
        if segment.arm == "dual":
            # A genuinely simultaneous segment already carries both arms' columns -- write them
            # straight through rather than through one arm's slice, which is exactly the
            # distinction Segment.__post_init__ enforces on this arm value.
            positions[:, :] = segment.positions
            velocities[:, :] = segment.velocities
            grippers[:, :] = segment.gripper
        else:
            arm_slice = ARM_SLICES[segment.arm]
            positions[:, arm_slice] = segment.positions
            velocities[:, arm_slice] = segment.velocities
            grippers[:, GRIPPER_INDEX[segment.arm]] = segment.gripper

        pos_chunks.append(positions)
        vel_chunks.append(velocities)
        grip_chunks.append(grippers)
        t_chunks.append(np.asarray(segment.t_wall, dtype=np.float64))

        held_q = positions[-1].copy()
        held_g = grippers[-1].copy()
        previous_end = float(segment.t_wall[-1])

    return {
        "positions": np.concatenate(pos_chunks, axis=0).astype(np.float32),
        "velocities": np.concatenate(vel_chunks, axis=0).astype(np.float32),
        "gripper": np.concatenate(grip_chunks, axis=0).astype(np.float32),
        "t_wall": np.concatenate(t_chunks, axis=0).astype(np.float64),
    }


def dump_bimanual_episode(
    save_dir: Path,
    *,
    segments: list[Segment],
    joint_samples: list,
    instruction: str,
    cameras: dict[str, str],
    q_start: np.ndarray,
    gripper_start: np.ndarray,
    fps: int = 15,
    config_id: str | None = None,
    record_start: float | None = None,
    record_stop: float | None = None,
    arms_used: list[str] | None = None,
) -> Path | None:
    """Write a bimanual ``robot_state.npz`` + ``_meta.json`` for one executed rollout.

    Arrays, on a uniform ``fps`` wall-clock grid over the whole episode:

    ===========================  ==========  ================================================
    key                          shape       meaning
    ===========================  ==========  ================================================
    ``joint_position``           [F, 12]     MEASURED encoders, both arms
    ``gripper_position``         [F, 2]      MEASURED closedness in [0,1] (0 open, 1 closed)
    ``cmd_joint_position``       [F, 12]     COMMANDED joint targets
    ``cmd_joint_velocity``       [F, 12]     COMMANDED joint velocities
    ``cmd_gripper``              [F, 2]      COMMANDED gripper, binary 0/1
    ``frame_time``               [F]         float64 epoch seconds — the master timeline
    ===========================  ==========  ================================================

    ``frame_time`` must stay float64: float32 near the current epoch (~1.78e9) resolves to 128 s,
    which silently collapses every frame to one timestamp.

    Returns the npz path, or None when there is nothing usable — no segment carried an execution
    timeline, or the measured trace is missing. It will NOT fall back to writing plan positions as
    proprioception; that silent substitution is the bug the DROID capture rewrite removed.
    """
    save_dir = Path(save_dir)
    dense = _dense_bimanual(segments, q_start, gripper_start)
    t_wall = dense["t_wall"]
    m = len(t_wall)
    if m < 2:
        _log.warning("No executed segments with a wall-clock timeline; skipping raw episode dump")
        return None

    t0, t1 = float(t_wall[0]), float(t_wall[-1])
    n = max(2, int(round((t1 - t0) * fps)) + 1)
    grid = np.minimum(t0 + np.arange(n) / float(fps), t1)

    # COMMANDED: nearest-preceding dense row (t_wall is non-decreasing by construction).
    order = np.argsort(t_wall, kind="stable")
    t_sorted = t_wall[order]
    idx = order[np.clip(np.searchsorted(t_sorted, grid, side="right") - 1, 0, m - 1)]
    cmd_joint_position = dense["positions"][idx].astype(np.float32)
    cmd_joint_velocity = dense["velocities"][idx].astype(np.float32)
    cmd_gripper = dense["gripper"][idx].astype(np.float32)
    if not np.all((cmd_gripper == 0.0) | (cmd_gripper == 1.0)):
        offenders = np.unique(cmd_gripper[(cmd_gripper != 0.0) & (cmd_gripper != 1.0)])[:8]
        raise ValueError(f"commanded gripper is not binary 0/1 (offending values {offenders.tolist()})")

    samples = list(joint_samples or [])
    if not samples:
        _log.error(
            "MEASURED joint trace is EMPTY; refusing to dump raw episode (would falsely record plan "
            "positions as proprioception). save_dir=%s", save_dir,
        )
        return None
    sample_t = np.asarray([s[0] for s in samples], dtype=np.float64)
    sample_q = np.stack([np.asarray(s[1], dtype=np.float32).reshape(-1) for s in samples])
    sample_g = np.stack([np.asarray(s[3], dtype=np.float32).reshape(-1) for s in samples])
    if sample_q.shape[1] != BIMANUAL_ARM_DOF or sample_g.shape[1] != len(ARMS):
        _log.error(
            "MEASURED trace is unusable (q %s, gripper %s; expected [K,%d] and [K,%d]); refusing to "
            "dump raw episode. save_dir=%s",
            sample_q.shape, sample_g.shape, BIMANUAL_ARM_DOF, len(ARMS), save_dir,
        )
        return None

    # The measured trace has to actually SPAN the episode. `_nearest_by_wall` resolves any grid time
    # past the last sample to that sample, so a sampler that died part-way through yields an episode
    # whose proprioception is frozen from that point on — plausible-looking and silently wrong.
    # Refuse rather than write it.
    gap_before = float(sample_t[0] - t0)
    gap_after = float(t1 - sample_t[-1])
    max_gap = max(2.0 / fps, 0.5)
    if gap_before > max_gap or gap_after > max_gap:
        _log.error(
            "MEASURED trace covers %.1f-%.1f s of a %.1f s episode (%.1f s missing at the start, "
            "%.1f s at the end); refusing to dump raw episode rather than record frozen "
            "proprioception. Did the state sampler die? save_dir=%s",
            sample_t[0] - t0, sample_t[-1] - t0, t1 - t0, gap_before, gap_after, save_dir,
        )
        return None

    joint_position = _nearest_by_wall(sample_t, sample_q, grid).astype(np.float32)
    gripper_position = np.clip(_nearest_by_wall(sample_t, sample_g, grid), 0.0, 1.0).astype(np.float32)

    save_dir.mkdir(parents=True, exist_ok=True)
    npz_path = save_dir / "robot_state.npz"
    np.savez(
        npz_path,
        joint_position=joint_position,
        gripper_position=gripper_position,
        cmd_joint_position=cmd_joint_position,
        cmd_joint_velocity=cmd_joint_velocity,
        cmd_gripper=cmd_gripper,
        frame_time=grid.astype(np.float64),
    )
    meta = {
        "instruction": instruction,
        "fps": int(fps),
        "n_frames": int(n),
        "config_id": config_id,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "source": "tiptop-yam",
        "embodiment": "bimanual_yam",
        # What the columns mean, recorded alongside the data so a consumer never has to guess the
        # arm order or the DOF from the shapes alone.
        "layout": {
            "arms": list(ARMS),
            "arm_dof": ARM_DOF,
            "joint_columns": {arm: [ARM_SLICES[arm].start, ARM_SLICES[arm].stop] for arm in ARMS},
            "gripper_columns": dict(GRIPPER_INDEX),
            "gripper_convention": "0 = open, 1 = closed",
        },
        "arms_used": arms_used or sorted({s.arm for s in segments}),
        "cameras": cameras,
        "record_start": float(record_start) if record_start is not None else None,
        "record_stop": float(record_stop) if record_stop is not None else None,
    }
    (save_dir / "_meta.json").write_text(json.dumps(meta, indent=2))
    _log.info("Wrote bimanual raw episode (%d frames @ %d Hz, %.1fs) to %s", n, fps, t1 - t0, npz_path)
    return npz_path

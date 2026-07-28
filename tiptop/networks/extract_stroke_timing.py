"""DROID stroke segmentation: the shared gripper-event detector and parquet helpers.

A *stroke* is the motion between two gripper events, the unit every learned timing component is defined
on. This module holds the segmentation itself -- the Schmitt trigger + min dwell + 4-frame actuator lag
detector (mirrors ``analysis2`` exactly, scored F1=0.944 on DROID) plus the DROID streaming constants and
the pyarrow -> numpy helpers -- so that a stroke here is delimited the SAME way a cuTAMP operation is
delimited at plan time (a gripper close and a gripper open both bound strokes; the episode start/end bound
the first/last).

It imports only numpy, so it runs in the openpi venv (which has ``datasets`` / ``pyarrow``) without a
cuRobo / tiptop-package install -- the DROID streamers import it by path.

Consumers: ``extract_stroke_flow`` (flow-model training data), ``droid_timing_stats`` (the pace /
event-speed conditionals), ``eval_transitions`` / ``eval_blend_transitions``. The module name is
historical -- it used to also build the deterministic timing network's training set, which was removed
with that model.
"""

from __future__ import annotations

import numpy as np

# --------------------------------------------------------------------------- #
# DROID streaming config (mirrors analysis2/config.py)                          #
# --------------------------------------------------------------------------- #
DROID_REPO = "lerobot/droid_1.0.1"
DROID_FPS = 15.0
DT = 1.0 / DROID_FPS
COLS = ["observation.state.joint_position", "action", "episode_index"]
ACT_GRIPPER_COL = 7  # action[:, 7] is the commanded gripper in [0, 1] (0 = open, 1 = closed)

# Gripper-event detection (identical to analysis2/config.py -- the source of truth for these constants).
GRIP_LO, GRIP_HI = 0.20, 0.60
GRIP_MIN_DWELL = 5
GRIP_LAG = 4

# Stroke filter: a usable stroke has enough frames to define a profile and real net motion.
MIN_STROKE_LEN = 6            # frames
MIN_STROKE_ARC = 5e-3         # rad, total joint-space path length


# --------------------------------------------------------------------------- #
# Gripper events (compact copy of analysis2/events.py -- keep in sync with it) #
# --------------------------------------------------------------------------- #
def _schmitt(g: np.ndarray, lo: float, hi: float) -> np.ndarray:
    g = np.asarray(g, np.float64)
    up, dn = g > hi, g < lo
    ev = np.flatnonzero(up | dn)
    if ev.size == 0:
        return np.zeros(g.shape, bool)
    pos = np.searchsorted(ev, np.arange(g.size), side="right") - 1
    out = np.zeros(g.shape, bool)
    hit = pos >= 0
    out[hit] = up[ev[pos[hit]]]
    return out


def _closed_segments(state: np.ndarray, min_dwell: int) -> list[tuple[int, int]]:
    if not state.any():
        return []
    d = np.diff(state.astype(np.int8), prepend=0, append=0)
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1) - 1
    keep = (ends - starts + 1) >= min_dwell
    return list(zip(starts[keep].tolist(), ends[keep].tolist()))


def gripper_events(g: np.ndarray) -> list[tuple[int, str]]:
    """(frame, kind) for every gripper event, kind in {"close", "open"} -- the interior stroke boundaries.

    A stroke decelerates differently into a grasp than into a release (in DROID the close event sits at
    ~0.54 of local cruise speed, the open event at ~0.64), so the kind is part of the training signal --
    see ``extract_stroke_flow`` and ``flow_timing.TrajFlow``.
    """
    g = np.asarray(g, np.float64)
    n = g.size
    if n == 0:
        return []
    segs = _closed_segments(_schmitt(g, GRIP_LO, GRIP_HI), GRIP_MIN_DWELL)
    if not segs:
        return []
    idx = np.arange(n)
    last_below = np.maximum.accumulate(np.where(g <= 0.5, idx, -1))
    last_above = np.maximum.accumulate(np.where(g >= 0.5, idx, -1))
    events = []
    for s, e in segs:
        if last_below[s] >= 0:
            events.append((min(last_below[s] + 1 + GRIP_LAG, n - 1), "close"))  # grasp
        t50_open = last_above[e] + 1
        if t50_open + GRIP_LAG <= n - 1:
            events.append((t50_open + GRIP_LAG, "open"))  # release
    return events


# --------------------------------------------------------------------------- #
# pyarrow -> numpy helpers (mirror analysis2/droid_stream.py)                   #
# --------------------------------------------------------------------------- #
def _list2d(col, width: int) -> np.ndarray:
    flat = col.combine_chunks().flatten().to_numpy(zero_copy_only=False)
    return np.asarray(flat, np.float32).reshape(-1, width)


def _scalar(col) -> np.ndarray:
    return col.combine_chunks().to_numpy(zero_copy_only=False)


def _episode_bounds(ep: np.ndarray) -> list[tuple[int, int]]:
    if ep.size == 0:
        return []
    cut = np.flatnonzero(np.diff(ep)) + 1
    starts = np.concatenate([[0], cut])
    stops = np.concatenate([cut, [ep.size]])
    return list(zip(starts.tolist(), stops.tolist()))

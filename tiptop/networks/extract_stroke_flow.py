"""Extract per-stroke TRAJECTORY pairs from DROID for the flow-matching blender (streaming; no download).

The flow model generates a whole human stroke as joint positions over normalized TIME, conditioned on the
timing-stripped PATH (positions over arc length). This script produces its training set: it streams DROID
1.0.1, segments every episode at its gripper events into strokes (same detector as ``extract_stroke_timing``
/ analysis2), and for each stroke saves two start-centered, fixed-length arrays:

  * ``x`` [T_TIME, 7]  -- joint positions resampled uniform in TIME (the generation TARGET: geometry+timing)
  * ``c`` [S_ARC, 7]   -- joint positions resampled uniform in ARC LENGTH (the CONDITION: the path, no timing)

``c`` is timing-free by construction (arc-length parameterized), so conditioning on it cannot leak the
target timing. Both are centered at the stroke start. Each parquet file is reduced to a few thousand small
records and dropped; the 27.6 M frames never land on disk.

Run in the openpi venv::

    openpi/.venv/bin/python -m tiptop.networks.extract_stroke_flow --files 4

Output: shards + merged ``stroke_flow.npz`` (x, c) under ``checkpoints/flow_data/``.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_stroke_timing import (  # noqa: E402
    ACT_GRIPPER_COL,
    COLS,
    DROID_REPO,
    MIN_STROKE_ARC,
    MIN_STROKE_LEN,
    _episode_bounds,
    _list2d,
    _scalar,
    gripper_events,
)

from flow_timing import DOF, KIND, S_ARC, T_TIME  # noqa: E402  (the model-side contract)

OUT_DIR = Path(__file__).resolve().parents[1] / "checkpoints" / "flow_data"
SHARD_DIR = OUT_DIR / "shards"
MERGED = OUT_DIR / "stroke_flow.npz"


def _resample_time(q: np.ndarray, n: int) -> np.ndarray:
    """Resample q [T,7] to n points uniform in time (frame index)."""
    src = np.linspace(0.0, 1.0, len(q))
    dst = np.linspace(0.0, 1.0, n)
    return np.stack([np.interp(dst, src, q[:, j]) for j in range(q.shape[1])], axis=1)


def _resample_arc(q: np.ndarray, n: int) -> np.ndarray | None:
    """Resample q [T,7] to n points uniform in joint-space arc length (timing-stripped path)."""
    step = np.linalg.norm(np.diff(q, axis=0), axis=1)
    arc = step.sum()
    if arc < 1e-6:
        return None
    u = np.concatenate([[0.0], np.cumsum(step)])
    dst = np.linspace(0.0, arc, n)
    return np.stack([np.interp(dst, u, q[:, j]) for j in range(q.shape[1])], axis=1)


def _episode_pairs(q: np.ndarray, g: np.ndarray) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Strokes of one episode -> (x, c, k), k = the [start, end] event kinds as KIND codes."""
    T = len(q)
    events = dict(gripper_events(g))
    bounds = sorted({0, T - 1, *events})
    pairs = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        if b - a + 1 < MIN_STROKE_LEN:
            continue
        seg = q[a : b + 1]
        if np.linalg.norm(np.diff(seg, axis=0), axis=1).sum() < MIN_STROKE_ARC:
            continue
        c = _resample_arc(seg, S_ARC)
        if c is None:
            continue
        x = _resample_time(seg, T_TIME)
        x = (x - x[0:1]).astype(np.float32)   # start-centered trajectory over time
        c = (c - c[0:1]).astype(np.float32)   # start-centered path over arc length
        # The stroke boundaries ARE the gripper events, so these labels say what the generated stroke
        # must decelerate into (or accelerate out of). "none" = an episode end, not a gripper event.
        k = np.array([KIND[events.get(a, "none")], KIND[events.get(b, "none")]], np.int64)
        pairs.append((x, c, k))
    return pairs


def reduce_file(tbl) -> dict:
    ep = np.asarray(_scalar(tbl["episode_index"]), np.int64)
    J = _list2d(tbl["observation.state.joint_position"], 7)
    A = _list2d(tbl["action"], 8)
    if not np.isfinite(J).all():
        raise ValueError("NaN/inf in DROID joints")
    xs, cs, ks = [], [], []
    n_eps = 0
    for a, b in _episode_bounds(ep):
        qseg = J[a:b]
        if len(qseg) < MIN_STROKE_LEN:
            continue
        n_eps += 1
        for x, c, k in _episode_pairs(qseg, A[a:b, ACT_GRIPPER_COL]):
            xs.append(x)
            cs.append(c)
            ks.append(k)
    return dict(
        x=np.asarray(xs, np.float32).reshape(-1, T_TIME, DOF),
        c=np.asarray(cs, np.float32).reshape(-1, S_ARC, DOF),
        k=np.asarray(ks, np.int64).reshape(-1, 2),
        n_episodes=np.int64(n_eps),
        n_strokes=np.int64(len(xs)),
    )


def reduce_cached_shards(shard_dir: Path, max_files: int = None) -> dict:
    """Same reduction over the proprio shards ``vae/data_full.py fetch`` already wrote.

    Those shards hold exactly the columns this extractor needs (joints + the 8-wide action whose last
    column is the commanded gripper), so when they are present the training set can be rebuilt without
    re-streaming the Hub.
    """
    shards = sorted(Path(shard_dir).glob("shard_*.npz"))[:max_files]
    if not shards:
        raise FileNotFoundError(f"no shard_*.npz under {shard_dir}")
    xs, cs, ks = [], [], []
    n_eps = 0
    for sp in shards:
        blob = np.load(sp, allow_pickle=True)
        J, A = blob["joints"], blob["actions"]
        for i in range(int(blob["n_eps"])):
            q = np.asarray(J[i], np.float64)
            a = np.asarray(A[i], np.float64)
            if len(q) < MIN_STROKE_LEN or a.ndim != 2 or a.shape[1] <= ACT_GRIPPER_COL:
                continue
            n_eps += 1
            for x, c, k in _episode_pairs(q, a[:, ACT_GRIPPER_COL]):
                xs.append(x)
                cs.append(c)
                ks.append(k)
        print(f"[{sp.name}] cumulative: {n_eps} episodes, {len(xs)} strokes", flush=True)
    return dict(
        x=np.asarray(xs, np.float32).reshape(-1, T_TIME, DOF),
        c=np.asarray(cs, np.float32).reshape(-1, S_ARC, DOF),
        k=np.asarray(ks, np.int64).reshape(-1, 2),
        n_episodes=np.int64(n_eps),
        n_strokes=np.int64(len(xs)),
    )


def stream(n_files: int, start: int = 0, sleep: float = 0.5, overwrite: bool = False):
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    SHARD_DIR.mkdir(parents=True, exist_ok=True)
    fs = HfFileSystem()
    t0 = time.time()
    for fi in range(start, start + n_files):
        out = SHARD_DIR / f"file_{fi:03d}.npz"
        if out.exists() and not overwrite:
            print(f"[{fi:3d}] cached", flush=True)
            continue
        path = f"datasets/{DROID_REPO}/data/chunk-000/file-{fi:03d}.parquet"
        tbl = None
        for attempt in range(6):
            try:
                with fs.open(path, "rb") as f:
                    tbl = pq.read_table(f, columns=COLS)
                break
            except Exception as exc:  # noqa: BLE001
                if attempt == 5:
                    raise
                back = 90.0 if ("429" in str(exc).lower() or "rate" in str(exc).lower()) else 3.0 * (attempt + 1)
                print(f"    [{fi}] attempt {attempt + 1} failed ({type(exc).__name__}); backoff {back:.0f}s", flush=True)
                time.sleep(back)
        agg = reduce_file(tbl)
        del tbl
        np.savez_compressed(out, **agg)
        el = time.time() - t0
        print(f"[{fi:3d}] {int(agg['n_episodes']):5d} eps  {int(agg['n_strokes']):6d} strokes  ({el/60:.1f} min)", flush=True)
        time.sleep(sleep)
    merge()


def merge():
    shards = sorted(SHARD_DIR.glob("file_*.npz"))
    if not shards:
        raise FileNotFoundError(f"no shards under {SHARD_DIR}")
    x = np.concatenate([np.load(s)["x"] for s in shards], 0)
    c = np.concatenate([np.load(s)["c"] for s in shards], 0)
    k = np.concatenate([np.load(s)["k"] for s in shards], 0)
    np.savez_compressed(MERGED, x=x, c=c, k=k)
    print(f"\nmerged {len(shards)} shards -> {MERGED}   x{x.shape} c{c.shape} k{k.shape}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", type=int, default=4)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--sleep", type=float, default=0.5)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--merge-only", action="store_true")
    ap.add_argument("--from-cache", type=str, default=None,
                    help="Build from vae/data_cache/droid_full_proprio shards instead of streaming.")
    a = ap.parse_args()
    if a.from_cache:
        agg = reduce_cached_shards(Path(a.from_cache), a.files if a.files > 0 else None)
        MERGED.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(MERGED, x=agg["x"], c=agg["c"], k=agg["k"])
        print(f"\n{int(agg['n_episodes'])} episodes -> {int(agg['n_strokes'])} strokes -> {MERGED}")
    elif a.merge_only:
        merge()
    else:
        stream(a.files, a.start, a.sleep, a.overwrite)

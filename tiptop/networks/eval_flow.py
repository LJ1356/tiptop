"""Evaluate the flow-matching blender on HELD-OUT DROID: plausibility, smoothness, multimodality, match.

A generative timing model must NOT be judged by per-stroke MSE (that rewards the mean and punishes the
diversity we want). This streams disjoint DROID files and reports:

  1. Smoothness -- mean |2nd-difference of speed| of generated strokes vs DROID (the earlier MLP model was
     ~5x jerkier; the temporal U-Net should close this).
  2. Multimodality -- for a FIXED path, sample many trajectories and measure the spread of the end-speed
     ratio (mean speed over the last 20% / overall). A large within-condition spread means the model
     expresses the decelerate-into-grasp vs constant-velocity styles a deterministic regressor collapses.
  3. Distribution match -- the marginal end-speed-ratio distribution over generated strokes (one sample
     per held-out path) vs held-out DROID (mean/std/quantiles + 1-D W1). Should reproduce DROID's SPREAD.

Run in the openpi venv::

    openpi/.venv/bin/python -m tiptop.networks.eval_flow --start 45 --files 2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from flow_timing import FlowModel  # noqa: E402
from extract_stroke_flow import COLS, DROID_REPO, reduce_file  # noqa: E402


def _speed(x: np.ndarray) -> np.ndarray:
    return np.linalg.norm(np.diff(x, axis=0), axis=1)


def _end_ratio(x: np.ndarray, frac: float = 0.2) -> float:
    s = _speed(x)
    if s.mean() < 1e-9:
        return 1.0
    k = max(1, int(round(frac * len(s))))
    return float(s[-k:].mean() / s.mean())


def _jerk(x: np.ndarray) -> float:
    return float(np.abs(np.diff(_speed(x), 2)).mean())


def _w1(a: np.ndarray, b: np.ndarray, n: int = 200) -> float:
    qs = (np.arange(n) + 0.5) / n
    return float(np.abs(np.quantile(a, qs) - np.quantile(b, qs)).mean())


def _load_heldout(start: int, n_files: int) -> dict:
    import os

    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    fs = HfFileSystem()
    xs, cs = [], []
    for fi in range(start, start + n_files):
        path = f"datasets/{DROID_REPO}/data/chunk-000/file-{fi:03d}.parquet"
        with fs.open(path, "rb") as f:
            tbl = pq.read_table(f, columns=COLS)
        agg = reduce_file(tbl)
        del tbl
        xs.append(agg["x"]); cs.append(agg["c"])
        print(f"  held-out file-{fi:03d}: {len(agg['x'])} strokes", flush=True)
    return {"x": np.concatenate(xs, 0), "c": np.concatenate(cs, 0)}


def _spark(v):
    s = " .:-=+*#%@"
    lo, hi = v.min(), v.max()
    return "".join(s[min(len(s) - 1, int((z - lo) / (hi - lo + 1e-9) * (len(s) - 1)))] for z in v)


def _unit(s):
    return s / (s.mean() + 1e-9)


def evaluate(args):
    model = FlowModel(args.model)
    ho = _load_heldout(args.start, args.files)
    X, C = ho["x"], ho["c"]
    m = len(X)
    print(f"\nHeld-out: {m} strokes from files [{args.start},{args.start + args.files}).  steps={args.steps}")

    rng = np.random.default_rng(0)
    sub = rng.choice(m, size=min(args.n_dist, m), replace=False)
    gen = np.concatenate([model.sample_batch(C[sub][i:i + 512], n_steps=args.steps) for i in range(0, len(sub), 512)], 0)

    # (1) smoothness + endpoint fidelity
    droid_jerk = np.mean([_jerk(x) for x in X[sub]])
    gen_jerk = np.mean([_jerk(g) for g in gen])
    endpoint_err = float(np.linalg.norm(gen[:, -1] - (C[sub] - C[sub, 0:1])[:, -1], axis=1).max())

    # (2) multimodality
    idx = rng.choice(m, size=min(40, m), replace=False)
    within = []
    for j, i in enumerate(idx):
        samp = model.sample(C[i], n_samples=args.k, n_steps=args.steps)
        ratios = np.array([_end_ratio(s) for s in samp])
        within.append(ratios.std())
        if j < 2:
            print(f"\n  path {j}: {args.k} samples end-ratios [min {ratios.min():.2f} med {np.median(ratios):.2f} max {ratios.max():.2f}]")
            for s in samp[:3]:
                print(f"    speed |{_spark(_speed(s))}|  end_ratio={_end_ratio(s):.2f}")
    within = np.array(within)

    # (3) distribution match
    droid_r = np.array([_end_ratio(x) for x in X[sub]])
    gen_r = np.array([_end_ratio(g) for g in gen])

    def st(a):
        return f"mean {a.mean():.2f} std {a.std():.2f} q10/50/90 {np.quantile(a,0.1):.2f}/{np.median(a):.2f}/{np.quantile(a,0.9):.2f}"

    droid_prof = np.stack([_unit(_speed(x)) for x in X[sub]]).mean(0)
    gen_prof = np.stack([_unit(_speed(g)) for g in gen]).mean(0)

    print("\n=== (1) Smoothness + endpoints ===")
    print(f"  jerk (|2nd-diff speed|): DROID {droid_jerk:.4f}  flow {gen_jerk:.4f}  ratio {gen_jerk/droid_jerk:.1f}x  (was 5.2x)")
    print(f"  endpoint pinning max error = {endpoint_err:.2e} rad  (0 = exact)")
    print(f"  mean unit speed profile:  DROID |{_spark(droid_prof)}| end/mid {droid_prof[-3:].mean()/droid_prof[13:19].mean():.2f}")
    print(f"                            flow  |{_spark(gen_prof)}| end/mid {gen_prof[-3:].mean()/gen_prof[13:19].mean():.2f}")
    print("\n=== (2) Multimodality: within-condition end-ratio std ===")
    print(f"  mean over {len(idx)} fixed paths = {within.mean():.3f}   (deterministic = 0.000)")
    print("\n=== (3) End-speed-ratio distribution: generated vs DROID ===")
    print(f"  DROID : {st(droid_r)}")
    print(f"  flow  : {st(gen_r)}")
    print(f"  W1(DROID, flow) = {_w1(droid_r, gen_r):.3f}   (was 0.48)")
    print(f"  decelerating (<0.8): DROID {(droid_r<0.8).mean():.1%} flow {(gen_r<0.8).mean():.1%}   "
          f"constant (0.8-1.2): DROID {((droid_r>=0.8)&(droid_r<=1.2)).mean():.1%} flow {((gen_r>=0.8)&(gen_r<=1.2)).mean():.1%}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="checkpoints/flow_net_t64.pt")
    ap.add_argument("--start", type=int, default=45)
    ap.add_argument("--files", type=int, default=2)
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--n-dist", type=int, default=1500)
    evaluate(ap.parse_args())

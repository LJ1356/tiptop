"""Do generated strokes enter grasps and releases the way DROID does?

The stroke boundaries ARE the gripper events (``extract_stroke_flow`` cuts there), so a stroke's speed
at tau=0 / tau=1 is exactly its speed at the release it left and the grasp it arrives at. This scores a
checkpoint on HELD-OUT DROID episodes:

  1. Boundary speed by event kind -- unit-mean speed at each end, split by close / open / none, model vs
     ground truth. DROID decelerates harder into a grasp (~0.55 of the stroke's own mean speed) than into
     a release (~0.80); a model that ignores the kind lands on one number for both.
  2. The close/open ratio -- the single number that says whether the asymmetry survived.
  3. Approach shape -- the speed profile over the last 25% of the stroke, which is what a viewer reads as
     "decelerates into the grasp" rather than "arrives at full speed".

Held-out means shards this checkpoint was not trained on; pass ``--shards`` accordingly.

Run in the tiptop pixi env (GPU) or the openpi venv (CPU)::

    .pixi/envs/default/bin/python -m tiptop.networks.eval_transitions --shards 10,11,12
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_stroke_flow import _episode_pairs  # noqa: E402
from extract_stroke_timing import ACT_GRIPPER_COL, MIN_STROKE_LEN  # noqa: E402
from flow_timing import KIND, FlowModel  # noqa: E402

KIND_NAME = {v: k for k, v in KIND.items()}
# Held-out DROID reference (shards 10-12, 6000 strokes): unit-mean speed at the stroke END by kind.
# When scoring a non-DROID stroke set, THIS stays the target -- that set's own timing is not one.
DROID_END = {"close": 0.4327, "open": 0.7741, "none": 0.5788}
DEFAULT_SHARD_DIR = Path("/home/prpl/tamp-vla/vae/data_cache/droid_full_proprio")


def load_strokes(shard_dir: Path, shard_ids: list[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Held-out shards -> (x [M,T,7] time-domain strokes, c [M,S,7] paths, k [M,2] event kinds)."""
    xs, cs, ks = [], [], []
    for si in shard_ids:
        sp = shard_dir / f"shard_{si:03d}.npz"
        if not sp.exists():
            print(f"  (missing {sp.name}, skipped)")
            continue
        blob = np.load(sp, allow_pickle=True)
        J, A = blob["joints"], blob["actions"]
        for i in range(int(blob["n_eps"])):
            q = np.asarray(J[i], np.float64)
            a = np.asarray(A[i], np.float64)
            if len(q) < MIN_STROKE_LEN or a.ndim != 2 or a.shape[1] <= ACT_GRIPPER_COL:
                continue
            for x, c, k in _episode_pairs(q, a[:, ACT_GRIPPER_COL]):
                xs.append(x)
                cs.append(c)
                ks.append(k)
    return np.asarray(xs, np.float32), np.asarray(cs, np.float32), np.asarray(ks, np.int64)


def profiles(x: np.ndarray) -> np.ndarray:
    """[M,T,7] strokes -> [M,T-1] unit-mean speed profiles (1.0 = that stroke's own average speed)."""
    s = np.linalg.norm(np.diff(x, axis=1), axis=2)
    return s / np.clip(s.mean(axis=1, keepdims=True), 1e-9, None)


def main(args):
    if args.strokes_npz:
        # A prepared stroke set (e.g. cuTAMP plans). Its own timing is NOT the target -- the DROID
        # column below stays the reference, so this asks "does the model still produce DROID-like
        # transitions when conditioned on THIS path distribution?".
        blob = np.load(args.strokes_npz, allow_pickle=True)
        x, c, k = blob["x"].astype(np.float32), blob["c"].astype(np.float32), blob["k"].astype(np.int64)
        print(f"stroke set: {args.strokes_npz}")
    else:
        shard_ids = [int(s) for s in args.shards.split(",")]
        print(f"Held-out shards: {shard_ids}")
        x, c, k = load_strokes(Path(args.shard_dir), shard_ids)
    if len(x) == 0:
        raise SystemExit("no strokes found -- fetch the shards first (vae/data_full.py fetch)")
    if args.max_strokes and len(x) > args.max_strokes:
        sel = np.random.default_rng(0).choice(len(x), args.max_strokes, replace=False)
        x, c, k = x[sel], c[sel], k[sel]
    print(f"{len(x)} held-out strokes")

    model = FlowModel(args.ckpt, device=args.device)
    print(f"checkpoint: {Path(args.ckpt).name}  (n_kinds={model.n_kinds}, "
          f"guidance={args.guidance if args.guidance is not None else model.guidance})")
    # Each sample is a fresh draw, so a small stroke set needs repeats before the per-kind means settle.
    gen = []
    for _ in range(args.repeats):
        for i in range(0, len(c), args.batch_size):
            kb = k[i : i + args.batch_size] if model.n_kinds else None
            gen.append(model.sample_batch(c[i : i + args.batch_size].astype(np.float64), args.steps, kb,
                                          args.guidance))
    gen = np.concatenate(gen)
    k_gen = np.tile(k, (args.repeats, 1))

    p_data, p_gen = profiles(x), profiles(gen)

    print("\n=== boundary speed (unit-mean; 1.0 = the stroke's own average speed) ===")
    src = "SET" if args.strokes_npz else "DROID"
    print(f"{'end':<6} {'kind':<6} {'n':>6} {src:>9} {'MODEL':>9} {'gap':>8}")
    res = {}
    for end, col in (("start", 0), ("end", -1)):
        for code in (KIND["close"], KIND["open"], KIND["none"]):
            col_i = 0 if end == "start" else 1
            sel = k[:, col_i] == code
            sel_g = k_gen[:, col_i] == code
            if sel.sum() < 10:
                continue
            d, g = p_data[sel, col].mean(), p_gen[sel_g, col].mean()
            res[(end, KIND_NAME[code])] = (d, g)
            print(f"{end:<6} {KIND_NAME[code]:<6} {int(sel.sum()):>6} {d:>9.4f} {g:>9.4f} {g - d:>+8.4f}")

    droid_ratio = DROID_END["open"] / DROID_END["close"]
    print("\n=== close/open asymmetry (the headline) ===")
    for end in ("start", "end"):
        if (end, "close") in res and (end, "open") in res:
            dc, gc = res[(end, "close")]
            do, go = res[(end, "open")]
            # On a non-DROID set the reference is the fixed DROID ratio, not that set's own timing.
            use_droid = bool(args.strokes_npz)
            ref = droid_ratio if use_droid else do / dc
            label = "DROID" if use_droid else "set  "
            print(f"  {end:<6} {label} open/close = {ref:.3f}   MODEL open/close = {go / gc:.3f}   "
                  f"({100 * (go / gc) / ref:.0f}% of DROID's)")
    if args.strokes_npz:
        print("  model end levels vs DROID: " + "  ".join(
            f"{kd}={res[('end', kd)][1]:.3f} (DROID {DROID_END[kd]:.3f})"
            for kd in ("close", "open", "none") if ("end", kd) in res))

    print("\n=== approach shape: speed over the last 25% of the stroke, by end kind ===")
    tail = max(2, int(round(0.25 * p_data.shape[1])))
    hdr = " ".join(f"{f'-{tail - 1 - j}':>7}" for j in range(tail))
    print(f"{'kind':<7} {'src':<6} {hdr}   (frames before the boundary)")
    for code in (KIND["close"], KIND["open"]):
        sel = k[:, 1] == code
        if sel.sum() < 10:
            continue
        for tag, p, kk in (("DROID", p_data, k), ("model", p_gen, k_gen)):
            print(f"{KIND_NAME[code]:<7} {tag:<6} " +
                  " ".join(f"{v:>7.3f}" for v in p[kk[:, 1] == code, -tail:].mean(axis=0)))

    d2_data = np.abs(np.diff(p_data, n=2, axis=1)).mean()
    d2_gen = np.abs(np.diff(p_gen, n=2, axis=1)).mean()
    print("\n=== smoothness (mean |2nd difference| of the speed profile) ===")
    print(f"  DROID {d2_data:.4f}   model {d2_gen:.4f}   ratio {d2_gen / d2_data:.2f}x "
          f"(>1 = jerkier than DROID)")

    if args.save:
        np.savez(args.save, p_data=p_data, p_gen=p_gen, k=k)
        print(f"\nwrote {args.save}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str,
                    default="/home/prpl/tamp-vla/tiptop/tiptop/checkpoints/flow_net_t64.pt")
    ap.add_argument("--shards", type=str, default="10,11,12")
    ap.add_argument("--shard-dir", type=str, default=str(DEFAULT_SHARD_DIR))
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--max-strokes", type=int, default=6000)
    ap.add_argument("--repeats", type=int, default=1,
                    help="Draw this many samples per stroke; raise it for small stroke sets.")
    ap.add_argument("--strokes-npz", type=str, default=None,
                    help="Evaluate on a prepared (x, c, k) stroke set instead of DROID shards.")
    ap.add_argument("--guidance", type=float, default=None,
                    help="CFG scale; default = whatever the checkpoint baked in.")
    ap.add_argument("--save", type=str, default=None)
    main(ap.parse_args())

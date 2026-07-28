"""The two DROID scalars the flow model does not generate: stroke PACE and gripper-event SPEED.

``flow_timing.TrajFlow`` generates a stroke in NORMALIZED time, so its output is a unit-mean speed
*shape* and nothing more. Two numbers per stroke are therefore missing, and the blender used to take
both from somewhere other than DROID:

* **pace** -- the stroke's mean joint speed. Was inherited from the cuTAMP plan's own wall-clock (i.e.
  from ``time_dilation_factor``), which makes the emitted absolute velocity a planner setting rather
  than a human statistic, and makes it DETERMINISTIC where DROID's is spread over ~5x.
* **event speed** -- the joint speed at a gripper open/close. Was a single config constant
  (``blend_boundary_speed``) applied to every event of both kinds, which flattens DROID's 1.6x
  grasp-vs-release asymmetry to 1.0 and pins a distribution to a point.

This module fits both as small empirical conditionals on DROID and samples them at plan time. They are
plain regressions with an empirical residual pool rather than extra channels on the flow model: they are
scalars, the pool reproduces DROID's skew (-0.8) and heavy tails (kurtosis 6.2) that a Gaussian would
not, and being a 3-line model it stays inspectable.

Measured over the 13 cached DROID proprio shards, position-derived at 15 Hz -- ``action.joint_velocity``
is hard-clipped at +-1 rad/s and is NOT unit-comparable:

* pace:      ``log vbar = a + b log L + c (log K - mean log K) + off[k_start, k_end] + eps``, where K is
  :func:`path_curvature`. The CURVATURE term carries this model: ``c`` is about -0.7 (humans obey a
  speed-curvature law) and adding it cuts the residual sd from 0.456 to 0.296 -- 61% of the variance
  the length-only law leaves -- after which the kind offsets are small (-0.13 to +0.10), because
  curvature was most of what they had been standing in for. Residuals are skewed and heavy-tailed, so
  they are sampled from an empirical pool rather than a Gaussian.
* boundary:  ``log v_e = a_k + b_k log vbar + eps_k`` per event kind, residual sd ~1.04-1.13. Note the
  weak dependence on pace (corr 0.41-0.46): the event speed is very nearly its own random variable.
  It is still conditioned on pace because drawing the two independently over-disperses the
  boundary/pace ratio by ~18% (sd 1.24 vs DROID's 1.05).

Both samplers clip to the range DROID actually exhibits: a regression with a heavy-tailed empirical
residual extrapolates past its own data once the conditioning value is itself a sample, and unclipped
this asked for boundary speeds above 4 rad/s.

Fit (writes ``checkpoints/droid_timing_stats.npz``)::

    openpi/.venv/bin/python -m tiptop.networks.droid_timing_stats --fit
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

_log = logging.getLogger(__name__)

_PKG_DIR = Path(__file__).resolve().parents[1]
DEFAULT_STATS = _PKG_DIR / "checkpoints" / "droid_timing_stats.npz"

KIND_PAIRS = [(a, b) for a in ("none", "close", "open") for b in ("none", "close", "open")]
# Residual pools are subsampled to this many draws before shipping (keeps the file small; the
# empirical quantiles are already converged at this size).
_POOL = 20000
# Arc-length spacing (rad) the path-curvature feature is measured on. Must match between the fit and
# plan time, so both live on this constant.
CURV_STEP = 0.01
_MIN_CURV_ARC = 0.2   # below this a path has too few grid points for a stable curvature


def path_curvature(q: np.ndarray, step: float = CURV_STEP) -> float:
    """Mean |d2q/ds2| of a joint path, on a fixed arc-length grid -- how WIGGLY the path is.

    Timing-free and resolution-independent by construction, so the same number can be measured on a
    DROID stroke and on a cuTAMP operation group and compared. Returns 0.0 for a path too short to
    support the grid.
    """
    q = np.asarray(q, np.float64)
    u = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(q, axis=0), axis=1))])
    arc = float(u[-1])
    n = int(arc / step) + 1
    if arc < _MIN_CURV_ARC or n < 8:
        return 0.0
    g = np.stack([np.interp(np.linspace(0.0, arc, n), u, q[:, j]) for j in range(q.shape[1])], axis=1)
    return float(np.abs(np.diff(g, n=2, axis=0) / step ** 2).sum(axis=1).mean())


class TimingStats:
    """Sample DROID stroke pace and gripper-event speed. Load once per plan; pass an rng for repeatability."""

    def __init__(self, path: str | Path | None = None):
        p = Path(path) if path else DEFAULT_STATS
        if not p.is_absolute():
            p = (_PKG_DIR / p).resolve()
        b = np.load(p)
        self.path = p
        self.pace_a, self.pace_b = float(b["pace_a"]), float(b["pace_b"])
        # Speed-curvature term, centred on the DROID mean log-curvature so that omitting curvature
        # falls back to "assume a typically-wiggly path" rather than to a broken intercept.
        self.pace_c = float(b["pace_c"]) if "pace_c" in b else 0.0
        self.curv_log_mean = float(b["curv_log_mean"]) if "curv_log_mean" in b else 0.0
        self.pace_off = {tuple(k.split("|")): float(v)
                         for k, v in zip(b["pace_off_keys"], b["pace_off_vals"])}
        self.pace_resid = b["pace_resid"]
        # Median DROID stroke duration. The flow model generates in NORMALIZED time, so its profile is
        # implicitly calibrated to a stroke of about this length; longer strokes need its end regions
        # compressed to keep the approach at a human PHYSICAL width (see flow_blending._end_remap).
        self.dur_median = float(b["dur_median"]) if "dur_median" in b else 4.07
        self.pace_range = tuple(b["pace_range"]) if "pace_range" in b else (1e-3, 1e3)
        self.bnd = {}
        for kind in ("close", "open"):
            rng_ = tuple(b[f"bnd_range_{kind}"]) if f"bnd_range_{kind}" in b else (1e-4, 1e3)
            self.bnd[kind] = (float(b[f"bnd_a_{kind}"]), float(b[f"bnd_b_{kind}"]),
                              b[f"bnd_resid_{kind}"], *rng_)
        _log.info("Loaded DROID timing stats %s (pace resid sd %.3f, %d strokes)", p.name,
                  float(self.pace_resid.std()), int(b["n_strokes"]))

    def sample_pace(self, arc_length: float, curvature: float, kinds: tuple[str, str],
                    rng: np.random.Generator) -> float:
        """Mean joint speed (rad/s) for a stroke of this arc length, curvature and pair of event kinds.

        ``curvature`` is :func:`path_curvature` of the same path. It matters far more than length:
        humans obey a speed-curvature law (the exponent here is about -0.7), and conditioning on it
        cuts the pace residual sd from 0.456 to 0.292 -- 59% of the residual variance. Ignoring it
        draws fast paces for long-but-winding planner paths, which is both un-human and physically
        infeasible: the acceleration goes as curvature x speed^2, so the vel/accel caps then stretch
        the stroke back down and the sampled pace never reaches the robot. Pass 0 to fall back to the
        length-only law (what a pre-curvature stats file supports).
        """
        L = max(float(arc_length), 1e-6)
        mu = self.pace_a + self.pace_b * np.log(L) + self.pace_off.get(tuple(kinds), 0.0)
        if curvature > 0.0 and self.pace_c:
            mu += self.pace_c * (np.log(max(float(curvature), 1e-6)) - self.curv_log_mean)
        return float(np.clip(np.exp(mu + rng.choice(self.pace_resid)), *self.pace_range))

    def sample_event_speed(self, kind: str, pace: float, rng: np.random.Generator) -> float:
        """Joint speed (rad/s) at a gripper event of this kind, for a stroke running at ``pace``."""
        if kind not in self.bnd:
            return 0.0
        a, b, resid, lo, hi = self.bnd[kind]
        v = np.exp(a + b * np.log(max(pace, 1e-6)) + rng.choice(resid))
        # Clip to the range DROID actually exhibits. A regression with a heavy-tailed empirical residual
        # extrapolates past its own data when the conditioning value is itself sampled: unclipped this
        # asks for boundary speeds over 4 rad/s, which the caps then claw back by stretching the whole
        # stroke -- so one outlier draw slows an entire operation.
        return float(np.clip(v, lo, hi))


# --------------------------------------------------------------------------- #
# Fitting                                                                       #
# --------------------------------------------------------------------------- #
def _strokes(shard_dir: Path, shard_ids):
    """(L, duration, vbar, end speed, k_start, k_end) per DROID stroke, position-derived at 15 Hz."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from extract_stroke_timing import (ACT_GRIPPER_COL, DT, MIN_STROKE_ARC, MIN_STROKE_LEN,
                                       gripper_events)

    rows = []
    for si in shard_ids:
        sp = Path(shard_dir) / f"shard_{si:03d}.npz"
        if not sp.exists():
            print(f"  (missing {sp.name}, skipped)")
            continue
        blob = np.load(sp, allow_pickle=True)
        J, A = blob["joints"], blob["actions"]
        for i in range(int(blob["n_eps"])):
            q, a = np.asarray(J[i], np.float64), np.asarray(A[i], np.float64)
            if len(q) < MIN_STROKE_LEN or a.ndim != 2 or a.shape[1] <= ACT_GRIPPER_COL:
                continue
            spd = np.linalg.norm(np.diff(q, axis=0), axis=1) / DT
            ev = dict(gripper_events(a[:, ACT_GRIPPER_COL]))
            bounds = sorted({0, len(q) - 1, *ev})
            for s_, e_ in zip(bounds[:-1], bounds[1:]):
                seg = q[s_:e_ + 1]
                arc = float(np.linalg.norm(np.diff(seg, axis=0), axis=1).sum())
                if len(seg) < MIN_STROKE_LEN or arc < MIN_STROKE_ARC:
                    continue
                d = (e_ - s_) * DT
                # 3-frame median at the end: one interval is noisy, and it is the quantity the
                # blender has to reproduce at the event.
                rows.append((arc, d, arc / d, float(np.median(spd[max(e_ - 3, 0):e_])),
                             ev.get(s_, "none"), ev.get(e_, "none"), path_curvature(seg)))
        print(f"  shard {si}: {len(rows)} strokes", flush=True)
    return rows


def fit(shard_dir: Path, shard_ids, out: Path, seed: int = 0):
    rows = _strokes(shard_dir, shard_ids)
    if not rows:
        raise SystemExit("no strokes found -- fetch the shards first (vae/data_full.py fetch)")
    rows = [r for r in rows if r[6] > 0.0]   # need a measurable curvature
    L = np.array([r[0] for r in rows]); vbar = np.array([r[2] for r in rows])
    v_e = np.array([r[3] for r in rows])
    k0 = np.array([r[4] for r in rows]); k1 = np.array([r[5] for r in rows])
    K = np.array([r[6] for r in rows])
    rng = np.random.default_rng(seed)
    print(f"\n{len(rows)} strokes with a measurable curvature")

    lo, lv, lk = np.log(L), np.log(vbar), np.log(K)
    lk_mean = float(lk.mean())
    b0, a0 = np.polyfit(lo, lv, 1)
    res0 = lv - (a0 + b0 * lo)
    print(f"pace, length only:  log vbar = {a0:.4f} + {b0:.4f} log L          resid sd {res0.std():.4f}")
    # Humans obey a speed-curvature law: the wigglier the path, the slower they go. Without this the
    # sampler draws fast paces for long-but-winding planner paths, which the acceleration caps then
    # claw back (acceleration ~ curvature x speed^2), so the pace never reaches the robot.
    design = np.stack([np.ones_like(lo), lo, lk - lk_mean], axis=1)
    coef, *_ = np.linalg.lstsq(design, lv, rcond=None)
    a, b, c = (float(x) for x in coef)
    res = lv - design @ coef
    print(f"pace, + curvature:  log vbar = {a:.4f} + {b:.4f} log L {c:+.4f} (log K - {lk_mean:.4f})")
    print(f"                    resid sd {res.std():.4f}   "
          f"({100 * (1 - res.var() / res0.var()):.0f}% of the length-only residual variance explained)")
    print(f"  DROID curvature: p5={np.percentile(K,5):.1f} p50={np.percentile(K,50):.1f} "
          f"p95={np.percentile(K,95):.1f}   (mean log = {lk_mean:.4f})")
    keys, vals = [], []
    for pair in KIND_PAIRS:
        s = (k0 == pair[0]) & (k1 == pair[1])
        if s.sum() >= 200:
            keys.append("|".join(pair)); vals.append(float(res[s].mean()))
    res -= np.array([dict(zip(keys, vals)).get("|".join(p), 0.0) for p in zip(k0, k1)])
    print(f"  + kind offsets                          resid sd {res.std():.4f}")
    for k, v in sorted(zip(keys, vals), key=lambda t: t[1]):
        print(f"    {k:<14} {v:+.4f}  (x{np.exp(v):.2f})")

    dur = np.array([r[1] for r in rows])
    blob = {"pace_a": a, "pace_b": b, "pace_c": c, "curv_log_mean": lk_mean,
            "pace_off_keys": np.array(keys), "pace_off_vals": np.array(vals),
            "pace_resid": rng.choice(res, min(_POOL, len(res)), replace=False).astype(np.float32),
            "dur_median": float(np.median(dur)),
            # Empirical support, used to clip samples back to what DROID actually does.
            "pace_range": np.array([np.percentile(vbar, 0.5), np.percentile(vbar, 99.5)]),
            "n_strokes": np.int64(len(rows))}
    print(f"\nmedian stroke duration {np.median(dur):.3f} s -- the reference the flow model's "
          f"normalized-time profile is implicitly calibrated to (see _retime_stroke's end remap)")

    print("\nboundary: log v_e = a + b log vbar, per event kind")
    for kind in ("close", "open"):
        s = (k1 == kind) & (v_e > 1e-4)
        x, y = np.log(vbar[s]), np.log(v_e[s])
        bb, aa = np.polyfit(x, y, 1)
        r = y - (aa + bb * x)
        print(f"  {kind:<6} n={int(s.sum()):>6}  a={aa:+.4f} b={bb:.4f}  resid sd={r.std():.4f}  "
              f"(sd if unconditioned {y.std():.4f})   median v_e={np.median(v_e[s]):.4f}")
        blob[f"bnd_a_{kind}"] = aa
        blob[f"bnd_b_{kind}"] = bb
        blob[f"bnd_resid_{kind}"] = rng.choice(r, min(_POOL, len(r)), replace=False).astype(np.float32)
        blob[f"bnd_range_{kind}"] = np.array([np.percentile(v_e[s], 0.5), np.percentile(v_e[s], 99.5)])

    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **blob)
    print(f"\nwrote {out}")

    # Sanity: draws from the fitted model should reproduce the source marginals.
    st = TimingStats(out)
    g = np.random.default_rng(1)
    idx = g.choice(len(L), 20000)
    sp = np.array([st.sample_pace(L[i], K[i], (k0[i], k1[i]), g) for i in idx])
    print("\ncheck -- sampled vs source (p5/p50/p95):")
    print(f"  pace   sampled {np.percentile(sp,5):.3f}/{np.percentile(sp,50):.3f}/"
          f"{np.percentile(sp,95):.3f}   source {np.percentile(vbar,5):.3f}/"
          f"{np.percentile(vbar,50):.3f}/{np.percentile(vbar,95):.3f}")
    for kind in ("close", "open"):
        s = (k1 == kind) & (v_e > 1e-4)
        sub = g.choice(np.flatnonzero(s), 20000)
        sv = np.array([st.sample_event_speed(kind, vbar[i], g) for i in sub])
        print(f"  {kind:<6} sampled {np.percentile(sv,5):.3f}/{np.percentile(sv,50):.3f}/"
              f"{np.percentile(sv,95):.3f}   source {np.percentile(v_e[s],5):.3f}/"
              f"{np.percentile(v_e[s],50):.3f}/{np.percentile(v_e[s],95):.3f}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", action="store_true")
    ap.add_argument("--shard-dir", type=str,
                    default="/home/prpl/tamp-vla/vae/data_cache/droid_full_proprio")
    ap.add_argument("--shards", type=str, default="0,1,2,3,4,5,6,7,8,9,10,11,12")
    ap.add_argument("--out", type=str, default=str(DEFAULT_STATS))
    a = ap.parse_args()
    if not a.fit:
        ap.error("pass --fit")
    fit(Path(a.shard_dir), [int(s) for s in a.shards.split(",")], Path(a.out))

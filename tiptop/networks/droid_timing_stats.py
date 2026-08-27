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

The PACE is measured over the 13 cached DROID proprio shards, position-derived at 15 Hz:

* pace:      ``log vbar = a + b log L + c (log K - mean log K) + off[k_start, k_end] + eps``, where K is
  :func:`path_curvature`. The CURVATURE term carries this model: ``c`` is about -0.7 (humans obey a
  speed-curvature law) and adding it cuts the residual sd from 0.456 to 0.296 -- 61% of the variance
  the length-only law leaves -- after which the kind offsets are small (-0.13 to +0.10), because
  curvature was most of what they had been standing in for. Residuals are skewed and heavy-tailed, so
  they are sampled from an empirical pool rather than a Gaussian.
The pace sampler clips to the range DROID actually exhibits: a regression with a heavy-tailed empirical
residual extrapolates past its own data once the conditioning value is itself a sample.

The BOUNDARY is a different DROID COLUMN from the pace -- commanded, not achieved
--------------------------------------------------------------------------------
The gripper-event speed is fitted on DROID's ``action.joint_velocity`` (see
:func:`droid_boundary_ratios`), the COMMANDED channel, while the pace above is position-derived
(ACHIEVED). That split is deliberate, and getting it wrong was a real regression:

* The two diverge exactly at a gripper event. A human's ARM slows to ~0.29 of its cruise into a
  grasp; their COMMAND only to ~0.65, because the command leads the arm and does not wait for it.
* cuRobo tracks our commands ~1:1, so writing an achieved-motion statistic into the command channel
  emitted a grasp command ~3x slower than any human's -- the stall a velocity-action policy learns
  (``analysis_gripper_events/GRIPPER_EVENT_VELOCITY.md`` measured min/cruise 0.15 against teleop's
  0.62 on data generated that way).

Two traps worth naming, since both cost a wrong conclusion here:

* ``action.joint_velocity`` is NOT the aggregate ``action`` column that the proprio shards cache.
  That one is a composite: it correlates r~0.1 with joint motion and ~33% of its entries sit at
  +-1, which reads as a hard-clipped velocity signal but is not one. ``action.joint_velocity`` is
  genuine rad/s -- 0.2% at the clip, r=0.82-0.89 against d(q)/dt, matching teleop's 0.83-0.94.
* The grasp/release asymmetry differs between the channels: open/close is 1.55 in achieved motion
  but ~1.2 in the command, so a boundary model fitted on one does not transfer to the other.

``--boundary-source teleop`` refits the same quantity from the local teleop dataset instead. It is a
small-n cross-check (~61 events per kind against DROID's ~9k), useful because teleop is this robot and
this task; the two agree that the command does not collapse at a grasp, and differ in detail.

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
DOF_TELEOP = 7
# Per-event boundary ratios streamed from DROID's action.joint_velocity (see droid_boundary_ratios).
DEFAULT_BND_CACHE = _PKG_DIR / "checkpoints" / "boundary_cache"
# Teleop, available as a cross-check boundary source (--boundary-source teleop).
DEFAULT_TELEOP = Path.home() / ".cache/huggingface/lerobot/SamratSahoo/teleop_prpl"


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
        # Boundary: per-event COMMANDED-speed / stroke-pace ratios (DROID action.joint_velocity by
        # default -- see sample_event_speed). Absent in an older stats file, which falls back to the
        # achieved-motion regression below.
        self.bnd_ratio = {k: b[f"bnd_ratio_{k}"] for k in ("close", "open")
                          if f"bnd_ratio_{k}" in b}
        self.bnd = {}
        for kind in ("close", "open"):
            if f"bnd_a_{kind}" not in b:
                continue
            rng_ = tuple(b[f"bnd_range_{kind}"]) if f"bnd_range_{kind}" in b else (1e-4, 1e3)
            self.bnd[kind] = (float(b[f"bnd_a_{kind}"]), float(b[f"bnd_b_{kind}"]),
                              b[f"bnd_resid_{kind}"], *rng_)
        # Record WHICH source the ratios came from -- the two differ (DROID close 0.63 of cruise vs
        # teleop 1.05), so a run log that cannot tell them apart is worse than no log.
        self.bnd_source = str(b["bnd_source"]) if "bnd_source" in b else "unknown"
        src = (f"{self.bnd_source} commanded ratios (n={len(next(iter(self.bnd_ratio.values())))})"
               if self.bnd_ratio else "DROID achieved motion (legacy)")
        _log.info("Loaded timing stats %s (pace: DROID, resid sd %.3f, %d strokes; boundary: %s)",
                  p.name, float(self.pace_resid.std()), int(b["n_strokes"]), src)

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
        """COMMANDED joint speed (rad/s) at a gripper event of this kind, for a stroke running at ``pace``.

        Drawn as a FRACTION of the stroke's own pace, from the teleop ratios measured on
        ``action.joint_velocity`` -- the same channel this value ends up in. See the module docstring
        for why the DROID-fitted version (kept below as a fallback) was the wrong reference: it
        described how fast the human's ARM moved, which is ~3x slower at a grasp than what the human
        COMMANDED, and our commands are tracked almost 1:1.

        A ratio rather than an absolute speed because the teleop operators run ~1.5x slower than our
        plans, so their absolute rad/s would not transfer; the ratio does, and it makes the anchor
        scale with whatever pace this stroke happened to draw. The draw is a plain inverse-CDF sample
        over the ~61 measured events per kind -- n that size supports an empirical distribution, not a
        pace-conditional law (and on DROID the pace slope explained only ~17% of the variance anyway).
        """
        if kind not in ("close", "open"):
            return 0.0
        if kind in self.bnd_ratio:
            return float(max(pace, 1e-6) * rng.choice(self.bnd_ratio[kind]))
        # Fallback: a stats file fitted before the teleop ratios existed (DROID achieved motion).
        if kind not in self.bnd:
            return 0.0
        a, b, resid, lo, hi = self.bnd[kind]
        v = np.exp(a + b * np.log(max(pace, 1e-6)) + rng.choice(resid))
        # Clip to the range DROID actually exhibits. A regression with a heavy-tailed empirical residual
        # extrapolates past its own data when the conditioning value is itself sampled: unclipped this
        # asks for boundary speeds over 4 rad/s, which the caps then claw back by stretching the whole
        # stroke -- so one outlier draw slows an entire operation.
        return float(np.clip(v, lo, hi))


def droid_boundary_ratios(cache_dir: Path = None) -> dict[str, np.ndarray]:
    """Per-gripper-event COMMANDED speed / stroke pace, from DROID's ``action.joint_velocity``.

    This is the shipped boundary reference. Note it is a DIFFERENT DROID column from the one the pace
    law uses: the pace comes from position-derived (achieved) speed, this comes from the commanded
    channel, because that is the channel the blend writes into and the two differ sharply at a grasp
    (a human's arm slows to ~0.29 of cruise there; their command only to ~0.65).

    Not to be confused with the aggregate ``action`` column the proprio shards cache -- that one is a
    composite that correlates only r~0.1 with joint motion and is useless as a velocity reference.
    ``action.joint_velocity`` is genuine rad/s: 0.2% at the +-1 clip, r=0.82-0.89 against d(q)/dt.

    Pre-extracted by ``boundary_cache/bnd_*.npz`` (see the streaming note in the fit CLI help), since
    the column is not in the cached proprio shards and must be streamed from the Hub.
    """
    d = Path(cache_dir) if cache_dir else DEFAULT_BND_CACHE
    files = sorted(Path(d).glob("bnd_*.npz"))
    if not files:
        raise FileNotFoundError(
            f"no bnd_*.npz under {d}. The boundary reference needs DROID's action.joint_velocity, "
            f"which the proprio shards do not carry -- stream it first (see --help)."
        )
    out = {}
    for kind in ("close", "open"):
        pool = np.concatenate([np.load(f)[kind] for f in files])
        out[kind] = pool[np.isfinite(pool) & (pool > 0)].astype(np.float32)
    return out


def teleop_boundary_ratios(teleop_dir: Path) -> dict[str, np.ndarray]:
    """Per-gripper-event COMMANDED speed / stroke pace, from a teleop LeRobot dataset.

    Reads ``action.joint_velocity`` -- genuine rad/s, unlike DROID's action channel, which is
    normalized with 35% of entries at the +-1 clip and cannot serve as a velocity reference at all.
    That is why the boundary comes from teleop while the pace law still comes from DROID: this is the
    only human source we have for the channel the blend actually writes.

    For each event: the stroke ending at it, its mean commanded speed, and a 3-frame median of the
    commanded speed at the event (one frame is noisy, and the 3-frame window straddles the edge, which
    matches how the blender's single shared anchor serves both adjacent strokes).
    """
    import pyarrow.parquet as pq

    files = sorted(Path(teleop_dir).glob("data/**/*.parquet"))
    if not files:
        raise FileNotFoundError(f"no parquet under {teleop_dir}/data")
    out = {"close": [], "open": []}
    for fp in files:
        t = pq.read_table(fp)
        av = np.asarray(t["action.joint_velocity"].combine_chunks().flatten()
                        .to_numpy(zero_copy_only=False), np.float64).reshape(-1, DOF_TELEOP)
        g = np.asarray(t["action.gripper_position"].combine_chunks()
                       .to_numpy(zero_copy_only=False), np.float64).reshape(-1)
        ep = np.asarray(t["episode_index"].combine_chunks().to_numpy(zero_copy_only=False))
        for e in np.unique(ep):
            m = ep == e
            v, gg = np.linalg.norm(av[m], axis=1), g[m]
            closed = (gg > 0.5).astype(int)
            edges = np.flatnonzero(np.diff(closed)) + 1
            bounds = sorted({0, len(v) - 1, *edges.tolist()})
            for i in edges:
                kind = "close" if closed[i] else "open"
                start = max([x for x in bounds if x < i], default=0)
                if i - start < 6:
                    continue
                pace = float(v[start:i + 1].mean())
                if pace < 1e-6:
                    continue
                out[kind].append(float(np.median(v[max(i - 1, 0):i + 2])) / pace)
    return {k: np.asarray(v, np.float32) for k, v in out.items() if v}


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


def fit(shard_dir: Path, shard_ids, out: Path, seed: int = 0, boundary_source: str = "droid",
        teleop_dir: Path | None = None, bnd_cache: Path | None = None):
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

    # The boundary the blender actually uses: teleop COMMANDED speed as a fraction of stroke pace.
    # The DROID block above is kept only as a fallback for older stats files -- it measures achieved
    # motion, which at a grasp is ~3x slower than what a human commands.
    ratios = (droid_boundary_ratios(bnd_cache) if boundary_source == "droid"
              else teleop_boundary_ratios(teleop_dir or DEFAULT_TELEOP))
    blob["bnd_source"] = boundary_source
    print(f"\nboundary (SHIPPED): {boundary_source} COMMANDED speed / stroke pace, per event kind")
    print(f"  {'kind':<7}{'n':>5} {'p5':>7} {'p25':>7} {'p50':>7} {'p75':>7} {'p95':>7}  {'sd(log)':>8}")
    for kind in ("close", "open"):
        r = ratios[kind]
        blob[f"bnd_ratio_{kind}"] = r
        print(f"  {kind:<7}{len(r):>5} " + " ".join(f"{np.percentile(r, q):>7.3f}"
                                                   for q in (5, 25, 50, 75, 95))
              + f"  {np.log(r).std():>8.3f}")
    print(f"  open/close (median ratio) = {np.median(ratios['open']) / np.median(ratios['close']):.2f}"
          f"   -- against 1.55 for DROID ACHIEVED motion. The command dips far less at a grasp than the"
          f" arm does (0.65 of cruise vs 0.29), which is the whole reason this is a separate fit.")

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
    # The boundary is a RATIO now, so check it against the teleop ratios, not against DROID's
    # absolute achieved speeds -- those are a different channel and would look like a huge miss.
    for kind in ("close", "open"):
        sv = np.array([st.sample_event_speed(kind, 1.0, g) for _ in range(20000)])
        r = ratios[kind]
        print(f"  {kind:<6} boundary/pace sampled {np.percentile(sv,5):.3f}/"
              f"{np.percentile(sv,50):.3f}/{np.percentile(sv,95):.3f}   source "
              f"{np.percentile(r,5):.3f}/{np.percentile(r,50):.3f}/{np.percentile(r,95):.3f}")
    print(f"  -> at our ~0.35 rad/s cruise that is "
          f"{0.35*np.median(ratios['close']):.2f} rad/s at a grasp and "
          f"{0.35*np.median(ratios['open']):.2f} at a release "
          f"(the DROID-ACHIEVED fit gave 0.10 / 0.17 -- the regression this replaces).")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", action="store_true")
    ap.add_argument("--shard-dir", type=str,
                    default="/home/prpl/tamp-vla/vae/data_cache/droid_full_proprio")
    ap.add_argument("--shards", type=str, default="0,1,2,3,4,5,6,7,8,9,10,11,12")
    ap.add_argument("--out", type=str, default=str(DEFAULT_STATS))
    ap.add_argument("--boundary-source", type=str, default="droid", choices=("droid", "teleop"),
                    help="whose COMMANDED gripper-event speed to sample. 'droid' (default) reads the "
                         "pre-streamed boundary_cache/bnd_*.npz; 'teleop' reads the local teleop "
                         "LeRobot cache and is a small-n cross-check (~61 events/kind vs ~10k).")
    ap.add_argument("--bnd-cache", type=str, default=str(DEFAULT_BND_CACHE),
                    help="dir of bnd_*.npz holding DROID action.joint_velocity event ratios")
    ap.add_argument("--teleop-dir", type=str, default=str(DEFAULT_TELEOP))
    a = ap.parse_args()
    if not a.fit:
        ap.error("pass --fit")
    fit(Path(a.shard_dir), [int(s) for s in a.shards.split(",")], Path(a.out),
        boundary_source=a.boundary_source, teleop_dir=Path(a.teleop_dir),
        bnd_cache=Path(a.bnd_cache))

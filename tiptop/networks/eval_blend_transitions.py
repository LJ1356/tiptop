"""Score what the BLENDER EMITS against DROID -- the gap ``eval_transitions`` cannot see.

``eval_transitions`` calls ``FlowModel.sample_batch`` directly, so it never runs ``_retime_stroke``:
no boundary override, no vel/accel caps, no resample to the plan ``dt``. It therefore scored the model
well while the emitted plan's boundary was a config constant realized to +-1%. This script scores the
COMMANDED joint velocity that actually reaches the dataset (``lerobot_capture._flatten_plan`` samples
the plan's ``velocities`` onto the 15 Hz grid), against held-out DROID.

Two modes:

  --plans '<glob>'    score saved ``tiptop_plan.json`` runs as emitted.
  --replay '<glob>'   re-run ``flow_blend_group`` offline over those plans' group paths, so a code or
                      config change can be scored without a robot (``--boundary-speed`` sweeps the knob).

DROID references are position-derived: ``action.joint_velocity`` is hard-clipped at +-1 rad/s and is
NOT unit-comparable. Run in the tiptop pixi env, or the openpi venv for --plans only::

    tiptop/.pixi/envs/default/bin/python -m tiptop.networks.eval_blend_transitions \
        --replay 'data-collection/runs/prpl/tamp/vae_tdf_jd_flowblend/*/*/tiptop_plan.json'
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_stroke_timing import (  # noqa: E402
    ACT_GRIPPER_COL, MIN_STROKE_ARC, MIN_STROKE_LEN, gripper_events,
)

FPS = 15.0
DT15 = 1.0 / FPS
WIN = 8  # frames either side of a gripper event for the notch profile
DEFAULT_SHARDS = Path("/home/prpl/tamp-vla/vae/data_cache/droid_full_proprio")

# Held-out DROID (shards 10-12), position-derived medians. Regenerate with --droid-only.
DROID_NOTCH = {
    "close": [0.183, 0.167, 0.152, 0.137, 0.123, 0.110, 0.101, 0.094, 0.089, 0.089, 0.092, 0.097,
              0.108, 0.123, 0.145, 0.168, 0.201],
    "open": [0.223, 0.206, 0.189, 0.172, 0.160, 0.154, 0.149, 0.150, 0.156, 0.172, 0.194, 0.222,
             0.260, 0.301, 0.342, 0.384, 0.417],
}
DROID_BOUNDARY_MED = {"close": 0.101, "open": 0.156}   # abs rad/s at the stroke END
DROID_RATIO_MED = {"close": 0.289, "open": 0.475}      # as a fraction of the stroke's own mean
DROID_PACE = (-1.1622, 0.4182)                          # log vbar = a + b log L
DROID_TEXTURE = (19.00, 11.61)                          # (jerk RMS, curvature on the _CURV_STEP grid)


def pct(x, ps=(5, 25, 50, 75, 95)):
    return "  ".join(f"p{p}={np.percentile(x, p):.3f}" for p in ps)


# Arc-length spacing (rad) the geometric-texture metric is evaluated on. It MUST be fixed rather than
# derived from the sample count: a curvature computed on "however many frames this stroke happens to
# have" shrinks as the stroke gets faster, so a pace change alone would look like a texture change.
_CURV_STEP = 0.01


def texture(pos, dt):
    """(jerk RMS in time, mean |d2q/du2| on a fixed arc-length grid) for a [N,dof] joint path.

    The second value is timing-free and resolution-independent, so it compares the PATH's wiggle --
    the configuration-space texture the VAE-manifold cost puts there -- across blend settings, paces
    and against DROID on equal terms.
    """
    if len(pos) < 8:
        return None
    jerk = float(np.sqrt(((np.diff(pos, n=3, axis=0) / dt ** 3) ** 2).sum(axis=1).mean()))
    u = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pos, axis=0), axis=1))])
    if u[-1] < 4 * _CURV_STEP:
        return None
    n = int(u[-1] / _CURV_STEP) + 1
    q = np.stack([np.interp(np.linspace(0, u[-1], n), u, pos[:, k]) for k in range(pos.shape[1])], 1)
    return jerk, float(np.abs(np.diff(q, n=2, axis=0) / _CURV_STEP ** 2).sum(axis=1).mean())


def plan_groups(path):
    """-> [(positions [N,dof], velocities [N,dof], dt, kind_start, kind_end)] per operation group."""
    steps = json.load(open(path))["steps"]
    runs, cur, prev = [], [], "none"
    for s in steps:
        if s["type"] == "trajectory":
            cur.append(s)
        else:
            if cur:
                runs.append((cur, prev, s["action"]))
                cur = []
            prev = s["action"]
    if cur:
        runs.append((cur, prev, "none"))
    out = []
    for run, k0, k1 in runs:
        out.append((np.concatenate([np.asarray(s["positions"], np.float64) for s in run], 0),
                    np.concatenate([np.asarray(s["velocities"], np.float64) for s in run], 0),
                    float(run[0]["dt"]), k0, k1))
    return out


def path_deviation(emitted, original):
    """Max per-joint |q| deviation (deg) of the emitted path from the collision-checked waypoints.

    Smoothing rounds the corners at the old segment joins, so the executed path is NOT the one cuRobo
    collision-checked. ``trajectory_blending``'s docstring quotes ~1 deg at the default smoothing; this
    measures it for whatever smoothing is actually set, which is what makes raising it defensible (or
    not) near clutter. Both paths are compared at matched arc-length fraction.
    """
    a = np.asarray(emitted, np.float64)
    b = np.asarray(original, np.float64)
    if len(a) < 2 or len(b) < 2:
        return None
    # NEAREST-POINT distance to the original polyline, not a matched-parameter difference: corner
    # rounding shifts the arc-length parameterisation, so comparing at matched fraction reports a
    # phase offset as if it were a spatial excursion. What collision safety cares about is how far the
    # executed path strays from the checked one, i.e. distance to the polyline as a SET.
    if len(a) > 400:
        a = a[np.linspace(0, len(a) - 1, 400).astype(int)]
    seg = b[1:] - b[:-1]
    ss = np.einsum("ij,ij->i", seg, seg)
    ss[ss < 1e-18] = 1e-18
    w = a[:, None, :] - b[None, :-1, :]
    t = np.clip(np.einsum("kij,ij->ki", w, seg) / ss[None], 0.0, 1.0)
    proj = b[None, :-1, :] + t[..., None] * seg[None]
    d = np.linalg.norm(a[:, None, :] - proj, axis=2).min(axis=1)
    return float(d.max()) * 180.0 / np.pi


def score(groups, label):
    """``groups`` = [(pos, vel, dt, k0, k1)] in plan order per episode -> print the comparison."""
    L, dur, vbar, v0, v1, K0, K1, tex = [], [], [], [], [], [], [], []
    notch = {"close": [], "open": []}
    for ep in groups:
        s15 = []
        for pos, vel, dt, k0, k1 in ep:
            spd = np.linalg.norm(vel, axis=1)
            d = (len(pos) - 1) * dt
            arc = float(np.linalg.norm(np.diff(pos, axis=0), axis=1).sum())
            L.append(arc); dur.append(d); vbar.append(arc / d if d > 0 else 0.0)
            v0.append(float(spd[0])); v1.append(float(spd[-1])); K0.append(k0); K1.append(k1)
            t = np.arange(len(spd)) * dt
            grid = np.arange(0.0, d + 1e-9, DT15)
            s15.append((np.interp(grid, t, spd), k1))
            p15 = np.stack([np.interp(grid, t, pos[:, j]) for j in range(pos.shape[1])], 1)
            r = texture(p15, DT15)
            if r:
                tex.append(r)
        for i in range(len(s15) - 1):
            a, k = s15[i]
            b, _ = s15[i + 1]
            if k in notch and len(a) > WIN and len(b) > WIN:
                notch[k].append(np.concatenate([a[-(WIN + 1):], b[1:WIN + 1]]))

    L, dur, vbar = np.array(L), np.array(dur), np.array(vbar)
    v0, v1 = np.array(v0), np.array(v1)
    K0, K1, tex = np.array(K0), np.array(K1), np.array(tex)
    print(f"\n########## {label}   ({len(L)} groups, "
          f"{sum(len(v) for v in notch.values())} gripper events)")

    print("\n=== pace ===")
    print(f"  joint arc L (rad):  {pct(L)}      DROID p50 1.645")
    print(f"  duration (s):       {pct(dur)}      DROID p50 4.07")
    print(f"  mean speed (rad/s): {pct(vbar)}      DROID p50 0.391")
    ok = (L > 1e-6) & (vbar > 1e-6)
    if ok.sum() > 5:
        pred = np.exp(DROID_PACE[0] + DROID_PACE[1] * np.log(L[ok]))
        print(f"  pace / DROID-predicted-for-this-L: {pct(vbar[ok] / pred)}   (1.0 = DROID's pace law)")

    print("\n=== boundary speed at the stroke END (rad/s) ===")
    for kind in ("close", "open"):
        s = K1 == kind
        if s.sum() < 3:
            continue
        print(f"  {kind:<6} n={int(s.sum()):<4} med={np.median(v1[s]):.3f} "
              f"(DROID {DROID_BOUNDARY_MED[kind]:.3f})   ratio-to-pace med="
              f"{np.median(v1[s] / np.clip(vbar[s], 1e-9, None)):.3f} "
              f"(DROID {DROID_RATIO_MED[kind]:.3f})   {pct(v1[s])}")
    if (K1 == "close").sum() > 2 and (K1 == "open").sum() > 2:
        print(f"  open/close asymmetry = "
              f"{np.median(v1[K1 == 'open']) / max(np.median(v1[K1 == 'close']), 1e-9):.3f}   "
              f"(DROID 1.545)")

    print(f"\n=== commanded speed through the event, +-{WIN} frames @15Hz ===")
    print("kind    src     " + " ".join(f"{j - WIN:>6}" for j in range(2 * WIN + 1)))
    for kind in ("close", "open"):
        if not notch[kind]:
            continue
        m = np.median(np.array(notch[kind]), 0)
        print(f"{kind:<8}{'ours':<8}" + " ".join(f"{v:>6.3f}" for v in m))
        print(f"{'':<8}{'DROID':<8}" + " ".join(f"{v:>6.3f}" for v in DROID_NOTCH[kind]))
        trough_o, trough_d = m.min(), min(DROID_NOTCH[kind])
        print(f"{'':<8}trough {trough_o:.3f} vs DROID {trough_d:.3f} ({trough_o / trough_d:.2f}x), "
              f"exit(+8) {m[-1]:.3f} vs {DROID_NOTCH[kind][-1]:.3f} "
              f"({m[-1] / DROID_NOTCH[kind][-1]:.2f}x)")

    if len(tex):
        print(f"\n=== path texture (criterion: keep the VAE-manifold wiggle) ===")
        print(f"  jerk RMS med {np.median(tex[:, 0]):.2f} (DROID {DROID_TEXTURE[0]:.2f})   "
              f"curvature/arc med {np.median(tex[:, 1]):.3f} (DROID {DROID_TEXTURE[1]:.3f})")


def droid_reference(shard_dir, shard_ids):
    """Recompute the DROID reference tables from the cached proprio shards."""
    rows, notch = [], {"close": [], "open": []}
    for si in shard_ids:
        sp = Path(shard_dir) / f"shard_{si:03d}.npz"
        if not sp.exists():
            print(f"  (missing {sp.name})")
            continue
        blob = np.load(sp, allow_pickle=True)
        J, A = blob["joints"], blob["actions"]
        for i in range(int(blob["n_eps"])):
            q, a = np.asarray(J[i], np.float64), np.asarray(A[i], np.float64)
            if len(q) < MIN_STROKE_LEN or a.ndim != 2 or a.shape[1] <= ACT_GRIPPER_COL:
                continue
            spd = np.linalg.norm(np.diff(q, axis=0), axis=1) / DT15
            evs = gripper_events(a[:, ACT_GRIPPER_COL])
            bounds = sorted({0, len(q) - 1, *[f for f, _ in evs]})
            for s_, e_ in zip(bounds[:-1], bounds[1:]):
                seg = q[s_:e_ + 1]
                arc = float(np.linalg.norm(np.diff(seg, axis=0), axis=1).sum())
                if len(seg) < MIN_STROKE_LEN or arc < MIN_STROKE_ARC:
                    continue
                d = (e_ - s_) * DT15
                tx = texture(seg, DT15)
                rows.append((arc, d, arc / d, float(np.median(spd[s_:s_ + 3])),
                             float(np.median(spd[max(e_ - 3, 0):e_])),
                             dict(evs).get(s_, "none"), dict(evs).get(e_, "none"),
                             tx[0] if tx else np.nan, tx[1] if tx else np.nan))
            for f, kind in evs:
                if f - WIN >= 0 and f + WIN + 1 <= len(spd):
                    notch[kind].append(spd[f - WIN:f + WIN + 1])
    L = np.array([r[0] for r in rows]); vb = np.array([r[2] for r in rows])
    v1 = np.array([r[4] for r in rows]); K1 = np.array([r[6] for r in rows])
    print(f"\nDROID reference, {len(rows)} strokes")
    print(f"  L {pct(L)}\n  vbar {pct(vb)}")
    b, a0 = np.polyfit(np.log(L), np.log(vb), 1)
    print(f"  pace law: log vbar = {a0:.4f} + {b:.4f} log L   "
          f"resid sd={(np.log(vb) - a0 - b * np.log(L)).std():.4f}")
    for kind in ("close", "open"):
        s = K1 == kind
        print(f"  end {kind:<6} abs med={np.median(v1[s]):.4f}  "
              f"ratio med={np.median(v1[s] / vb[s]):.4f}   {pct(v1[s])}")
    print("  notch medians:")
    for kind in ("close", "open"):
        print(f"    {kind:<6} " + " ".join(f"{v:.3f}" for v in np.median(np.array(notch[kind]), 0)))
    tx = np.array([(r[7], r[8]) for r in rows])
    tx = tx[np.isfinite(tx).all(axis=1)]
    print(f"  texture: jerk RMS med={np.median(tx[:, 0]):.2f}  "
          f"curvature/arc med={np.median(tx[:, 1]):.3f}  (update DROID_TEXTURE with these)")


# FR3 arm limits, so an offline replay is capped the way a live run is (a live blend reads these from
# motion_gen via arm_joint_limits; a saved plan does not carry them). Acceleration matches
# cuTAMP/cutamp/robots/assets/fr3_franka.yml (max_acceleration: 15.0, uniform).
FR3_VEL = np.array([2.62, 2.62, 2.62, 2.62, 5.26, 4.18, 5.26])
FR3_ACC = np.full(7, 15.0)


def _as_cutamp_plan(path):
    """Saved tiptop_plan.json -> the in-memory cuTAMP plan shape ``flow_blend_cutamp_plan`` expects."""
    import torch
    from curobo.types.state import JointState

    out = []
    for s in json.load(open(path))["steps"]:
        if s["type"] != "trajectory":
            out.append(dict(s))
            continue
        pos = torch.as_tensor(np.asarray(s["positions"], np.float32))
        vel = torch.as_tensor(np.asarray(s["velocities"], np.float32))
        out.append({"type": "trajectory", "dt": float(s["dt"]), "label": s.get("label", ""),
                    "plan": JointState(position=pos, velocity=vel,
                                       acceleration=torch.zeros_like(pos),
                                       jerk=torch.zeros_like(pos), joint_names=None)})
    return out


def replay(files, overrides, model_path, repeats=1):
    """Re-blend saved plans through the REAL entry point -> episodes in ``score`` format.

    Goes through ``flow_blend_cutamp_plan`` rather than ``flow_blend_group`` so the replay exercises the
    whole-plan timing pass -- the shared per-event boundary draw is decided there, and it is exactly the
    part a per-stroke harness cannot see.
    """
    from tiptop.flow_blending import flow_blend_cutamp_plan
    from tiptop.networks.flow_timing import FlowModel
    from tiptop.trajectory_blending import resolve_blend_config

    cfg = resolve_blend_config({"blend_trajectory": True, "blend_mode": "flow", **overrides})
    model = FlowModel(model_path, device="cpu")
    out = []
    # Every draw is fresh, and the pace / boundary conditionals are heavy-tailed (sd of log ~1.0), so a
    # single pass over ~100 events leaves the per-kind medians with an appreciable standard error --
    # enough to move the open/close ratio by ~1.3x run to run. Repeat to settle them.
    files = list(files) * max(1, repeats)
    devs = []
    for f in files:
        try:
            blended = flow_blend_cutamp_plan(_as_cutamp_plan(f), cfg, model, FR3_VEL, FR3_ACC)
        except Exception as exc:  # noqa: BLE001
            print(f"  replay failed on {Path(f).parent.name} ({type(exc).__name__}: {exc})")
            continue
        orig = [g[0] for g in plan_groups(f)]   # same order as the blended runs
        ep = []
        prev = "none"
        for i, s in enumerate(blended):
            if s.get("type") != "trajectory":
                prev = s.get("action", "none")
                continue
            nxt = blended[i + 1].get("action", "none") if i + 1 < len(blended) else "none"
            if len(ep) < len(orig):
                d = path_deviation(s["plan"].position.numpy().astype(np.float64), orig[len(ep)])
                if d is not None:
                    devs.append(d)
            ep.append((s["plan"].position.numpy().astype(np.float64),
                       s["plan"].velocity.numpy().astype(np.float64), float(s["dt"]),
                       prev if prev in ("close", "open") else "none",
                       nxt if nxt in ("close", "open") else "none"))
            prev = "none"
        if ep:
            out.append(ep)
    if devs:
        d = np.array(devs)
        print(f"\n=== path deviation from the collision-checked waypoints (deg, max per stroke) ===")
        print(f"  {pct(d)}  max={d.max():.2f}   -- nearest-point distance to the checked polyline. "
              f"trajectory_blending quotes ~1 deg at its 1e-4 default; validate on the viz near clutter")
    return out


def main(a):
    if a.droid_only:
        droid_reference(a.shard_dir, [int(s) for s in a.shards.split(",")])
        return
    pattern = a.plans or a.replay
    if pattern.startswith("@"):   # @file -> newline-separated list of plan paths
        files = [ln.strip() for ln in open(pattern[1:]) if ln.strip()]
    else:
        files = sorted(glob.glob(pattern))
    if a.limit:
        files = files[: a.limit]
    if not files:
        raise SystemExit("no plans matched")
    if a.plans:
        score([plan_groups(f) for f in files], f"AS EMITTED  ({len(files)} plans)")
        return
    overrides = {"blend_boundary_speed": a.boundary_speed, "blend_smoothing": a.smoothing,
                 "blend_flow_retime_only": not a.no_retime_only, "blend_pace": a.pace,
                 "blend_pace_scale": a.pace_scale, "blend_boundary_mode": a.boundary_mode,
                 "blend_profile_end_sec": a.profile_end_sec,
                 "blend_boundary_window_sec": a.boundary_window_sec,
                 "blend_ops": a.ops.split(",") if a.ops else None,
                 "blend_acc_slack": a.acc_slack}
    if a.seed is not None:
        overrides["blend_seed"] = a.seed
    score(replay(files, overrides, a.model_path, a.repeats),
          f"REPLAY  pace={a.pace}x{a.pace_scale} boundary={a.boundary_mode} "
          f"(floor {a.boundary_speed}) smoothing={a.smoothing} "
          f"retime_only={not a.no_retime_only} model={Path(a.model_path).name if a.model_path else 'default'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--plans", type=str, default=None,
                    help="glob (or @listfile) of tiptop_plan.json to score as emitted")
    ap.add_argument("--replay", type=str, default=None,
                    help="glob (or @listfile) of plans to re-blend offline. Prefer UNBLENDED plans: "
                         "re-blending an already-blended plan smooths its geometry twice and "
                         "understates the texture the live blender actually sees.")
    ap.add_argument("--boundary-speed", type=float, default=0.02,
                    help="constant boundary speed, or the FLOOR when --boundary-mode droid")
    ap.add_argument("--boundary-mode", type=str, default="droid", choices=("const", "droid"))
    ap.add_argument("--pace", type=str, default="droid", choices=("plan", "droid"))
    ap.add_argument("--pace-scale", type=float, default=1.0)
    ap.add_argument("--ops", type=str, default="Pick,Place,GoToInitial",
                    help="comma-separated operations to blend; empty blends every operation")
    ap.add_argument("--seed", type=int, default=None,
                    help="reproduce one plan exactly; do NOT use when scoring a distribution -- it "
                         "makes every plan draw identically")
    ap.add_argument("--limit", type=int, default=None, help="score only the first N plans")
    ap.add_argument("--repeats", type=int, default=1,
                    help="re-blend each plan this many times; the draws are heavy-tailed, so a single "
                         "pass leaves the per-kind medians noisy")
    ap.add_argument("--profile-end-sec", type=float, default=1.0,
                    help="seconds the model's normalized-time end regions are compressed to; 0 disables")
    ap.add_argument("--boundary-window-sec", type=float, default=0.5)
    ap.add_argument("--acc-slack", type=float, default=1.0,
                    help="fraction of the robot acceleration limit the blend may use")
    ap.add_argument("--smoothing", type=float, default=1e-5)
    ap.add_argument("--no-retime-only", action="store_true", help="let the flow supply geometry too")
    ap.add_argument("--model-path", type=str, default=None)
    ap.add_argument("--droid-only", action="store_true", help="recompute the DROID reference tables")
    ap.add_argument("--shards", type=str, default="10,11,12")
    ap.add_argument("--shard-dir", type=str, default=str(DEFAULT_SHARDS))
    args = ap.parse_args()
    if not (args.plans or args.replay or args.droid_only):
        ap.error("pass --plans, --replay or --droid-only")
    main(args)

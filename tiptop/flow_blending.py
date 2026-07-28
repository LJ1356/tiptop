"""Flow-matching trajectory blending: SAMPLE a whole human stroke per operation, then constrain it.

This is the ``blend_mode: flow`` backend. Where ``blend_mode: spline`` applies a single analytic time law,
this SAMPLES a full stroke from a conditional flow-matching model (:class:`~tiptop.networks.
flow_timing.FlowModel`) for each operation -- so every generated episode draws a different human-like
realization (decelerate-into-grasp vs constant-velocity ...), reproducing the DISTRIBUTION of teleoperator
styles instead of one averaged blend.

Three things define a human episode's commanded speed signal, and the flow model only supplies one of
them. It generates a stroke in NORMALIZED time, so its output is a unit-mean speed SHAPE; the stroke's
absolute PACE and the joint speed AT each gripper event are separate scalars. Both used to come from
somewhere other than DROID -- pace from the cuTAMP plan's own wall-clock (i.e. from
``time_dilation_factor``) and the event speed from the ``blend_boundary_speed`` constant -- which is why
the emitted transitions did not look human however good the model was. They now come from
:mod:`tiptop.networks.droid_timing_stats`:

  * **pace** -- ``blend_pace: droid`` draws each stroke's mean joint speed from DROID conditioned on its
    joint arc length, its path CURVATURE and its two event kinds, so the absolute velocity is a human
    statistic and varies per stroke instead of being a fixed function of the planner's clock. Curvature
    dominates: humans slow down on winding paths, and it is also what keeps the draw reachable, since
    acceleration goes as curvature x speed^2.
  * **event speed** -- one draw per gripper event from DROID conditioned on the kind, used as the trail
    of the incoming stroke AND the lead of the outgoing one. Sharing the draw makes the commanded speed
    continuous across the event by construction; conditioning on the kind restores DROID's ~1.6x
    grasp-vs-release asymmetry, which a single constant flattens to 1.0.
  * **shape** -- the flow sample, warped multiplicatively in log space to meet those two anchors
    (:func:`_warp_profile`) so the model's approach shape and texture survive the constraint.

Pipeline per plan:
  1. Group the trajectory segments between gripper events into operation strokes, and note which will be
     blended (``blend_ops``) -- an unblended neighbour rests at zero, so the shared anchor must be zero
     too or the two sides would not meet.
  2. Draw a pace per stroke, then one boundary anchor per interior gripper event.
  3. Sample the flow model per stroke and run it through :func:`_retime_stroke`, which enforces the FR3
     vel/accel caps, pins rest ends to zero, warps to the anchors, and resamples to the plan ``dt``.

The flow sample provides the geometry only if you ask for it: ``blend_flow_retime_only`` (default True)
keeps the cuTAMP collision-checked path -- which is also what preserves the VAE-manifold texture the
planner's cost put there -- and uses ONLY the flow's timing.

Robustness: a per-stroke failure falls back to the analytic ``blend_group`` for that stroke, a missing
stats file falls back to the constant boundary and the plan clock, and a model-load failure falls back to
spline mode for the whole plan (handled in ``planning.run_planning``).
Enable per config::

    blend_trajectory: true
    blend_mode: flow
    blend_pace: droid
    blend_boundary_mode: droid
"""

from __future__ import annotations

import logging

import numpy as np

from tiptop.networks.droid_timing_stats import path_curvature
from tiptop.networks.flow_timing import FlowModel
from tiptop.trajectory_blending import (
    _DEFAULT_BOUNDARY_WINDOW_SEC,
    _DEFAULT_SMOOTHING,
    _MAX_BOUNDARY_RATIO,
    _MIN_CHORD,
    BlendConfig,
    _dedup_path,
    _eval_geometry,
    _finish_stroke,
    _fit_geometry,
    _op_name,
    _resolve_caps,
    _smootherstep,
    blend_group,
)

_log = logging.getLogger(__name__)

# Dense grid the sampled profile is evaluated + integrated on (matches _asymmetric_stroke's 2048).
_PROFILE_SAMPLES = 2048
# Width (seconds) of the window the boundary correction acts over at each end. The old window was a
# FRACTION of the duration (0.15, i.e. 1.3 s on an 8.7 s stroke) -- 2.5x DROID's entire +-0.53 s notch,
# so it repainted the whole approach instead of adjusting an endpoint. Clamped in _warp_profile.
_END_WINDOW_SEC = _DEFAULT_BOUNDARY_WINDOW_SEC
# Fixed-point iterations for the warp (see _warp_profile). Converges to <1% in 2; 4 is free insurance.
_WARP_ITERS = 4


def _end_remap(tau: np.ndarray, w: float, phi: float) -> np.ndarray:
    """Monotone [0,1]->[0,1] remap sending ``[0, w] -> [0, phi]`` and ``[1-w, 1] -> [1-phi, 1]``.

    The flow model generates in NORMALIZED time, so everything it produces stretches with the stroke's
    duration. Human deceleration into a gripper event does NOT: measured on DROID, the speed dip around
    a close is 1.47 s wide at half depth on 3-6 s strokes and 1.67 s on >6 s ones -- the same feature in
    SECONDS, not the same fraction. Read straight off, the model's profile is already at 0.54 of cruise
    with a quarter of the stroke left, which on a 4 s DROID stroke is a 1.0 s approach (right) but on a
    6.5 s planner stroke is a 1.6 s one (a dip roughly twice as wide as any human's).

    So evaluate the model on a remapped axis: whatever it does in its last ``phi`` of normalized time --
    ``phi = end_sec / (DROID's median stroke duration)``, the fraction that WAS ``end_sec`` seconds in
    training -- is compressed into the last ``w = end_sec / duration`` of this stroke, and its cruise is
    stretched over the middle. On a stroke of the reference duration this is the identity. The shape is
    reparameterised, never resampled away, so the model's style and texture are untouched.
    """
    w = float(np.clip(w, 0.02, 0.4))
    phi = float(np.clip(phi, 0.02, 0.4))
    return np.interp(tau, [0.0, w, 1.0 - w, 1.0], [0.0, phi, 1.0 - phi, 1.0])


def _warp_profile(p: np.ndarray, tau: np.ndarray, r_lead: float, r_trail: float,
                  window: float) -> np.ndarray:
    """Scale ``p`` to hit the end levels ``r_lead`` / ``r_trail`` while keeping its unit mean.

    The correction is MULTIPLICATIVE in log space over a smootherstep window at each end::

        p' = p * exp(h_lead * S_lead(tau) + h_trail * S_trail(tau)) / Z

    which (a) cannot drive the profile negative, (b) is smooth at the ends and where the window closes,
    so it introduces no acceleration step, and (c) RESCALES the learned shape instead of adding to it,
    so the model's texture and the relative shape of its approach survive and a stroke already near
    target is barely touched. The previous additive lift did none of these, and being one-sided (fire
    only when the model was too slow) it could bias the boundary in one direction only.

    Unit mean and the end levels interact -- rescaling to mean 1 moves the ends again -- so
    ``(h_lead, h_trail)`` and the normaliser ``Z`` are solved together by fixed point. The constraint
    held exact is the MEAN, since that sets the stroke's pace; the ends land within ~1%.

    A zero end level means rest, which log space cannot express; the caller ramps those to zero.
    """
    w = float(np.clip(window, 0.03, 0.35))
    s_lead = np.where(tau < w, 1.0 - _smootherstep(tau / w), 0.0)
    s_trail = np.where(tau > 1.0 - w, 1.0 - _smootherstep((1.0 - tau) / w), 0.0)
    z, out = 1.0, p
    for _ in range(_WARP_ITERS):
        h_lead = np.log(r_lead * z / p[0]) if r_lead > 0.0 else 0.0
        h_trail = np.log(r_trail * z / p[-1]) if r_trail > 0.0 else 0.0
        out = p * np.exp(h_lead * s_lead + h_trail * s_trail)
        z = float(np.trapezoid(out, tau)) or 1.0
        out = out / z
    return out


def _arclength_map(geom: list, length: float, n: int = 4096) -> tuple[np.ndarray, np.ndarray]:
    """(spline parameter grid, cumulative TRUE arc length along the fitted spline).

    ``_fit_geometry`` is parameterised by the POLYLINE arc length of the waypoints, so ``|dq/du|`` is
    only approximately 1: wherever the smoothing spline rounds a corner its path is shorter than the
    polyline, and the tangent collapses. That matters most exactly where we care -- at a grasp the
    planner's path reverses (down onto the object, then up), and rounding that cusp drops ``|dq/du|`` to
    ~0.75 over the last tenth of the stroke. Timing built on ``u`` then produces a joint speed ~25%
    below what the speed profile asked for, spread over the whole approach, which widens the dip around
    the gripper event well beyond the human one.

    Re-parameterising by the spline's own arc length makes ``|dq/ds| == 1`` everywhere, so the speed
    profile alone determines joint speed and no endpoint-tangent compensation is needed.
    """
    u = np.linspace(0.0, length, n)
    spd = np.linalg.norm(_eval_geometry(geom, u, 1), axis=1)
    s = np.concatenate([[0.0], np.cumsum((spd[1:] + spd[:-1]) / 2.0 * np.diff(u))])
    return u, s


def _retime_stroke(
    geom: list,
    length: float,
    dt: float,
    target_duration: float,
    vel_cap: np.ndarray,
    acc_cap: np.ndarray,
    lead_speed: float,
    trail_speed: float,
    profile_fn,
    mean_speed: float | None = None,
    window_sec: float = _END_WINDOW_SEC,
    end_sec: float = 0.0,
    ref_duration: float = 0.0,
    max_duration_mult: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, dict]:
    """Re-time a stroke using a SAMPLED speed profile, then enforce the boundary / limit constraints.

    Builds a strictly-positive speed-vs-time profile ``p(tau)`` from the flow sample (``profile_fn``),
    warps it to the requested end levels (:func:`_warp_profile`), integrates to the time law, samples,
    and stretches the duration to fit the vel/accel caps.

    ``max_duration_mult`` bounds the stroke at that multiple of ``target_duration``; without it a
    sampled pace has no upper bound on duration at all, since ``target_duration`` is otherwise unused
    once ``mean_speed`` is given.

    ``mean_speed`` (rad/s) sets the stroke's PACE. ``None`` keeps the historical behaviour -- inherit
    the cuTAMP group's own wall-clock through ``target_duration`` -- which makes the emitted absolute
    velocity a function of ``time_dilation_factor`` rather than of the human data, and deterministic
    where DROID's is spread over ~5x. Passing a value (see ``networks.droid_timing_stats``) sets the
    duration from it instead, so the model supplies the shape and DROID supplies the clock.

    Returns ``(pos, vel, acc, duration, info)``. ``info`` reports how hard the warp had to push and
    whether the vel/accel caps bound -- that stretch silently slows the stroke back down, undoing both
    the requested pace and the requested boundary, so the caller logs it.
    """
    # Re-parameterise by the spline's TRUE arc length so |dq/ds| == 1 and the speed profile alone sets
    # the joint speed (see _arclength_map).
    u_grid, s_grid = _arclength_map(geom, length)
    s_total = float(s_grid[-1]) or float(length)
    v_ref = float(mean_speed) if mean_speed else s_total / target_duration

    # The end remap and the warp window are real timescales, so both need the duration this profile
    # implies, before the caps stretch it.
    nominal = s_total / v_ref

    tau_d = np.linspace(0.0, 1.0, _PROFILE_SAMPLES)
    # Compress the model's normalized-time end regions to a human PHYSICAL width (see _end_remap).
    tau_p = tau_d
    if end_sec > 0.0 and ref_duration > 0.0:
        tau_p = _end_remap(tau_d, end_sec / max(nominal, 1e-9), end_sec / ref_duration)
    p_nn = np.asarray(profile_fn(tau_p), dtype=np.float64)
    p_nn = np.clip(p_nn, 1e-3, None)
    p_nn = p_nn / (np.trapezoid(p_nn, tau_d) or 1.0)  # unit-mean over tau: pace reference = 1

    # End level relative to the pace reference. With the arc-length re-parameterisation the tangent is
    # exactly 1, so this is just the requested speed over the pace -- no endpoint compensation needed.
    r_lead = 0.0 if lead_speed == 0.0 else min(_MAX_BOUNDARY_RATIO, lead_speed / v_ref)
    r_trail = 0.0 if trail_speed == 0.0 else min(_MAX_BOUNDARY_RATIO, trail_speed / v_ref)

    w = float(np.clip(window_sec / max(nominal, 1e-9), 0.03, 0.35))
    p = _warp_profile(p_nn, tau_d, r_lead, r_trail, w)

    # A rest end cannot be expressed multiplicatively in log space; ramp it to exactly zero.
    if lead_speed == 0.0:
        p = p * (1.0 - np.where(tau_d < w, 1.0 - _smootherstep(tau_d / w), 0.0))
    if trail_speed == 0.0:
        p = p * (1.0 - np.where(tau_d > 1.0 - w, 1.0 - _smootherstep((1.0 - tau_d) / w), 0.0))
    p = np.maximum(p, 0.0)

    pbar = float(np.trapezoid(p, tau_d)) or 1.0
    # h(tau) in [0,1]: normalized arc length vs normalized time (cumulative p, scaled so h(1)=1).
    h = np.concatenate([[0.0], np.cumsum((p[1:] + p[:-1]) / 2.0 * np.diff(tau_d))]) / pbar

    def sample(duration: float, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        ts = np.linspace(0.0, 1.0, n)
        st = s_total * np.interp(ts, tau_d, h)          # true arc length travelled by time ts
        ut = np.interp(st, s_grid, u_grid)               # -> the spline's own parameter
        ps = np.interp(ts, tau_d, p)
        pos = _eval_geometry(geom, ut, 0)
        d1 = _eval_geometry(geom, ut, 1)
        # dq/ds is the UNIT tangent, so joint speed is exactly ds/dt = s_total * p / (pbar * duration).
        tan = d1 / np.clip(np.linalg.norm(d1, axis=1, keepdims=True), 1e-12, None)
        ds_dt = s_total * (ps / pbar) / duration
        vel = tan * ds_dt[:, None]
        acc = np.gradient(vel, duration / (n - 1), axis=0)
        return pos, vel, acc

    # Duration so the mean joint speed equals v_ref through the (unit-mean) profile, then stretch to fit
    # the caps. That stretch scales the whole profile down, so it undoes the requested pace AND the
    # requested boundary together -- report it instead of letting it pass silently.
    duration = s_total / (pbar * v_ref)
    # Rail on the sampled pace. With mean_speed set, target_duration otherwise constrains nothing, so a
    # bottom-tail pace draw on a long path can produce a stroke several times longer than the planner
    # intended -- which on a real robot is a very slow operation nobody asked for. Clamp BEFORE the cap
    # check so the vel/accel stretch below still gets the last word.
    capped_slow = False
    if max_duration_mult > 0.0 and target_duration > 0.0:
        ceiling = max_duration_mult * target_duration
        if duration > ceiling:
            duration, capped_slow = ceiling, True
    _, v0, a0 = sample(duration, 512)
    alpha_v = float((np.abs(v0).max(axis=0) / vel_cap).max())
    alpha_a = float(np.sqrt((np.abs(a0).max(axis=0) / acc_cap).max()))
    stretch = max(1.0, alpha_v, alpha_a)
    duration *= stretch
    n = max(2, int(round(duration / dt)) + 1)
    pos, vel, acc = sample(duration, n)
    info = {"cap_stretch": stretch, "pace_ref": v_ref, "window_frac": w, "slow_capped": capped_slow,
            "duration": duration,
            "warp_lead": float(p[0] / p_nn[0]) if r_lead > 0 else 1.0,
            "warp_trail": float(p[-1] / p_nn[-1]) if r_trail > 0 else 1.0}
    return pos, vel, acc, duration, info


def _flow_speed_profile(x_flow: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Unit-mean speed-vs-normalized-time profile of a sampled stroke -> (tau_mid, p)."""
    s = np.linalg.norm(np.diff(x_flow, axis=0), axis=1)  # arc-length speed over the T-1 intervals
    tau_mid = (np.arange(len(s)) + 0.5) / len(s)
    p = np.clip(s / (s.mean() or 1.0), 1e-3, None)
    return tau_mid, p


def flow_blend_group(
    positions: np.ndarray,
    dt: float,
    target_duration: float,
    vel_cap: np.ndarray,
    acc_cap: np.ndarray,
    flow_model: FlowModel,
    n_steps: int = 60,
    smoothing: float = _DEFAULT_SMOOTHING,
    lead_speed: float = 0.0,
    trail_speed: float = 0.0,
    retime_only: bool = True,
    kinds: tuple[str, str] = ("none", "none"),
    mean_speed: float | None = None,
    window_sec: float = _END_WINDOW_SEC,
    end_sec: float = 0.0,
    ref_duration: float = 0.0,
    max_duration_mult: float = 0.0,
    geometry: dict | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, dict]:
    """Sample the flow model conditioned on the group's path, then re-time it under the hard constraints.

    ``positions`` are the group's joined cuTAMP waypoints; ``lead_speed`` / ``trail_speed`` the boundary
    joint speeds to arrive at and leave with (0 = rest). ``kinds`` is the (start, end) gripper event this
    stroke runs between -- "close" / "open" / "none" -- which is what tells the model to decelerate into a
    grasp harder than into a release (DROID does, by ~1.6x). ``mean_speed`` (rad/s) sets the pace; None
    inherits the cuTAMP group's wall-clock via ``target_duration``. With ``retime_only`` the geometry
    stays the collision-checked cuTAMP path and only the flow's sampled TIMING is applied.

    NOTE the defaults here are the UNCONFIGURED path, not the shipped one: ``end_sec``/``ref_duration``
    default to 0, which disables the physical end remap, and ``geometry`` to None, which refits. The
    shipped path goes through :func:`flow_blend_cutamp_plan`, which passes all of them from the config.
    A standalone caller that wants shipped behaviour has to supply them.

    Returns (pos, vel, acc, dt_out, info). Raises on failure so the caller can fall back to ``blend_group``.
    """
    positions = _dedup_path(np.asarray(positions, dtype=np.float64))
    if len(positions) < 2:
        pos = np.repeat(positions[:1], 2, axis=0)
        zeros = np.zeros_like(pos)
        return pos, zeros, zeros, dt, {}

    # Sample a full human-like stroke conditioned on this path (endpoint-anchored to positions[-1]).
    samp = flow_model.sample(positions, n_samples=1, n_steps=n_steps, kinds=kinds)
    if samp is None:
        raise ValueError("flow model returned no sample (degenerate path)")
    x_flow = samp[0] + positions[0]  # start-centered sample -> absolute joint positions [T, dof]

    # Geometry source: the collision-checked cuTAMP path by default (retime_only), taking ONLY the flow's
    # timing; otherwise the flow sample's own learned path. `positions` is already de-duplicated above;
    # `x_flow` supplies the speed profile in both cases.
    if retime_only and geometry is not None:
        geom, length = geometry["geom"], geometry["length"]   # fitted once in the caller's first pass
    else:
        geom_pos = positions if retime_only else _dedup_path(x_flow)
        if len(geom_pos) < 2:
            raise ValueError("degenerate flow geometry")
        u = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(geom_pos, axis=0), axis=1))])
        length = float(u[-1])
        if length < _MIN_CHORD:
            pos = np.repeat(geom_pos[:1], 2, axis=0)
            zeros = np.zeros_like(pos)
            return pos, zeros, zeros, dt, {}
        geom = _fit_geometry(u, geom_pos, smoothing)

    tau_mid, p = _flow_speed_profile(x_flow)

    def profile_fn(tau: np.ndarray) -> np.ndarray:
        return np.interp(np.asarray(tau, dtype=np.float64), tau_mid, p)

    out_pos, out_vel, out_acc, duration, info = _retime_stroke(
        geom, length, dt, target_duration, vel_cap, acc_cap, lead_speed, trail_speed,
        profile_fn, mean_speed, window_sec, end_sec, ref_duration, max_duration_mult,
    )
    pos, vel, acc, dt_out = _finish_stroke(out_pos, out_vel, out_acc, duration, lead_speed, trail_speed)
    return pos, vel, acc, dt_out, info


def _neighbor_gripper_action(plan: list[dict], idx: int) -> str:
    """The gripper action at ``idx`` ("close"/"open"), or "none" when that neighbour is not a gripper step.

    A stroke's boundaries are gripper events, so the adjacent gripper steps name what the generated stroke
    leaves and arrives at -- the conditioning ``flow_timing.TrajFlow`` was trained on.
    """
    if 0 <= idx < len(plan) and plan[idx].get("type") == "gripper":
        action = plan[idx].get("action")
        if action in ("close", "open"):
            return action
    return "none"


def _plan_runs(cutamp_plan: list[dict], ops: tuple[str, ...] | None) -> list[dict]:
    """Split the plan into operation strokes: maximal runs of consecutive ``trajectory`` steps.

    Each run records the gripper-event kinds at its two ends and whether ``blend_ops`` will actually
    blend it -- an unblended run keeps its raw cuRobo segments, which decelerate to rest at every
    waypoint, so its neighbours must meet it at zero rather than at a nonzero anchor.
    """
    n = len(cutamp_plan)
    runs: list[dict] = []
    steps: list[dict] = []
    start = 0

    def close(end: int):
        if not steps:
            return
        label = steps[0]["label"]
        runs.append({
            "steps": list(steps), "start": start, "end": end,
            "k_lead": _neighbor_gripper_action(cutamp_plan, start - 1),
            "k_trail": _neighbor_gripper_action(cutamp_plan, end + 1),
            "blended": ops is None or _op_name(label) in ops,
        })
        steps.clear()

    for idx, step in enumerate(cutamp_plan):
        if step.get("type") == "trajectory":
            if not steps:
                start = idx
            steps.append(step)
        else:
            close(idx - 1)
    close(n - 1)
    return runs


def _run_geometry(run: dict, smoothing: float) -> dict | None:
    """De-duplicate, fit, and measure the path this stroke will actually follow.

    Built once per run, before any timing is drawn, because the PACE depends on it: humans obey a
    speed-curvature law, so how wiggly the path is decides how fast the stroke should go. The curvature
    must be measured on the SMOOTHED path, not on the raw waypoints -- the smoothed one is what the arm
    traverses and what sets the acceleration, and the planner's raw path is about twice as wiggly as
    the spline that replaces it (K ~ 23 vs ~ 10 at the default smoothing, against DROID's ~ 11). Feeding
    the raw number to the pace law would halve every stroke's speed for wiggle that is never executed.
    """
    joined, vel, dt, wall_clock = _joined_path(run)
    pts = _dedup_path(joined)
    if len(pts) < 2:
        return None
    u = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
    length = float(u[-1])
    if length < _MIN_CHORD:
        return None
    geom = _fit_geometry(u, pts, smoothing)
    # Sample the fitted spline uniformly in its parameter, then measure curvature on the arc grid.
    dense = _eval_geometry(geom, np.linspace(0.0, length, max(64, min(4096, int(length / 0.002)))), 0)
    return {"positions": pts, "geom": geom, "length": length, "vel": vel, "dt": dt,
            "wall_clock": wall_clock, "curvature": path_curvature(dense)}


def _joined_path(run: dict) -> tuple[np.ndarray, np.ndarray, float, float]:
    """(joined positions, joined velocities, dt, original wall-clock) for one run."""
    seg = [s["plan"].position.detach().cpu().numpy().astype(np.float64) for s in run["steps"]]
    joined = np.concatenate([seg[0]] + [p[1:] for p in seg[1:]], axis=0)
    vel = np.concatenate(
        [s["plan"].velocity.detach().cpu().numpy().astype(np.float64) for s in run["steps"]], axis=0
    )
    dt = float(run["steps"][0]["dt"])
    return joined, vel, dt, sum((len(p) - 1) for p in seg) * dt


def _assign_timing(runs: list[dict], config: BlendConfig, n_steps: int, stats, rng) -> None:
    """Fill each run's ``pace`` and its ``lead`` / ``trail`` boundary anchors, in place.

    Pace is drawn per stroke from its arc length AND its path curvature; the boundary is drawn ONCE per
    gripper event and written to both sides of
    it, which is what makes the commanded speed continuous there. The event is conditioned on the
    geometric mean of the two strokes' paces, because DROID's event speed does depend on the local pace
    -- weakly (corr ~0.44), but drawing the two independently over-disperses the boundary/pace ratio.
    """
    use_droid_pace = stats is not None and config.pace_mode == "droid"
    use_droid_bnd = stats is not None and config.boundary_mode == "droid"

    for run in runs:
        run["pace"] = None
        if not run["blended"]:
            continue
        if use_droid_pace:
            run["pace"] = config.pace_scale * stats.sample_pace(
                run["arc"], run["curvature"], (run["k_lead"], run["k_trail"]), rng
            )

    def draw(kind: str, *paces) -> float:
        """Boundary speed for one gripper event, floored by ``blend_boundary_speed``."""
        if kind not in ("close", "open"):
            return config.boundary_speed
        if not use_droid_bnd:
            return config.boundary_speed
        known = [p for p in paces if p]
        ref = float(np.exp(np.mean(np.log(known)))) if known else 0.4
        return max(config.boundary_speed, stats.sample_event_speed(kind, ref, rng))

    n_plan_steps = runs[-1]["end"] + 1 if runs else 0
    for i, run in enumerate(runs):
        # Rest only where the arm is genuinely stationary: the plan's very first and last samples.
        run["lead"] = 0.0 if run["start"] == 0 else draw(run["k_lead"], run["pace"])
        run["trail"] = 0.0 if run["end"] == n_plan_steps - 1 else draw(run["k_trail"], run["pace"])

    # One shared draw per interior event, overwriting the per-run defaults above.
    for a, b in zip(runs[:-1], runs[1:]):
        if not (a["blended"] and b["blended"]):
            # One side keeps raw cuRobo segments and rests at its waypoints; meet it at zero.
            a["trail"] = b["lead"] = 0.0
            continue
        v = draw(a["k_trail"], a["pace"], b["pace"])
        a["trail"] = b["lead"] = v


def _blend_trajectory_steps_flow(
    run: dict,
    config: BlendConfig,
    flow_model: FlowModel,
    vel_limit: np.ndarray | None,
    acc_limit: np.ndarray | None,
    ref_duration: float = 0.0,
) -> tuple[dict, dict]:
    """Blend one run of consecutive ``trajectory`` steps into a single flow-sampled, re-timed step."""
    from curobo.types.state import JointState  # lazy: keep the module importable without cuRobo
    import torch

    template = run["steps"][0]["plan"]
    device = template.position.device
    joint_names = template.joint_names
    geo = run["geometry"]
    joined, orig_velocities, dt, wall_clock = geo["positions"], geo["vel"], geo["dt"], geo["wall_clock"]
    target_duration = wall_clock / config.speed_scale

    vel_cap, acc_cap = _resolve_caps(
        orig_velocities, dt, vel_limit, acc_limit, config.vel_slack, config.acc_slack
    )
    try:
        pos, vel, acc, dt_out, info = flow_blend_group(
            joined, dt, target_duration, vel_cap, acc_cap, flow_model, config.flow_steps,
            config.smoothing, run["lead"], run["trail"], config.flow_retime_only,
            (run["k_lead"], run["k_trail"]), run["pace"], config.boundary_window_sec,
            config.profile_end_sec, ref_duration, config.max_duration_mult, geo,
        )
    except Exception:
        _log.exception("Flow sampling failed for a stroke; falling back to the analytic spline time law")
        pos, vel, acc, dt_out = blend_group(
            joined, dt, target_duration, vel_cap, acc_cap, config.smoothing, run["lead"], run["trail"],
            config.boundary_window,
        )
        info = {"fallback": True}

    plan = JointState(
        position=torch.as_tensor(pos, dtype=torch.float32, device=device),
        velocity=torch.as_tensor(vel, dtype=torch.float32, device=device),
        acceleration=torch.as_tensor(acc, dtype=torch.float32, device=device),
        jerk=torch.zeros(pos.shape, dtype=torch.float32, device=device),
        joint_names=list(joint_names) if joint_names is not None else None,
    )
    return {"type": "trajectory", "plan": plan, "dt": dt_out, "label": run["steps"][0]["label"]}, info


def _load_stats(config: BlendConfig):
    """The DROID pace / event-speed conditionals, or None (caller falls back to the constant behaviour)."""
    if config.pace_mode != "droid" and config.boundary_mode != "droid":
        return None
    try:
        from tiptop.networks.droid_timing_stats import TimingStats

        return TimingStats(config.stats_path)
    except Exception:
        _log.exception(
            "DROID timing stats unavailable (fit them with `python -m tiptop.networks."
            "droid_timing_stats --fit`); falling back to the plan clock and the constant boundary speed"
        )
        return None


def flow_blend_cutamp_plan(
    cutamp_plan: list[dict],
    config: BlendConfig,
    flow_model: FlowModel,
    vel_limit: np.ndarray | None = None,
    acc_limit: np.ndarray | None = None,
) -> list[dict]:
    """Return ``cutamp_plan`` with each operation's trajectory segments replaced by a FLOW-SAMPLED stroke.

    Gripper steps pass through and delimit the strokes; ``config.ops`` restricts which operations are
    blended. Timing is assigned across the WHOLE plan first (see :func:`_assign_timing`) so that the two
    sides of every gripper event share one boundary speed -- that is what makes the commanded velocity
    continuous there, and it cannot be done one stroke at a time. A no-op when ``config.enabled`` is False.
    """
    if not config.enabled:
        return cutamp_plan

    runs = _plan_runs(cutamp_plan, config.ops)
    if not runs:
        return cutamp_plan

    rng = np.random.default_rng(config.seed)
    if config.seed is not None:
        import torch

        torch.manual_seed(config.seed)
    stats = _load_stats(config)
    # The duration the flow model's normalized-time profile is implicitly calibrated to (DROID's median
    # stroke). Without it the end remap is skipped and the approach stretches with the stroke.
    ref_duration = getattr(stats, "dur_median", 0.0) if stats else 0.0
    for run in runs:
        # Geometry first: the pace law needs this path's curvature, and re-timing needs the same fit.
        g = _run_geometry(run, config.smoothing)
        run["geometry"] = g
        run["arc"] = g["length"] if g else 0.0
        run["curvature"] = g["curvature"] if g else 0.0
    _assign_timing(runs, config, len(cutamp_plan), stats, rng)

    blended_out: dict[int, dict] = {}
    stats_ct = {"blended": 0, "skipped": 0, "fallback": 0}
    infos: list[dict] = []
    for run in runs:
        if not run["blended"]:
            stats_ct["skipped"] += 1
            continue
        try:
            step, info = _blend_trajectory_steps_flow(run, config, flow_model, vel_limit, acc_limit,
                                                      ref_duration)
            blended_out[run["start"]] = step
            infos.append(info)
            stats_ct["blended"] += 1
        except Exception:
            _log.exception("Flow blending failed for a segment run; keeping original segments")
            stats_ct["fallback"] += 1

    out: list[dict] = []
    replaced = {i for run in runs if run["start"] in blended_out
                for i in range(run["start"], run["end"] + 1)}
    for idx, step in enumerate(cutamp_plan):
        if idx in blended_out:
            out.append(blended_out[idx])
        elif idx not in replaced:
            out.append(step)

    scope = "all operations" if config.ops is None else f"operations {list(config.ops)}"
    pace_src = "DROID" if (stats and config.pace_mode == "droid") else "the plan clock"
    bnd_src = "DROID per event" if (stats and config.boundary_mode == "droid") else \
        f"the constant {config.boundary_speed}"
    _log.info(
        "Flow trajectory blending (%s): %d operation groups sampled + re-timed, %d left unblended, %d "
        "fell back to originals (%d trajectory segments -> %d steps). Pace from %s; boundary from %s.",
        scope, stats_ct["blended"], stats_ct["skipped"], stats_ct["fallback"],
        sum(1 for s in cutamp_plan if s.get("type") == "trajectory"),
        sum(1 for s in out if s.get("type") == "trajectory"), pace_src, bnd_src,
    )
    # The vel/accel caps stretch a stroke's duration, which scales the whole profile down and so undoes
    # BOTH the sampled pace and the sampled boundary. A high bind rate means the emitted velocities are
    # the robot's limits rather than DROID's distribution -- work the geometry (blend_smoothing, the
    # grasp-reversal cusp), not the pace knob.
    binds = [i.get("cap_stretch", 1.0) for i in infos if i.get("cap_stretch")]
    if binds:
        bound = [b for b in binds if b > 1.01]
        (_log.warning if len(bound) > 0.25 * len(binds) else _log.info)(
            "Vel/accel caps bound on %d/%d strokes (max stretch %.2fx); those strokes run slower than "
            "the sampled pace.", len(bound), len(binds), max(binds),
        )
    return out

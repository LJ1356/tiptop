"""Blend + re-time a cuTAMP plan so consecutive arm motions read like one continuous stroke.

A cuTAMP plan is a sequence of independently-planned trajectory segments (retract -> approach ->
grasp, etc.) separated by gripper open/close steps. Every segment is a cuRobo *reaching* motion, so
it accelerates from rest and decelerates back to rest at each of its endpoints. Concatenated, the
arm therefore comes to a full stop at every interior waypoint -- a stop-and-go, per-segment velocity
profile with ~20 zero-velocity dips, quite unlike a human teleop demo, which flows through the
reach and only slows near the grasp/place itself.

This module collapses that: it groups the consecutive trajectory segments *between gripper events*
and replaces each group with a single smooth stroke that comes to rest only at its ends -- which are
exactly the gripper-actuation points, where the arm genuinely should be still for a reliable
grasp/place. Each stroke is built by path-velocity decomposition:

* Geometry. The group's joined waypoints are fit with a penalized cubic SMOOTHING spline in joint
  space, parameterized by arc length. The penalty rounds the sharp direction changes at the old
  segment joins (which an interpolating spline would overshoot, producing large phantom
  accelerations) while staying close to the collision-checked path (~1 deg joint deviation at the
  default smoothing).

* Timing. A quintic time law sweeps along the path with a smooth (min-jerk-like) bell-shaped speed
  profile. At the plan's true rest points (episode start/end) it comes fully to rest (this is the
  6t^5-15t^4+10t^3 minimum-jerk quintic, the standard model of human reaching). At gripper-adjacent
  boundaries it instead ends/starts at a small NONZERO boundary speed: the min-jerk quintic ramps so
  gently through zero that it leaves a long run of near-zero-velocity frames on both sides of the
  gripper event, and the DROID/openpi non-idle training filter drops idle joint-velocity runs (>=7
  frames at 15 Hz), which would take the gripper open/close timestep with them. Carrying a nonzero
  boundary speed keeps the only zero-velocity frames at a gripper event to the short (~3-frame)
  stationary hold the export inserts for the actuation itself -- comfortably under the filter's run
  length. The arm still slows into the grasp/place (the boundary speed is a small fraction of cruise)
  and its final commanded POSITION is exactly the grasp, so there is no overshoot.

  A symmetric bell makes the endpoint the stroke's SLOWEST point, so the boundary speed it can carry is
  capped at ``_MAX_BOUNDARY_SLOPE`` * the stroke average (monotonicity): on a slow (heavily time-dilated)
  operation that clamps the transition back into the idle band, and an endpoint-tangent collapse at a
  grasp reversal clamps it further. Setting ``blend_boundary_window`` > 0 switches to an asymmetric time
  law (:func:`_asymmetric_stroke`) that holds the plan pace through the middle and carries the boundary
  speed only within that window at each end -- a strictly-positive (monotonic) speed profile with no
  such cap, so the transition clears the idle band while the careful mid-approach stays slow.

* Pace / limits. The stroke's duration starts at the group's original wall-clock (so a
  ``time_dilation_factor``-slowed plan stays slow, just continuous) and is then shortened PER STROKE
  only as much as its own gripper-adjacent boundary speed requires -- never globally. This matters
  because the boundary joint speed is capped at ``_MAX_BOUNDARY_SLOPE`` times the stroke AVERAGE
  (the min-jerk endpoints are its slowest point), so at a slow pace ``blend_boundary_speed`` is
  silently clamped and the arm crawls through gripper open/close -- which a velocity-command policy
  learns to stall on. Shortening only the strokes that would clamp lifts their transition speed to the
  request while leaving strokes already fast enough (and the cruise of the rest of the plan) at their
  original careful pace. The shortening is bounded by the robot's REAL joint velocity/acceleration
  limits (from ``motion_gen``); a transition unreachable within those limits (e.g. a rounded grasp
  cusp) is left at the original pace and logged, not chased by over-speeding the whole stroke.
  ``blend_speed_scale`` (default 1.0) is an optional EXTRA global multiplier on top, rarely needed now.

Calibration: on this repo's teleop demos the human joint-jerk RMS is ~3-6; the default smoothing is
tuned so the blended strokes land in that same band (i.e. as smooth as the human data), not smoother
and blurrier, nor sharper and more robotic.

Collision caveat: smoothing rounds the sharp corners at the old segment joins, so the blended path is
NOT identical to the one cuRobo collision-checked (largest deviation at those corners, ~1 deg joint at
the default). This is the same rounding a human demo has, but validate in sim / on the viz before
trusting it near tight obstacles.

Blending is OFF by default and opt-in per config: set ``blend_trajectory: true`` under
``tamp_overrides`` in a ``cfg/tamp/*.yml`` (see :func:`resolve_blend_config`). This is a pure
post-process over the joint-waypoint arrays (no planner changes); it runs in ``run_planning`` so the
saved plan and the executed plan are the identical blended object.
"""

import logging
from dataclasses import dataclass

import numpy as np
import torch
from scipy.interpolate import CubicSpline, make_smoothing_spline

# NOTE: ``from curobo.types.state import JointState`` is imported lazily inside
# ``_blend_trajectory_steps`` (its only user) rather than at module top, so this module's pure
# geometry/timing helpers (``_fit_geometry``, ``_finish_stroke``, ``blend_group`` ...) can be
# imported -- and unit-tested -- without a cuRobo install. ``flow_blending`` reuses them.

_log = logging.getLogger(__name__)

# Defaults for the tunable knobs (overridable per config -- see resolve_blend_config).
_DEFAULT_SMOOTHING = 1e-4  # smoothing-spline penalty (lam) in the arc-length (radian) parameterization
_DEFAULT_VEL_SLACK = 1.0  # fraction of the robot's real velocity limit the blend may use
_DEFAULT_ACC_SLACK = 1.0  # fraction of the robot's real acceleration limit the blend may use
# Joint speed (rad/s, L2) each stroke carries at gripper-adjacent boundaries so those frames are not
# idle (see module docstring). ~0.15 clears the ~0.02 non-idle threshold with wide margin while
# staying a small fraction of cruise. Set to 0 to fall back to fully rest-to-rest (min-jerk) strokes.
_DEFAULT_BOUNDARY_SPEED = 0.15

# Optional EXTRA global speed-up applied to every blended stroke (target_duration is divided by this).
# Normally left at 1.0: blend_group already shortens each stroke just enough to realize its own
# gripper-adjacent boundary speed (see blend_group), so the transitions clear the idle band without
# over-speeding the cruise of the rest of the plan. Set >1.0 only to deliberately run the whole plan
# faster on top of that; the vel/accel caps still bound it to the robot's real limits.
_DEFAULT_SPEED_SCALE = 1.0

# The boundary slope is capped at this fraction of the stroke's average slope so the time law stays
# monotonic (a boundary faster than the mean would invert the bell / make the arm backtrack). This cap
# only applies to the default (symmetric min-jerk) time law; the asymmetric profile (blend_boundary_window
# > 0) is monotonic by construction and is not subject to it.
_MAX_BOUNDARY_SLOPE = 0.8

# Asymmetric time law (Option B): length of the fast/slow end window in seconds. 0 disables it (use the
# symmetric min-jerk bell). >0 carries blend_boundary_speed only within this window at each gripper end
# and holds the plan pace through the middle -- decoupling the transition speed from the careful cruise.
_DEFAULT_BOUNDARY_WINDOW = 0.0

# Ceiling on the asymmetric profile's end speed as a multiple of the cruise, so a near-vanishing endpoint
# tangent can't blow the profile up. The vel/accel caps bound the actual motion; this only keeps the
# normalized speed profile numerically sane.
_MAX_BOUNDARY_RATIO = 40.0

# Physical width (seconds) of the window the flow backend's boundary correction acts over at each end
# (flow_blending._warp_profile). DROID's speed dip around a gripper event spans about +-0.53 s, so a
# wider correction stops adjusting an endpoint and starts repainting the whole approach.
_DEFAULT_BOUNDARY_WINDOW_SEC = 0.5

# Waypoints closer than this (Euclidean, joint space, radians) to the previous kept one are dropped
# before fitting so the arc-length parameter is strictly increasing. Consecutive segments share an
# exact endpoint, and each segment's decel tail is a cluster of near-identical points -- this removes
# both.
_MIN_CHORD = 1e-4

# Below this many distinct waypoints the smoothing spline is not well-conditioned; fall back to a
# plain (natural) cubic through the points -- the min-jerk time law still makes it rest-to-rest.
_MIN_SMOOTH_PTS = 6

# Fallback caps when the robot's real limits are unavailable, as multiples of the plan's OWN peak
# per-joint velocity / acceleration. The acceleration multiple is generous because a slowed plan sits
# well below the true acceleration limit (see module docstring), so its own peak is a poor ceiling.
_FALLBACK_VEL_MULT = 1.5
_FALLBACK_ACC_MULT = 4.0


@dataclass
class BlendConfig:
    """Resolved trajectory-blending settings (from a config's ``tamp_overrides``)."""

    enabled: bool = False
    smoothing: float = _DEFAULT_SMOOTHING
    vel_slack: float = _DEFAULT_VEL_SLACK
    acc_slack: float = _DEFAULT_ACC_SLACK
    boundary_speed: float = _DEFAULT_BOUNDARY_SPEED
    speed_scale: float = _DEFAULT_SPEED_SCALE
    boundary_window: float = _DEFAULT_BOUNDARY_WINDOW
    # Operation names to restrict blending to (e.g. ("Pick", "Place")); None blends every operation.
    ops: tuple[str, ...] | None = None
    # Timing backend: "spline" (the analytic min-jerk / asymmetric time law here, the default) or "flow"
    # (a DROID-learned conditional flow-matching model SAMPLES a full human stroke per operation -- see
    # flow_blending; each sample is a different human-like realization, so generated data reproduces the
    # distribution of teleoperator styles). In both modes the GEOMETRY smoothing, vel/accel caps,
    # endpoint pinning and non-idle boundary speed are enforced by the same engine.
    # or "vae" (the VAE motion-manifold cost picks each stroke's clock directly -- see vae_retiming;
    # no learned timing model, and no DROID timing stats, just the same Mahalanobis distance the
    # cuRobo cost already minimizes, scored on whole strokes instead of individual cuRobo legs).
    mode: str = "spline"
    # Path to the learned flow checkpoint (flow mode). Relative paths resolve under the tiptop package
    # dir; when omitted the model uses its own default (flow_net_t64.pt).
    model_path: str | None = None
    # VAE-manifold checkpoint (vae mode). Taken from the SAME `vae_path` override the cuRobo manifold
    # cost uses, so the re-timer scores against the encoder and DROID cluster that shaped the geometry.
    vae_path: str | None = None
    # Flow mode only: number of Euler ODE steps when sampling the flow model (more = finer, slower).
    flow_steps: int = 60
    # Flow mode only: use ONLY the flow's sampled timing and keep the collision-checked cuTAMP geometry,
    # instead of also adopting the flow sample's own path. Default True: the flow sample's own path is a
    # 32-point resample that measurably STRAIGHTENS the planner's, deleting both the collision guarantee
    # and the configuration-space texture the VAE-manifold cost put there (arc-length curvature drops
    # from the planner's ~64 to ~6, against DROID's ~10 -- i.e. straighter than human, the wrong way).
    flow_retime_only: bool = True
    # Where each stroke's PACE (mean joint speed) comes from. "plan" (default) inherits the cuTAMP
    # group's own wall-clock, so absolute velocity follows time_dilation_factor and is deterministic.
    # "droid" draws it from the measured DROID conditional on arc length + event kinds (see
    # networks.droid_timing_stats), which is what makes the emitted speed a human statistic.
    pace_mode: str = "plan"
    # Global multiplier on the sampled pace -- the safety lever for "droid" pace, replacing the role
    # time_dilation_factor plays for "plan" pace. 1.0 = run at DROID's measured speed.
    pace_scale: float = 1.0
    # Where the joint speed AT a gripper event comes from. "const" uses blend_boundary_speed for every
    # event of both kinds. "droid" draws once per event from the measured conditional on the event KIND,
    # restoring DROID's ~1.6x grasp-vs-release asymmetry that one constant flattens to 1.0; the draw is
    # shared by the strokes on both sides, which is what makes the commanded speed continuous there.
    boundary_mode: str = "const"
    # Seconds at each end over which the boundary correction is applied (see flow_blending._warp_profile).
    boundary_window_sec: float = _DEFAULT_BOUNDARY_WINDOW_SEC
    # Seconds at each end over which the flow model's normalized-time profile is compressed to a human
    # PHYSICAL width (flow_blending._end_remap). DROID's dip around a gripper event is ~1.5 s wide
    # whatever the stroke's length, but the model generates in normalized time, so on a stroke longer
    # than DROID's median its approach stretches with it -- measured, the model is at 0.54 of cruise
    # with a quarter of the stroke still to go, a 1.6 s decel on a 6.5 s stroke. 0 disables the remap.
    profile_end_sec: float = 1.0
    # Upper bound on a stroke's duration, as a multiple of the cuTAMP group's own wall-clock. Only bites
    # under blend_pace: droid, where target_duration otherwise constrains nothing and a bottom-tail pace
    # draw on a long path can run several times longer than the planner intended. The vel/accel caps are
    # applied after it, so they still win.
    max_duration_mult: float = 2.0
    # Path to the fitted DROID timing stats (relative paths resolve under the tiptop package dir).
    stats_path: str | None = None
    # Seed for the pace / boundary / flow draws. None (default) means fresh entropy per plan, which is
    # the point in dataset generation -- every episode a different human-like realization. Set it to
    # reproduce one plan exactly.
    seed: int | None = None


def resolve_blend_config(overrides: dict | None) -> BlendConfig:
    """Read the blending knobs from a config's ``tamp_overrides`` dict (OFF unless opted in).

    Recognized keys (all optional):
        blend_trajectory: bool       -- master enable (default False)
        blend_smoothing:  float      -- smoothing-spline penalty; higher = smoother, less faithful
        blend_vel_slack:  float      -- fraction of the real velocity limit to use (default 1.0)
        blend_acc_slack:  float      -- fraction of the real acceleration limit to use (default 1.0)
        blend_boundary_speed: float  -- joint speed (rad/s) carried at gripper-adjacent boundaries so
                                        those frames are not idle-filtered (default 0.15; 0 = rest)
        blend_speed_scale: float     -- optional EXTRA global speed-up on top of the automatic per-stroke
                                        boundary speed-up (default 1.0 = none). Rarely needed now: each
                                        stroke is already shortened just enough to realize
                                        blend_boundary_speed; >1 additionally over-speeds every stroke.
        blend_boundary_window: float -- 0 (default) uses the symmetric min-jerk bell. >0 enables the
                                        asymmetric time law (Option B): carry blend_boundary_speed only
                                        within this many seconds at each gripper end while holding the
                                        plan pace through the middle, so the transition is non-idle
                                        without speeding up the careful cruise (see _asymmetric_stroke).
        blend_ops:        list[str]  -- restrict blending to these operations by name, e.g.
                                        [Pick, Place]; omitted/empty blends every operation
        blend_mode:       str        -- "spline" (default; the analytic time law here), "vae" (the VAE
                                        motion-manifold cost picks each stroke's clock; needs
                                        `vae_retiming: false`, see vae_retiming) or "flow" (a
                                        DROID-learned flow-matching model samples the stroke's speed
                                        profile -- see flow_blending). Only the TIMING differs; geometry +
                                        limits + endpoint/non-idle handling are identical between the two.
        blend_model_path: str        -- checkpoint for the learned flow model (flow mode); relative paths
                                        resolve under the tiptop package dir. Defaults to the model's own
                                        default checkpoint when omitted.
        blend_flow_retime_only: bool -- flow mode only: keep the collision-checked cuTAMP geometry and use
                                        ONLY the flow model's sampled timing (default True). False adopts
                                        the flow sample's own path, which straightens it and deletes the
                                        VAE-manifold texture. See flow_blending.
        blend_pace:       str        -- flow mode only: where each stroke's mean joint speed comes from.
                                        "plan" (default) = the cuTAMP group's own wall-clock, so absolute
                                        velocity follows time_dilation_factor. "droid" = drawn from the
                                        measured DROID conditional on arc length + event kinds, so the
                                        emitted speed is a human statistic and varies per stroke.
        blend_pace_scale: float      -- global multiplier on the "droid" pace (default 1.0). This is the
                                        safety lever for blend_pace: droid, the way time_dilation_factor
                                        is for blend_pace: plan.
        blend_boundary_mode: str     -- flow mode only: where the joint speed AT a gripper event comes
                                        from. "const" (default) = blend_boundary_speed everywhere.
                                        "droid" = one draw per event from the measured conditional on the
                                        event KIND, shared by the strokes on both sides so the commanded
                                        speed is continuous there. Under "droid", blend_boundary_speed
                                        becomes a FLOOR only.
        blend_boundary_window_sec: float -- seconds at each end over which the learned backends' boundary
                                        correction acts (default 0.5, matching DROID's ~+-0.53 s dip).
        blend_stats_path: str        -- fitted DROID timing stats for the two modes above; relative paths
                                        resolve under the tiptop package dir. Defaults to
                                        checkpoints/droid_timing_stats.npz.
        blend_seed:       int        -- seed the pace / boundary / flow draws. Omitted (default) means a
                                        fresh draw per plan, which is the point in dataset generation.
    """
    o = overrides or {}
    # vae_retiming means the VAE manifold cost owns the clock (it optimizes each waypoint interval's
    # duration inside cuRobo's trajopt, and the result is baked into the emitted waypoints).
    # Blending would refit the geometry with a smoothing spline and re-time it from its own model,
    # discarding that. Gated HERE rather than at the call site so it covers every caller and both
    # backends -- and before the blend_* validation below, so a stale or malformed blend stanza in a
    # vae_retiming config cannot raise on values that are about to be ignored anyway.
    # local: keeps this module import-light
    from tiptop.motion_planning import resolve_vae_path, resolve_vae_retiming

    if resolve_vae_retiming(o):
        if o.get("blend_trajectory"):
            extra = (
                " -- note blend_mode: vae ALSO gives the manifold cost the clock, but per gripper-"
                "delimited stroke rather than per cuRobo leg; set vae_retiming: false to use it"
                if str(o.get("blend_mode", "")).strip().lower() == "vae"
                else ""
            )
            _log.warning(
                "vae_retiming active: trajectory blending (blend_*) is DISABLED for this run%s", extra
            )
        return BlendConfig(enabled=False)
    raw_ops = o.get("blend_ops")
    ops = tuple(str(x) for x in raw_ops) if raw_ops else None
    speed_scale = float(o.get("blend_speed_scale", _DEFAULT_SPEED_SCALE))
    if speed_scale <= 0.0:
        raise ValueError(f"blend_speed_scale must be > 0 (got {speed_scale})")
    boundary_window = float(o.get("blend_boundary_window", _DEFAULT_BOUNDARY_WINDOW))
    if boundary_window < 0.0:
        raise ValueError(f"blend_boundary_window must be >= 0 (got {boundary_window})")
    mode = str(o.get("blend_mode", "spline")).strip().lower()
    if mode not in ("spline", "flow", "vae"):
        raise ValueError(f"blend_mode must be 'spline', 'flow' or 'vae' (got {mode!r})")
    raw_model_path = o.get("blend_model_path")
    model_path = str(raw_model_path) if raw_model_path else None
    pace_mode = str(o.get("blend_pace", "plan")).strip().lower()
    if pace_mode not in ("plan", "droid"):
        raise ValueError(f"blend_pace must be 'plan' or 'droid' (got {pace_mode!r})")
    pace_scale = float(o.get("blend_pace_scale", 1.0))
    if pace_scale <= 0.0:
        raise ValueError(f"blend_pace_scale must be > 0 (got {pace_scale})")
    boundary_mode = str(o.get("blend_boundary_mode", "const")).strip().lower()
    if boundary_mode not in ("const", "droid"):
        raise ValueError(f"blend_boundary_mode must be 'const' or 'droid' (got {boundary_mode!r})")
    window_sec = float(o.get("blend_boundary_window_sec", _DEFAULT_BOUNDARY_WINDOW_SEC))
    if window_sec <= 0.0:
        raise ValueError(f"blend_boundary_window_sec must be > 0 (got {window_sec})")
    raw_stats = o.get("blend_stats_path")
    raw_seed = o.get("blend_seed")
    return BlendConfig(
        enabled=bool(o.get("blend_trajectory", False)),
        smoothing=float(o.get("blend_smoothing", _DEFAULT_SMOOTHING)),
        vel_slack=float(o.get("blend_vel_slack", _DEFAULT_VEL_SLACK)),
        acc_slack=float(o.get("blend_acc_slack", _DEFAULT_ACC_SLACK)),
        boundary_speed=float(o.get("blend_boundary_speed", _DEFAULT_BOUNDARY_SPEED)),
        speed_scale=speed_scale,
        boundary_window=boundary_window,
        ops=ops,
        mode=mode,
        model_path=model_path,
        # Resolved the same way the cuRobo manifold cost resolves it, so both ends of the pipeline
        # score against one checkpoint even when the config gives a repo-relative path.
        vae_path=resolve_vae_path(str(o["vae_path"])) if o.get("vae_path") else None,
        flow_steps=int(o.get("blend_flow_steps", 60)),
        flow_retime_only=bool(o.get("blend_flow_retime_only", True)),
        pace_mode=pace_mode,
        pace_scale=pace_scale,
        boundary_mode=boundary_mode,
        boundary_window_sec=window_sec,
        profile_end_sec=float(o.get("blend_profile_end_sec", 1.0)),
        max_duration_mult=float(o.get("blend_max_duration_mult", 2.0)),
        stats_path=str(raw_stats) if raw_stats else None,
        seed=int(raw_seed) if raw_seed is not None else None,
    )


def _op_name(label: str) -> str:
    """Operation name from a step label, e.g. 'Pick(brown_toy, grasp1, q1)' -> 'Pick'."""
    return (label or "").split("(", 1)[0].strip()


def _dedup_path(positions: np.ndarray) -> np.ndarray:
    """Drop waypoints within ``_MIN_CHORD`` of the previous kept one so chord length is strictly increasing.

    Each segment decelerates to rest, so its final few waypoints are near-identical; the true group
    endpoint must survive, so the final point is forced in and any trailing kept points too close to
    it are dropped first (keeping the endpoint exact and the last interval strictly positive).
    """
    keep = [0]
    for i in range(1, len(positions) - 1):
        if np.linalg.norm(positions[i] - positions[keep[-1]]) > _MIN_CHORD:
            keep.append(i)
    final = len(positions) - 1
    while len(keep) > 1 and np.linalg.norm(positions[final] - positions[keep[-1]]) <= _MIN_CHORD:
        keep.pop()
    keep.append(final)
    return positions[keep]


def _fit_geometry(u: np.ndarray, positions: np.ndarray, smoothing: float) -> list:
    """Per-joint geometry spline q_j(u) over arc length: penalized smoothing spline, cubic fallback."""
    if len(positions) >= _MIN_SMOOTH_PTS:
        try:
            return [make_smoothing_spline(u, positions[:, j], lam=smoothing) for j in range(positions.shape[1])]
        except Exception:
            _log.exception("Smoothing-spline geometry fit failed; falling back to a plain cubic")
    return [CubicSpline(u, positions[:, j], bc_type="natural") for j in range(positions.shape[1])]


def _eval_geometry(geom: list, u: np.ndarray, nu: int) -> np.ndarray:
    """Evaluate the ``nu``-th derivative of every joint's geometry spline -> [len(u), dof]."""
    return np.stack([g(u, nu) for g in geom], axis=1)


def _time_law(tau: np.ndarray, lead_slope: float, trail_slope: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normalized quintic h(tau) mapping [0,1]->[0,1] with, at both ends, the given first-derivative
    (slope) and zero second-derivative. Returns (h, h', h'').

    With lead_slope == trail_slope == 0 this is exactly the minimum-jerk quintic 10t^3-15t^4+6t^5
    (rest to rest). A nonzero slope makes the arc-length sweep -- hence the joint speed -- start/end
    at a controlled nonzero value instead of dwelling near zero. h(0)=0, h(1)=1, h'(0)=lead_slope,
    h'(1)=trail_slope, h''(0)=h''(1)=0.
    """
    # h = lead_slope*tau + c3 tau^3 + c4 tau^4 + c5 tau^5  (c0=0, c1=lead_slope, c2=0 satisfy the tau=0 BCs)
    a = np.array([[1.0, 1.0, 1.0], [3.0, 4.0, 5.0], [6.0, 12.0, 20.0]])
    rhs = np.array([1.0 - lead_slope, trail_slope - lead_slope, 0.0])
    c3, c4, c5 = np.linalg.solve(a, rhs)
    h = lead_slope * tau + c3 * tau**3 + c4 * tau**4 + c5 * tau**5
    hp = lead_slope + 3 * c3 * tau**2 + 4 * c4 * tau**3 + 5 * c5 * tau**4
    hpp = 6 * c3 * tau + 12 * c4 * tau**2 + 20 * c5 * tau**3
    return h, hp, hpp


def _smootherstep(x: np.ndarray) -> np.ndarray:
    """Quintic smoothstep 10x^3-15x^4+6x^5 on [0,1] (0 at 0, 1 at 1, zero slope+accel at both ends)."""
    x = np.clip(x, 0.0, 1.0)
    return x * x * x * (x * (x * 6.0 - 15.0) + 10.0)


def _asymmetric_stroke(
    geom: list,
    length: float,
    t0: float,
    t1: float,
    dt: float,
    target_duration: float,
    vel_cap: np.ndarray,
    acc_cap: np.ndarray,
    lead_speed: float,
    trail_speed: float,
    window_sec: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Option B time law: cruise at the plan pace through the middle, carry the requested boundary speed
    only within a ~``window_sec`` window at each end.

    Instead of a single symmetric min-jerk bell (whose slowest point is the endpoint, so the boundary is
    capped at ``_MAX_BOUNDARY_SLOPE`` * the stroke average), this builds a strictly-positive speed-vs-time
    profile ``p(tau)`` that equals the cruise in the middle and ramps to the boundary speed at the ends
    (a "dumbbell" when the ends are faster than the cruise, or a gentle dip when they are slower). Because
    ``p > 0`` everywhere the sweep is monotonic with no ``_MAX_BOUNDARY_SLOPE`` cap, and the end level is
    divided by the endpoint tangent so a collapsed grasp-reversal tangent is compensated. The transition
    thus clears the idle band while the careful mid-stroke pace is preserved. Returns (pos, vel, acc,
    duration); the boundary falls short only if the vel/accel caps (or ``_MAX_BOUNDARY_RATIO``) bind.
    """
    v_cruise = length / target_duration  # the plan's average pace = the middle cruise target
    t_mid = float(np.median(np.linalg.norm(_eval_geometry(geom, np.linspace(0.0, length, 256), 1), axis=1))) or 1.0
    # Relative end speed (vs cruise); divide by the endpoint tangent so the realized joint speed matches
    # the request even where that tangent has collapsed. 0 keeps a rest end (episode start/end).
    r_lead = 0.0 if lead_speed == 0.0 else min(_MAX_BOUNDARY_RATIO, (lead_speed / v_cruise) * (t_mid / t0))
    r_trail = 0.0 if trail_speed == 0.0 else min(_MAX_BOUNDARY_RATIO, (trail_speed / v_cruise) * (t_mid / t1))
    w = float(np.clip(window_sec / target_duration, 0.03, 0.35))  # window as a fraction of normalized time
    tau_d = np.linspace(0.0, 1.0, 2048)
    shape_lead = np.where(tau_d < w, 1.0 - _smootherstep(tau_d / w), 0.0)
    shape_trail = np.where(tau_d > 1.0 - w, 1.0 - _smootherstep((1.0 - tau_d) / w), 0.0)
    p = np.maximum(1.0 + (r_lead - 1.0) * shape_lead + (r_trail - 1.0) * shape_trail, 0.0)  # speed vs time
    pbar = float(np.trapezoid(p, tau_d))
    # h(tau) in [0,1]: normalized arc length vs normalized time (cumulative p, scaled so h(1)=1).
    h = np.concatenate([[0.0], np.cumsum((p[1:] + p[:-1]) / 2.0 * np.diff(tau_d))]) / pbar

    def sample(duration: float, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        ts = np.linspace(0.0, 1.0, n)
        ut = length * np.interp(ts, tau_d, h)
        ps = np.interp(ts, tau_d, p)
        pos = _eval_geometry(geom, ut, 0)
        du_dt = length * (ps / pbar) / duration
        vel = _eval_geometry(geom, ut, 1) * du_dt[:, None]
        acc = np.gradient(vel, duration / (n - 1), axis=0)
        return pos, vel, acc

    # Duration so the middle (p == 1) joint speed equals v_cruise, then stretch to fit the vel/accel caps.
    duration = t_mid * length / (pbar * v_cruise)
    _, v0, a0 = sample(duration, 512)
    alpha_v = float((np.abs(v0).max(axis=0) / vel_cap).max())
    alpha_a = float(np.sqrt((np.abs(a0).max(axis=0) / acc_cap).max()))
    duration *= max(1.0, alpha_v, alpha_a)
    n = max(2, int(round(duration / dt)) + 1)
    pos, vel, acc = sample(duration, n)
    return pos, vel, acc, duration


def blend_group(
    positions: np.ndarray,
    dt: float,
    target_duration: float,
    vel_cap: np.ndarray,
    acc_cap: np.ndarray,
    smoothing: float = _DEFAULT_SMOOTHING,
    lead_speed: float = 0.0,
    trail_speed: float = 0.0,
    boundary_window: float = _DEFAULT_BOUNDARY_WINDOW,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Re-time one group of joined waypoints into a single smooth stroke.

    Args:
        positions: [N, dof] concatenated waypoints of the group (segment joins de-duplicated upstream).
        dt: control timestep to re-sample at.
        target_duration: UPPER bound on duration (the group's original wall-clock). The stroke is sped
            up below this only as much as its own boundary speed needs (never slowed past it), and only
            when that boundary is reachable within the vel/accel caps -- so a stroke already fast enough
            keeps its original pace, and only a genuinely slow one is shortened (see the body).
        vel_cap, acc_cap: [dof] per-joint velocity / acceleration ceilings.
        smoothing: geometry smoothing-spline penalty (lam).
        lead_speed, trail_speed: joint speed (rad/s, L2) to carry at the first / last sample. 0 means
            rest (min-jerk); >0 keeps the boundary out of the idle band (see module docstring).
        boundary_window: 0 -> symmetric min-jerk bell (the default; boundary <= _MAX_BOUNDARY_SLOPE *
            average). >0 -> Option B asymmetric time law: cruise at the plan pace through the middle and
            carry the boundary speed only within this many seconds at each end (see _asymmetric_stroke).

    Returns:
        (positions, velocities, accelerations, dt_out) re-sampled at ~uniform ``dt_out`` (== dt up to
        rounding). Velocity is exactly zero at an end whose *_speed is 0, else that boundary speed.
    """
    positions = _dedup_path(np.asarray(positions, dtype=np.float64))

    # Degenerate group (no motion): a single held pose -> two stationary, zero-velocity samples.
    if len(positions) < 2:
        pos = np.repeat(positions[:1], 2, axis=0)
        zeros = np.zeros_like(pos)
        return pos, zeros, zeros, dt

    # Arc-length parameter (joint space) for the geometry spline.
    u = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(positions, axis=0), axis=1))])
    length = float(u[-1])
    if length < _MIN_CHORD:  # no meaningful net motion (e.g. returns to start): hold the pose.
        pos = np.repeat(positions[:1], 2, axis=0)
        zeros = np.zeros_like(pos)
        return pos, zeros, zeros, dt

    geom = _fit_geometry(u, positions, smoothing)

    # Tangent magnitude |dq/du| at each end. Arc length makes this ~1, but the smoothing spline
    # rounds corners, so it can be < 1 near a boundary; dividing the requested boundary joint speed
    # by it makes the actual end speed match the request rather than falling short.
    t0 = float(np.linalg.norm(_eval_geometry(geom, np.array([0.0]), 1)[0])) or 1.0
    t1 = float(np.linalg.norm(_eval_geometry(geom, np.array([length]), 1)[0])) or 1.0

    if boundary_window > 0.0:
        # Option B: asymmetric time law -- plan pace in the middle, boundary speed only near the ends.
        out_pos, out_vel, out_acc, duration = _asymmetric_stroke(
            geom, length, t0, t1, dt, target_duration, vel_cap, acc_cap, lead_speed, trail_speed, boundary_window
        )
        return _finish_stroke(out_pos, out_vel, out_acc, duration, lead_speed, trail_speed)

    def sample(duration: float, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        tau = np.linspace(0.0, 1.0, n)
        # Boundary slopes in normalized (tau) units so that the joint speed at each end equals the
        # requested value. Capped below the mean slope (1.0) to keep the sweep monotonic.
        lead_slope = min(_MAX_BOUNDARY_SLOPE, (lead_speed / t0) * duration / length)
        trail_slope = min(_MAX_BOUNDARY_SLOPE, (trail_speed / t1) * duration / length)
        h, hp, hpp = _time_law(tau, lead_slope, trail_slope)
        ut = length * h
        du_dt = length * hp / duration
        d2u_dt2 = length * hpp / duration**2
        d1 = _eval_geometry(geom, ut, 1)  # dq/du
        d2 = _eval_geometry(geom, ut, 2)  # d2q/du2
        pos = _eval_geometry(geom, ut, 0)
        vel = d1 * du_dt[:, None]
        acc = d2 * (du_dt**2)[:, None] + d1 * d2u_dt2[:, None]
        return pos, vel, acc

    # Duration that just realizes each requested boundary speed WITHOUT hitting the slope cap: the
    # boundary joint speed is t_end * length * slope / duration and slope maxes at _MAX_BOUNDARY_SLOPE,
    # so the request is met iff duration <= _MAX_BOUNDARY_SLOPE * length * t_end / speed. Take the tightest
    # (smallest) over the nonzero-speed ends; rest ends (speed 0) impose no constraint.
    dur_reqs = []
    if lead_speed > 0.0:
        dur_reqs.append(_MAX_BOUNDARY_SLOPE * length * t0 / lead_speed)
    if trail_speed > 0.0:
        dur_reqs.append(_MAX_BOUNDARY_SLOPE * length * t1 / trail_speed)
    dur_boundary = min(dur_reqs) if dur_reqs else target_duration

    # vel/accel-cap floor: the shortest duration the robot can physically run this stroke. Under
    # duration -> alpha*duration, velocity scales as 1/alpha and acceleration as 1/alpha^2, so
    # alpha*duration is invariant and cap_min is independent of the probe duration.
    _, v0, a0 = sample(target_duration, 400)
    alpha_v = float((np.abs(v0).max(axis=0) / vel_cap).max())
    alpha_a = float(np.sqrt((np.abs(a0).max(axis=0) / acc_cap).max()))
    cap_min = target_duration * max(alpha_v, alpha_a)

    # Speed the stroke up ONLY as much as its own boundary needs -- not globally. A stroke already fast
    # enough to carry the boundary keeps its original (careful) pace; a slow one is shortened just to
    # dur_boundary so the transition clears the idle band without over-speeding the cruise. When the
    # boundary is unreachable within the robot's limits (dur_boundary < cap_min -- e.g. a rounded grasp
    # cusp whose tangent collapses), don't chase it: keep the original pace rather than running the whole
    # stroke at the limit for a transition it still can't hit (the warning below flags it instead).
    if dur_boundary >= cap_min:
        duration = min(target_duration, dur_boundary)
    else:
        duration = target_duration
    duration = max(duration, cap_min)

    n = max(2, int(round(duration / dt)) + 1)
    out_pos, out_vel, out_acc = sample(duration, n)
    return _finish_stroke(out_pos, out_vel, out_acc, duration, lead_speed, trail_speed)


def _finish_stroke(out_pos, out_vel, out_acc, duration, lead_speed, trail_speed):
    """Pin rest ends to exactly zero, warn on a shortfall, and return (pos, vel, acc, dt_out)."""
    # Pin a boundary to exact rest only where a zero boundary speed was requested (episode ends);
    # a nonzero boundary is left as computed so the gripper-adjacent frames stay out of the idle band.
    if lead_speed == 0.0:
        out_vel[0] = 0.0
    if trail_speed == 0.0:
        out_vel[-1] = 0.0
    # Surface a boundary that fell short of the request: it is limited by the robot's vel/accel limits.
    # Default (symmetric) law: a slow stroke whose tangent collapses at a grasp/place cusp -- fix on the
    # geometry side (blend_smoothing / dwell). Asymmetric law (blend_boundary_window): the end ramp needs
    # more room -- widen blend_boundary_window. Either way, running faster alone won't clear it.
    for end, req in ((0, lead_speed), (-1, trail_speed)):
        realized = float(np.linalg.norm(out_vel[end]))
        if req > 0.0 and realized < 0.7 * req:
            _log.warning(
                "Blended stroke boundary speed %.3f rad/s is below the requested %.3f, limited by the "
                "robot's vel/accel limits (a collapsed grasp-reversal tangent, or too tight a boundary "
                "window). These gripper-adjacent frames stay slow.",
                realized, req,
            )
    dt_out = duration / (len(out_pos) - 1)
    return out_pos, out_vel, out_acc, dt_out


def _resolve_caps(
    orig_velocities: np.ndarray,
    dt: float,
    vel_limit: np.ndarray | None,
    acc_limit: np.ndarray | None,
    vel_slack: float,
    acc_slack: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-joint velocity/acceleration caps: the robot's real limits (scaled by slack) when available,
    else a generous multiple of the plan's own peaks (a slowed plan sits well below the true limits)."""
    if vel_limit is not None and acc_limit is not None:
        return vel_slack * np.abs(vel_limit), acc_slack * np.abs(acc_limit)
    _log.warning(
        "Trajectory blending: robot limits unavailable; capping at %.1fx/%.1fx the plan's own peak "
        "velocity/acceleration instead.",
        _FALLBACK_VEL_MULT,
        _FALLBACK_ACC_MULT,
    )
    ov = np.asarray(orig_velocities, dtype=np.float64)
    vel_cap = _FALLBACK_VEL_MULT * np.abs(ov).max(axis=0)
    orig_acc = np.gradient(ov, dt, axis=0) if len(ov) > 2 else np.zeros_like(ov)
    acc_cap = _FALLBACK_ACC_MULT * np.abs(orig_acc).max(axis=0)
    # Floor tiny-motion joints at a fraction of the largest cap so their near-zero peak does not make
    # the limit check hypersensitive and stretch the whole group.
    for cap in (vel_cap, acc_cap):
        m = float(cap.max())
        if m > 0:
            np.maximum(cap, 0.25 * m, out=cap)
    return vel_cap, acc_cap


def _blend_trajectory_steps(
    steps: list[dict],
    config: BlendConfig,
    vel_limit: np.ndarray | None,
    acc_limit: np.ndarray | None,
    lead_speed: float,
    trail_speed: float,
) -> dict:
    """Blend a run of consecutive ``trajectory`` steps into one re-timed trajectory step."""
    from curobo.types.state import JointState  # lazy: keep the module importable without cuRobo

    template = steps[0]["plan"]
    device = template.position.device
    joint_names = template.joint_names
    dt = float(steps[0]["dt"])

    # Join waypoints, dropping each later segment's first row (an exact copy of the previous
    # segment's last row -- see motion_solver: the next plan starts from the prior end position).
    seg_positions = [s["plan"].position.detach().cpu().numpy().astype(np.float64) for s in steps]
    joined = np.concatenate([seg_positions[0]] + [p[1:] for p in seg_positions[1:]], axis=0)
    orig_velocities = np.concatenate(
        [s["plan"].velocity.detach().cpu().numpy().astype(np.float64) for s in steps], axis=0
    )
    # The group's original wall-clock -- an UPPER bound on the stroke's duration. blend_group shortens it
    # per stroke only as much as that stroke's boundary speed needs (never slower, and only within the
    # robot's real limits), so the cruise of strokes already fast enough stays at the original pace.
    # speed_scale is an optional extra global multiplier on top (default 1.0 -> no global change).
    target_duration = sum((len(p) - 1) for p in seg_positions) * dt / config.speed_scale

    vel_cap, acc_cap = _resolve_caps(
        orig_velocities, dt, vel_limit, acc_limit, config.vel_slack, config.acc_slack
    )
    if config.mode == "vae":
        # The VAE manifold cost picks this stroke's clock. Unlike the spline/flow laws it gets the
        # group's wall-clock only as a RANGE BOUND, not a target -- choosing the pace is the point.
        from tiptop.vae_retiming import vae_retime_group

        pos, vel, acc, dt_out = vae_retime_group(
            joined, dt, target_duration, vel_cap, acc_cap, config.smoothing, lead_speed, trail_speed,
            config.max_duration_mult, config.vae_path,
        )
    else:
        pos, vel, acc, dt_out = blend_group(
            joined, dt, target_duration, vel_cap, acc_cap, config.smoothing, lead_speed, trail_speed,
            config.boundary_window,
        )

    plan = JointState(
        position=torch.as_tensor(pos, dtype=torch.float32, device=device),
        velocity=torch.as_tensor(vel, dtype=torch.float32, device=device),
        acceleration=torch.as_tensor(acc, dtype=torch.float32, device=device),
        jerk=torch.zeros(pos.shape, dtype=torch.float32, device=device),
        joint_names=list(joint_names) if joint_names is not None else None,
    )
    # Keep the run's first label; the merged stroke spans the same logical action(s).
    return {"type": "trajectory", "plan": plan, "dt": dt_out, "label": steps[0]["label"]}


def blend_cutamp_plan(
    cutamp_plan: list[dict],
    config: BlendConfig,
    vel_limit: np.ndarray | None = None,
    acc_limit: np.ndarray | None = None,
) -> list[dict]:
    """Return ``cutamp_plan`` with consecutive trajectory segments blended + re-timed, if enabled.

    A no-op returning the plan unchanged when ``config.enabled`` is False (the default -- blending is
    opt-in per config via ``blend_trajectory: true``). When enabled, gripper steps pass through
    untouched and delimit the groups; the trajectory steps between two gripper events (or a plan end)
    -- always a single operation, e.g. one Pick or Place -- are merged into a single smooth,
    minimum-jerk stroke that only comes to rest at those gripper events. If ``config.ops`` is set,
    only operations named in it are blended; every other operation's original segments pass through
    untouched. ``vel_limit`` / ``acc_limit`` are the robot's per-arm-joint velocity / acceleration
    limits (see :func:`arm_joint_limits`); when omitted, a fallback derived from the plan's own peaks
    is used.
    """
    if not config.enabled:
        return cutamp_plan

    n_steps = len(cutamp_plan)
    out: list[dict] = []
    run: list[dict] = []
    run_start_idx = 0
    stats = {"blended": 0, "skipped": 0}

    def flush(run_end_idx: int):
        if not run:
            return
        # A run is one operation's trajectory segments (grippers delimit operations). Skip it if an
        # ops filter is set and this operation is not in it -- its original segments pass through.
        if config.ops is not None and _op_name(run[0]["label"]) not in config.ops:
            out.extend(run)
            stats["skipped"] += 1
            run.clear()
            return
        # Rest (zero boundary speed) only where the arm is genuinely stationary: the episode's very
        # start (this run opens the plan) and end (it closes the plan). Every other boundary abuts a
        # gripper event, so carry a nonzero speed there to keep those frames out of the idle filter.
        lead_speed = 0.0 if run_start_idx == 0 else config.boundary_speed
        trail_speed = 0.0 if run_end_idx == n_steps - 1 else config.boundary_speed
        try:
            out.append(_blend_trajectory_steps(run, config, vel_limit, acc_limit, lead_speed, trail_speed))
            stats["blended"] += 1
        except Exception:
            # Best-effort: on any numerical/shape surprise, keep the original segments for this run
            # rather than failing the whole plan.
            _log.exception("Trajectory blending failed for a segment run; keeping original segments")
            out.extend(run)
        run.clear()

    for idx, step in enumerate(cutamp_plan):
        if step.get("type") == "trajectory":
            if not run:
                run_start_idx = idx
            run.append(step)
        else:
            flush(idx - 1)
            out.append(step)
    flush(n_steps - 1)

    scope = "all operations" if config.ops is None else f"operations {list(config.ops)}"
    _log.info(
        "Trajectory blending (%s): %d operation groups blended into single strokes, %d left unblended "
        "(%d trajectory segments -> %d steps)",
        scope,
        stats["blended"],
        stats["skipped"],
        sum(1 for s in cutamp_plan if s.get("type") == "trajectory"),
        sum(1 for s in out if s.get("type") == "trajectory"),
    )
    return out


def arm_joint_limits(motion_gen, dof: int) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Best-effort (velocity_limit, acceleration_limit) arrays for the first ``dof`` arm joints.

    Reads the robot's joint limits from ``motion_gen`` and returns the upper (positive) velocity and
    acceleration bounds for the arm joints (the plan's dof), or (None, None) if they can't be read --
    in which case the blend falls back to plan-derived caps.
    """
    try:
        jl = motion_gen.kinematics.get_joint_limits()
        vel = jl.velocity[1].detach().cpu().numpy().astype(np.float64)  # upper bound per joint
        acc = jl.acceleration[1].detach().cpu().numpy().astype(np.float64)
        # cuRobo cspace lists the arm joints first, then any gripper joints; the plan is the arm dof.
        return vel[:dof], acc[:dof]
    except Exception:
        _log.exception("Could not read robot joint limits for trajectory blending; using fallback caps")
        return None, None

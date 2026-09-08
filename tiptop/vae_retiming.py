"""Stroke re-timing driven entirely by the VAE motion-manifold cost -- no flow model, no search.

WHY THIS EXISTS, AND WHY IT IS NOT THE `vae_retiming` TAMP OVERRIDE
------------------------------------------------------------------
Both put the VAE manifold cost in charge of the clock; they differ in what a "segment" is, and that
difference is the whole ballgame.

The in-trajopt `vae_retiming` override optimizes duration knots inside each cuRobo ``plan_single``.
But cuTAMP issues 2-3 ``plan_single`` calls per gripper-to-gripper motion (retract, approach, grasp
-- see cutamp/motion_solver.py), while DROID's latent cluster was built on whole human strokes. The
cost then pulls every LEG to a human STROKE duration. Measured on a 3-toy pick-and-place: 19 legs,
each emitted at 4.58-4.72 s (a 1.03x spread) while their joint path lengths spanned 12.4x -- a 5 cm
retract given the same 4.6 s as a 3.5 rad transit. Seven strokes took 13.9 s each against teleop's
6.8 s median, for an 88 s episode against teleop's 48.8 s.

This module scores the unit DROID actually has: one gripper-delimited stroke. The grouping,
vel/accel caps and endpoint pinning are the existing blender's (see trajectory_blending); only the
TIME LAW is ours.

WHAT IS OPTIMIZED, AND WHY THERE IS NO DURATION SEARCH
------------------------------------------------------
One batched Adam run over two things at once:

* ``theta`` -- one duration knot per arc-length interval, the SHAPE of the speed profile.
* ``span``  -- a single scalar, projected onto the allowed duration range after each step: how many
               15 Hz frames the stroke spans, and hence its duration.

Duration used to be a grid search (a coarse geomspace sweep plus a refine bracket, each candidate
getting its own independently-converged ``theta``) because the obvious parameterization makes it
non-differentiable. If the clock is normalized by duration -- knot shares scaled to sum to D -- then
every sampled position is scale-invariant in D, and D reaches the score ONLY through the integer
sample count ``round(D * 15) + 1``. The objective is then a step function of duration: literally zero
gradient inside a bin, so nothing to descend, and a finite-difference secant across bins is
noise-dominated (measured: the sign of the adjacent-bin difference flips on ~55% of steps).

Here the stroke is instead sampled on a FIXED 15 Hz grid whose frame COUNT is what duration
controls. Spreading the same path over more frames shrinks the per-frame step, so velocity,
acceleration and jerk -- and therefore the manifold score -- vary smoothly and analytically with
duration. The leftover discreteness (a stroke ends between frames) is absorbed by a soft mask, exact
thanks to ``_FilterbankVAE.encode_mu_masked``.

That same masking fix is what lets ``_N_STARTS`` initial durations share ONE padded forward pass, so
multi-start costs no more wall-clock than a single start. It also disposes of the old grid's most
delicate property: candidate durations no longer need hand-matched optimization budgets to be
comparable, because they now literally step together in one optimizer.

THREE THINGS THAT HAD TO BE RIGHT (each was measured, each alternative was worse)
--------------------------------------------------------------------------------
1. TRUE 15 Hz ENCODING. ``droid_mean``/``droid_prec`` were built from motion sampled at 15 Hz
   (vae/data.py COMMON_RATE), and the encoder is a filterbank whose kernels therefore mean a fixed
   number of SECONDS. Holding the frame count fixed and letting the spacing float instead (which
   would make duration differentiable the easy way, through the 1/h scaling of the derivative
   channels) feeds that filterbank the wrong rate: against a true 15 Hz encode it correlates
   anywhere from +0.02 to +0.99 depending on the stroke, moves the best duration by up to 3.9 s, and
   compresses the score range 10-100x. Do not "simplify" the clock that way.
2. ARC-LENGTH CANVAS, NOT INDEX. Resampling the joined stroke by index inherits cuRobo's own profile
   -- including the full stop at each interior leg join -- so theta can only rescale it. Arc length
   hands the speed profile to theta.
3. THETA, NOT JUST A DURATION. Arc length alone is worse than index: a constant-speed traversal is
   less human than cuRobo's bell. The cost wants an accelerate/cruise/decelerate shape, and theta is
   what lets it build one.

WHAT IS DELIBERATELY ABSENT (each was in an earlier version, each was ablated out)
---------------------------------------------------------------------------------
The graph is two parameters, one sampler, one mask and two loss terms. Removing the following
made the score BETTER on both test plans, not merely equal, so do not reintroduce them:

* An in-loop velocity/acceleration hinge. ``_emit_raw``'s hard check already decides feasibility.
* A sigmoid mask taper. ``clamp(span - i, 0, 1)`` scores better and drops a width constant.
* Boundary speeds measured at the executor timestep, which needed a ``dt / duration`` branch and a
  second progress convention. One 15 Hz frame in from each end is as good.
* ``progress.clamp(max=1)``. ``_sample`` already clamps the interpolation weight.
* A separate ``n_eff`` with ``n_eff - 1`` written at three call sites; ``span`` is the same number
  once.

Things that ARE load-bearing and were confirmed by trying to remove them: the boundary term (without
it the emitted lead speed is 0.42-0.49 rad/s against a 0.10 target), ``_KNOTS``, ``_N_STARTS``, and
theta's normalization + tanh rail (an unnormalized "duration is just the sum of interval times"
parameterization deletes the duration parameter entirely and is genuinely simpler, but scores
~19% worse on one plan).
"""

from __future__ import annotations

import hashlib
import logging
import time

import numpy as np
import torch
import torch.nn.functional as F

_log = logging.getLogger(__name__)

#: Arc-length control knots the stroke's clock is parameterized over. One duration knot per interval,
#: so this also sets how finely the speed profile can be shaped -- and it IS the binding resolution,
#: despite being well above the ~10-25 samples a 15 Hz encoding of a human stroke has. Halving it
#: costs badly (total maha^2 5.6 -> 8.7 and 12.6 -> 18.1 at 32 knots; 17.5 and 47.1 at 16), so do not
#: trim it for speed.
_KNOTS = 64

#: Rail on each interval's duration: within exp(+-_RAIL) of the stroke's uniform-arc-length pace,
#: i.e. exp(1.0) ~ 2.7x either way. Swept, and the value matters in both directions -- it trades the
#: profile's expressiveness against how many starts survive the cap check:
#:    0.7   6.69 / 13.41   11/16 starts   tightest interval 0.21 frames
#:    1.0   6.12 / 12.07   10/16          0.16
#:    1.5   5.92 / 13.02    6/16          0.057
#:    2.5   6.54 / 22.11    1/16          0.005
#:    4.0  74.1  / 433      0/16          0.0007   (every start infeasible)
#: Past ~1.5 the softmax starves an interval toward zero time, which is an unbounded acceleration
#: that _emit_raw then throws away. Dropping the tanh entirely (unbounded softmax logits) gives
#: 6.31 / 13.29 at 5/16 -- worse than 1.0 on both plans, so the rail is not merely a safety net.
_RAIL = 1.0

#: One joint optimization over (theta, span). Adam steps, and the two learning rates.
#:
#: The duration rate is a FRACTION OF THE ALLOWED RANGE per step, not an absolute frame count: Adam
#: moves a parameter by ~lr in its own units, and a stroke's [s_lo, s_hi] range spans an order of
#: magnitude across strokes, so an absolute rate is far too slow for a wide range and too coarse for
#: a narrow one. Ablated (analysis_dataset_diff/NU_PARAM_ABLATION.txt): with the rate scaled this
#: way, optimizing the frame count directly matches the previous sigmoid-squashed parameterization
#: (median maha^2 0.79 vs 0.82, paired -0.03) with more cap-feasible candidates (8.5 vs 7.0 of 16);
#: with an unscaled rate it is far worse (1.44, 2.0 of 16) because 500 steps cannot cross the range.
#: 0.0375 reproduces the old sigmoid's midpoint sensitivity, 0.15 * (s_hi - s_lo) / 4, exactly.
#:
#: It is NOT a free tuning knob to raise: duration is a single scalar competing with 63 profile
#: knots, so on its own a larger step just walks the duration somewhere bad before theta has shaped
#: anything. Measured on a 7-stroke plan under the old parameterization, raising it alone
#: monotonically hurt (total maha^2 13.4 -> 17.7 -> 26.1 -> 205 for 0.05/0.15/0.30/0.60 at a single
#: start, in logit units). What fixes the imbalance is starting from several durations at once --
#: see _N_STARTS.
_ITERS = 500
_LR_THETA = 0.08
_LR_DURATION = 0.0375

#: Initial durations, log-spaced across the allowed range, optimized SIMULTANEOUSLY as one batch.
#: Strokes differ in where their basin sits -- most land near the middle of the range, but a Place
#: whose score explodes past ~6 s has its optimum near 3.5 s -- and a single start converges into
#: whichever basin it began in. Because the batch shares one forward pass this is nearly free: 16
#: starts cost the same wall-clock as 1, and took a 7-stroke plan from 13.4 to 5.9 total maha^2.
_N_STARTS = 16

#: Weight pulling each stroke end onto the requested lead/trail boundary speed (a two-sided squared
#: target -- the end speed is held AT it, not merely above it).
#:
#: This term governs the arm's dwell at gripper events, which is the dominant stall in the emitted
#: data. Every cuTAMP gripper event is a near-exact direction reversal -- measured cos(v_before,
#: v_after) = -0.75 to -0.95, because the arm approaches along the tool axis and the Place retract
#: ladder lifts back out along it -- so |v| has to pass through a minimum there no matter how the
#: stroke is timed. Human teleop, which does NOT reverse (cos +0.85, only 6% of events), holds a
#: 0.168 rad/s median minimum through its own gripper events. What the timing controls is not
#: whether the dip happens but how long it lasts, and that is set by `blend_boundary_speed`.
#:
#: A/B'd on one raw 19-leg plan, counting emitted frames at 15 Hz:
#:    target @ 0.01   58 frames < 0.10 rad/s, longest near-zero run 17   junction acc 0.48x limit
#:    target @ 0.10    6 frames < 0.10 rad/s, longest run 5             junction acc 0.91x limit
#:    band   @ 0.10    7 frames < 0.10 rad/s, longest run 5             junction acc 1.16x limit
#: So the VALUE is the whole fix. Replacing the target with a floor (or a [floor, 3x floor] band) was
#: tried and rejected: it does not reduce the stall further, and the extra end speed it permits pushes
#: the reversal's acceleration -- which spans TWO strokes and is therefore invisible to _emit_raw's
#: per-stroke cap check -- past the FR3's 15 rad/s^2 at two of six junctions. 0.10 already sits at
#: 0.91x, so it is close to the ceiling: raising `blend_boundary_speed` further trades a stall the
#: robot can track for a commanded reversal it cannot.
_BOUNDARY_WEIGHT = 50.0

#: Allowed duration range, as multiples of the stroke's own cuRobo wall-clock, and the absolute
#: seconds it is clipped to. These are BOX BOUNDS on the optimized duration (``span`` is projected
#: onto them after each Adam step), not a set of candidates. The lower multiple is what lets the VAE
#: speed a stroke up; the upper one is `blend_max_duration_mult`, passed in.
_D_LO_MULT = 0.2
_D_ABS_LO, _D_ABS_HI = 0.8, 20.0

_VAE_RATE_HZ = 15.0

# Shrinkage ladder for the sampled target draw (see target_latent). The draw is a TARGET, not a
# feasibility statement, so an extreme one asks for a stroke the robot's velocity/acceleration caps
# then reject -- measured 1-2 strokes in 10 unemittable at full scale. Each entry is tried in turn
# and the first that emits wins; 0.0 is the cluster MEAN, i.e. the original objective.
#
# Graduated rather than all-or-nothing because the fallback costs exactly the variance this change
# exists to restore: measured over 10 planner strokes, residual log-duration sd was 0.153 with the
# mean target, 0.350 sampled with no guard, but only 0.207 when the 2 cap-failing strokes dropped
# straight to the mean. Shrinking keeps those strokes sampled, just less far out.
_TARGET_SCALES = (1.0, 0.6, 0.3, 0.0)

#: Passes _stretch_to_caps gets to slow a stroke into the caps. Each pass divides the overshoot by
#: roughly itself, so this is many more than the two or three it takes; it exists only so a pathology
#: (a caps array with a zero in it, say) terminates instead of looping.
_STRETCH_TRIES = 12


class _Scorer:
    """Loads the VAE manifold pack once and scores 15 Hz-sampled strokes against the DROID cluster."""

    def __init__(self, checkpoint_path: str | None, n_joints: int):
        # Imported lazily: cuRobo is heavy and this module is only reached when blend_mode is "vae".
        from curobo.rollout.cost.vae_manifold_cost import load_vae_manifold
        from curobo.types.base import TensorDeviceType

        self.tp = TensorDeviceType()
        self.device = self.tp.device
        self.pack = dict(load_vae_manifold(checkpoint_path, self.tp), n_joints=n_joints)
        self.n_joints = n_joints

    def features(self, q: torch.Tensor) -> torch.Tensor:
        """q: [B, T, J] sampled at exactly 15 Hz -> [B, C, T] standardized [q|v|a|jerk]."""
        from curobo.rollout.cost.vae_manifold_cost import _grad_time

        h = 1.0 / _VAE_RATE_HZ
        v = _grad_time(q, h)
        a = _grad_time(v, h)
        jk = _grad_time(a, h)
        feats = torch.cat([q, v, a, jk], dim=-1)
        feats = (feats - self.pack["chan_mu"]) / self.pack["chan_sd"]
        return feats.transpose(1, 2).contiguous()

    @property
    def chol_cov(self) -> torch.Tensor:
        """Cholesky factor of the DROID latent covariance, for drawing target latents.

        ``load_vae_manifold`` exposes the precision, not the covariance; they are exact inverses
        (verified ``cov @ prec == I`` to 1e-3 on the shipped checkpoint), so inverting here avoids
        touching the vendored cuRobo loader.
        """
        if getattr(self, "_chol", None) is None:
            cov = torch.linalg.inv(self.pack["droid_prec"])
            cov = 0.5 * (cov + cov.T)                       # symmetrise away inversion round-off
            self._chol = torch.linalg.cholesky(cov)
        return self._chol

    def maha2(self, x: torch.Tensor, m: torch.Tensor, target: torch.Tensor | None = None):
        """Squared Mahalanobis distance on a masked (zero-padded) window.

        ``target`` defaults to the DROID cluster MEAN, which is the mode-seeking objective: its
        optimum is the centroid, a point no real motion occupies (the checkpoint bakes
        ``maha2_droid_mean = 7.04`` over 94,774 real DROID segments -- real motion sits at ~7, not 0).
        Passing a target SAMPLED from the cluster is what makes this a distribution match instead;
        see :func:`target_latent`.
        """
        z = self.pack["model"].encode_mu_masked(x, m)
        dz = z - (self.pack["droid_mean"] if target is None else target)
        return torch.einsum("ni,ij,nj->n", dz, self.pack["droid_prec"], dz)

    def score_emitted(self, positions: np.ndarray, duration: float,
                      target: torch.Tensor | None = None) -> float:
        """Score an EMITTED stroke: resample to exactly 15 Hz and encode it unpadded.

        Deliberately independent of the optimizer's own machinery -- this is what actually ships, so
        it is what starts are ranked by. ``target`` must be the SAME latent the optimizer used: with
        a sampled target, ranking starts by distance to the cluster mean would re-select the most
        typical candidate and undo the sampling (an argmin over _N_STARTS is an extreme-value pick).
        """
        n = max(6, int(round(duration * _VAE_RATE_HZ)) + 1)
        src = np.linspace(0.0, 1.0, len(positions))
        tgt = np.linspace(0.0, 1.0, n)
        q = np.stack([np.interp(tgt, src, positions[:, j]) for j in range(positions.shape[1])], axis=1)
        qt = torch.as_tensor(q[None], device=self.device, dtype=torch.float32)
        x = self.features(qt)
        with torch.no_grad():
            z = self.pack["model"].encode_mu(x, torch.ones(1, 1, n, device=self.device))
            dz = z - (self.pack["droid_mean"] if target is None else target)
            return float(torch.einsum("ni,ij,nj->n", dz, self.pack["droid_prec"], dz))


_SCORER: _Scorer | None = None


def _scorer(checkpoint_path: str | None, n_joints: int) -> _Scorer:
    """Process-wide scorer. Rebuilt if the checkpoint or dof changes (they don't, within a run)."""
    global _SCORER
    if _SCORER is None or _SCORER.n_joints != n_joints:
        _SCORER = _Scorer(checkpoint_path, n_joints)
    return _SCORER


def target_latent(scorer: "_Scorer", knots_np: np.ndarray, enabled: bool,
                  scale: float = 1.0) -> torch.Tensor | None:
    """One latent for this stroke to aim at: a DRAW from the DROID cluster, or None for the mean.

    WHY THIS EXISTS. The objective minimises Mahalanobis distance to ``droid_mean``, so its optimum
    is the cluster CENTROID -- and ``vae_retime_group`` then takes an argmin over ``_N_STARTS``
    candidates, which pushes further into "more typical than typical". Both levels select for
    typicality, and the result is that every stroke of a given path length gets nearly the same
    clock. Measured on gripper-delimited strokes matched for path length (1.5-4.0 rad), residual
    log-duration sd after removing the duration~length law: DROID 0.414, this lab's teleop 0.288,
    generated 0.148 -- between-stroke timing variance roughly 2.8x too small.

    Minimising distance to a distribution's mean returns the mean. A distribution MATCH needs a
    sample, so this draws ``z* ~ N(droid_mean, droid_cov)`` and the caller aims at that instead.

    Feasibility was the real risk -- only the time law is free here, so a sampled target could just
    project back onto the same latent. It does not: re-timing one fixed real DROID path over 5
    durations x 5 profiles spans 4.98 Mahalanobis units against a real between-stroke spread of 2.18,
    i.e. 229% of the variation that needs reproducing (analysis_dataset_diff/VAE_RETIMING_FIX.md).

    The seed is derived from the stroke GEOMETRY, so a re-plan of the same scene reproduces the same
    timing, different strokes draw independently, and no caller has to thread a stroke index through.
    The hash is chaotic by design: near-identical geometries must give independent draws, or the
    target becomes a smooth function of the path and re-introduces exactly the correlation being
    removed.
    """
    if not enabled or scale <= 0.0:
        return None
    digest = hashlib.blake2b(np.ascontiguousarray(np.round(knots_np, 6)).tobytes(), digest_size=8)
    seed = int.from_bytes(digest.digest(), "little") % (2 ** 63 - 1)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    d = scorer.pack["droid_mean"].shape[-1]
    eps = torch.randn(d, generator=gen, dtype=torch.float64).to(
        device=scorer.device, dtype=scorer.pack["droid_mean"].dtype
    )
    return scorer.pack["droid_mean"] + float(scale) * (scorer.chol_cov @ eps).unsqueeze(0)


def _time_knots(theta: torch.Tensor) -> torch.Tensor:
    """theta [B, M-1] -> tau [B, M], the normalized clock time at which each arc knot is reached.

    ``softmax(_RAIL * tanh(theta))`` gives each equal-distance arc interval a share of the stroke's
    time, so theta owns the SHAPE of the speed profile. The tanh rail keeps every interval within
    exp(+-1.5) ~ 4.5x of the uniform pace; without it intervals collapse toward zero time (measured
    down to 0.0000 frames), which is an unbounded acceleration the cap check then has to throw away.
    """
    return F.pad(torch.cumsum(torch.softmax(_RAIL * torch.tanh(theta), dim=-1), dim=-1), (1, 0))


def _sample(q_knots: torch.Tensor, tau: torch.Tensor, frac: torch.Tensor) -> torch.Tensor:
    """Read the arc-length canvas ``q_knots`` [B, M, J] at normalized clock progress ``frac`` [B, T].

    ``frac`` outside [0, 1] needs no clamping: ``u`` is clamped below, which already pins anything
    past the end to the final knot.
    """
    idx = ((frac.unsqueeze(-1) >= tau[:, :-1].unsqueeze(1)).sum(-1) - 1).clamp(0, tau.shape[1] - 2)
    t0 = torch.gather(tau, 1, idx)
    t1 = torch.gather(tau, 1, idx + 1)
    u = ((frac - t0) / (t1 - t0).clamp(min=1e-9)).clamp(0.0, 1.0).unsqueeze(-1)
    gi = idx.unsqueeze(-1).expand(-1, -1, q_knots.shape[-1])
    q0 = torch.gather(q_knots, 1, gi)
    q1 = torch.gather(q_knots, 1, gi + 1)
    return q0 + (q1 - q0) * u


def _optimize(scorer, q_knots, d_lo, d_hi, lead_speed, trail_speed, target=None):
    """Joint Adam over (theta, span) from ``_N_STARTS`` durations at once. -> [(duration, tau)].

    ``span`` -- how many 15 Hz frames the stroke covers -- is the one duration quantity, and it is
    the optimized parameter itself: progress is ``i/span``, the mask is ``clamp(span - i, 0, 1)``,
    and the emitted duration is ``span/15``. The range is held by projecting onto [s_lo, s_hi] after
    each step rather than by squashing a logit through a sigmoid; the two are equivalent in solution
    quality once the learning rate is scaled to the range (see _LR_DURATION), and the projection is
    one fewer transform to reason about. The sigmoid's flat tails were never reached in practice
    (measured: 0 of 16 starts ended at a bound), so nothing is lost by dropping it.
    """
    dev = q_knots.device
    s_lo, s_hi = d_lo * _VAE_RATE_HZ, d_hi * _VAE_RATE_HZ
    n_frames = int(np.ceil(s_hi)) + 8

    starts = np.geomspace(d_lo, d_hi, _N_STARTS + 2)[1:-1]
    span = torch.tensor(starts * _VAE_RATE_HZ, device=dev, dtype=torch.float32, requires_grad=True)
    theta = torch.zeros(_N_STARTS, q_knots.shape[1] - 1, device=dev, requires_grad=True)
    knots = q_knots.expand(_N_STARTS, -1, -1)
    frames = torch.arange(n_frames, device=dev, dtype=q_knots.dtype)

    opt = torch.optim.Adam([{"params": [theta], "lr": _LR_THETA},
                            {"params": [span], "lr": _LR_DURATION * (s_hi - s_lo)}])
    for _ in range(_ITERS):
        opt.zero_grad(set_to_none=True)
        tau = _time_knots(theta)

        # 15 Hz scoring pass. Duration enters through the progress denominator: the same path spread
        # over more frames means a smaller step per frame, hence smaller v/a/jerk. Zeroing the
        # features past the mask is what makes this padded window score identically to the prefix
        # alone (encode_mu_masked). A linear ramp beats a sigmoid taper here, measured.
        q = _sample(knots, tau, frames[None] / span[:, None])
        mask = (span[:, None, None] - frames[None, None]).clamp(0.0, 1.0)
        loss = scorer.maha2(scorer.features(q) * mask, mask, target).sum()

        # Boundary speeds, read one 15 Hz frame in from each end off the same clock. Measuring at the
        # executor timestep instead costs an extra `dt / duration` branch and scored no better.
        zero = torch.zeros_like(span)
        ends = _sample(knots, tau,
                       torch.stack([zero, zero + 1.0, span - 1.0, span], dim=1) / span[:, None])
        bnd = q.new_zeros(())
        if lead_speed > 0.0:
            bnd = bnd + (((ends[:, 1] - ends[:, 0]).norm(dim=-1) * _VAE_RATE_HZ - lead_speed) ** 2).sum()
        if trail_speed > 0.0:
            bnd = bnd + (((ends[:, 3] - ends[:, 2]).norm(dim=-1) * _VAE_RATE_HZ - trail_speed) ** 2).sum()

        (loss + _BOUNDARY_WEIGHT * bnd).backward()
        opt.step()
        with torch.no_grad():                       # projection: the box, without a squash
            span.clamp_(s_lo, s_hi)

    with torch.no_grad():
        spans = span.detach().cpu().numpy()
        tau = _time_knots(theta)
    return [(float(spans[b] / _VAE_RATE_HZ), tau[b : b + 1].detach()) for b in range(_N_STARTS)]


def _emit_raw(q_knots, tau, duration, dt, vel_cap_np, acc_cap_np):
    """Resample one optimized stroke at the control timestep. -> (pos, vel, acc, over).

    ``over`` is the factor the DURATION would have to grow by for this stroke to fit inside the caps;
    <= 1 means it already does. The geometry is fixed and only the clock changes, so spreading the
    same path over ``k`` times the wall-clock divides velocity by ``k`` and acceleration by ``k^2`` --
    hence ``max(v/v_cap, sqrt(a/a_cap))``, and hence a long enough duration ALWAYS exists. See
    :func:`_stretch_to_caps`, which is what makes that guarantee good.

    This is the ONLY thing enforcing the robot's limits. An in-loop hinge was tried and removed: it
    scored strictly worse (6.62 -> 6.34 and 13.12 -> 12.37 total maha^2 without it) and changed how
    many starts survive this test by less than one.
    """
    n_out = max(3, int(round(duration / dt)) + 1)
    dt_out = duration / (n_out - 1)
    frac = torch.linspace(0.0, 1.0, n_out, device=q_knots.device, dtype=q_knots.dtype)[None]
    with torch.no_grad():
        pos = _sample(q_knots, tau, frac)[0].cpu().numpy().astype(np.float64)
    vel = np.gradient(pos, dt_out, axis=0)
    acc = np.gradient(vel, dt_out, axis=0)
    over = max(
        float(np.max(np.abs(vel) / vel_cap_np)),
        float(np.sqrt(np.max(np.abs(acc) / acc_cap_np))),
    )
    return pos, vel, acc, over


def _stretch_to_caps(q_knots, tau, duration, dt, vel_cap_np, acc_cap_np):
    """Slow one candidate stroke down until it fits inside the caps. -> (pos, vel, acc, duration).

    The last resort when no duration inside the searched range produced an admissible stroke -- an
    optimizer that lands on a peaked time-warp can want a dash the robot cannot make in twice the
    planner's wall-clock. Resampling makes ``over`` only approximately the exact scale factor (the
    frame count changes with the duration), so this iterates on the measured value rather than
    trusting one step of it; each pass overshoots by 2% so it converges from above rather than
    creeping up on the cap. Two or three passes in practice.

    Returning a slow stroke matters more than it looks: the alternative is an exception, and
    ``blend_cutamp_plan`` answers an exception by passing that operation's ORIGINAL cuRobo segments
    through at the plan's time-dilation factor -- untouched by any of these caps, and by far the most
    aggressive motion in the episode.
    """
    for _ in range(_STRETCH_TRIES):
        pos, vel, acc, over = _emit_raw(q_knots, tau, duration, dt, vel_cap_np, acc_cap_np)
        if over <= 1.0:
            return pos, vel, acc, duration
        duration *= over * 1.02
    raise RuntimeError(
        f"VAE re-timing could not fit a stroke inside the velocity/acceleration caps even at "
        f"{duration:.2f}s over {_STRETCH_TRIES} passes (last overshoot {over:.2f}x)"
    )


def vae_retime_group(
    positions: np.ndarray,
    dt: float,
    orig_duration: float,
    vel_cap: np.ndarray,
    acc_cap: np.ndarray,
    smoothing: float,
    lead_speed: float,
    trail_speed: float,
    max_duration_mult: float,
    checkpoint_path: str | None = None,
    sample_target: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Re-time one joined stroke so its motion sits as close as possible to the DROID manifold.

    Same contract as :func:`trajectory_blending.blend_group` -- (pos, vel, acc, dt_out) -- so it drops
    into the same call site. ``orig_duration`` is the stroke's cuRobo wall-clock, used only to bound
    the duration range; unlike the spline/flow laws it is NOT a target, because the point here is to
    let the cost pick the pace.

    Geometry is the planner's, re-parameterized by arc length and smoothed with the shared
    ``blend_smoothing`` spline (``smoothing`` = 0 keeps the exact planner polyline, hence its exact
    collision status, at the price of the polyline's corner accelerations).
    """
    from tiptop.trajectory_blending import _dedup_path, _eval_geometry, _fit_geometry

    t_start = time.perf_counter()
    pos = _dedup_path(np.asarray(positions, dtype=np.float64))
    if len(pos) < 3:
        raise ValueError(f"VAE re-timing needs at least 3 distinct waypoints, got {len(pos)}")
    dof = pos.shape[1]

    # Arc-length canvas: the same curve, sampled uniformly in distance rather than in cuRobo's time.
    # Index sampling would carry cuRobo's stop at every interior leg join into the stroke, leaving
    # theta able only to rescale a profile that already comes to rest twice in the middle.
    chord = np.linalg.norm(np.diff(pos, axis=0), axis=1)
    u = np.concatenate([[0.0], np.cumsum(chord)])
    geom = _fit_geometry(u, pos, smoothing)
    knots_np = _eval_geometry(geom, np.linspace(0.0, u[-1], _KNOTS), 0)
    # The spline is fit through, not to, the endpoints; pin them so the stroke still starts and ends
    # exactly where cuTAMP's plan says (the next step's start position and any gripper action depend
    # on it).
    knots_np[0] = pos[0]
    knots_np[-1] = pos[-1]

    if dof != 7:
        # The VAE encodes a 28-D [q|v|a|j] metric for 7 joints; there is no meaningful score for a
        # 12-DOF bimanual chain, and silently scoring its first 7 columns would re-time BOTH arms
        # from one arm's motion. blend_cutamp_plan catches this per stroke and keeps the original
        # segments, so an unsupported embodiment degrades to "no re-timing" rather than to bad timing.
        raise ValueError(
            f"VAE stroke re-timing is 7-DOF only (the VAE was trained on 7-DOF Franka joint "
            f"metrics); this plan has dof={dof}"
        )
    scorer = _scorer(checkpoint_path, dof)
    q_knots = torch.as_tensor(knots_np[None], device=scorer.device, dtype=torch.float32)

    d_lo = max(_D_ABS_LO, _D_LO_MULT * orig_duration)
    d_hi = min(_D_ABS_HI, max_duration_mult * orig_duration)
    if d_hi <= d_lo:
        d_hi = d_lo * 1.5

    # One target latent for this stroke, shared by all _N_STARTS candidates and by the ranking below.
    # None (the default) reproduces the original mode-seeking objective exactly.
    #
    # A sampled target can ask for a stroke no admissible clock realizes -- it is a draw from the
    # human cluster, not a feasibility statement, and _emit_raw rejects anything past the robot's caps
    # (measured 1 stroke in 10 at _TARGET_SCALE 1.0). Falling back to the mean target is strictly
    # better than losing the plan: that stroke keeps the old, over-typical timing while every other
    # stroke keeps its sampled one.
    def _best_for(target):
        """-> (best admissible candidate or None, cheapest candidate to SLOW into the caps).

        The second is the material for the fallback below, and is tracked here because it is only
        available where the optimizer's candidates are: ranked by ``duration * over``, the wall-clock
        each one would end up at once slowed, so the fallback stretches the candidate that needs the
        least stretching rather than whichever happened to come last.
        """
        best = None
        relax = None
        for duration, tau in _optimize(scorer, q_knots, d_lo, d_hi, lead_speed, trail_speed, target):
            pos, vel, acc, over = _emit_raw(q_knots, tau, duration, dt, np.abs(vel_cap), np.abs(acc_cap))
            if over > 1.0:
                stretched = duration * over
                if relax is None or stretched < relax[0]:
                    relax = (stretched, duration, tau)
                continue
            # Ranked against the SAME target the optimizer used. Ranking by distance to the cluster
            # mean instead would re-select the most typical candidate and undo the sampling entirely.
            m2 = scorer.score_emitted(pos, duration, target)
            if best is None or m2 < best[0]:
                best = (m2, duration, (pos, vel, acc))
        return best, relax

    scales = _TARGET_SCALES if sample_target else (0.0,)
    best, relax, target = None, None, None
    for scale in scales:
        target = target_latent(scorer, knots_np, sample_target, scale)
        best, scale_relax = _best_for(target)
        # Keep the cheapest stretch seen across scales; the last scale is the mean target, whose
        # candidates are the most typical, so its relax is normally also the mildest.
        if scale_relax is not None and (relax is None or scale_relax[0] < relax[0]):
            relax = scale_relax
        if best is not None:
            if scale not in (scales[0], 0.0):
                _log.info("VAE re-timing: sampled target shrunk to %.1fx to meet the caps", scale)
            break
        if scale > 0.0:
            _log.debug("VAE re-timing: target scale %.1f had no admissible clock; shrinking", scale)

    if best is None:
        # No clock inside [d_lo, d_hi] fits the caps. Slow the mildest candidate past d_hi until it
        # does, rather than raising: an exception here is not "no re-timing", it is
        # blend_cutamp_plan falling back to this operation's raw cuRobo segments at the plan's
        # time-dilation factor -- the fastest motion in the episode, and the one nothing has checked
        # against the caps. A stroke slower than max_duration_mult asked for is the lesser cost, and
        # it is announced loudly enough to be tuned (raise blend_max_duration_mult, or the slacks).
        if relax is None:
            raise RuntimeError(
                f"VAE re-timing produced no candidate clock at all in [{d_lo:.2f}, {d_hi:.2f}]s"
            )
        _, duration, tau = relax
        out_pos, out_vel, out_acc, duration = _stretch_to_caps(
            q_knots, tau, duration, dt, np.abs(vel_cap), np.abs(acc_cap)
        )
        _log.warning(
            "VAE re-timing found no clock in [%.2f, %.2f]s within the velocity/acceleration caps; "
            "slowed this stroke to %.2fs (%.1fx the planner's %.2fs) to stay inside them. Raise "
            "blend_max_duration_mult above %.1f if strokes like this should be searched, not slowed.",
            d_lo, d_hi, duration, duration / max(orig_duration, 1e-6), orig_duration, max_duration_mult,
        )
        from tiptop.trajectory_blending import _finish_stroke

        return _finish_stroke(out_pos, out_vel, out_acc, duration, lead_speed, trail_speed)

    m2, duration, (out_pos, out_vel, out_acc) = best
    _log.debug(
        "VAE stroke re-timing: %.2f s -> %.2f s (%s %.2f, %d waypoints, %.1f s to fit)",
        orig_duration, duration, "d2(sampled target)" if target is not None else "maha2",
        m2, len(out_pos), time.perf_counter() - t_start,
    )
    from tiptop.trajectory_blending import _finish_stroke

    return _finish_stroke(out_pos, out_vel, out_acc, duration, lead_speed, trail_speed)

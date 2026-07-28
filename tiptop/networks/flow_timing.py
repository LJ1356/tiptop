"""Conditional flow-matching blender: generate a whole human stroke, no analytic time law or spline.

Where the (removed) deterministic timing regressor learned only a speed profile -- and, being MSE-trained,
collapsed the different teleoperator styles to their mean -- this module learns the ENTIRE blend
generatively, so
sampling different noise yields different human-like realizations of the same path (decelerate-into-grasp
vs constant-velocity), expressing the multimodality a regressor averages away.

Representation (endpoint-anchored deviation)
--------------------------------------------
A stroke is ``x(tau)`` = start-centered joint positions over normalized TIME, ``x[0]=0`` and ``x[-1]=e``
(the net displacement / endpoint). We generate the DEVIATION from the straight-line motion between the
two endpoints:

    base(tau) = tau * e            (constant-velocity straight line, 0 -> e)
    delta     = x - base           (delta[0] = delta[-1] = 0 by construction)

The flow models ``delta``; at sampling we add ``base`` back and pin ``x[0]=0, x[-1]=e`` EXACTLY. This
hard-anchors both endpoints (no grasp overshoot), makes the target smaller/smoother, and -- crucially --
gives the model a fixed endpoint to decelerate into (the earlier position-space model captured accel from
the anchored start but not decel into the un-anchored end). The conditioning is the timing-stripped PATH
``c`` (positions over ARC LENGTH) plus the endpoint ``e`` -- conditioning on an arc-length path cannot leak
the target timing -- and the (start, end) GRIPPER-EVENT KINDS ``k``.

Why the event kinds
-------------------
A stroke's boundaries ARE gripper events (``extract_stroke_flow`` cuts there), so ``x``'s speed at tau=0
and tau=1 is its speed at a release and at a grasp. DROID decelerates much harder into a grasp than into
a release -- on held-out episodes the boundary sits at 0.42 of the stroke's own mean speed for a close vs
0.76 for an open, a 1.8x ratio. Conditioned only on ``(c, e)`` the net cannot tell the two apart where
their paths look alike, and averages them: an otherwise identical model trained without ``k`` reaches
only 78% of that ratio, against 96% with it (``eval_transitions.py``).

Because the path is a strong cue, the net will lean on it and ignore ``k`` on out-of-distribution
geometry -- on cuTAMP plan paths the label alone moved the boundary not at all. Training therefore drops
``k`` to ``KIND_NULL`` at random (``--kind-dropout``) so the unconditioned field is learned too, and
sampling applies classifier-free guidance (``guidance``, baked into the checkpoint) to amplify what the
label contributes. Guidance 2.0 is what the shipped checkpoint carries.

Architecture
------------
``TrajFlow`` is a small 1-D temporal U-Net over the time axis (T -> T/2 -> T/4 -> T/2 -> T with skip
connections), FiLM-conditioned on ``(encode(c), e, embed(k), timestep(t))``. The temporal convolutions
give the smoothness inductive bias the earlier MLP-over-flattened-positions lacked (its samples were ~5x
jerkier than human).

Training uses conditional (rectified) flow matching on ``delta``: draw ``d0 ~ N(0,I)``, ``d1 = data``,
``t ~ U(0,1)``, ``dt = (1-t) d0 + t d1``, regress ``v(dt, t, c, e, k)`` onto ``d1 - d0``, with the
per-timestep weighting of :func:`boundary_weights` so the ends are not under-fit. Sampling integrates
``dx/dt = v`` from Gaussian noise over ``t: 0->1``. Training: ``train_flow.py``; extraction:
``extract_stroke_flow.py``; transition scoring: ``eval_transitions.py``. NOTE: this is the generative
model; wiring its samples into a cuTAMP plan (vel/accel caps, non-idle) is a separate integration step --
and ``blend_boundary_speed`` there will overwrite the learned approach if set above it.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_log = logging.getLogger(__name__)

# Timesteps per generated stroke. This is a NORMALIZED-time grid, so its resolution in seconds
# depends on the stroke's duration: at T=32 a 8 s stroke is described at 4 Hz, and DROID's speed dip
# around a gripper event is a +-0.53 s feature -- about two samples wide, too coarse to reproduce.
# T=64 puts our 6-9 s operation groups at 7-10 Hz, close to the 15 Hz command rate. Checkpoints carry
# their own ``t_time`` in meta, so ``FlowModel`` still loads older T=32 ones.
T_TIME = 64
S_ARC = 32
DOF = 7

# Gripper-event kind at a stroke boundary. A stroke runs BETWEEN two events, so each one carries a
# (start, end) pair. "none" is an episode start/end, where the arm is genuinely at rest.
KIND = {"none": 0, "close": 1, "open": 2}
# A 4th slot the data never uses: training drops the label to it at random so the net must also model the
# UNCONDITIONED stroke. That makes classifier-free guidance available at sampling, which is what carries
# the grasp/release distinction onto cuTAMP paths -- on those out-of-distribution geometries the net
# otherwise reads the path and ignores the label entirely (measured open/close ratio 1.00 vs DROID 1.80).
KIND_NULL = 3
N_KINDS = 4

_PKG_DIR = Path(__file__).resolve().parents[1]
# The shipped checkpoint. T=64: the earlier T=32 one described DROID's +-0.53 s notch with about two
# samples on a planner-length stroke, so the approach it generated was a straight ramp.
DEFAULT_CKPT = _PKG_DIR / "checkpoints" / "flow_net_t64.pt"


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal embedding of the flow time t in [0,1] -> [B, dim]."""
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device, dtype=t.dtype) / max(half - 1, 1))
    args = t[:, None] * freqs[None] * 100.0
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class _FiLM(nn.Module):
    """Per-channel scale/shift of a [B,C,L] feature map from a conditioning vector [B,cond_dim]."""

    def __init__(self, cond_dim: int, ch: int):
        super().__init__()
        self.lin = nn.Linear(cond_dim, 2 * ch)

    def forward(self, h: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        s, b = self.lin(g).chunk(2, dim=-1)
        return h * (1.0 + s[..., None]) + b[..., None]


class _ResBlock(nn.Module):
    """Two 1-D convs with a FiLM in between and a residual connection."""

    def __init__(self, cin: int, cout: int, cond_dim: int, k: int = 5):
        super().__init__()
        self.conv1 = nn.Conv1d(cin, cout, k, padding=k // 2)
        self.film = _FiLM(cond_dim, cout)
        self.conv2 = nn.Conv1d(cout, cout, k, padding=k // 2)
        self.skip = nn.Conv1d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.film(self.conv1(x), g))
        h = F.silu(self.conv2(h))
        return h + self.skip(x)


class TrajFlow(nn.Module):
    """1-D temporal U-Net velocity field v(delta_t, t, c, e, k) for conditional flow matching.

    ``k`` is the [start, end] gripper-event kind (see ``extract_stroke_flow.KIND``). Without it the path
    ``c`` and endpoint ``e`` are the only cues for how to enter a boundary, and since a grasp and a
    release can share a path shape the model averages the two -- it reproduced only ~1.25x of DROID's
    1.45x open/close speed ratio at the boundary. ``n_kinds=0`` restores the original unconditioned net
    so checkpoints trained before this existed still load.
    """

    def __init__(self, dof: int = DOF, cond_dim: int = 128, t_dim: int = 64, base_ch: int = 64,
                 n_kinds: int = 0, k_dim: int = 16):
        super().__init__()
        self.t_dim = t_dim
        self.n_kinds = n_kinds
        ch0, ch1 = base_ch, base_ch * 2
        # Condition encoder over the path c [B,dof,S] -> global vector.
        self.c_enc = nn.Sequential(
            nn.Conv1d(dof, 64, 5, padding=2), nn.SiLU(),
            nn.Conv1d(64, 64, 5, padding=2), nn.SiLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        # Separate embeddings for the two ends: entering a grasp and leaving one are different motions.
        k_total = 0
        if n_kinds:
            self.k_start = nn.Embedding(n_kinds, k_dim)
            self.k_end = nn.Embedding(n_kinds, k_dim)
            k_total = 2 * k_dim
        self.g_mlp = nn.Sequential(nn.Linear(64 + dof + t_dim + k_total, cond_dim), nn.SiLU(),
                                   nn.Linear(cond_dim, cond_dim))
        # U-Net over the time axis.
        self.in_conv = nn.Conv1d(dof, ch0, 5, padding=2)
        self.d1 = _ResBlock(ch0, ch0, cond_dim)
        self.down1 = nn.Conv1d(ch0, ch0, 4, stride=2, padding=1)
        self.d2 = _ResBlock(ch0, ch1, cond_dim)
        self.down2 = nn.Conv1d(ch1, ch1, 4, stride=2, padding=1)
        self.mid = _ResBlock(ch1, ch1, cond_dim)
        self.up2 = nn.ConvTranspose1d(ch1, ch1, 4, stride=2, padding=1)
        self.u2 = _ResBlock(ch1 + ch1, ch1, cond_dim)
        self.up1 = nn.ConvTranspose1d(ch1, ch0, 4, stride=2, padding=1)
        self.u1 = _ResBlock(ch0 + ch0, ch0, cond_dim)
        self.out_conv = nn.Conv1d(ch0, dof, 5, padding=2)

    def forward(self, delta_t: torch.Tensor, t: torch.Tensor, c: torch.Tensor, e: torch.Tensor,
                k: torch.Tensor | None = None) -> torch.Tensor:
        xt = delta_t.transpose(1, 2)          # [B,dof,T]
        cg = self.c_enc(c.transpose(1, 2)).squeeze(-1)  # [B,64]
        parts = [cg, e, timestep_embedding(t, self.t_dim)]
        if self.n_kinds:
            if k is None:
                raise ValueError("this TrajFlow was built with event-kind conditioning; k [B,2] is required")
            parts += [self.k_start(k[:, 0]), self.k_end(k[:, 1])]
        g = self.g_mlp(torch.cat(parts, dim=-1))
        h = self.in_conv(xt)
        h1 = self.d1(h, g)
        h2 = self.d2(self.down1(h1), g)
        h = self.mid(self.down2(h2), g)
        h = self.u2(torch.cat([self.up2(h), h2], dim=1), g)
        h = self.u1(torch.cat([self.up1(h), h1], dim=1), g)
        return self.out_conv(h).transpose(1, 2)  # [B,T,dof]


def augment_path(c: np.ndarray, sigma: float, lam: float) -> np.ndarray:
    """Smooth and straighten an arc-length path [S,dof], keeping its endpoints (and so ``e``) fixed.

    Planner paths are geometrically much cleaner than human ones: on matched strokes, cuTAMP sits at
    0.70x DROID's turning-per-radian and 1.37x its straightness. Conditioned only on real DROID paths
    the net keys on that human wiggle and, handed a planner path, falls back on the geometry and ignores
    the event label entirely. Randomising the condition over this range during training is what keeps the
    grasp/release distinction alive on cuTAMP geometry.
    """
    out = np.asarray(c, np.float64)
    if sigma > 0:
        from scipy.ndimage import gaussian_filter1d
        sm = gaussian_filter1d(out, sigma, axis=0, mode="nearest")
        sm[0], sm[-1] = out[0], out[-1]
        out = sm
    if lam > 0:
        tau = np.linspace(0.0, 1.0, len(out))[:, None]
        out = (1.0 - lam) * out + lam * (out[0][None] + tau * (out[-1] - out[0])[None])
    step = np.linalg.norm(np.diff(out, axis=0), axis=1)
    arc = step.sum()
    if arc < 1e-9:                       # degenerate: leave the path alone
        return np.asarray(c, np.float64)
    u = np.concatenate([[0.0], np.cumsum(step)])
    dst = np.linspace(0.0, arc, len(out))
    return np.stack([np.interp(dst, u, out[:, j]) for j in range(out.shape[1])], axis=1)


def boundary_weights(n: int, strength: float, scale: float = 0.1) -> torch.Tensor:
    """Per-timestep loss weights [n] that emphasise the two stroke ends. ``strength=0`` -> all ones.

    ``delta`` is pinned to zero at both ends, so the model has the least freedom exactly where the
    boundary speed is decided (that speed is set by the SECOND sample in from each end). Unweighted, the
    ends are 2 of 32 timesteps and get under-fit: samples arrived at a grasp ~19% too fast. The weight
    decays over ``scale`` of the stroke, so it lifts the approach, not the cruise.
    """
    tau = torch.linspace(0.0, 1.0, n)
    return 1.0 + strength * (torch.exp(-tau / scale) + torch.exp(-(1.0 - tau) / scale))


def cfm_loss(net: TrajFlow, d1: torch.Tensor, c: torch.Tensor, e: torch.Tensor,
             k: torch.Tensor | None = None, w: torch.Tensor | None = None) -> torch.Tensor:
    """Conditional flow-matching loss on the endpoint-anchored deviation delta.

    ``w`` is an optional [T] per-timestep weight (see :func:`boundary_weights`); it is normalised to
    mean 1 so the loss stays comparable across weightings.
    """
    d0 = torch.randn_like(d1)
    t = torch.rand(d1.shape[0], device=d1.device)
    dt = (1.0 - t)[:, None, None] * d0 + t[:, None, None] * d1
    err = (net(dt, t, c, e, k) - (d1 - d0)) ** 2
    if w is None:
        return err.mean()
    return (err * (w / w.mean())[None, :, None]).mean()


def _anchor(delta: np.ndarray, tau: np.ndarray) -> np.ndarray:
    """Force ``delta[:, 0] = delta[:, -1] = 0`` by subtracting the ramp between its own endpoints.

    In the DATA both ends of ``delta`` are exactly zero; a sample only lands NEAR zero. Overwriting
    the endpoint POSITIONS (``x[:, 0] = 0; x[:, -1] = e``) also pins the ends, but it dumps the whole
    endpoint error into the first and last INTERVAL -- and those two intervals are what set the
    boundary speed the blender reads off the sample. Subtracting a straight ramp instead spreads the
    same correction over the stroke as a constant-velocity offset, so the approach into a grasp stays
    the model's own.
    """
    return delta - ((1.0 - tau) * delta[:, :1] + tau * delta[:, -1:])


def _resample_arc(q: np.ndarray, n: int) -> np.ndarray | None:
    step = np.linalg.norm(np.diff(np.asarray(q, np.float64), axis=0), axis=1)
    arc = step.sum()
    if arc < 1e-6:
        return None
    u = np.concatenate([[0.0], np.cumsum(step)])
    dst = np.linspace(0.0, arc, n)
    return np.stack([np.interp(dst, u, q[:, j]) for j in range(q.shape[1])], axis=1)


class FlowModel:
    """Plan-time wrapper: load a checkpoint and SAMPLE endpoint-anchored human-like strokes given a path."""

    def __init__(self, ckpt_path: str | Path | None = None, device: str = "cpu"):
        path = Path(ckpt_path) if ckpt_path else DEFAULT_CKPT
        if not path.is_absolute():
            path = (_PKG_DIR / path).resolve()
        blob = torch.load(path, map_location=device, weights_only=False)
        meta = blob["meta"]
        self.meta = meta
        self.device = device
        self.T, self.S, self.dof = int(meta["t_time"]), int(meta["s_arc"]), int(meta["dof"])
        self.c_std, self.delta_std, self.e_std = float(meta["c_std"]), float(meta["delta_std"]), float(meta["e_std"])
        self.n_kinds = int(meta.get("n_kinds", 0))          # 0 = a checkpoint from before kind conditioning
        self.guidance = float(meta.get("guidance", 1.0))    # default CFG scale baked in at training time
        self.net = TrajFlow(self.dof, meta["cond_dim"], meta["t_dim"], meta["base_ch"],
                            self.n_kinds, int(meta.get("k_dim", 16)))
        self.net.load_state_dict(blob["state_dict"])
        self.net.to(device).eval()
        self._tau = np.linspace(0.0, 1.0, self.T)[:, None]
        _log.info("Loaded flow model %s (T=%d S=%d dof=%d kinds=%d)", path.name, self.T, self.S,
                  self.dof, self.n_kinds)

    def _guided(self, d, t, c, e, k, k_null, guidance: float):
        """Velocity field, extrapolated away from the label-free prediction (classifier-free guidance).

        ``guidance=1`` is the plain conditional field. Higher values amplify what the event label
        contributes, which is what makes the grasp/release distinction survive on cuTAMP paths.
        """
        v = self.net(d, t, c, e, k)
        if k is None or guidance == 1.0:
            return v
        v_null = self.net(d, t, c, e, k_null)
        return v_null + guidance * (v - v_null)

    def _kinds(self, kinds, n: int) -> torch.Tensor | None:
        """[start, end] kind codes (or names) -> [n,2] long tensor; None when the net is unconditioned."""
        if not self.n_kinds:
            return None
        if kinds is None:
            kinds = ("none", "none")
        pair = [KIND[k] if isinstance(k, str) else int(k) for k in kinds]
        return torch.as_tensor(pair, device=self.device, dtype=torch.long)[None].repeat(n, 1)

    def _cond(self, path_positions: np.ndarray):
        """Raw stroke path [N,7] -> (c_norm [S,dof], e [dof]) or None if degenerate."""
        c = _resample_arc(np.asarray(path_positions, np.float64), self.S)
        if c is None:
            return None
        c = c - c[0:1]
        return (c / self.c_std).astype(np.float32), c[-1].astype(np.float32)  # e = net displacement

    @torch.no_grad()
    def sample(self, path_positions: np.ndarray, n_samples: int = 1, n_steps: int = 60,
               kinds: tuple = None, guidance: float = None) -> np.ndarray | None:
        """Sample ``n_samples`` strokes for a raw path -> [n_samples, T, dof] (start-centered, endpoint-pinned).

        ``kinds`` is the (start, end) gripper-event kind this stroke runs between -- "none" / "close" /
        "open" -- and controls how the sample decelerates into its boundaries.
        """
        cond = self._cond(path_positions)
        if cond is None:
            return None
        c_norm, e = cond
        c = torch.as_tensor(c_norm[None], device=self.device).repeat(n_samples, 1, 1)
        e_t = torch.as_tensor((e / self.e_std)[None], device=self.device).repeat(n_samples, 1)
        k_t = self._kinds(kinds, n_samples)
        guidance = self.guidance if guidance is None else float(guidance)
        k_null = None if k_t is None else torch.full_like(k_t, KIND_NULL)
        d = torch.randn(n_samples, self.T, self.dof, device=self.device)
        ts = torch.linspace(0.0, 1.0, n_steps + 1, device=self.device)
        for i in range(n_steps):
            v = self._guided(d, ts[i].repeat(n_samples), c, e_t, k_t, k_null, guidance)
            d = d + v * (ts[i + 1] - ts[i])
        delta = d.cpu().numpy() * self.delta_std
        return self._tau[None] * e[None, None, :] + _anchor(delta, self._tau[None])

    @torch.no_grad()
    def sample_batch(self, arc_paths: np.ndarray, n_steps: int = 60, kinds: np.ndarray = None,
                     guidance: float = None) -> np.ndarray:
        """One trajectory for each of B distinct arc-length paths [B,S,7] -> [B,T,dof] (endpoint-pinned).

        ``kinds`` is an optional [B,2] array of (start, end) event-kind codes, one row per path.
        """
        c = np.asarray(arc_paths, np.float64)
        c = c - c[:, 0:1]
        e = c[:, -1].astype(np.float32)                      # [B,7] endpoints
        b = len(c)
        ct = torch.as_tensor((c / self.c_std).astype(np.float32), device=self.device)
        et = torch.as_tensor((e / self.e_std).astype(np.float32), device=self.device)
        kt = None
        if self.n_kinds:
            kk = np.zeros((b, 2), np.int64) if kinds is None else np.asarray(kinds, np.int64)
            kt = torch.as_tensor(kk, device=self.device)
        guidance = self.guidance if guidance is None else float(guidance)
        k_null = None if kt is None else torch.full_like(kt, KIND_NULL)
        d = torch.randn(b, self.T, self.dof, device=self.device)
        ts = torch.linspace(0.0, 1.0, n_steps + 1, device=self.device)
        for i in range(n_steps):
            v = self._guided(d, ts[i].repeat(b), ct, et, kt, k_null, guidance)
            d = d + v * (ts[i + 1] - ts[i])
        delta = d.cpu().numpy() * self.delta_std
        return self._tau[None] * e[:, None, :] + _anchor(delta, self._tau[None])

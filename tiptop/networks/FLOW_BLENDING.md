# Flow-matching trajectory blending

A generative replacement for the hand-tuned trajectory blend: instead of imposing one analytic (or one
deterministic learned) time law, it **samples a full human-like stroke** from a conditional flow-matching
model trained on DROID teleop — so every generated episode draws a different, plausible human timing.

**Code:** `flow_timing.py` (model + `FlowModel`), `extract_stroke_flow.py` (data), `train_flow.py`,
`eval_flow.py`, and `../flow_blending.py` (the `blend_mode: flow` backend). Checkpoint: `../checkpoints/flow_net_t64.pt`.

---

## 1. Why we need it

A cuTAMP plan is a sequence of independently-planned reaching segments separated by gripper open/close
steps. Each segment accelerates from and decelerates back to rest, so the concatenated plan is **stop-and-go**:
near-zero joint velocity at every pick/place, high velocity through free space. That velocity signature is
unlike human teleop and hurts a velocity-command VLA policy.

The evolution of fixes:

1. **Analytic spline blend** (`../trajectory_blending.py`) — smooths the segments into one continuous stroke
   with a min-jerk / asymmetric quintic time law. Fixes the discontinuity, but the *timing* is an
   **arbitrary hand-chosen** profile, not measured from humans.
2. **Deterministic learned timing** (removed) — predicted a human speed profile from geometry. Better, but
   it was MSE-trained, so it learned the **conditional mean**. Different teleoperators time the same motion
   differently (some *decelerate into a grasp*, some hold *constant velocity*); the mean of those is a
   blurred profile **neither operator produces**, and it strips the diversity that makes imitation data
   useful. Superseded by (3), and deleted along with its network and training scripts.
3. **Flow matching (this)** — a *generative* model of the human stroke distribution. Sampling reproduces the
   **spread of teleoperator styles**, not their mean. On held-out DROID the end-speed-ratio distribution
   matches to Wasserstein-1 **0.055** (deterministic-mean baseline: 0.48), with 64% of samples decelerating
   into the endpoint vs DROID's 67% — i.e. it recovers both modes.

---

## 2. How it works

Per **operation group** (the trajectory segments between two gripper events):

1. **Condition** on the group's path (timing-stripped) and its endpoint.
2. **Sample** a full stroke by integrating the flow ODE from Gaussian noise (`t: 0 → 1`).
3. **Constrain**: feed the sample's geometry + its own speed profile into the shared engine
   `flow_blending._retime_stroke`, which enforces the FR3 vel/accel caps, pins rest ends to zero, lifts
   gripper-adjacent ends to the non-idle boundary speed, pins the endpoint, and resamples to the plan `dt`.

So the model supplies the (multimodal, human) geometry + timing; the hard constraints stay analytic and
guaranteed.

**Representation — endpoint-anchored deviation.** A stroke `x(τ)` is joint positions over normalized time,
start-centered (`x[0]=0`) and ending at the net displacement `x[-1]=e`. We model the **deviation from the
straight line** to the endpoint:

```
base(τ) = τ · e            # constant-velocity line 0 → e
δ       = x − base         # δ[0] = δ[-1] = 0 by construction
```

The flow generates `δ`; at sampling we add `base` back and pin `x[0]=0, x[-1]=e` **exactly**. This hard-anchors
both endpoints (no grasp overshoot), shrinks/​smooths the target, and gives the model a fixed endpoint to
decelerate into.

**Architecture (`TrajFlow`).** A 1-D **temporal U-Net** over the time axis (T: 32→16→8→16→32 with skip
connections), FiLM-conditioned on `encode(c) ⊕ e ⊕ timestep(t)`. The temporal convolutions supply the
smoothness prior a flat MLP lacked (early MLP samples were ~5× jerkier than human; the U-Net is ~0.8×,
i.e. as smooth as human data). ~1.06M params.

---

## 3. Inputs and outputs of the network

**Inputs** — `TrajFlow(δ_t, t, c, e)`:

| symbol | shape | meaning |
|---|---|---|
| `δ_t` | [B, 32, 7] | flow ODE **state** — the current (noisy) deviation being transported |
| `t`   | [B] | **flow-matching time** ∈ [0,1] (the ODE variable — *not* physical time) |
| `c`   | [B, 32, 7] | **condition**: the path, positions over **arc length**, start-centered + standardized (timing-stripped, so it can't leak the target timing) |
| `e`   | [B, 7] | **condition**: the endpoint / net displacement, standardized |

**Output** — velocity field `v` [B, 32, 7]. To generate, integrate `dδ/dt = v(δ, t, c, e)` from `δ ~ N(0,I)`
over `t: 0→1`, then reconstruct the stroke `x = τ·e + δ` [32, 7] (joint positions over normalized time,
endpoints pinned). Multimodality comes from the initial noise: different draws → different human timings.

---

## 4. How we get the labels for training (`extract_stroke_flow.py` + `train_flow.py`)

1. **Stream DROID** `lerobot/droid_1.0.1` (never downloaded — column-projected parquet).
2. **Segment** every episode at its gripper events (Schmitt trigger + dwell + 4-frame lag, mirroring
   `analysis2`); the pieces between events (and episode start/end) are the strokes.
3. Per stroke, resample the joint path two ways, both start-centered:
   * `x` [32, 7] — uniform in **time** (the generation target, has geometry **and** timing);
   * `c` [32, 7] — uniform in **arc length** (the condition, timing-free).
4. `e = x[-1]`; `base = τ·e`; the **data target is the deviation** `d₁ = δ = x − base`. Standardize `δ, c, e`.
5. **Flow-matching loss** (rectified / conditional FM), per sample: draw noise `d₀ ~ N(0,I)`, `t ~ U(0,1)`,
   interpolate `δ_t = (1−t)·d₀ + t·d₁`; the regression **target is `d₁ − d₀`** (straight-line velocity from
   noise to data):

   ```
   L = ‖ v(δ_t, t, c, e) − (d₁ − d₀) ‖²
   ```

   The loss plateaus at a positive value (the target has irreducible variance) — judge the model by its
   **samples** (`eval_flow.py`), not the loss.

Scale: 100,094 strokes (25 DROID files); trained on an RTX 5090 (see `train_flow.py`, ~4 min, CFM val 0.21).

---

## 5. How it's used for trajopt currently

It is a **pure post-process over the cuTAMP plan** (no planner change), run inside `run_planning` so the
saved and executed plans are the identical blended object. Enable per `cfg/tamp/*.yml`:

```yaml
tamp_overrides:
  blend_trajectory: true
  blend_mode: flow
  blend_model_path: "checkpoints/flow_net_t64.pt"   # optional; defaults to flow_net_t64.pt
  blend_flow_steps: 60                           # Euler ODE steps per sample
  blend_boundary_speed: 0.3                       # non-idle boundary at gripper events
  blend_ops: [Pick, Place, GoToInitial]
```

Dispatch path: `planning.run_planning` → `_apply_blend` (on `blend_mode == "flow"`) →
`flow_blending.flow_blend_cutamp_plan`:

- Gripper steps pass through untouched and **delimit** the operation groups (`config.ops` restricts which
  operations are blended).
- For each group (`flow_blend_group`): sample the flow conditioned on the group's joined path → `x_flow`;
  take **its** geometry + speed profile and run `flow_blending._retime_stroke` with the boundary speeds —
  **rest (0)** at the plan's start/end, **`boundary_speed`** at gripper events — plus the **FR3 vel/accel
  caps**, endpoint pinning, and resample to the control `dt`. Emit as the group's single trajectory step.
- **Robustness:** a per-stroke failure falls back to the analytic `blend_group` for that stroke; a model-load
  failure falls back to spline for the whole plan.

Because each `flow_blend_group` draws fresh noise, every operation of every episode gets an independent
human-like realization — so a dataset generated with `blend_mode: flow` reproduces the *distribution* of
teleoperator timings. Verified on the emitted plan: endpoints pinned (~0.01 rad), rest ends at 0,
gripper ends non-idle, vel/accel within caps, and multimodality survives the constraint pipeline
(6 draws on one path → end-ratios spanning 0.35–1.40).

**Caveats.**
- Uses the flow's **own** generated geometry by default. For collision-sensitive scenes, one line in
  `flow_blend_group` swaps in the cuTAMP collision-checked path (keeping only the flow's timing).
- Like the analytic blend, the boundary speed can land below the requested `blend_boundary_speed` (~0.19 vs
  0.3) when the caps or a collapsed grasp-tangent bind; it still clears the non-idle threshold (>0.02) and logs
  a warning.

### Related files
`flow_timing.py` · `extract_stroke_flow.py` · `train_flow.py` · `eval_flow.py` · `../flow_blending.py` ·
`../trajectory_blending.py` (spline, shared constraint engine)

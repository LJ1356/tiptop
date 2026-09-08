"""Tests for the velocity/acceleration cap guarantees in stroke re-timing.

The failure these pin down: ``vae_retime_group`` used to raise when no duration inside its searched
range met the caps, and ``blend_cutamp_plan`` answers an exception by passing that operation's raw
cuRobo segments through at the plan's time-dilation factor -- the fastest, least-checked motion in
the episode. Both halves now slow down instead.
"""

import numpy as np
import pytest
import torch

from tiptop.trajectory_blending import BlendConfig, _slow_to_caps
from tiptop.vae_retiming import _emit_raw, _stretch_to_caps, _time_knots

DOF = 7
DT = 0.02


def _stroke(knots=32, sweep=2.0, device="cpu"):
    """A straight sweep of every joint through ``sweep`` radians, as an arc-length knot canvas."""
    path = np.linspace(0.0, sweep, knots)[:, None].repeat(DOF, axis=1)
    return torch.as_tensor(path[None], dtype=torch.float32, device=device)


def _uniform_tau(knots=32, device="cpu"):
    """The even time-warp: each equal-distance interval gets an equal share of the clock."""
    return _time_knots(torch.zeros(1, knots - 1, device=device))


class TestEmitRaw:
    def test_over_is_one_at_the_cap(self):
        q_knots, tau = _stroke(), _uniform_tau()
        _, vel, acc, _ = _emit_raw(q_knots, tau, 4.0, DT, np.full(DOF, 1e3), np.full(DOF, 1e3))
        vel_cap = np.full(DOF, np.abs(vel).max())
        acc_cap = np.full(DOF, np.abs(acc).max())
        _, _, _, over = _emit_raw(q_knots, tau, 4.0, DT, vel_cap, acc_cap)
        assert over == pytest.approx(1.0, abs=1e-6)

    def test_over_scales_with_the_duration(self):
        q_knots, tau = _stroke(), _uniform_tau()
        caps = np.full(DOF, 1.0)
        _, _, _, slow = _emit_raw(q_knots, tau, 8.0, DT, caps, caps)
        _, _, _, fast = _emit_raw(q_knots, tau, 2.0, DT, caps, caps)
        # Four times the wall-clock is four times gentler. Velocity goes as 1/T (and acceleration
        # as 1/T^2, which only makes the ratio larger) -- that monotonicity is the whole reason a
        # long enough duration always exists, so it is worth stating as a law rather than a bound.
        assert fast / slow == pytest.approx(4.0, rel=1e-3)


class TestStretchToCaps:
    def test_slows_a_violating_stroke_until_it_fits(self):
        q_knots, tau = _stroke(sweep=3.0), _uniform_tau()
        vel_cap, acc_cap = np.full(DOF, 0.5), np.full(DOF, 2.0)
        # The stroke is far too fast at the requested duration.
        _, _, _, over = _emit_raw(q_knots, tau, 1.0, DT, vel_cap, acc_cap)
        assert over > 1.0

        pos, vel, acc, duration = _stretch_to_caps(q_knots, tau, 1.0, DT, vel_cap, acc_cap)
        assert duration > 1.0
        assert np.abs(vel).max() <= vel_cap.max()
        assert np.abs(acc).max() <= acc_cap.max()
        # Same geometry, just spread over more time: the endpoints are where they were.
        assert pos[0] == pytest.approx(q_knots[0, 0].numpy(), abs=1e-5)
        assert pos[-1] == pytest.approx(q_knots[0, -1].numpy(), abs=1e-5)

    def test_leaves_an_already_admissible_stroke_alone(self):
        q_knots, tau = _stroke(), _uniform_tau()
        caps = np.full(DOF, 1e3)
        _, _, _, duration = _stretch_to_caps(q_knots, tau, 3.0, DT, caps, caps)
        assert duration == 3.0

    def test_a_peaked_time_warp_still_converges(self):
        # The shape that produced the reported failure: an optimizer landing on a time-warp that
        # dashes through part of the path, so twice the planner's wall-clock is nowhere near enough.
        theta = torch.linspace(-3.0, 3.0, 31)[None]
        q_knots, tau = _stroke(), _time_knots(theta)
        vel_cap, acc_cap = np.full(DOF, 0.4), np.full(DOF, 1.0)
        _, vel, acc, duration = _stretch_to_caps(q_knots, tau, 1.5, DT, vel_cap, acc_cap)
        assert np.abs(vel).max() <= vel_cap.max()
        assert np.abs(acc).max() <= acc_cap.max()
        assert duration > 1.5


class _Plan:
    """The fields _slow_to_caps reads off a cuRobo JointState."""

    def __init__(self, position, velocity, acceleration, jerk):
        self.position = position
        self.velocity = velocity
        self.acceleration = acceleration
        self.jerk = jerk
        self.joint_names = [f"j{i}" for i in range(DOF)]


def _steps(peak_vel, n=50, dt=DT):
    t = torch.linspace(0.0, 1.0, n)[:, None].repeat(1, DOF)
    velocity = peak_vel * torch.sin(torch.pi * t)
    plan = _Plan(position=torch.cumsum(velocity * dt, dim=0), velocity=velocity,
                 acceleration=torch.zeros_like(velocity), jerk=torch.zeros_like(velocity))
    return [{"type": "trajectory", "plan": plan, "dt": dt, "label": "Place(x)"}]


class TestSlowToCaps:
    config = BlendConfig(enabled=True)

    def test_passes_admissible_segments_through_untouched(self):
        steps = _steps(peak_vel=0.5)
        out = _slow_to_caps(steps, self.config, np.full(DOF, 2.0), np.full(DOF, 10.0))
        assert out is steps

    def test_scales_the_clock_and_the_profile_together(self):
        steps = _steps(peak_vel=4.0)
        vel_limit, acc_limit = np.full(DOF, 1.0), np.full(DOF, 1e3)
        out = _slow_to_caps(steps, self.config, vel_limit, acc_limit)
        assert out is not steps

        scale = out[0]["dt"] / steps[0]["dt"]
        assert scale > 1.0
        # dt up by the scale, velocity down by it, acceleration down by its square -- one consistent
        # profile for the controller, which is handed both the durations and the velocities.
        assert float(out[0]["plan"].velocity.abs().max()) <= vel_limit.max() * self.config.vel_slack
        assert out[0]["plan"].velocity == pytest.approx(steps[0]["plan"].velocity / scale, abs=1e-6)
        assert out[0]["plan"].position is steps[0]["plan"].position  # geometry untouched
        assert out[0]["label"] == steps[0]["label"]

    def test_no_limits_available_falls_back_to_the_plans_own_peaks(self):
        # With no robot limits the caps come from the plan itself, which by construction it meets.
        out = _slow_to_caps(_steps(peak_vel=4.0), self.config, None, None)
        assert out[0]["dt"] == DT

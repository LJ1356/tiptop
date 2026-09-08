"""Tests for build_tamp_config -> TAMPConfiguration plumbing.

Run with: pytest tests/test_tamp_config.py -v
"""

from pathlib import Path

import pytest
import yaml

from cutamp.config import validate_tamp_config
from tiptop.motion_planning import resolve_placement_support
from tiptop.planning import build_tamp_config

BASE = dict(
    num_particles=256,
    max_planning_time=30.0,
    opt_steps=500,
    robot_type="panda",
    time_dilation_factor=0.2,
)
Q_HOME = [0.0, -0.628, 0.0, -2.513, 0.0, 1.885, 0.0]  # cfg/tiptop robot.q_home


def test_q_home_reaches_the_config():
    """cuTAMP ends every plan at config.q_home when it is set, instead of the pose it planned from."""
    config = build_tamp_config(**BASE, q_home=Q_HOME)
    assert config.q_home == tuple(Q_HOME)


def test_q_home_is_hashable():
    """TAMPConfiguration is a frozen dataclass, so a list here would break hashing it."""
    config = build_tamp_config(**BASE, q_home=Q_HOME)
    assert isinstance(config.q_home, tuple)
    hash(config)


def test_q_home_defaults_to_none():
    """Omitted, cuTAMP keeps its original behaviour of returning to q0."""
    assert build_tamp_config(**BASE).q_home is None


def test_a_plan_returns_home_unless_the_caller_says_it_is_one_leg_of_an_episode():
    """run_planning drops cuTAMP's final drive to q_home for a mid-task leg, and only for that."""
    import dataclasses

    config = build_tamp_config(**BASE, q_home=Q_HOME)
    assert config.return_home, "the standalone default is unchanged -- a plan parks the arm"
    # What run_planning(return_home=False) does: a copy for this leg, leaving the shared config (and
    # q_home, which is what keeps a hand-off resume plannable) alone.
    leg = dataclasses.replace(config, return_home=False)
    assert not leg.return_home
    assert leg.q_home == tuple(Q_HOME), "q_home stays set; only the go-home SEGMENT is dropped"
    assert config.return_home, "the shared config is not mutated"
    hash(leg)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))


class TestPlacementSupport:
    """resolve_placement_support -> build_tamp_config, the cfg/tamp `placement_support` knob."""

    def test_absent_keeps_the_bounding_box_region(self):
        for overrides in (None, {}, {"placement_support": False}, {"grasp_center_weight": 30}):
            config = build_tamp_config(**BASE, placement=resolve_placement_support(overrides))
            assert config.placement_check == "obb"
            assert config.placement_shrink_dist == 0.01
            assert config.placement_ignores_target_surface is False
            validate_tamp_config(config)

    def test_enabled_switches_the_region_and_lets_objects_into_containers(self):
        config = build_tamp_config(**BASE, placement=resolve_placement_support({"placement_support": True}))
        assert config.placement_check == "support"
        # The support region applies its own clearance; cuTAMP rejects the two together.
        assert config.placement_shrink_dist is None
        assert config.support_margin == 0.01
        assert config.placement_support_required is True
        assert config.placement_ignores_target_surface is True
        validate_tamp_config(config)

    def test_occluded_fill_is_off_unless_asked_for(self):
        config = build_tamp_config(**BASE, placement=resolve_placement_support({"placement_support": True}))
        assert config.support_fill_occluded is False
        on = build_tamp_config(
            **BASE,
            placement=resolve_placement_support(
                {"placement_support": True, "placement_fill_occluded": True, "placement_min_seen_frac": 0.4}
            ),
        )
        assert on.support_fill_occluded is True
        assert on.support_min_seen_frac == 0.4
        validate_tamp_config(on)

    def test_knobs_are_read_from_the_overrides(self):
        config = build_tamp_config(
            **BASE,
            placement=resolve_placement_support(
                {
                    "placement_support": True,
                    "placement_support_margin": 0.025,
                    "placement_support_required": False,
                    "placement_into_surface": False,
                }
            ),
        )
        assert config.support_margin == 0.025
        assert config.placement_support_required is False
        assert config.placement_ignores_target_surface is False
        validate_tamp_config(config)

    def test_the_bread_box_config_opts_in(self):
        cfg_path = Path(__file__).resolve().parents[2] / "data-collection/cfg/tamp/4_bread_box.yml"
        overrides = yaml.safe_load(cfg_path.read_text())["tamp_overrides"]
        config = build_tamp_config(**BASE, placement=resolve_placement_support(overrides))
        assert config.placement_check == "support"
        # This task's box is only placeable with its occluded floor counted; see the yml.
        assert config.support_fill_occluded is True
        validate_tamp_config(config)

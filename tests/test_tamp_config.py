"""Tests for build_tamp_config -> TAMPConfiguration plumbing.

Run with: pytest tests/test_tamp_config.py -v
"""

import pytest

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


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))

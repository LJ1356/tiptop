"""``apply_perception_overrides`` must actually reach the config perception reads.

A cfg/tamp ``tamp_overrides`` key normally lands on a cuRobo cost weight or a TAMPConfiguration
knob. ``contact_threshold_m`` is neither -- it lives in tiptop.yml and is read at scene-processing
time as ``tiptop_cfg().perception.contact_threshold_m``. The only thing that makes it settable per
data-gen config is mutating the cached DictConfig before perception runs, so these tests pin that
the mutation happens, is visible to every later ``tiptop_cfg()`` reader, and stays off unless the
config asks for it -- a silent no-op here would look exactly like "the knob did not help".
"""

import pytest

from tiptop.config import tiptop_cfg
from tiptop.motion_planning import apply_perception_overrides


@pytest.fixture
def cfg():
    """The process-wide config, with perception restored afterwards (it is a cached singleton)."""
    c = tiptop_cfg()
    original = c.perception.contact_threshold_m
    original_gt = c.perception.m2t2.grasp_threshold
    original_nr = c.perception.m2t2.num_runs
    original_vox = c.perception.voxel_downsample_size
    yield c
    c.perception.voxel_downsample_size = original_vox
    c.perception.contact_threshold_m = original
    c.perception.m2t2.grasp_threshold = original_gt
    c.perception.m2t2.num_runs = original_nr


def test_absent_key_changes_nothing(cfg):
    before = cfg.perception.contact_threshold_m
    assert apply_perception_overrides(cfg, None) == {}
    assert apply_perception_overrides(cfg, {}) == {}
    # A dict of solver-only keys must not touch perception either.
    assert apply_perception_overrides(cfg, {"grasp_center_weight": 30, "num_particles": 256}) == {}
    assert cfg.perception.contact_threshold_m == before


def test_override_is_applied_and_reported(cfg):
    before = cfg.perception.contact_threshold_m
    target = before * 2
    assert apply_perception_overrides(cfg, {"contact_threshold_m": target}) == {
        "contact_threshold_m": (before, target)
    }
    assert cfg.perception.contact_threshold_m == target


def test_override_is_visible_to_later_readers(cfg):
    """process_scene reads tiptop_cfg() itself; it must see the mutated value, not a stale copy."""
    target = cfg.perception.contact_threshold_m * 2
    apply_perception_overrides(cfg, {"contact_threshold_m": target})
    assert tiptop_cfg().perception.contact_threshold_m == target


def test_reapplying_the_same_value_reports_no_change(cfg):
    """Only actual changes are reported, so the caller's log line means something."""
    target = cfg.perception.contact_threshold_m * 2
    assert apply_perception_overrides(cfg, {"contact_threshold_m": target})
    assert apply_perception_overrides(cfg, {"contact_threshold_m": target}) == {}


@pytest.mark.parametrize("bad", [0, -0.01])
def test_non_positive_threshold_is_rejected(cfg, bad):
    """A threshold of zero would associate no grasps at all -- fail loudly rather than plan blind."""
    before = cfg.perception.contact_threshold_m
    with pytest.raises(ValueError, match="must be positive"):
        apply_perception_overrides(cfg, {"contact_threshold_m": bad})
    assert cfg.perception.contact_threshold_m == before


def test_json_round_trip_survives(cfg):
    """tamp_overrides reach tiptop as JSON via sessions.js -> --curobo-overrides."""
    import json

    before = cfg.perception.contact_threshold_m
    overrides = json.loads(json.dumps({"contact_threshold_m": 0.02}))
    assert apply_perception_overrides(cfg, overrides) == {"contact_threshold_m": (before, 0.02)}


def test_nested_path_override_reaches_the_m2t2_block(cfg):
    """grasp_threshold lives at perception.m2t2.grasp_threshold, two levels down."""
    before = cfg.perception.m2t2.grasp_threshold
    assert apply_perception_overrides(cfg, {"grasp_threshold": 0.02}) == {
        "grasp_threshold": (before, 0.02)
    }
    assert tiptop_cfg().perception.m2t2.grasp_threshold == 0.02


def test_every_declared_path_exists_in_the_config(cfg):
    """A key in the table with no home in tiptop.yml would silently never apply."""
    from tiptop.motion_planning import _PERCEPTION_OVERRIDE_KEYS

    for key, (path, _cast) in _PERCEPTION_OVERRIDE_KEYS.items():
        node = cfg
        for part in path:
            assert part in node, f"{key}: {'.'.join(path)} missing from tiptop.yml"
            node = node[part]
        assert isinstance(node, (int, float)), f"{key} resolves to {type(node).__name__}, not a number"


def test_both_keys_apply_together(cfg):
    applied = apply_perception_overrides(cfg, {"contact_threshold_m": 0.02, "grasp_threshold": 0.02})
    assert set(applied) == {"contact_threshold_m", "grasp_threshold"}
    assert cfg.perception.contact_threshold_m == 0.02
    assert cfg.perception.m2t2.grasp_threshold == 0.02


def test_int_knob_lands_as_an_int(cfg):
    """m2t2_num_runs reaches M2T2 as a count. JSON hands back 20.0 for a YAML 20 in some paths, and
    a float would ride into the request payload."""
    applied = apply_perception_overrides(cfg, {"m2t2_num_runs": 20.0})
    assert applied["m2t2_num_runs"][1] == 20
    value = tiptop_cfg().perception.m2t2.num_runs
    assert value == 20 and isinstance(value, int), f"got {value!r} ({type(value).__name__})"


def test_perception_wrapper_passes_both_m2t2_knobs():
    """The overrides are pointless if the M2T2 call still uses the function defaults."""
    import inspect

    from tiptop import perception_wrapper

    src = inspect.getsource(perception_wrapper.predict_depth_and_grasps)
    assert 'grasp_threshold=cfg.perception.m2t2.get("grasp_threshold"' in src
    assert 'num_runs=cfg.perception.m2t2.get("num_runs"' in src

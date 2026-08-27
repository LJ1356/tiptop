"""A scene reset must move what it can, not all-or-nothing.

The reset goal is ONE cuTAMP plan over N objects, and every object's Pick and Place has to be
satisfied simultaneously — so a single unreachable object leaves 0/256 particles satisfying and
NOTHING moves. That is the common case, not a corner: an object nestled on the surface it has to
come off gets few clean M2T2 grasps. Measured on
``runs/prpl/tamp/tiptop/resets/2026-07-31_13-18-13``: pink_toy had 4 grasp candidates and its Pick
reported 0/256, against blue_toy's 73 grasps → 129/256, and all three toys stayed on the plate.

``_plan_largest_solvable_reset`` therefore drops the fewest-grasps object and re-plans the rest.
"""

import pytest
from curobo.geom.types import Cuboid, Mesh

from tiptop import tiptop_run

UNIT_QUAT = [1.0, 0.0, 0.0, 0.0]
TABLE = Cuboid(name="table", dims=[1.2, 1.6, 0.1], pose=[0.5, 0.0, -0.07, *UNIT_QUAT])


def _mesh(name, center, half=0.03) -> Mesh:
    corners = [[sx * half, sy * half, sz * half] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
    return Mesh(name=name, vertices=corners, faces=[[0, 1, 2]], pose=[*center, *UNIT_QUAT])


class _Scene:
    """Stands in for ProcessedScene: three toys on a plate, with the measured grasp counts."""

    def __init__(self, grasp_counts):
        self.table_cuboid = TABLE
        self.object_meshes = {
            "plate": _mesh("plate", (0.5, 0.0, 0.01), half=0.12),
            "blue_toy": _mesh("blue_toy", (0.52, 0.03, 0.06)),
            "brown_toy": _mesh("brown_toy", (0.47, -0.02, 0.06)),
            "pink_toy": _mesh("pink_toy", (0.50, 0.05, 0.06)),
        }
        self.grasps = {label: {"poses": [None] * n} for label, n in grasp_counts.items()}


class _Container:
    """Only the solver handles the ladder forwards to run_planning, which is stubbed out."""

    ik_solver = None
    motion_gen = None
    cost_overrides: dict = {}


# The real counts from that run.
GRASPS = {"blue_toy": 73, "brown_toy": 7, "pink_toy": 4, "plate": 28}
GOAL = [{"predicate": "on", "args": [label, "table"]} for label in ("blue_toy", "brown_toy", "pink_toy")]


@pytest.fixture
def stub_planner(monkeypatch):
    """Replace run_planning with one that fails whenever `unplannable` is in the goal."""
    calls = []

    def make(unplannable):
        def fake_run_planning(env, config, **kw):
            goal = {atom.values[0] for atom in env.goal_state if atom.name == "On"}
            calls.append(sorted(goal))
            if goal & unplannable:
                return None, 0.0, "No satisfying particles found"
            return ["a-plan"], 0.0, None

        monkeypatch.setattr(tiptop_run, "run_planning", fake_run_planning)
        return calls

    return make


def test_one_unreachable_object_no_longer_sinks_the_whole_reset(stub_planner):
    calls = stub_planner({"pink_toy"})
    scene = _Scene(GRASPS)
    env, surfaces = tiptop_run.create_tamp_environment(
        scene.object_meshes, TABLE, GOAL, include_workspace=True, extra_surface_labels={"plate"}
    )

    plan, moved, skipped = tiptop_run._plan_largest_solvable_reset(
        container=_Container(),
        config=None,
        processed_scene=scene,
        q_init=None,
        goal_atoms=GOAL,
        env=env,
        all_surfaces=surfaces,
        save_dir=tiptop_run.Path("/tmp/unused"),
    )

    assert plan == ["a-plan"]
    assert skipped == ["pink_toy"], "the fewest-grasps object is the one dropped"
    assert sorted(moved) == ["blue_toy", "brown_toy"]
    # Exactly one retry: the full goal, then the goal minus pink_toy.
    assert calls == [["blue_toy", "brown_toy", "pink_toy"], ["blue_toy", "brown_toy"]]


def test_objects_are_dropped_fewest_grasps_first(stub_planner):
    """Only the single most-graspable object is plannable, so it retreats all the way down."""
    calls = stub_planner({"pink_toy", "brown_toy"})
    scene = _Scene(GRASPS)
    env, surfaces = tiptop_run.create_tamp_environment(
        scene.object_meshes, TABLE, GOAL, include_workspace=True, extra_surface_labels={"plate"}
    )

    plan, moved, skipped = tiptop_run._plan_largest_solvable_reset(
        container=_Container(),
        config=None,
        processed_scene=scene,
        q_init=None,
        goal_atoms=GOAL,
        env=env,
        all_surfaces=surfaces,
        save_dir=tiptop_run.Path("/tmp/unused"),
    )

    assert plan == ["a-plan"] and moved == ["blue_toy"]
    assert skipped == ["pink_toy", "brown_toy"], "dropped in ascending grasp count (4 then 7)"
    assert len(calls) == 3


def test_nothing_plannable_reports_every_object(stub_planner):
    stub_planner({"pink_toy", "brown_toy", "blue_toy"})
    scene = _Scene(GRASPS)
    env, surfaces = tiptop_run.create_tamp_environment(
        scene.object_meshes, TABLE, GOAL, include_workspace=True, extra_surface_labels={"plate"}
    )

    plan, moved, skipped = tiptop_run._plan_largest_solvable_reset(
        container=_Container(),
        config=None,
        processed_scene=scene,
        q_init=None,
        goal_atoms=GOAL,
        env=env,
        all_surfaces=surfaces,
        save_dir=tiptop_run.Path("/tmp/unused"),
    )

    assert plan is None and moved == []
    assert sorted(skipped) == ["blue_toy", "brown_toy", "pink_toy"]


def test_the_plate_is_never_a_candidate_to_move():
    """It has 28 grasps — more than two of the toys — so only the goal keeps it out of the ladder."""
    scene = _Scene(GRASPS)
    env, _ = tiptop_run.create_tamp_environment(
        scene.object_meshes, TABLE, GOAL, include_workspace=True, extra_surface_labels={"plate"}
    )
    assert "plate" not in {m.name for m in env.movables}

"""Goal-surface clearing (tiptop.goal_clearing).

The fixture is the scene that motivated the module:
``runs/prpl/tamp/4_pack_coffee_pods_vae_stroke/failure/2026-08-16_22-55-22`` -- a 14.5 cm wooden tray
on the table with a banana lying across it, four coffee pods standing on the table beside it, and the
goal "pack the coffee pods onto the wooden tray". Dimensions and centroids are the ones perception
actually reported (``scene_objects.json``), so a rule that happens to work on rounder numbers but not
on that scene fails here.

Built against the real cuRobo Cuboid/Mesh for the same reason ``test_scene_reset`` is: that is what
perception hands over, and the support relation both modules share leans on how those two carry their
geometry.
"""

import numpy as np
import pytest
from curobo.geom.types import Cuboid, Mesh

from tiptop.goal_clearing import (
    blocking_objects,
    build_clearing_goal,
    drop_return_to_initial,
    final_configuration,
    goal_movables,
    goal_surfaces,
    matrix_to_pose,
    move_meshes,
    placed_poses,
    pose_to_matrix,
)
from tiptop.perception.segmentation import TABLE_BOX_CLEARANCE
from tiptop.scene_reset import footprint_area, supporting_surfaces, world_aabb

UNIT_QUAT = [1.0, 0.0, 0.0, 0.0]
RESTING_Z = 0.0
TABLE = Cuboid(
    name="table", dims=[0.42, 1.11, 0.02], pose=[0.486, -0.025, RESTING_Z - TABLE_BOX_CLEARANCE - 0.01, *UNIT_QUAT]
)


def _box(name: str, center, size) -> Mesh:
    """A Mesh shaped like perception's output: vertices centred on the origin, centre in the pose."""
    cx, cy, cz = np.asarray(size, dtype=float) / 2.0
    corners = np.array([[sx * cx, sy * cy, sz * cz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], dtype=float)
    return Mesh(name=name, vertices=corners.tolist(), faces=[[0, 1, 2]], pose=[*map(float, center), *UNIT_QUAT])


def _prism(name: str, center, outline, height: float) -> Mesh:
    """A Mesh whose XY footprint is ``outline`` rather than its bounding box.

    The distinction is the whole ballgame for a banana: it spans a 16.3 x 14.0 cm bbox -- WIDER than
    the 14.5 cm tray it lies on -- while its convex hull covers only 104 cm2 against the tray's 199.
    A box-shaped banana therefore reads as *supporting* the tray and the blocker is never found, so
    the fixture has to carry the real outline. ``test_the_fixture_matches_the_measured_scene``
    pins those numbers.
    """
    outline = np.asarray(outline, dtype=float)
    verts = np.vstack(
        [np.c_[outline, np.full(len(outline), -height / 2)], np.c_[outline, np.full(len(outline), height / 2)]]
    )
    return Mesh(name=name, vertices=verts.tolist(), faces=[[0, 1, 2]], pose=[*map(float, center), *UNIT_QUAT])


# The perceived scene, verbatim from scene_objects.json (centroids and extents), with the banana's
# outline standing in for the crescent its point cloud reconstructs as.
TRAY = _box("wooden_tray", center=(0.4817, -0.1207, -0.0231), size=(0.1454, 0.1463, 0.0083))
BANANA = _prism(
    "banana",
    center=(0.4573, -0.1342, 0.0066),
    outline=[(-0.0817, -0.020), (0.0, -0.0700), (0.0817, -0.020), (0.0, 0.0400)],
    height=0.0401,
)
PODS = {
    "top_left_blue_coffee_pod": _box("top_left_blue_coffee_pod", (0.4264, 0.0181, -0.0091), (0.0527, 0.0520, 0.0247)),
    "top_right_blue_coffee_pod": _box("top_right_blue_coffee_pod", (0.4317, 0.1157, -0.0097), (0.0512, 0.0513, 0.0239)),
    "bottom_left_blue_coffee_pod": _box(
        "bottom_left_blue_coffee_pod", (0.5185, 0.0294, -0.0068), (0.0500, 0.0504, 0.0267)
    ),
    "bottom_right_blue_coffee_pod": _box(
        "bottom_right_blue_coffee_pod", (0.5196, 0.1040, -0.0077), (0.0501, 0.0508, 0.0319)
    ),
}
SCENE = {"wooden_tray": TRAY, "banana": BANANA, **PODS}
PACK_THE_PODS = [{"predicate": "on", "args": [label, "wooden_tray"]} for label in sorted(PODS)]


def test_the_fixture_matches_the_measured_scene():
    """Guard on the fixture itself: the banana's bbox is wider than the tray, its footprint smaller.

    Both halves matter. If the banana's footprint ever came out the LARGER of the two -- which is
    exactly what happens if it is modelled as a box -- ``supporting_surfaces`` would read the banana
    as holding the tray up, every clearing assertion below would go quietly green for the wrong
    reason, and the suite would stop covering the scene it was written for.
    """
    tray_area, banana_area = footprint_area(TRAY), footprint_area(BANANA)
    assert banana_area < tray_area
    assert banana_area == pytest.approx(0.0104, abs=0.0025)  # measured: 104 cm2
    assert tray_area == pytest.approx(0.0199, abs=0.0025)  # measured: 199 cm2
    banana_lo, banana_hi = world_aabb(BANANA)
    tray_lo, tray_hi = world_aabb(TRAY)
    assert (banana_hi - banana_lo)[0] > (tray_hi - tray_lo)[0]  # wider than the tray it sits on
    assert supporting_surfaces(SCENE) == {"banana": "wooden_tray"}


def test_goal_surfaces_only_counts_perceived_objects():
    assert goal_surfaces(PACK_THE_PODS, SCENE) == {"wooden_tray"}
    # The table is a legitimate goal surface but is not something that can be blocked: there is
    # nowhere to clear it to.
    assert goal_surfaces([{"predicate": "on", "args": ["banana", "table"]}], SCENE) == set()
    assert goal_surfaces([{"predicate": "holding", "args": ["banana"]}], SCENE) == set()
    assert goal_surfaces(None, SCENE) == set()


def test_goal_movables_covers_every_predicate():
    atoms = [{"predicate": "on", "args": ["a", "tray"]}, {"predicate": "holding", "args": ["b"]}]
    assert goal_movables(atoms) == {"a", "b"}
    assert goal_movables(None) == set()


def test_banana_on_the_tray_is_the_blocker():
    """The regression case: nothing in the instruction mentions the banana, the geometry does."""
    assert blocking_objects(SCENE, PACK_THE_PODS) == {"banana": "wooden_tray"}


def test_goal_objects_are_never_cleared():
    """A pod already on the tray is left alone -- the plan moves it anyway, or it is already done."""
    scene = {**SCENE}
    # On a clear corner of the tray, not on the banana -- a pod resting on the banana would make the
    # banana a non-leaf, which test_only_leaves_are_cleared covers separately.
    scene["top_left_blue_coffee_pod"] = _box("top_left_blue_coffee_pod", (0.52, -0.06, 0.005), (0.0527, 0.0520, 0.0247))
    blockers = blocking_objects(scene, PACK_THE_PODS)
    assert "top_left_blue_coffee_pod" not in blockers
    assert blockers == {"banana": "wooden_tray"}


def test_empty_goal_surface_clears_nothing():
    scene = {label: mesh for label, mesh in SCENE.items() if label != "banana"}
    assert blocking_objects(scene, PACK_THE_PODS) == {}


def test_only_leaves_are_cleared():
    """A loaded container is never picked up, so the arm cannot be asked to carry it."""
    # A saucer on the tray with a sugar cube on the saucer: the saucer holds something up.
    scene = {
        **SCENE,
        "saucer": _box("saucer", (0.4817, -0.1207, -0.014), (0.10, 0.10, 0.01)),
        "sugar_cube": _box("sugar_cube", (0.4817, -0.1207, -0.004), (0.02, 0.02, 0.02)),
    }
    del scene["banana"]
    blockers = blocking_objects(scene, PACK_THE_PODS)
    assert "saucer" not in blockers
    # The cube is on the saucer, not on the goal surface, so it is not a blocker either.
    assert blockers == {}


def test_nothing_on_a_surface_the_goal_does_not_name():
    """An object resting on some other object is irrelevant unless the goal wants that surface."""
    other_goal = [{"predicate": "on", "args": ["top_left_blue_coffee_pod", "table"]}]
    assert blocking_objects(SCENE, other_goal) == {}


def test_the_clearing_goal_holds_only_the_clearing():
    """Phase one's goal. The instruction's own atoms belong to phase two, planned separately."""
    atoms, surfaces = build_clearing_goal(SCENE, TABLE, PACK_THE_PODS)
    assert atoms == [{"predicate": "on", "args": ["banana", "table"]}]
    # The tray must still be designated a surface even though the clearing goal never names it,
    # or create_tamp_environment classifies it movable and the planner may pick the tray up.
    assert surfaces == {"wooden_tray"}


def test_no_clearing_goal_when_nothing_blocks():
    """An empty atom list is the caller's signal to plan the task directly, as it always did."""
    scene = {label: mesh for label, mesh in SCENE.items() if label != "banana"}
    atoms, _ = build_clearing_goal(scene, TABLE, PACK_THE_PODS)
    assert atoms == []


def test_clearing_atoms_are_sorted_so_the_goal_is_deterministic():
    scene = {
        **SCENE,
        # Far enough apart that neither lands inside the other's footprint: two independent leaves.
        "spoon": _box("spoon", (0.515, -0.08, 0.005), (0.03, 0.03, 0.01)),
        "apple": _box("apple", (0.45, -0.16, 0.02), (0.05, 0.05, 0.05)),
    }
    del scene["banana"]
    atoms, _ = build_clearing_goal(scene, TABLE, PACK_THE_PODS)
    assert [a["args"][0] for a in atoms] == ["apple", "spoon"]


def test_clearing_goal_uses_the_scenes_own_table_name():
    table = Cuboid(name="workbench", dims=[0.42, 1.11, 0.02], pose=[0.486, -0.025, -0.03, *UNIT_QUAT])
    atoms, _ = build_clearing_goal(SCENE, table, PACK_THE_PODS)
    assert atoms == [{"predicate": "on", "args": ["banana", "workbench"]}]


@pytest.mark.parametrize("atoms", [None, [], [{"predicate": "holding", "args": ["banana"]}]])
def test_goals_with_no_placement_surface_clear_nothing(atoms):
    result, surfaces = build_clearing_goal(SCENE, TABLE, atoms)
    assert result == []
    assert surfaces == set()


# --- phase two: where the clearing plan leaves the world ------------------------------------------


def _traj(label, positions):
    """A trajectory step shaped like cuTAMP's, whose `plan.position` is indexable per waypoint."""

    class _Plan:
        def __init__(self, pos):
            self.position = np.asarray(pos, dtype=float)

    return {"type": "trajectory", "label": label, "plan": _Plan(positions), "dt": 0.1}


def _grip(label, action, placed=None, world_from_obj=None):
    step = {"type": "gripper", "label": label, "action": action}
    if placed is not None:
        # cuTAMP stamps every Place with where it left the object (motion_solver).
        step["placed_object"] = placed
        step["world_from_obj"] = world_from_obj
    return step


BANANA_ON_TABLE = np.eye(4)
BANANA_ON_TABLE[:3, 3] = [0.4564, 0.2571, -0.0075]

# One blocker's worth of plan, in the order cuTAMP emits: approach, close, transport, open, go home.
CLEARING_PLAN = [
    _traj("Pick(banana, grasp1, q1)", [[0.0] * 7, [0.1] * 7]),
    _grip("Pick(banana, grasp1, q1)", "close"),
    _traj("Place(banana, grasp1, pose1, table, q2)", [[0.1] * 7, [0.2] * 7]),
    _grip("Place(banana, grasp1, pose1, table, q2)", "open", "banana", BANANA_ON_TABLE),
    _traj("GoToInitial(q0)", [[0.2] * 7, [0.05] * 7]),
]


def test_final_configuration_is_where_the_next_plan_starts():
    assert np.allclose(final_configuration(CLEARING_PLAN), [0.05] * 7)
    assert final_configuration([_grip("Pick(x)", "close")]) is None


def test_dropping_the_return_home_leaves_the_arm_where_the_clearing_ended():
    """The next plan should set out from the place pose, not from home via a round trip."""
    trimmed = drop_return_to_initial(CLEARING_PLAN)
    assert [s["label"] for s in trimmed] == [s["label"] for s in CLEARING_PLAN[:-1]]
    # ...which moves the hand-off configuration off q0 and onto the end of the transport.
    assert np.allclose(final_configuration(trimmed), [0.2] * 7)
    assert np.allclose(final_configuration(CLEARING_PLAN), [0.05] * 7)


def test_dropping_the_return_home_is_a_no_op_without_one():
    plan = CLEARING_PLAN[:-1]
    assert drop_return_to_initial(plan) == plan
    assert drop_return_to_initial([]) == []


def test_only_a_trailing_return_home_is_dropped():
    """A GoToInitial mid-plan is somebody's deliberate step, not the tail cuTAMP appends."""
    plan = [_traj("GoToInitial(q0)", [[0.0] * 7]), *CLEARING_PLAN]
    trimmed = drop_return_to_initial(plan)
    assert trimmed[0]["label"] == "GoToInitial(q0)"
    assert len(trimmed) == len(plan) - 1


def test_dropping_the_return_home_does_not_mutate_the_plan():
    before = list(CLEARING_PLAN)
    drop_return_to_initial(CLEARING_PLAN)
    assert CLEARING_PLAN == before


def test_placed_pose_comes_from_the_plan_not_a_reconstruction():
    """cuTAMP stamps the Place step with the pose it updated the collision world to."""
    assert np.allclose(placed_poses(CLEARING_PLAN)["banana"], BANANA_ON_TABLE)


def test_a_plan_that_places_nothing_reports_nothing():
    """Not an error -- the caller then keeps every object at its perceived pose, which is correct."""
    assert placed_poses([_traj("MoveFree(q0, traj1, q1)", [[0.0] * 7])]) == {}
    assert placed_poses([_grip("Pick(banana, grasp1, q1)", "close")]) == {}


def test_the_last_placement_of_an_object_wins():
    """An object set down twice is wherever it finally came to rest."""
    second = np.eye(4)
    second[:3, 3] = [0.3, 0.3, 0.0]
    plan = [*CLEARING_PLAN, _grip("Place(banana, grasp2, pose2, table, q4)", "open", "banana", second)]
    assert np.allclose(placed_poses(plan)["banana"], second)


def test_pose_matrix_round_trip():
    pose = [0.4, -0.1, 0.02, 0.9238795, 0.0, 0.0, 0.3826834]  # 45 deg about z
    assert np.allclose(matrix_to_pose(pose_to_matrix(pose)), pose, atol=1e-6)


def test_move_meshes_writes_the_placed_world_pose_in():
    moved = move_meshes(SCENE, {"banana": BANANA_ON_TABLE})

    assert np.allclose(moved["banana"].pose[:3], [0.4564, 0.2571, -0.0075], atol=1e-4)
    # Everything else passes through, and the input is never mutated.
    assert moved["wooden_tray"] is SCENE["wooden_tray"]
    assert SCENE["banana"].pose[1] == pytest.approx(-0.1342)


def test_move_meshes_ignores_labels_that_are_not_in_the_scene():
    assert move_meshes(SCENE, {"ghost": np.eye(4)}) == SCENE

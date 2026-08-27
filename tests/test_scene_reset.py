"""Scene-reset goal building (tiptop.scene_reset).

Built against the real cuRobo Cuboid/Mesh because that is what perception hands over, and the module
leans on how those two carry their geometry: an identity rotation with the centre in ``pose[:3]``,
extents in ``dims`` for a Cuboid and re-centred ``vertices`` for a Mesh.

The fixture reproduces the ONE table geometry perception can actually produce: the collision cuboid's
top face sits ``TABLE_BOX_CLEARANCE`` (2 cm) BELOW the plane objects rest on, because
``segment_table_with_ransac`` sinks it there. A suite whose table top coincides with the resting
plane would pass while the shipped rule was inverted, so that offset is deliberate here.
"""

import numpy as np
import pytest
from curobo.geom.types import Cuboid, Mesh

from tiptop.perception.segmentation import TABLE_BOX_CLEARANCE
from tiptop.scene_reset import build_reset_goal, footprint_area, supporting_surfaces, world_aabb

UNIT_QUAT = [1.0, 0.0, 0.0, 0.0]

# Objects rest at z = 0; the table collision box therefore has its top at -TABLE_BOX_CLEARANCE.
RESTING_Z = 0.0
TABLE = Cuboid(name="table", dims=[1.2, 1.6, 0.1], pose=[0.5, 0.0, RESTING_Z - TABLE_BOX_CLEARANCE - 0.05, *UNIT_QUAT])


def _box(name: str, center, size) -> Mesh:
    """A Mesh shaped like perception's output: vertices centred on the origin, centre in the pose."""
    cx, cy, cz = np.asarray(size, dtype=float) / 2.0
    corners = np.array([[sx * cx, sy * cy, sz * cz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], dtype=float)
    return Mesh(name=name, vertices=corners.tolist(), faces=[[0, 1, 2]], pose=[*map(float, center), *UNIT_QUAT])


def _disc(name: str, center, radius: float, thickness: float, n: int = 24) -> Mesh:
    """A round plate: the case where the AABB (circumscribing square) overstates the footprint."""
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    ring = np.c_[radius * np.cos(angles), radius * np.sin(angles)]
    verts = np.vstack([np.c_[ring, np.full(n, -thickness / 2)], np.c_[ring, np.full(n, thickness / 2)]])
    return Mesh(name=name, vertices=verts.tolist(), faces=[[0, 1, 2]], pose=[*map(float, center), *UNIT_QUAT])


# A plate lying on the table at (0.5, 0.0), 24 cm across and 2 cm thick.
PLATE = _box("white_plate", center=(0.5, 0.0, 0.01), size=(0.24, 0.24, 0.02))


# A toy standing on the table well clear of the plate.
def _toy(name, xy):
    return _box(name, center=(*xy, 0.04), size=(0.06, 0.06, 0.08))


# A toy sitting ON the plate (its underside at the plate's top, z = 0.02).
def _toy_on_plate(name, xy):
    return _box(name, center=(*xy, 0.06), size=(0.06, 0.06, 0.08))


def test_world_aabb_cuboid_and_mesh():
    lo, hi = world_aabb(TABLE)
    assert np.allclose(lo, [-0.1, -0.8, -0.12])
    assert np.allclose(hi, [1.1, 0.8, -0.02])  # the box top is BELOW the resting plane, by design

    lo, hi = world_aabb(PLATE)
    assert np.allclose(lo, [0.38, -0.12, 0.0])
    assert np.allclose(hi, [0.62, 0.12, 0.02])


def test_world_aabb_rejects_geometry_it_cannot_bound():
    class Nothing:
        pose = [0.0, 0.0, 0.0, *UNIT_QUAT]

    with pytest.raises(TypeError):
        world_aabb(Nothing())


def test_footprint_area_uses_the_outline_not_the_bounding_box():
    """A disc's area is pi*r^2, not the (2r)^2 its AABB would give."""
    disc = _disc("round_plate", center=(0.5, 0.0, 0.01), radius=0.12, thickness=0.02)
    assert footprint_area(disc) == pytest.approx(np.pi * 0.12**2, rel=0.02)
    assert footprint_area(TABLE) == pytest.approx(1.2 * 1.6)


def test_objects_on_the_plate_are_the_ones_that_move():
    scene = {
        "white_plate": PLATE,
        "blue_toy": _toy_on_plate("blue_toy", (0.52, 0.03)),
        "red_toy": _toy_on_plate("red_toy", (0.46, -0.04)),
        "green_toy": _toy("green_toy", (0.20, 0.30)),  # on the table, well clear
    }
    assert supporting_surfaces(scene) == {"blue_toy": "white_plate", "red_toy": "white_plate"}

    atoms, surfaces = build_reset_goal(scene, TABLE)
    assert atoms == [
        {"predicate": "on", "args": ["blue_toy", "table"]},
        {"predicate": "on", "args": ["red_toy", "table"]},
    ]
    # The plate must stay a SURFACE, or create_tamp_environment classifies it movable and cuTAMP is
    # free to carry the loaded plate off to the table instead of the toys.
    assert surfaces == {"white_plate"}


def test_the_surface_is_never_moved_itself():
    """The larger of an overlapping pair is the support, so the relation is never symmetric."""
    scene = {"white_plate": PLATE, "blue_toy": _toy_on_plate("blue_toy", (0.5, 0.0))}  # dead centre
    assert "white_plate" not in supporting_surfaces(scene)
    assert [a["args"][0] for a in build_reset_goal(scene, TABLE)[0]] == ["blue_toy"]


def test_a_clean_scene_produces_no_reset_plan():
    """Nothing stacked -> no atoms, so tiptop skips planning entirely rather than churning the arm."""
    scene = {
        "white_plate": PLATE,
        "green_toy": _toy("green_toy", (0.20, 0.30)),
        "tan_toy": _toy("tan_toy", (0.75, -0.25)),
    }
    assert supporting_surfaces(scene) == {}
    assert build_reset_goal(scene, TABLE) == ([], set())


def test_the_rule_does_not_depend_on_object_labels():
    """Perception renames the same plate between runs; the goal must not change when it does."""
    on_plate = _toy_on_plate("toy", (0.52, 0.03))
    a = build_reset_goal({"white_plate": PLATE, "toy": on_plate}, TABLE)
    b = build_reset_goal({"plate": PLATE, "toy": on_plate}, TABLE)
    assert a[0] == b[0] == [{"predicate": "on", "args": ["toy", "table"]}]
    assert a[1] == {"white_plate"} and b[1] == {"plate"}


def test_an_object_perched_on_the_rim_still_counts():
    """Just outside the plate's outline, inside the margin."""
    scene = {"white_plate": PLATE, "rim_toy": _box("rim_toy", center=(0.63, 0.0, 0.05), size=(0.06, 0.06, 0.06))}
    assert supporting_surfaces(scene, xy_margin=0.02) == {"rim_toy": "white_plate"}
    assert supporting_surfaces(scene, xy_margin=0.0) == {}


def test_a_toy_diagonally_beside_a_ROUND_plate_is_left_alone():
    """Its centroid is inside the plate's bounding box but outside the actual disc."""
    plate = _disc("round_plate", center=(0.5, 0.0, 0.01), radius=0.12, thickness=0.02)
    corner = 0.12 / np.sqrt(2) + 0.035  # past the rim on the diagonal, still inside the AABB
    scene = {"round_plate": plate, "corner_toy": _toy("corner_toy", (0.5 + corner, corner))}
    lo, hi = world_aabb(plate)
    centroid = np.asarray(scene["corner_toy"].pose)[:2]
    assert np.all(centroid >= lo[:2]) and np.all(centroid <= hi[:2]), "fixture must sit inside the AABB"
    assert supporting_surfaces(scene) == {}


def test_an_object_inside_a_deep_bowl_still_counts():
    """The measured spray-in-bowl scene from runs/prpl/tamp/tdf_jd_blend_bowl/eval/2026-07-28_01-04-18.

    Reproduced from that env's real dimensions rather than invented, because it is the one archived
    stacked scene and it is the case the rule exists for: the spray's underside is 7.2 cm BELOW the
    bowl's rim, so no height comparison of any kind could pair them. Only the footprint areas
    (167 vs 91 cm^2) say which is holding which.
    """
    bowl = _box("green_bowl", center=(0.510, -0.239, 0.0105), size=(0.1458, 0.1458, 0.0710))
    spray = _box("spray_bottle", center=(0.500, -0.250, 0.0325), size=(0.0956, 0.0956, 0.1150))
    assert world_aabb(spray)[0][2] - world_aabb(bowl)[1][2] == pytest.approx(-0.072, abs=5e-3), "fixture: sunk"
    assert footprint_area(bowl) > footprint_area(spray)
    assert supporting_surfaces({"green_bowl": bowl, "spray_bottle": spray}) == {"spray_bottle": "green_bowl"}


def test_a_loaded_surface_is_never_carried():
    """Toy on a plate on a pad: the toy is attributed to the plate (the innermost support), and the
    plate — which is itself stacked — stays put rather than being carried off with the toy on it."""
    pad = _box("blue_pad", center=(0.5, 0.0, 0.005), size=(0.40, 0.40, 0.01))
    plate = _box("white_plate", center=(0.5, 0.0, 0.02), size=(0.24, 0.24, 0.02))
    scene = {"blue_pad": pad, "white_plate": plate, "blue_toy": _toy_on_plate("blue_toy", (0.5, 0.0))}
    assert supporting_surfaces(scene) == {"blue_toy": "white_plate", "white_plate": "blue_pad"}

    atoms, surfaces = build_reset_goal(scene, TABLE)
    assert atoms == [{"predicate": "on", "args": ["blue_toy", "table"]}]
    assert surfaces == {"white_plate", "blue_pad"}

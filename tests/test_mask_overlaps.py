"""Can a container's mask still swallow the object resting on it?

SAM2 is seeded from Gemini's boxes, and a box around a book contains the marker lying on it -- so
the book's mask came back covering 81% of the marker's pixels. Each mask became its own point cloud
and its own convex hull, so the book's collision box spanned z <= 0.052 while the marker spanned
0.036-0.051: the marker sat entirely inside an unpickable obstacle, every pick pose was in
collision, and collision-aware IK returned 1 of 512 seeds. cuTAMP then failed the pick's pos_err
(0.0136 against a 0.005 tolerance) on run 3_pen_open_book/failure/2026-09-06_23-06-41.

    tiptop/.pixi/envs/default/bin/python -m pytest tiptop/tests/test_mask_overlaps.py -q
"""

from __future__ import annotations

import numpy as np

from tiptop.perception.segmentation import resolve_mask_overlaps


def _masks(*specs, shape=(40, 40)):
    """Build (n, H, W) boolean masks from (y0, y1, x0, x1) slices."""
    out = np.zeros((len(specs), *shape), dtype=bool)
    for i, (y0, y1, x0, x1) in enumerate(specs):
        out[i, y0:y1, x0:x1] = True
    return out


def test_small_object_keeps_every_pixel_the_container_also_claimed():
    """The failing case: a marker fully contained in the book's mask must not lose a single pixel."""
    marker, book = 0, 1
    masks = _masks((10, 14, 10, 14), (0, 30, 0, 30))
    before = masks.sum(axis=(1, 2))

    out = resolve_mask_overlaps(masks, ["marker", "book"])

    assert out[marker].sum() == before[marker]
    assert (out[marker] == masks[marker]).all()
    # The book keeps everything else and gives up exactly the contested region.
    assert out[book].sum() == before[book] - before[marker]
    assert not (out[book] & masks[marker]).any()


def test_no_pixel_is_claimed_twice_and_the_union_is_unchanged():
    """Disjointness is the postcondition; no pixel may be invented or dropped from the scene."""
    masks = _masks((0, 30, 0, 30), (10, 14, 10, 14), (5, 25, 20, 35), (36, 39, 36, 39))

    out = resolve_mask_overlaps(masks, ["book", "marker", "tray", "cube"])

    assert out.sum(axis=0).max() <= 1
    assert (out.any(axis=0) == masks.any(axis=0)).all()


def test_three_way_nesting_resolves_to_the_innermost_claimant():
    """A marker on a book on a tray: each contested pixel goes to the smallest mask claiming it."""
    tray, book, marker = 0, 1, 2
    masks = _masks((0, 30, 0, 30), (5, 20, 5, 20), (10, 14, 10, 14))

    out = resolve_mask_overlaps(masks, ["tray", "book", "marker"])

    assert (out[marker] == masks[marker]).all()
    assert (out[book] == (masks[book] & ~masks[marker])).all()
    assert (out[tray] == (masks[tray] & ~masks[book])).all()


def test_disjoint_masks_are_returned_untouched():
    masks = _masks((0, 10, 0, 10), (20, 30, 20, 30))
    assert (resolve_mask_overlaps(masks, ["a", "b"]) == masks).all()


def test_single_mask_is_a_no_op():
    masks = _masks((0, 10, 0, 10))
    assert (resolve_mask_overlaps(masks, ["a"]) == masks).all()


def test_empty_mask_survives_without_claiming_anything():
    """A detection SAM2 found nothing for has area 0, so it wins every tie -- it must still stay empty."""
    empty, other = 0, 1
    masks = _masks((0, 0, 0, 0), (0, 20, 0, 20))

    out = resolve_mask_overlaps(masks, ["ghost", "book"])

    assert out[empty].sum() == 0
    assert (out[other] == masks[other]).all()


def test_more_masks_than_labels_does_not_raise():
    """The caller trims labels to len(bboxes); logging must not index past it."""
    masks = _masks((0, 30, 0, 30), (10, 14, 10, 14))
    out = resolve_mask_overlaps(masks, ["book"])
    assert out.sum(axis=0).max() <= 1

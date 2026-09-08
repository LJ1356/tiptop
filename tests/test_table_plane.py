"""Tests for how the table plane is chosen from RANSAC's candidates.

The failure these pin down (eval/2026-09-07_21-31-41): a plane fitted ~4 cm ABOVE the tabletop won
the vote, because the score used the absolute distance from each object's contact point and so
counted objects that were *below* the plane as resting on it. Everything downstream is silent about
it -- the near-table filter then dropped the plate and the box out of the scene entirely, and the
task planner was asked to put bread in a box that was no longer there.
"""

import numpy as np
import pytest

from tiptop.perception.segmentation import _CONTACT_BELOW_TOL, _plane_support_score, _upward

# The three object contact points from that run: the loaf sitting on the closed box, the box, and
# the plate. World frame, metres.
CONTACTS = np.array(
    [
        [0.5912, 0.2830, 0.03458],  # bread, up on the box
        [0.6692, 0.2535, -0.00745],  # box, on the table
        [0.4559, -0.0793, -0.00032],  # plate, on the table
    ]
)
TABLE_PLANE = [-0.01085814, 0.00223375, 0.99993855, 0.01866988]  # z ~ -0.019, the real tabletop
HIGH_PLANE = [0.03245069, -0.04092822, 0.99863498, -0.02405905]  # z ~ +0.024, the impostor


def _level(z):
    """A level plane at height ``z``: 0x + 0y + 1z - z = 0."""
    return [0.0, 0.0, 1.0, -z]


class TestUpward:
    def test_a_downward_normal_is_flipped(self):
        assert _upward([0.0, 0.0, -1.0, 0.5]) == pytest.approx([0.0, 0.0, 1.0, -0.5])

    def test_an_upward_normal_is_left_alone_and_normalised(self):
        assert _upward([0.0, 0.0, 2.0, -0.04]) == pytest.approx([0.0, 0.0, 1.0, -0.02])


class TestPlaneSupportScore:
    def test_the_real_tabletop_beats_the_plane_above_it(self):
        assert _plane_support_score(TABLE_PLANE, CONTACTS, 0.03) == 2
        assert _plane_support_score(HIGH_PLANE, CONTACTS, 0.03) == 1

    def test_absolute_distance_is_what_got_this_wrong(self):
        """The old scoring, reproduced. It did not merely tie -- the impostor OUTSCORED the table.

        These are the two numbers the run logged: "Plane 2: objects_on_plane=2/3" for the real
        tabletop and "Plane 4: objects_on_plane=3/3" for the plane 4 cm above it, whose 3/3 came
        from counting two objects that were underneath it.
        """
        def old_score(model):
            a, b, c, d = model
            dists = np.abs(CONTACTS @ np.array([a, b, c]) + d) / np.linalg.norm([a, b, c])
            return int((dists < 0.03).sum())

        assert old_score(HIGH_PLANE) == 3
        assert old_score(TABLE_PLANE) == 2
        # Signed distance reverses the verdict.
        assert _plane_support_score(HIGH_PLANE, CONTACTS, 0.03) < _plane_support_score(
            TABLE_PLANE, CONTACTS, 0.03
        )

    def test_an_object_below_the_plane_does_not_count(self):
        # It would have to be inside the table.
        contacts = np.array([[0.5, 0.0, -0.03]])
        assert _plane_support_score(_level(0.0), contacts, 0.03) == 0

    def test_noise_below_the_plane_is_tolerated(self):
        just_under = np.array([[0.5, 0.0, -0.5 * _CONTACT_BELOW_TOL]])
        well_under = np.array([[0.5, 0.0, -2.0 * _CONTACT_BELOW_TOL]])
        assert _plane_support_score(_level(0.0), just_under, 0.03) == 1
        assert _plane_support_score(_level(0.0), well_under, 0.03) == 0

    def test_an_object_stacked_on_another_does_not_vote_for_the_table(self):
        # The loaf on the closed box is 4.7 cm up: on the box, not on the table.
        stacked = np.array([[0.5, 0.0, 0.047]])
        assert _plane_support_score(_level(0.0), stacked, 0.03) == 0
        # But it does vote for the box's own top face.
        assert _plane_support_score(_level(0.045), stacked, 0.03) == 1

    def test_objects_resting_on_the_plane_all_count(self):
        contacts = np.array([[0.4, 0.0, 0.0], [0.5, 0.1, 0.002], [0.6, -0.1, 0.012]])
        assert _plane_support_score(_level(0.0), contacts, 0.03) == 3

    def test_a_downward_fitted_plane_scores_the_same(self):
        """RANSAC returns whichever orientation it likes; the score must not depend on it."""
        contacts = np.array([[0.4, 0.0, 0.004], [0.5, 0.1, -0.002]])
        up = _plane_support_score([0.0, 0.0, 1.0, 0.0], contacts, 0.03)
        down = _plane_support_score([0.0, 0.0, -1.0, 0.0], contacts, 0.03)
        assert up == down == 2

"""Tests for the Ferrari-Canny grasp quality metric."""

import numpy as np

from tampanda.planners.pointcloud_grasp_planner import (
    AntipodalGraspPlanner, ferrari_canny, robust_ferrari_canny,
)
from tampanda.planners.grasp_planner import GRASP_CONTACT_OFFSET, quat_to_rotmat

from test_antipodal_grasp_planner import TABLE_Z, _cylinder

COM = np.zeros(3)
PINCH = np.array([[0.0, 0.03, 0.0], [0.0, -0.03, 0.0]])
INWARD = np.array([[0.0, -1.0, 0.0], [0.0, 1.0, 0.0]])


def test_centred_antipodal_grasp_is_force_closure():
    assert ferrari_canny(PINCH, INWARD, COM, mu=0.7) > 0.0


def test_offset_from_com_lowers_quality():
    offset = PINCH + np.array([0.04, 0.0, 0.0])
    assert ferrari_canny(offset, INWARD, COM, 0.7) < ferrari_canny(PINCH, INWARD, COM, 0.7)


def test_point_contacts_cannot_resist_rotation_about_finger_axis():
    assert ferrari_canny(PINCH, INWARD, COM, 0.7, torsion_radius=0.0) == 0.0


def test_normals_outside_friction_cone_fail():
    a = np.radians(60)                      # cone half-angle atan(0.7) = 35 deg
    slanted = np.array([[np.sin(a), -np.cos(a), 0.0], [np.sin(a), np.cos(a), 0.0]])
    assert ferrari_canny(PINCH, slanted, COM, 0.7) == 0.0


def test_more_friction_is_better():
    assert ferrari_canny(PINCH, INWARD, COM, 1.0) > ferrari_canny(PINCH, INWARD, COM, 0.3)


def test_robust_is_at_most_nominal_for_a_good_grasp():
    rng = np.random.default_rng(0)
    nominal = ferrari_canny(PINCH, INWARD, COM, 0.7)
    robust = robust_ferrari_canny(PINCH, INWARD, COM, 0.7, rng, n_samples=20)
    assert 0.0 < robust <= nominal * 1.05


def test_planner_ranks_by_quality():
    """With the metric, candidate scores are robust epsilons in descending order."""
    cands = AntipodalGraspPlanner(table_z=TABLE_Z).generate_candidates(_cylinder(0.03, 0.12))
    scores = [c.score for c in cands]
    assert len(cands) > 0 and all(s > 0 for s in scores)
    assert scores == sorted(scores, reverse=True) and max(scores) > min(scores)
    # the best grasp passes close to the cylinder axis (the centre of mass)
    best = cands[0]
    contact = best.grasp_pos + quat_to_rotmat(best.grasp_quat)[:, 2] * GRASP_CONTACT_OFFSET
    assert np.linalg.norm(contact[:2] - np.array([0.4, 0.4])) < 0.02

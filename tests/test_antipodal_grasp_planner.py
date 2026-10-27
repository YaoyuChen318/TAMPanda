"""Tests for AntipodalGraspPlanner on synthetic point clouds (no rendering)."""

import numpy as np

from tampanda.planners.pointcloud_grasp_planner import AntipodalGraspPlanner
from tampanda.planners.grasp_planner import GRASP_CONTACT_OFFSET, GraspType, quat_to_rotmat

TABLE_Z = 0.27


def _cylinder(radius, height, center_xy=(0.4, 0.4), n=2500, seed=0):
    """Surface points of an upright cylinder standing on the table (side + top)."""
    rng = np.random.default_rng(seed)
    n_side = int(n * 0.8)
    a = rng.uniform(0, 2 * np.pi, n_side)
    side = np.column_stack([radius * np.cos(a), radius * np.sin(a), rng.uniform(0, height, n_side)])
    r = radius * np.sqrt(rng.uniform(0, 1, n - n_side))
    b = rng.uniform(0, 2 * np.pi, n - n_side)
    top = np.column_stack([r * np.cos(b), r * np.sin(b), np.full(n - n_side, height)])
    pts = np.vstack([side, top])
    return pts + np.array([center_xy[0], center_xy[1], TABLE_Z])


def _box(half, center=(0.4, 0.4), n=3000, seed=0):
    """Surface points of an axis-aligned box resting on the table."""
    rng = np.random.default_rng(seed)
    half = np.asarray(half)
    pts = rng.uniform(-1, 1, (n, 3)) * half
    face = rng.integers(0, 3, n)
    pts[np.arange(n), face] = np.sign(rng.uniform(-1, 1, n)) * half[face]
    return pts + np.array([center[0], center[1], TABLE_Z + half[2]])


def _contact(c):
    R = quat_to_rotmat(c.grasp_quat)
    return c.grasp_pos + R[:, 2] * GRASP_CONTACT_OFFSET, R


def test_tall_cylinder_grasps_are_feasible():
    """No top-down grasp deeper than the palm allows (the mustard-bottle failure)."""
    height, radius = 0.19, 0.028
    planner = AntipodalGraspPlanner(table_z=TABLE_Z)
    cands = planner.generate_candidates(_cylinder(radius, height))
    assert len(cands) > 0
    top_z = TABLE_Z + height
    has_side = False
    for c in cands:
        assert c.grasp_type == GraspType.SIX_DOF
        contact, R = _contact(c)
        assert np.linalg.norm(contact[:2] - np.array([0.4, 0.4])) < radius
        if R[2, 2] < -0.9:
            # palm bottom is 3.4 cm above the pad centre
            assert top_z - contact[2] < 0.034, f"top-down grasp {100 * (top_z - contact[2]):.1f} cm below top"
        has_side |= abs(R[2, 2]) < 0.3
    assert has_side, "a tall cylinder should get side grasps"


def test_candidates_stay_above_table():
    planner = AntipodalGraspPlanner(table_z=TABLE_Z, table_clearance=0.01)
    for c in planner.generate_candidates(_box([0.025, 0.02, 0.05])):
        assert c.grasp_pos[2] > TABLE_Z and c.approach_pos[2] > TABLE_Z
        assert c.lift_pos[2] > c.grasp_pos[2]


def test_closing_axis_respects_gripper_width():
    """A 12 cm cube does not fit the 8 cm gripper; a 4 cm cube does."""
    planner = AntipodalGraspPlanner(table_z=TABLE_Z)
    assert planner.generate_candidates(_box([0.06, 0.06, 0.06])) == []
    assert len(planner.generate_candidates(_box([0.02, 0.02, 0.02]))) > 0


def test_obstacle_points_block_grasps():
    """No approach through a wall of obstacle points next to the object."""
    planner = AntipodalGraspPlanner(table_z=TABLE_Z)
    obj = _box([0.02, 0.02, 0.04])
    free = planner.generate_candidates(obj)
    yy, zz = np.meshgrid(np.linspace(0.3, 0.5, 30), np.linspace(TABLE_Z, TABLE_Z + 0.2, 30))
    wall = np.column_stack([np.full(yy.size, 0.4 + 0.035), yy.ravel(), zz.ravel()])   # 1.5 cm from the +x face
    blocked = AntipodalGraspPlanner(table_z=TABLE_Z).generate_candidates(obj, obstacle_points=wall)
    assert len(blocked) > 0
    # a hand on the wall (+x) side approaches along -x
    assert any(_contact(c)[1][0, 2] < -0.9 for c in free), "setup: +x side is graspable without the wall"
    for c in blocked:
        _, R = _contact(c)
        assert R[0, 2] > -0.9, "no approach should come through the wall"


def _cup(r_out=0.04, wall=0.005, height=0.09, center_xy=(0.4, 0.4), n=6000, seed=0):
    """Open thin-walled cylinder and per-point viewpoints (outer wall from the side, rest from above)."""
    rng = np.random.default_rng(seed)
    c = np.array([center_xy[0], center_xy[1], TABLE_Z])
    a = rng.uniform(0, 2 * np.pi, n)
    radial = np.column_stack([np.cos(a), np.sin(a), np.zeros(n)])
    kind = rng.integers(0, 3, n)                      # 0 outer wall, 1 inner wall, 2 rim
    r = np.where(kind == 0, r_out, np.where(kind == 1, r_out - wall, rng.uniform(r_out - wall, r_out, n)))
    z = np.where(kind == 2, height, rng.uniform(0.005, height, n))
    pts = c + radial * r[:, None] + np.column_stack([np.zeros(n), np.zeros(n), z])
    above = c + np.array([0.0, 0.0, 1.0])
    views = np.where((kind == 0)[:, None], pts + radial * 1.0 + np.array([0, 0, 0.3]), above)
    return pts, views


def test_mug_rim_pinch_needs_viewpoints():
    """A thin cup is grasped top-down across the wall only with viewpoint-oriented normals."""
    pts, views = _cup()
    cands = AntipodalGraspPlanner(table_z=TABLE_Z).generate_candidates(pts, viewpoints=views)
    rim_pinches = 0
    for c in cands:
        contact, R = _contact(c)
        radial = np.r_[contact[:2] - np.array([0.4, 0.4]), 0.0]
        radial /= np.linalg.norm(radial)
        if R[2, 2] < -0.9 and abs(R[:, 1] @ radial) > 0.8:
            rim_pinches += 1
            assert abs(np.linalg.norm(contact[:2] - np.array([0.4, 0.4])) - 0.0375) < 0.01
    assert rim_pinches > 0
    # without viewpoints the inner-wall normals point outward and fail the friction cone
    assert AntipodalGraspPlanner(table_z=TABLE_Z).generate_candidates(pts) == []


def test_too_few_points():
    assert AntipodalGraspPlanner().generate_candidates(np.zeros((5, 3))) == []

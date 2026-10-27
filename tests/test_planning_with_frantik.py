"""Tests for planning with a multi-solution IK (candidate_goals in tampanda.planners.rrt_star)."""

import time

import mujoco
import numpy as np

from tampanda import RRTStar, FeasibilityRRT
from tampanda.controllers.position_controller import PositionController
from tampanda.planners.rrt_star import candidate_goals

from conftest import fk_error, make_tabletop

# Arm at the dropoff pose, then approaching cylinder_2 (examples/grasping_rrt_planner.py,
# seed 0).  The closest IK solution here presses the arm low over the table with
# q6 at its limit; RRT fails on it most of the time (~5 s per attempt).
DROPOFF_Q = np.array([-0.31, -0.2, 0.28, -1.88, 0.05, 1.68, 0.74])
TARGET_POS = np.array([0.27, 0.416, 0.35])
TARGET_QUAT = np.array([-0.5, 0.5, 0.5, 0.5])   # GraspType.FRONT


def _set_arm(env, q):
    env.data.qpos[:7] = q
    mujoco.mj_forward(env.model, env.data)


def test_interpolate_zero_length():
    q = np.zeros(7)
    traj = PositionController.interpolate_linear_points(q, q + 1e-4, step_size=0.01)
    assert len(traj) == 2 and np.allclose(traj[-1], q + 1e-4)


def test_candidate_goals():
    env = make_tabletop("frantik")
    _set_arm(env, DROPOFF_Q)
    goals = candidate_goals(env, env.ik, TARGET_POS, TARGET_QUAT, max_goals=3)
    assert 0 < len(goals) <= 3
    for i, g in enumerate(goals):
        assert env.is_collision_free(g)
        pos_err, ori_err = fk_error(env.model, g, TARGET_POS, TARGET_QUAT)
        assert pos_err < 2e-3 and ori_err < 2e-3
        for h in goals[:i]:
            assert np.linalg.norm(g - h) >= 1.0
    # Candidate selection must not move the simulated arm.
    np.testing.assert_allclose(env.data.qpos[:7], DROPOFF_Q)


def test_rrt_plan_to_pose_frantik_regression():
    env = make_tabletop("frantik")
    rrt = RRTStar(env)
    rrt.max_iterations, rrt.step_size, rrt.goal_sample_rate = 1000, 0.2, 0.2
    for seed in range(3):
        np.random.seed(seed)
        _set_arm(env, DROPOFF_Q)
        t0 = time.perf_counter()
        path = rrt.plan_to_pose(TARGET_POS, TARGET_QUAT, dt=0.005, max_iterations=2000)
        elapsed = time.perf_counter() - t0
        assert path is not None, f"seed {seed}"
        assert elapsed < 3.0, f"seed {seed}: {elapsed:.1f}s"
        np.testing.assert_allclose(path[0], DROPOFF_Q, atol=1e-6)
        pos_err, ori_err = fk_error(env.model, path[-1], TARGET_POS, TARGET_QUAT)
        assert pos_err < 2e-3 and ori_err < 2e-3


def test_feasibility_rrt_plan_to_pose_frantik():
    env = make_tabletop("frantik")
    planner = FeasibilityRRT(env)
    np.random.seed(0)
    _set_arm(env, DROPOFF_Q)
    path = planner.plan_to_pose(TARGET_POS, TARGET_QUAT, dt=0.005, max_iterations=2000)
    assert path is not None
    pos_err, _ = fk_error(env.model, path[-1], TARGET_POS, TARGET_QUAT)
    assert pos_err < 2e-3


def test_rrt_plan_to_pose_mink_unchanged():
    env = make_tabletop("mink")
    rrt = RRTStar(env)
    rrt.max_iterations, rrt.step_size, rrt.goal_sample_rate = 1000, 0.2, 0.2
    np.random.seed(0)
    _set_arm(env, DROPOFF_Q)
    path = rrt.plan_to_pose(TARGET_POS, TARGET_QUAT, dt=0.005, max_iterations=2000)
    assert path is not None
    pos_err, _ = fk_error(env.model, path[-1], TARGET_POS, TARGET_QUAT)
    assert pos_err < 1e-2   # mink converges to ~2 mm

"""Tests for the analytical Franka IK (tampanda.ik.frantik_ik)."""

import mujoco
import numpy as np
import pytest

from tampanda import FrankaEnvironment, SCENE_MJX
from tampanda.ik import FrantikIK, MinkIK
from tampanda.ik.frantik_ik import Q_MAX, Q_MIN, _T_LINK7_EE, franka_ik, franka_ik_batch

from conftest import fk_error, random_reachable_poses


def _random_ee_targets(model, n, seed):
    """(T_base_ee, q_seed) pairs: FRANTIK EE frames of random configurations."""
    rng = np.random.default_rng(seed)
    d = mujoco.MjData(model)
    link7 = model.body("link7").id
    lo, hi = model.jnt_range[:7, 0], model.jnt_range[:7, 1]
    out = []
    for _ in range(n):
        d.qpos[:7] = rng.uniform(lo, hi)
        mujoco.mj_kinematics(model, d)
        T = np.eye(4)
        T[:3, :3] = d.xmat[link7].reshape(3, 3)
        T[:3, 3] = d.xpos[link7]
        out.append((np.ascontiguousarray(T @ _T_LINK7_EE), rng.uniform(lo, hi)))
    return out


def test_batch_matches_scalar(frantik_env):
    rng = np.random.default_rng(1)
    for T, q_seed in _random_ee_targets(frantik_env.model, 100, seed=1):
        q7s = rng.uniform(-2.95, 2.95, 8)
        batch = franka_ik_batch(T, q7s, q_seed)
        scalar = np.concatenate([franka_ik(T, q7, q_seed) for q7 in q7s])
        np.testing.assert_array_equal(np.isnan(batch).any(axis=1), np.isnan(scalar).any(axis=1))
        valid = ~np.isnan(batch).any(axis=1)
        np.testing.assert_allclose(batch[valid], scalar[valid], atol=1e-10)


def test_matches_reference_frantik(frantik_env):
    """The numpy port matches the original C++ FRANTIK, if installed."""
    frantik = pytest.importorskip("frantik")
    rng = np.random.default_rng(2)
    for T, q_seed in _random_ee_targets(frantik_env.model, 200, seed=2):
        q7 = rng.uniform(-2.8, 2.8)
        ours = franka_ik(T, q7, q_seed)
        ref = np.asarray(frantik.ik(T, q7, q_seed))
        for a, b in zip(ours, ref):
            if not np.isnan(b).any() and b[1] == 0.0:
                continue  # original's shoulder-singularity fallback, deliberately changed
            assert np.isnan(a).any() == np.isnan(b).any()
            if not np.isnan(a).any():
                np.testing.assert_allclose(a, b, atol=1e-10)


def test_fk_ik_roundtrip(frantik_env):
    """Reachable poses are solved exactly (also checks the site/EE frame offset)."""
    env = frantik_env
    for pos, quat in random_reachable_poses(env.model, 50, seed=3):
        env.reset()
        env.ik.set_target_position(pos, quat)
        assert env.ik.converge_ik(0.005)
        pos_err, ori_err = fk_error(env.model, env.ik.configuration.q[:7], pos, quat)
        assert pos_err < 2e-3 and ori_err < 2e-3


def test_solve_all_solutions_valid(frantik_env):
    env = frantik_env
    for pos, quat in random_reachable_poses(env.model, 5, seed=4):
        env.reset()
        sols = env.ik.solve_all(pos, quat)
        assert len(sols) > 0
        assert np.all(sols > Q_MIN) and np.all(sols < Q_MAX)
        dist = np.linalg.norm(sols - env.ik.configuration.q[:7], axis=1)
        assert np.all(np.diff(dist) >= 0), "solutions must be sorted by distance"
        for q in sols[:: max(1, len(sols) // 20)]:
            pos_err, ori_err = fk_error(env.model, q, pos, quat)
            assert pos_err < 2e-3 and ori_err < 2e-3


def test_unreachable_target(frantik_env):
    env = frantik_env
    env.reset()
    env.ik.set_target_position(np.array([2.0, 0.0, 0.5]), np.array([0.0, 1.0, 0.0, 0.0]))
    assert not env.ik.converge_ik(0.005)
    assert len(env.ik.solve_all()) == 0


def test_prefers_solution_near_start(frantik_env):
    """basic_ik.py target from home: no q1/q3 counter-rotated ("twisted") arm."""
    env = frantik_env
    env.reset()
    # non-unit quaternion, as in the example
    env.ik.set_target_position(np.array([0.695, 0.0, 0.621]), np.array([0.001, 0.835, 0.002, 0.550]))
    assert env.ik.converge_ik(0.005)
    q = env.ik.configuration.q[:7]
    assert abs(q[0]) < 0.2 and abs(q[2]) < 0.2, q


def test_env_ik_solver_switch():
    assert isinstance(FrankaEnvironment(SCENE_MJX.as_posix()).ik, MinkIK)
    assert isinstance(FrankaEnvironment(SCENE_MJX.as_posix(), ik_solver="frantik").ik, FrantikIK)
    with pytest.raises(ValueError):
        FrankaEnvironment(SCENE_MJX.as_posix(), ik_solver="nope")

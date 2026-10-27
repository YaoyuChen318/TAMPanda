"""Shared fixtures for the IK / planning tests."""

import mujoco
import numpy as np
import pytest

from tampanda import FrankaEnvironment, SCENE_MJX
from tampanda.symbolic.domains.tabletop import GridDomain, StateManager
from tampanda.symbolic.domains.tabletop.env_builder import make_symbolic_builder


def site_pose(model, data, site="attachment_site"):
    """World position and WXYZ quaternion of a site."""
    sid = model.site(site).id
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, data.site_xmat[sid])
    return data.site_xpos[sid].copy(), quat


def fk_error(model, q_arm, pos, quat, site="attachment_site"):
    """(position error m, orientation error rad) of ``q_arm`` against a target pose."""
    d = mujoco.MjData(model)
    d.qpos[:7] = q_arm
    mujoco.mj_kinematics(model, d)
    p, q = site_pose(model, d, site)
    dot = abs(float(np.dot(q, np.asarray(quat) / np.linalg.norm(quat))))
    return np.linalg.norm(p - pos), 2.0 * np.arccos(min(1.0, dot))


def random_reachable_poses(model, n, seed=0, margin=0.05):
    """Target poses produced by FK of random in-limit configurations."""
    rng = np.random.default_rng(seed)
    lo, hi = model.jnt_range[:7, 0] + margin, model.jnt_range[:7, 1] - margin
    d = mujoco.MjData(model)
    poses = []
    for _ in range(n):
        d.qpos[:7] = rng.uniform(lo, hi)
        mujoco.mj_kinematics(model, d)
        poses.append(site_pose(model, d))
    return poses


@pytest.fixture(scope="module")
def frantik_env():
    return FrankaEnvironment(SCENE_MJX.as_posix(), ik_solver="frantik")


def make_tabletop(ik_solver, seed=0, n_cylinders=3):
    """Tabletop scene with cylinders placed as in examples/grasping_*_planner.py."""
    env = make_symbolic_builder().build_env(rate=200.0, ik_solver=ik_solver)
    grid = GridDomain(
        model=env.model, cell_size=0.04, working_area=(0.4, 0.3),
        table_body_name="simple_table", table_geom_name="simple_table_surface",
        grid_offset_x=0.05, grid_offset_y=0.25,
    )
    StateManager(grid, env).sample_random_state(n_cylinders=n_cylinders, seed=seed)
    return env

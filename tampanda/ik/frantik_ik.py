"""Analytical IK for the Franka Panda.

Numpy port of FRANTIK's ``Franka::ik`` (CoMMALab/frantik, Apache-2.0), based on
He & Liu, ICCMA 2021.  Unlike the original, the shoulder-singularity fallback is
only used when the q2 axis is exactly degenerate.
"""

from typing import Optional

import mujoco
import numpy as np
import mink

from tampanda.core.base_ik import BaseIK


D1 = 0.3330
D3 = 0.3160
D5 = 0.3840
D7E = 0.2104  # link7 -> Franka Hand EE (flange 0.107 + TCP 0.1034)
A4 = 0.0825
A7 = 0.0880

Q_MIN = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
Q_MAX = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])

_LL24 = A4 * A4 + D3 * D3
_LL46 = A4 * A4 + D5 * D5
_L24 = np.sqrt(_LL24)
_L46 = np.sqrt(_LL46)
_THETA_H46 = np.arctan(D5 / A4)
_THETA_342 = np.arctan(D3 / A4)
_THETA_46H = np.pi / 2 - np.arctan(D5 / A4)

_SINGULAR_EPS = 1e-9


def _normalized(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def franka_ik(T_base_ee: np.ndarray, q7: float, q_actual: np.ndarray) -> np.ndarray:
    """Solve for a fixed q7.  Returns (4, 7) solutions; invalid rows are NaN.

    ``T_base_ee`` is the hand EE frame in link0; ``q_actual`` is only used at
    the shoulder singularity.
    """
    q_all = np.full((4, 7), np.nan)
    if q7 <= Q_MIN[6] or q7 >= Q_MAX[6]:
        return q_all
    q_all[:, 6] = q7

    # compute p_6
    R_ee = T_base_ee[:3, :3]
    z_ee = T_base_ee[:3, 2]
    p_ee = T_base_ee[:3, 3]
    p_7 = p_ee - D7E * z_ee

    x_ee_6 = np.array([np.cos(q7 - np.pi / 4), -np.sin(q7 - np.pi / 4), 0.0])
    x_6 = _normalized(R_ee @ x_ee_6)
    p_6 = p_7 - A7 * x_6

    # compute q4
    p_2 = np.array([0.0, 0.0, D1])
    V26 = p_6 - p_2
    LL26 = V26 @ V26
    L26 = np.sqrt(LL26)
    if _L24 + _L46 < L26 or _L24 + L26 < _L46 or L26 + _L46 < _L24:
        return np.full((4, 7), np.nan)

    theta246 = np.arccos((_LL24 + _LL46 - LL26) / 2.0 / _L24 / _L46)
    q4 = theta246 + _THETA_H46 + _THETA_342 - 2.0 * np.pi
    if q4 <= Q_MIN[3] or q4 >= Q_MAX[3]:
        return np.full((4, 7), np.nan)
    q_all[:, 3] = q4

    # compute q6
    theta462 = np.arccos(np.clip((LL26 + _LL46 - _LL24) / 2.0 / L26 / _L46, -1.0, 1.0))
    theta26H = _THETA_46H + theta462
    D26 = -L26 * np.cos(theta26H)

    Z_6 = np.cross(z_ee, x_6)
    Y_6 = np.cross(Z_6, x_6)
    R_6 = np.column_stack([x_6, _normalized(Y_6), _normalized(Z_6)])
    V_6_62 = R_6.T @ (-V26)

    Phi6 = np.arctan2(V_6_62[1], V_6_62[0])
    sin_theta6 = D26 / np.hypot(V_6_62[0], V_6_62[1])
    if abs(sin_theta6) > 1.0:
        return np.full((4, 7), np.nan)
    Theta6 = np.arcsin(sin_theta6)

    q6 = [np.pi - Theta6 - Phi6, Theta6 - Phi6]
    for i in range(2):
        if q6[i] <= Q_MIN[5]:
            q6[i] += 2.0 * np.pi
        elif q6[i] >= Q_MAX[5]:
            q6[i] -= 2.0 * np.pi
        if q6[i] <= Q_MIN[5] or q6[i] >= Q_MAX[5]:
            q_all[2 * i:2 * i + 2] = np.nan
        else:
            q_all[2 * i:2 * i + 2, 5] = q6[i]

    if np.isnan(q_all[0, 5]) and np.isnan(q_all[2, 5]):
        return np.full((4, 7), np.nan)

    # compute q1 & q2
    thetaP26 = 3.0 * np.pi / 2 - theta462 - theta246 - _THETA_342
    thetaP = np.pi - thetaP26 - theta26H
    LP6 = L26 * np.sin(thetaP26) / np.sin(thetaP)

    z_5_all = [None] * 4
    V2P_all = [None] * 4
    for i in range(2):
        z_6_5 = np.array([np.sin(q6[i]), np.cos(q6[i]), 0.0])
        z_5 = R_6 @ z_6_5
        V2P = p_6 - LP6 * z_5 - p_2
        z_5_all[2 * i] = z_5_all[2 * i + 1] = z_5
        V2P_all[2 * i] = V2P_all[2 * i + 1] = V2P

        if np.hypot(V2P[0], V2P[1]) < _SINGULAR_EPS:
            q_all[2 * i:2 * i + 2, 0] = q_actual[0]
            q_all[2 * i:2 * i + 2, 1] = 0.0
        else:
            q1 = np.arctan2(V2P[1], V2P[0])
            q2 = np.arccos(np.clip(V2P[2] / np.linalg.norm(V2P), -1.0, 1.0))
            q_all[2 * i, 0] = q1
            q_all[2 * i, 1] = q2
            q_all[2 * i + 1, 0] = q1 + (np.pi if q1 < 0 else -np.pi)
            q_all[2 * i + 1, 1] = -q2

    for i in range(4):
        if np.isnan(q_all[i, 5]):
            continue
        if (q_all[i, 0] <= Q_MIN[0] or q_all[i, 0] >= Q_MAX[0]
                or q_all[i, 1] <= Q_MIN[1] or q_all[i, 1] >= Q_MAX[1]):
            q_all[i] = np.nan
            continue

        # compute q3
        z_3 = _normalized(V2P_all[i])
        y_3 = _normalized(-np.cross(V26, V2P_all[i]))
        x_3 = np.cross(y_3, z_3)
        c1, s1 = np.cos(q_all[i, 0]), np.sin(q_all[i, 0])
        R_1 = np.array([[c1, -s1, 0.0], [s1, c1, 0.0], [0.0, 0.0, 1.0]])
        c2, s2 = np.cos(q_all[i, 1]), np.sin(q_all[i, 1])
        R_1_2 = np.array([[c2, -s2, 0.0], [0.0, 0.0, 1.0], [-s2, -c2, 0.0]])
        R_2 = R_1 @ R_1_2
        x_2_3 = R_2.T @ x_3
        q_all[i, 2] = np.arctan2(x_2_3[2], x_2_3[0])
        if q_all[i, 2] <= Q_MIN[2] or q_all[i, 2] >= Q_MAX[2]:
            q_all[i] = np.nan
            continue

        # compute q5
        VH4 = p_2 + D3 * z_3 + A4 * x_3 - p_6 + D5 * z_5_all[i]
        c6, s6 = np.cos(q_all[i, 5]), np.sin(q_all[i, 5])
        R_5_6 = np.array([[c6, -s6, 0.0], [0.0, 0.0, -1.0], [s6, c6, 0.0]])
        R_5 = R_6 @ R_5_6.T
        V_5_H4 = R_5.T @ VH4
        q_all[i, 4] = -np.arctan2(V_5_H4[1], V_5_H4[0])
        if q_all[i, 4] <= Q_MIN[4] or q_all[i, 4] >= Q_MAX[4]:
            q_all[i] = np.nan
            continue

    return q_all


def _normalized_rows(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def franka_ik_batch(T_base_ee: np.ndarray, q7s: np.ndarray, q_actual: np.ndarray) -> np.ndarray:
    """Vectorised ``franka_ik``; row ``4 * k + i`` is solution ``i`` for ``q7s[k]``."""
    q7 = np.asarray(q7s, dtype=np.float64)
    K = len(q7)
    with np.errstate(invalid="ignore", divide="ignore"):
        valid = (q7 > Q_MIN[6]) & (q7 < Q_MAX[6])

        R_ee = T_base_ee[:3, :3]
        z_ee = T_base_ee[:3, 2]
        p_7 = T_base_ee[:3, 3] - D7E * z_ee
        x_ee_6 = np.stack([np.cos(q7 - np.pi / 4), -np.sin(q7 - np.pi / 4), np.zeros(K)], axis=1)
        x_6 = _normalized_rows(x_ee_6 @ R_ee.T)
        p_6 = p_7 - A7 * x_6

        p_2 = np.array([0.0, 0.0, D1])
        V26 = p_6 - p_2
        LL26 = np.einsum("ki,ki->k", V26, V26)
        L26 = np.sqrt(LL26)
        valid &= ~((_L24 + _L46 < L26) | (_L24 + L26 < _L46) | (L26 + _L46 < _L24))
        theta246 = np.arccos((_LL24 + _LL46 - LL26) / 2.0 / _L24 / _L46)
        q4 = theta246 + _THETA_H46 + _THETA_342 - 2.0 * np.pi
        valid &= (q4 > Q_MIN[3]) & (q4 < Q_MAX[3])

        theta462 = np.arccos(np.clip((LL26 + _LL46 - _LL24) / 2.0 / L26 / _L46, -1.0, 1.0))
        theta26H = _THETA_46H + theta462
        D26 = -L26 * np.cos(theta26H)
        Z_6 = np.cross(z_ee, x_6)
        Y_6 = np.cross(Z_6, x_6)
        R_6 = np.stack([x_6, _normalized_rows(Y_6), _normalized_rows(Z_6)], axis=2)  # (K, 3, 3)
        V_6_62 = np.einsum("kji,kj->ki", R_6, -V26)
        Phi6 = np.arctan2(V_6_62[:, 1], V_6_62[:, 0])
        sin_theta6 = D26 / np.hypot(V_6_62[:, 0], V_6_62[:, 1])
        valid &= np.abs(sin_theta6) <= 1.0
        Theta6 = np.arcsin(sin_theta6)

        q6 = np.stack([np.pi - Theta6 - Phi6, Theta6 - Phi6], axis=1)  # (K, 2)
        q6 = np.where(q6 <= Q_MIN[5], q6 + 2.0 * np.pi, np.where(q6 >= Q_MAX[5], q6 - 2.0 * np.pi, q6))
        q6_ok = (q6 > Q_MIN[5]) & (q6 < Q_MAX[5])

        thetaP26 = 3.0 * np.pi / 2 - theta462 - theta246 - _THETA_342
        thetaP = np.pi - thetaP26 - theta26H
        LP6 = L26 * np.sin(thetaP26) / np.sin(thetaP)

        z_6_5 = np.stack([np.sin(q6), np.cos(q6), np.zeros_like(q6)], axis=2)  # (K, 2, 3)
        z_5 = np.einsum("kij,kbj->kbi", R_6, z_6_5)
        V2P = p_6[:, None] - LP6[:, None, None] * z_5 - p_2  # (K, 2, 3)

        singular = np.hypot(V2P[..., 0], V2P[..., 1]) < _SINGULAR_EPS
        q1a = np.where(singular, q_actual[0], np.arctan2(V2P[..., 1], V2P[..., 0]))
        q2a = np.where(singular, 0.0,
                       np.arccos(np.clip(V2P[..., 2] / np.linalg.norm(V2P, axis=-1), -1.0, 1.0)))
        q1b = np.where(singular, q_actual[0], q1a + np.where(q1a < 0, np.pi, -np.pi))
        q2b = np.where(singular, 0.0, -q2a)

        # rows per q7: [q6 branch 0, q6 branch 0 flipped, branch 1, branch 1 flipped]
        q1 = np.stack([q1a, q1b], axis=2).reshape(K, 4)
        q2 = np.stack([q2a, q2b], axis=2).reshape(K, 4)
        q6r = np.repeat(q6, 2, axis=1)
        z5r = np.repeat(z_5, 2, axis=1)
        V2Pr = np.repeat(V2P, 2, axis=1)
        row_ok = valid[:, None] & np.repeat(q6_ok, 2, axis=1)
        row_ok &= (q1 > Q_MIN[0]) & (q1 < Q_MAX[0]) & (q2 > Q_MIN[1]) & (q2 < Q_MAX[1])

        z_3 = _normalized_rows(V2Pr)
        y_3 = _normalized_rows(-np.cross(V26[:, None], V2Pr))
        x_3 = np.cross(y_3, z_3)
        c1, s1 = np.cos(q1), np.sin(q1)
        c2, s2 = np.cos(q2), np.sin(q2)
        # x_2_3 = (R_1 @ R_1_2)^T x_3, expanded
        u = c1 * x_3[..., 0] + s1 * x_3[..., 1]
        w = x_3[..., 2]
        x23_0 = c2 * u - s2 * w
        x23_2 = (-s1 * x_3[..., 0] + c1 * x_3[..., 1])
        q3 = np.arctan2(x23_2, x23_0)
        row_ok &= (q3 > Q_MIN[2]) & (q3 < Q_MAX[2])

        VH4 = p_2 + D3 * z_3 + A4 * x_3 - p_6[:, None] + D5 * z5r
        c6, s6 = np.cos(q6r), np.sin(q6r)
        # V_5_H4 = R_5_6 (R_6^T VH4), expanded
        v6 = np.einsum("kji,kbj->kbi", R_6, VH4)
        v5_0 = c6 * v6[..., 0] - s6 * v6[..., 1]
        v5_1 = -v6[..., 2]
        q5 = -np.arctan2(v5_1, v5_0)
        row_ok &= (q5 > Q_MIN[4]) & (q5 < Q_MAX[4])

        q_all = np.stack([q1, q2, q3, np.repeat(q4[:, None], 4, axis=1), q5, q6r,
                          np.repeat(q7[:, None], 4, axis=1)], axis=2)
        row_ok &= np.isfinite(q_all).all(axis=2)
    q_all[~row_ok] = np.nan
    return q_all.reshape(K * 4, 7)


# Franka Hand EE frame used by the solver: link7 offset by D7E, rotated -45 deg about z.
_T_LINK7_EE = np.eye(4)
_T_LINK7_EE[:3, :3] = np.array([
    [np.cos(-np.pi / 4), -np.sin(-np.pi / 4), 0.0],
    [np.sin(-np.pi / 4),  np.cos(-np.pi / 4), 0.0],
    [0.0, 0.0, 1.0],
])
_T_LINK7_EE[2, 3] = D7E


def _pose_to_mat(pos: np.ndarray, quat: np.ndarray) -> np.ndarray:
    """Build a 4x4 transform from a position and a [w, x, y, z] quaternion."""
    mat = np.zeros(9)
    quat = np.asarray(quat, dtype=np.float64)
    mujoco.mju_quat2Mat(mat, quat / np.linalg.norm(quat))
    T = np.eye(4)
    T[:3, :3] = mat.reshape(3, 3)
    T[:3, 3] = pos
    return T


class FrantikIK(BaseIK):
    """IK by sampling joint 7 and solving the other joints in closed form.

    Same interface as MinkIK.  ``solve_all`` returns every solution sorted by
    distance to the current configuration; ``converge_ik`` keeps the closest.
    q7 is sampled densely: near the shoulder singularity (e.g. the home pose)
    the solutions close to the current configuration span only ~0.01 rad of q7.
    """

    def __init__(self, model, data, target_name: str = "target",
                 site_name: str = "attachment_site", n_q7_samples: int = 512,
                 n_q7_refine: int = 64):
        self.model = model
        self.data = data
        self.target_name = target_name
        self.site_name = site_name
        self.n_q7_samples = n_q7_samples
        self.n_q7_refine = n_q7_refine
        self.pos_threshold = 0.002
        self.ori_threshold = 1e-3
        self.max_iters = 1000  # unused; set by callers written for MinkIK

        self._target: Optional[np.ndarray] = None

        self.configuration = mink.Configuration(model)
        self._fk_data = mujoco.MjData(model)

        self._link0_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "link0")
        self._link7_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "link7")
        self._site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)

        cdata = self.configuration.data
        T_world_link7 = self._body_pose(cdata, self._link7_id)
        T_world_site = self._site_pose(cdata)
        T_link7_site = np.linalg.inv(T_world_link7) @ T_world_site
        self._T_site_ee = np.linalg.inv(T_link7_site) @ _T_LINK7_EE

    @staticmethod
    def _body_pose(data, body_id: int) -> np.ndarray:
        T = np.eye(4)
        T[:3, :3] = data.xmat[body_id].reshape(3, 3)
        T[:3, 3] = data.xpos[body_id]
        return T

    def _site_pose(self, data) -> np.ndarray:
        T = np.eye(4)
        T[:3, :3] = data.site_xmat[self._site_id].reshape(3, 3)
        T[:3, 3] = data.site_xpos[self._site_id]
        return T

    def set_target_position(self, pos: np.ndarray, quat: np.ndarray):
        self._target = _pose_to_mat(pos, quat)
        if self.model.nmocap > 0:
            self.data.mocap_pos[0] = pos
            self.data.mocap_quat[0] = quat

    def _target_in_base(self, T_world_site: np.ndarray) -> np.ndarray:
        T_world_link0 = self._body_pose(self.configuration.data, self._link0_id)
        T_base_ee = np.linalg.inv(T_world_link0) @ T_world_site @ self._T_site_ee
        return np.ascontiguousarray(T_base_ee, dtype=np.float64)

    def _candidates(self, T_world_site: np.ndarray) -> np.ndarray:
        """Unverified analytical solutions, sorted by distance to the current configuration."""
        T_base_ee = self._target_in_base(T_world_site)
        q_current = np.asarray(self.configuration.q[:7], dtype=np.float64)

        eps = 1e-3
        lo, hi = Q_MIN[6] + eps, Q_MAX[6] - eps
        grid = np.linspace(lo, hi, self.n_q7_samples)
        q7s = np.append(grid, np.clip(q_current[6], lo, hi))
        sols = franka_ik_batch(T_base_ee, q7s, q_current)
        sols = sols[~np.isnan(sols).any(axis=1)]
        if len(sols) == 0:
            return sols

        # refine q7 around the best coarse solution
        q7_best = sols[np.argmin(np.linalg.norm(sols - q_current, axis=1)), 6]
        h = grid[1] - grid[0]
        fine = np.linspace(max(lo, q7_best - h), min(hi, q7_best + h), self.n_q7_refine)
        fine_sols = franka_ik_batch(T_base_ee, fine, q_current)
        sols = np.concatenate([sols, fine_sols[~np.isnan(fine_sols).any(axis=1)]])

        return sols[np.argsort(np.linalg.norm(sols - q_current, axis=1))]

    def solve_all(self, pos: Optional[np.ndarray] = None,
                  quat: Optional[np.ndarray] = None) -> np.ndarray:
        """All (N, 7) solutions for the given pose, or the current target if omitted."""
        T_world_site = self._target if pos is None else _pose_to_mat(pos, quat)
        if T_world_site is None:
            raise ValueError("No IK target set.")
        # FK check drops approximate solutions at the shoulder singularity
        solutions = [q for q in self._candidates(T_world_site) if self._reaches(q, T_world_site)]
        return np.array(solutions) if solutions else np.empty((0, 7))

    def _reaches(self, q_arm: np.ndarray, T_world_site: np.ndarray) -> bool:
        self._fk_data.qpos[:] = self.configuration.q
        self._fk_data.qpos[:7] = q_arm
        mujoco.mj_kinematics(self.model, self._fk_data)
        T = self._site_pose(self._fk_data)
        pos_err = np.linalg.norm(T[:3, 3] - T_world_site[:3, 3])
        R_err = T[:3, :3].T @ T_world_site[:3, :3]
        ori_err = np.arccos(np.clip((np.trace(R_err) - 1.0) / 2.0, -1.0, 1.0))
        return pos_err <= self.pos_threshold and ori_err <= self.ori_threshold

    def converge_ik(self, dt: float = 0.0) -> bool:
        if self._target is None:
            raise ValueError("No IK target set.")
        for q_arm in self._candidates(self._target):
            if self._reaches(q_arm, self._target):
                q = self.configuration.q.copy()
                q[:7] = q_arm
                self.configuration.update(q)
                return True
        return False

    def update_configuration(self, qpos: np.ndarray):
        self.configuration.update(qpos)

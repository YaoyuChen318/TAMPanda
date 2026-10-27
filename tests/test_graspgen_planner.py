"""Tests for GraspGenPlanner with a fake server (no GPU, no network)."""

import threading

import mujoco
import numpy as np
import pytest

from tampanda.planners.graspgen_planner import GraspGenClient, GraspGenPlanner
from tampanda.planners.grasp_planner import GRASP_CONTACT_OFFSET, GraspType, quat_to_rotmat

PAD_Z = 0.1037          # pad centre above the hand origin, in GraspGen's frame as well
FINGER_BASE_Z = 0.0584  # finger body origin above the hand origin (base_panda.xml)


def graspgen_grasp(contact, closing_axis, approach=(0.0, 0.0, -1.0)):
    """A GraspGen-convention grasp (hand base, approach +Z, fingers along X) with pads at ``contact``."""
    z = np.asarray(approach, dtype=float)
    x = np.asarray(closing_axis, dtype=float)
    T = np.eye(4)
    T[:3, :3] = np.column_stack([x, np.cross(z, x), z])
    T[:3, 3] = np.asarray(contact) - PAD_Z * z
    return T


class FakeClient:
    def __init__(self, grasps, confidences, gripper="franka_panda", down=False):
        self.grasps, self.confidences = np.asarray(grasps), np.asarray(confidences)
        self.gripper, self.down = gripper, down

    def metadata(self, timeout_ms=None):
        if self.down:
            raise TimeoutError
        return {"gripper_name": self.gripper}

    def infer(self, points, **params):
        return self.grasps, self.confidences


def test_candidates_match_graspgen_grasp(frantik_env):
    """Converted poses put the MuJoCo fingers on either side of the GraspGen closing axis."""
    contact = np.array([0.45, 0.1, 0.35])
    planner = GraspGenPlanner(table_z=None, client=FakeClient([graspgen_grasp(contact, [1, 0, 0])], [0.9]))
    cands = planner.generate_candidates(np.zeros((10, 3)))
    assert len(cands) == 2                     # both Rz(+90) / Rz(-90) hand frames
    env, m = frantik_env, frantik_env.model
    for c in cands:
        assert c.grasp_type == GraspType.SIX_DOF and c.score == pytest.approx(0.9)
        R = quat_to_rotmat(c.grasp_quat)
        np.testing.assert_allclose(c.grasp_pos, contact - R[:, 2] * GRASP_CONTACT_OFFSET, atol=1e-6)
        assert c.approach_pos[2] > c.grasp_pos[2] and c.lift_pos[2] > c.grasp_pos[2]

        env.reset()
        env.ik.set_target_position(c.grasp_pos, c.grasp_quat)
        assert env.ik.converge_ik(0.005)
        d = mujoco.MjData(m)
        d.qpos[:7] = env.ik.configuration.q[:7]
        d.qpos[7:9] = 0.04
        mujoco.mj_forward(m, d)
        left, right = d.xpos[m.body("left_finger").id], d.xpos[m.body("right_finger").id]
        sep = (left - right) / np.linalg.norm(left - right)
        assert abs(sep[0]) > 0.99, f"fingers should close along world X, got {sep}"
        mid = (left + right) / 2
        np.testing.assert_allclose(mid[:2], contact[:2], atol=2e-3)
        assert abs(mid[2] - (contact[2] + PAD_Z - FINGER_BASE_Z)) < 2e-3   # pads at the contact height


def test_grasps_below_table_are_dropped():
    from_below = graspgen_grasp([0.45, 0.1, 0.28], [1, 0, 0], approach=[0, 0, 1])
    ok = graspgen_grasp([0.45, 0.1, 0.35], [0, 1, 0])
    planner = GraspGenPlanner(table_z=0.27, client=FakeClient([from_below, ok], [0.99, 0.5]))
    cands = planner.generate_candidates(np.zeros((10, 3)))
    assert len(cands) == 2 and all(c.score == pytest.approx(0.5) for c in cands)


def test_wrong_gripper_or_no_server_fails_fast():
    grasp = graspgen_grasp([0.45, 0.1, 0.35], [1, 0, 0])
    with pytest.raises(RuntimeError, match="robotiq"):
        GraspGenPlanner(client=FakeClient([grasp], [0.9], gripper="robotiq_2f_140")).generate_candidates(np.zeros((10, 3)))
    with pytest.raises(ConnectionError):
        GraspGenPlanner(client=FakeClient([grasp], [0.9], down=True)).generate_candidates(np.zeros((10, 3)))


def test_zmq_protocol_roundtrip():
    """GraspGenClient against a minimal server speaking GraspGen's msgpack protocol."""
    zmq = pytest.importorskip("zmq")
    msgpack = pytest.importorskip("msgpack")
    msgpack_numpy = pytest.importorskip("msgpack_numpy")
    grasp = graspgen_grasp([0.45, 0.1, 0.35], [1, 0, 0]).astype(np.float32)

    ctx = zmq.Context.instance()
    rep = ctx.socket(zmq.REP)
    port = rep.bind_to_random_port("tcp://127.0.0.1")

    def serve():
        for _ in range(2):
            req = msgpack.unpackb(rep.recv(), object_hook=msgpack_numpy.decode, raw=False)
            if req["action"] == "metadata":
                resp = {"gripper_name": "franka_panda"}
            else:
                assert req["point_cloud"].shape == (50, 3) and req["num_grasps"] == 200
                resp = {"grasps": grasp[None], "confidences": np.array([0.8], np.float32)}
            rep.send(msgpack.packb(resp, default=msgpack_numpy.encode, use_bin_type=True))

    t = threading.Thread(target=serve, daemon=True)
    t.start()
    client = GraspGenClient(port=port, timeout_ms=5000)
    cands = GraspGenPlanner(table_z=None, client=client).generate_candidates(np.zeros((50, 3)))
    t.join(timeout=5)
    client.close()
    rep.close()
    assert len(cands) == 2 and cands[0].score == pytest.approx(0.8)

"""Grasp candidates from a remote GraspGen server (Murali et al., 2025).

GraspGen needs a CUDA GPU, so it runs as its own ZMQ server (NVlabs/GraspGen,
``graspgen_franka_panda.yml``); this module only speaks its msgpack protocol.
GraspGen's code and weights are licensed for non-commercial use only.

GraspGen's Franka frame is the hand base with fingers along X; the MuJoCo hand
has them along Y, so ``T_hand = T_graspgen @ Rz(±90°)``.  Both signs are kept
since they differ in joint-7 reachability.
"""

from typing import List, Optional

import numpy as np

from tampanda.planners.grasp_planner import GraspCandidate, GraspPlanner, GraspType, filter_reachable

HAND_TO_SITE_Z: float = 0.09        # attachment_site above the hand origin (base_panda.xml)
_FINGERTIP_Z: float = 0.1126        # fingertip bottom, for the table check
_FINGER_HALF_OPEN: float = 0.04

_RZ_POS90 = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
_RZ_NEG90 = _RZ_POS90.T


class GraspGenClient:
    """Minimal ZMQ REQ client for the GraspGen inference server (msgpack protocol)."""

    def __init__(self, host: str = "localhost", port: int = 5556, timeout_ms: int = 60_000):
        try:
            import zmq
            import msgpack
            import msgpack_numpy
        except ImportError as e:
            raise ImportError(
                "GraspGenClient requires pyzmq, msgpack and msgpack-numpy "
                "(pip install 'tampanda[graspgen]')."
            ) from e
        self._zmq = zmq
        self._msgpack = msgpack
        self._encode = msgpack_numpy.encode
        self._decode = msgpack_numpy.decode

        self._addr = f"tcp://{host}:{port}"
        self._timeout_ms = timeout_ms
        self._ctx = zmq.Context()
        self._socket = None

    def _connect(self):
        sock = self._ctx.socket(self._zmq.REQ)
        sock.setsockopt(self._zmq.SNDTIMEO, self._timeout_ms)
        sock.setsockopt(self._zmq.LINGER, 0)
        sock.connect(self._addr)
        return sock

    def request(self, payload: dict, timeout_ms: Optional[int] = None) -> dict:
        if self._socket is None:
            self._socket = self._connect()
        self._socket.setsockopt(self._zmq.RCVTIMEO, timeout_ms or self._timeout_ms)
        try:
            self._socket.send(self._msgpack.packb(payload, default=self._encode, use_bin_type=True))
            response = self._msgpack.unpackb(self._socket.recv(), object_hook=self._decode, raw=False)
        except self._zmq.error.Again as e:
            # a REQ socket is unusable after a timeout
            self._socket.close()
            self._socket = None
            raise TimeoutError(f"GraspGen server at {self._addr} did not respond") from e
        if "error" in response:
            raise RuntimeError(f"GraspGen server error: {response['error']}")
        return response

    def metadata(self, timeout_ms: Optional[int] = None) -> dict:
        return self.request({"action": "metadata"}, timeout_ms)

    def infer(self, points: np.ndarray, **params) -> tuple:
        """Return (grasps (M, 4, 4), confidences (M,)) in the frame of ``points``."""
        payload = {"action": "infer", "point_cloud": np.asarray(points, dtype=np.float32)}
        payload.update(params)
        response = self.request(payload)
        grasps = np.asarray(response["grasps"], dtype=np.float64).reshape(-1, 4, 4)
        confidences = np.asarray(response["confidences"], dtype=np.float64).reshape(-1)
        return grasps, confidences

    def close(self):
        if self._socket is not None:
            self._socket.close()
            self._socket = None
        self._ctx.term()


class GraspGenPlanner:
    """Grasp candidates from a GraspGen server, same interface as PointCloudGraspPlanner.

    GraspGen only sees the object, so grasps reaching below ``table_z`` are dropped.
    ``grasp_threshold`` = -1 returns the top-k instead of thresholding.  The
    server must serve ``gripper``; it is checked on first use, since GraspGen's
    server defaults to a Robotiq model.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5556,
        num_grasps: int = 200,
        topk_num_grasps: int = 100,
        grasp_threshold: float = -1.0,
        approach_dist: float = 0.12,
        lift_height: float = 0.20,
        table_z: Optional[float] = 0.27,
        table_clearance: float = 0.01,
        gripper: str = "franka_panda",
        client: Optional[GraspGenClient] = None,
    ):
        self.num_grasps = num_grasps
        self.topk_num_grasps = topk_num_grasps
        self.grasp_threshold = grasp_threshold
        self.approach_dist = float(approach_dist)
        self.lift_height = float(lift_height)
        self.table_z = table_z
        self.table_clearance = float(table_clearance)
        self.gripper = gripper
        self._client = client or GraspGenClient(host, port)
        self._server_checked = False

    def check_server(self, timeout_ms: int = 5000) -> dict:
        """Fail fast if the server is down or serves another gripper; returns its metadata."""
        try:
            meta = self._client.metadata(timeout_ms=timeout_ms)
        except TimeoutError as e:
            raise ConnectionError(
                f"No GraspGen server answered: start it with "
                f"--gripper_config .../graspgen_{self.gripper}.yml") from e
        if meta.get("gripper_name") != self.gripper:
            raise RuntimeError(f"GraspGen server serves '{meta.get('gripper_name')}', "
                               f"expected '{self.gripper}'")
        self._server_checked = True
        return meta

    def generate_candidates(self, points: np.ndarray) -> List[GraspCandidate]:
        """Candidates for the segmented object cloud, best GraspGen confidence first."""
        if not self._server_checked:
            self.check_server()
        grasps, confidences = self._client.infer(
            points,
            num_grasps=self.num_grasps,
            topk_num_grasps=self.topk_num_grasps,
            grasp_threshold=self.grasp_threshold,
        )
        candidates: List[GraspCandidate] = []
        for T_gg, conf in zip(grasps, confidences):
            for Rz in (_RZ_POS90, _RZ_NEG90):
                c = self._make_candidate(T_gg, Rz, float(conf))
                if c is not None:
                    candidates.append(c)
        candidates.sort(key=lambda c: c.score, reverse=True)
        return candidates

    def filter_reachable(self, candidates: List[GraspCandidate], ik,
                         max_candidates: Optional[int] = None) -> List[GraspCandidate]:
        """Keep candidates whose grasp and approach poses both have an IK solution."""
        return filter_reachable(candidates, ik, max_candidates)

    def _make_candidate(self, T_gg: np.ndarray, Rz: np.ndarray, conf: float) -> Optional[GraspCandidate]:
        R_hand = T_gg[:3, :3] @ Rz
        p_hand = T_gg[:3, 3]
        approach_dir = R_hand[:, 2]

        if self.table_z is not None:
            floor = self.table_z + self.table_clearance
            fingertips = [p_hand + R_hand @ np.array([0.0, s * _FINGER_HALF_OPEN, _FINGERTIP_Z])
                          for s in (-1.0, 1.0)]
            if p_hand[2] < floor or min(f[2] for f in fingertips) < floor:
                return None

        grasp_pos = p_hand + approach_dir * HAND_TO_SITE_Z
        approach_pos = grasp_pos - approach_dir * self.approach_dist
        if self.table_z is not None and approach_pos[2] < self.table_z + self.table_clearance:
            return None
        lift_pos = grasp_pos + np.array([0.0, 0.0, self.lift_height])

        return GraspCandidate(
            grasp_type=GraspType.SIX_DOF,
            grasp_quat=GraspPlanner._rotmat_to_quat(R_hand),
            approach_pos=approach_pos,
            grasp_pos=grasp_pos,
            lift_pos=lift_pos,
            score=conf,
        )

"""Grasp candidates from a segmented point cloud.

Two planners share the ``points -> List[GraspCandidate]`` interface:
PointCloudGraspPlanner fits a box (PCA) and reuses GraspPlanner's templates;
AntipodalGraspPlanner samples 6-DOF grasps on the surface and ranks them by
robust Ferrari-Canny quality.

PointCloudGraspPlanner
======================
Fits an oriented bounding box (OBB) to a segmented point cloud via PCA,
then delegates candidate generation to GraspPlanner.  Acts as a drop-in
replacement wherever GraspPlanner is used when true object geometry is not
available (unseen objects, real-sensor pipelines, etc.).

Algorithm
---------
1. Compute centroid and centre the cloud.
2. SVD of the centred cloud → principal axes sorted by variance descending.
3. Re-order axes so the one most aligned with world-Z becomes body-Z
   (correct for both flat boxes and upright objects).
4. Fix rotation handedness (det = +1).
5. Project cloud onto each axis → half-extents and geometric centre.
6. Pass (pos, half_size, quat) to GraspPlanner.generate_candidates().

References
----------
- Point Cloud Projective Analysis for Part-Based Grasp Planning (2020)
  https://www.researchgate.net/publication/342320540
  Direct precedent: PCA on segmented point clouds for grasp axis estimation.

- Improved PCA + ICP for workpiece point cloud pose estimation (2025)
  https://doi.org/10.1007/s00371-025-04154-7
  Validates PCA as the correct foundation for object axis estimation.

- QuickGrasp: Lightweight Antipodal Grasp Planning with Point Clouds
  arXiv:2504.19716 / IEEE ICRA 2025
  https://arxiv.org/abs/2504.19716
  Validates that analytical, no-ML grasp planning matches learning-based
  methods for tabletop manipulation.

Future upgrade path
-------------------
QuickGrasp's region-growing plane segmentation + force-closure quality
metric can replace or augment the PCA OBB step here for non-convex objects.
The interface (points → List[GraspCandidate]) would remain unchanged, so
PickPlaceExecutor and all callers are unaffected.  See generate_candidates()
for the natural extension point.

"""

from dataclasses import replace
from typing import List, Optional

import numpy as np
from scipy.spatial import ConvexHull, QhullError

from tampanda.planners.grasp_planner import (
    GRASP_CONTACT_OFFSET, GraspCandidate, GraspPlanner, GraspType,
)


# Minimum number of points required for a reliable PCA fit.
_MIN_POINTS: int = 20


class PointCloudGraspPlanner:
    """Generate grasp candidates from a segmented point cloud.

    Mirrors the GraspPlanner constructor so the two are interchangeable
    depending on whether object geometry is known:

        # known geometry
        planner = GraspPlanner(table_z=0.27)
        candidates = planner.generate_candidates(pos, half_size, quat)

        # unknown geometry — only the point cloud is needed
        planner = PointCloudGraspPlanner(table_z=0.27)
        candidates = planner.generate_candidates(points)

    An existing GraspPlanner can be injected to share its configuration:

        pc_planner = PointCloudGraspPlanner(grasp_planner=existing_planner)

    Point clouds are obtained from MujocoCamera (simulation) or any RGBD
    sensor pipeline (real hardware) — the interface is identical:

        clouds = camera.get_multi_camera_segmented_pointcloud(
            ["top_camera", "side_camera"], total_samples_per_object=2000
        )
        pts, _ = clouds["object_name"]
        candidates = pc_planner.generate_candidates(pts)
    """

    def __init__(
        self,
        grasp_planner: GraspPlanner | None = None,
        **kwargs,
    ) -> None:
        """
        Args:
            grasp_planner: Optional pre-configured GraspPlanner to delegate to.
                           If None, a new GraspPlanner is created from **kwargs.
            **kwargs:      Forwarded to GraspPlanner.__init__ when no planner
                           is provided (approach_dist, lift_height, table_z,
                           table_clearance).
        """
        self._planner = grasp_planner or GraspPlanner(**kwargs)

    @property
    def table_z(self) -> float:
        return self._planner.table_z

    def generate_candidates(self, points: np.ndarray) -> List[GraspCandidate]:
        """Return grasp candidates sorted best-first.

        Args:
            points: (N, 3) world-frame point cloud of the target object.
                    Returns [] if the cloud is too sparse for reliable fitting.

        Shape-adaptive behaviour
        ------------------------
        For tall objects (height > widest horizontal dimension):

        * The grasp target is shifted upward so top-down fingers close around
          the upper portion of the object rather than descending through the
          full body height — which would tip the object sideways.

        * FRONT (horizontal approach) candidates are boosted above TOP_DOWN in
          the ranking, since they never need to pass through the object at all.

        Future: replace or augment _fit_obb() with QuickGrasp-style plane
        segmentation + force-closure scoring for non-convex object support.
        """
        if len(points) < _MIN_POINTS:
            return []
        pos, half_size, quat = _fit_obb(points)

        # For tall objects, shift the grasp target toward the upper portion.
        # GraspPlanner places contact at block_pos, so shifting pos[2] up means
        # the fingers close higher on the object — avoiding a full descent
        # through the body.  The shift scales with how much taller the object
        # is than its widest horizontal dimension.
        horiz_max = max(half_size[0], half_size[1])
        if half_size[2] > horiz_max:
            pos = pos.copy()
            pos[2] += (half_size[2] - horiz_max) * 0.4

        candidates = self._planner.generate_candidates(pos, half_size, quat)
        return _boost_front_for_tall(candidates, half_size)


def _boost_front_for_tall(
    candidates: List[GraspCandidate],
    half_size: np.ndarray,
) -> List[GraspCandidate]:
    """Re-rank FRONT candidates above TOP_DOWN for tall objects.

    GraspPlanner._score() always adds +20 for TOP_DOWN, which works well for
    flat blocks but is wrong for tall objects where a horizontal FRONT approach
    is geometrically safer (fingers never pass through the top surface).

    When the object is taller than it is wide (aspect ratio > 1.1), FRONT
    candidates receive +30 so they rank above TOP_DOWN's baseline of +25.
    For squat/flat objects the ranking is unchanged.
    """
    horiz_max = max(half_size[0], half_size[1])
    if half_size[2] <= horiz_max * 1.1:
        return candidates

    reranked = [
        replace(c, score=c.score + 30.0) if c.grasp_type == GraspType.FRONT else c
        for c in candidates
    ]
    reranked.sort(key=lambda c: c.score, reverse=True)
    return reranked


def _fit_obb(
    points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit an oriented bounding box to a point cloud via PCA.

    Args:
        points: (N, 3) array of 3D points in world frame.

    Returns:
        pos:       geometric centre of the OBB in world frame.
        half_size: half-extents along each OBB axis.
        quat:      WXYZ quaternion of the OBB frame relative to world.
    """
    centroid = points.mean(axis=0)
    centered = points - centroid

    # PCA: columns of Vt.T are principal axes, sorted by variance descending.
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    R = Vt.T  # (3, 3) — columns are body-X, body-Y, body-Z candidate axes

    # Re-order so the axis most aligned with world-Z becomes body-Z.
    # SVD places the lowest-variance axis last, which for flat tabletop objects
    # is already the vertical axis.  For tall/upright objects the highest-
    # variance axis is vertical and needs to be moved to the last column.
    dots = np.abs(R.T @ np.array([0.0, 0.0, 1.0]))
    z_idx = int(np.argmax(dots))
    order = [i for i in range(3) if i != z_idx] + [z_idx]
    R = R[:, order]

    # Ensure proper rotation (det must be +1).
    if np.linalg.det(R) < 0:
        R[:, 2] *= -1

    # Project the centred cloud onto each OBB axis.
    proj = centered @ R                           # (N, 3)
    lo, hi = proj.min(axis=0), proj.max(axis=0)
    half_size = (hi - lo) / 2.0
    # Shift from mean centroid to geometric centre (matters for occluded clouds).
    pos = centroid + R @ ((hi + lo) / 2.0)

    quat = GraspPlanner._rotmat_to_quat(R)
    return pos, half_size, quat


# --- AntipodalGraspPlanner ---

# Panda hand geometry in the hand body frame (base_panda.xml, fingers open).
_PAD_Z = (0.0954, 0.1119)        # fingertip pad extent along the approach axis
_PAD_CENTER_Z = 0.1037           # pad centre; attachment_site is at z = 0.09
_PAD_HALF_X = 0.0090
_FINGER_Y = (0.040, 0.062)       # |y| extent of one open finger (pad + finger body)
_FINGER_Z = (0.0584, 0.1126)     # finger base to tip
_FINGER_HALF_X = 0.011
_PALM_HALF = (0.040, 0.100)      # hand_capsule half extents in x, y
_PALM_Z = (-0.010, 0.070)


_JOGGLE_TOL = 1e-6  # Qhull "QJ" joggle; smaller epsilons count as zero


def _tangent_basis(n: np.ndarray):
    helper = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    t1 = np.cross(n, helper)
    t1 /= np.linalg.norm(t1)
    return t1, np.cross(n, t1)


def ferrari_canny(contacts: np.ndarray, normals: np.ndarray, com: np.ndarray,
                  mu: float, torsion_radius: float = 0.005, torque_scale: float = 0.05,
                  n_edges: int = 8) -> float:
    """Ferrari-Canny epsilon (Ferrari & Canny, ICRA 1992); 0 if not force closure.

    ``normals`` point into the object.  Soft-finger contacts: ``torsion_radius``
    bounds torsional friction (0 = point contact).  Torques are taken about
    ``com`` and divided by ``torque_scale`` (about the object radius).
    """
    wrenches = []
    for p, n in zip(np.asarray(contacts), np.asarray(normals)):
        n = n / np.linalg.norm(n)
        t1, t2 = _tangent_basis(n)
        r = p - com
        for a in np.linspace(0.0, 2.0 * np.pi, n_edges, endpoint=False):
            f = n + mu * (np.cos(a) * t1 + np.sin(a) * t2)
            wrenches.append(np.r_[f, np.cross(r, f) / torque_scale])
        for s in (-1.0, 1.0):
            tau = np.cross(r, n) + s * mu * torsion_radius * n
            wrenches.append(np.r_[n, tau / torque_scale])
    try:
        hull = ConvexHull(np.asarray(wrenches), qhull_options="QJ")
    except QhullError:
        return 0.0
    eps = float(np.min(-hull.equations[:, -1]))   # distance from origin to nearest facet
    return eps if eps > _JOGGLE_TOL else 0.0


def robust_ferrari_canny(contacts: np.ndarray, normals: np.ndarray, com: np.ndarray,
                         mu: float, rng: np.random.Generator, n_samples: int = 10,
                         pos_std: float = 0.003, normal_std_deg: float = 5.0,
                         mu_rel_std: float = 0.15, com_std: float = 0.01,
                         **kwargs) -> float:
    """Mean epsilon under contact, normal, friction and centre-of-mass noise (as in Dex-Net)."""
    contacts = np.asarray(contacts, dtype=np.float64)
    normals = np.asarray(normals, dtype=np.float64)
    scores = [ferrari_canny(contacts, normals, com, mu, **kwargs)]
    sigma = np.radians(normal_std_deg)
    for _ in range(n_samples - 1):
        c = contacts + rng.normal(0.0, pos_std, contacts.shape)
        n = normals + rng.normal(0.0, sigma, normals.shape)
        n /= np.linalg.norm(n, axis=1, keepdims=True)
        m = max(0.05, mu * (1.0 + rng.normal(0.0, mu_rel_std)))
        g = com + rng.normal(0.0, com_std, 3)
        scores.append(ferrari_canny(c, n, g, m, **kwargs))
    return float(np.mean(scores))


def _voxel_downsample(points: np.ndarray, voxel: float) -> np.ndarray:
    """Indices of one point per occupied voxel."""
    keys = np.floor(points / voxel).astype(np.int64)
    _, idx = np.unique(keys, axis=0, return_index=True)
    return np.sort(idx)


def _estimate_normals(points: np.ndarray, k: int = 12,
                      viewpoints: Optional[np.ndarray] = None) -> np.ndarray:
    """PCA normals, oriented towards each point's camera (else away from the centroid)."""
    d2 = np.sum((points[:, None, :] - points[None, :, :]) ** 2, axis=-1)
    nbrs = np.argpartition(d2, kth=min(k, len(points) - 1), axis=1)[:, :k]
    local = points[nbrs] - points[nbrs].mean(axis=1, keepdims=True)
    _, _, vt = np.linalg.svd(local, full_matrices=False)
    normals = vt[:, -1, :]
    if viewpoints is not None:
        flip = np.einsum("ij,ij->i", normals, viewpoints - points) < 0
    else:
        flip = np.einsum("ij,ij->i", normals, points - points.mean(axis=0)) < 0
    normals[flip] *= -1
    return normals


def _rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues rotation matrix about a unit axis."""
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K


class AntipodalGraspPlanner:
    """6-DOF grasps on the object surface, in the style of GPD (ten Pas et al., IJRR 2017).

    Hand orientations are sampled around each seed point's normal and curvature
    axis; the hand is pushed as deep as the cloud and table allow without
    collision.  Valid grasps are ranked by robust Ferrari-Canny quality.

    Args:
        collision_margin: Inflation (m) of palm and outer finger faces, to absorb tracking error.
        friction_coef:    Friction cone the contacted surfaces must lie in; also used by the metric.
        quality_metric:   "ferrari_canny" or "heuristic" (geometric score only).
        n_evaluate:       Candidates (best by geometric score) evaluated with the metric.
        depths:           Pad-centre offsets past the seed point tried by the depth search.
    """

    def __init__(
        self,
        approach_dist: float = 0.12,
        lift_height: float = 0.20,
        table_z: Optional[float] = 0.27,
        table_clearance: float = 0.01,
        collision_margin: float = 0.005,
        friction_coef: float = 0.7,
        quality_metric: str = "ferrari_canny",
        n_evaluate: int = 80,
        n_perturb: int = 10,
        voxel_size: float = 0.003,
        n_samples: int = 150,
        n_angles: int = 7,
        depths=(-0.01, 0.0, 0.01, 0.02, 0.03),
        n_shifts: int = 9,
        min_points: int = 3,
        max_candidates: int = 30,
        seed: Optional[int] = 0,
    ):
        self.approach_dist = float(approach_dist)
        self.lift_height = float(lift_height)
        self.table_z = table_z
        self.table_clearance = float(table_clearance)
        self.collision_margin = float(collision_margin)
        self.friction_coef = float(friction_coef)
        self.min_quality = float(np.cos(np.arctan(friction_coef)))
        if quality_metric not in ("ferrari_canny", "heuristic"):
            raise ValueError(f"Unknown quality_metric '{quality_metric}'")
        self.quality_metric = quality_metric
        self.n_evaluate = n_evaluate
        self.n_perturb = n_perturb
        self.voxel_size = float(voxel_size)
        self.n_samples = n_samples
        self.angles = np.linspace(-np.pi / 2, np.pi / 2, n_angles)
        self.depths = np.asarray(depths, dtype=np.float64)
        self.shifts = np.linspace(-_FINGER_Y[0], _FINGER_Y[0], n_shifts)
        self.min_points = min_points
        self.max_candidates = max_candidates
        self._rng = np.random.default_rng(seed)

    def generate_candidates(self, points: np.ndarray,
                            obstacle_points: Optional[np.ndarray] = None,
                            viewpoints: Optional[np.ndarray] = None) -> List[GraspCandidate]:
        """Return candidates for the object cloud ``points``, best first.

        ``obstacle_points`` are other objects the hand must not sweep through.
        ``viewpoints`` gives, per point, the camera it came from; needed to
        orient normals inside hollow objects such as a mug.
        """
        points = np.asarray(points, dtype=np.float64)
        if viewpoints is not None and np.shape(viewpoints) != points.shape:
            raise ValueError(f"viewpoints must give one camera position per point "
                             f"{points.shape}, got {np.shape(viewpoints)}")
        keep = _voxel_downsample(points, self.voxel_size)
        pts = points[keep]
        if len(pts) < 20:
            return []
        views = None if viewpoints is None else np.asarray(viewpoints, dtype=np.float64)[keep]
        normals = _estimate_normals(pts, viewpoints=views)
        centroid = pts.mean(axis=0)
        size = float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)))
        obstacles = np.zeros((0, 3)) if obstacle_points is None else np.asarray(obstacle_points)

        poses = []   # (geometric score, pad centre, R_hand, contacts, inward normals)
        seeds = self._rng.choice(len(pts), size=min(self.n_samples, len(pts)), replace=False)
        for i in seeds:
            frame = self._local_frame(pts, normals, i)
            if frame is None:
                continue
            for R in self._orientations(*frame):
                best = self._search_hand(R, pts[i], pts, normals, obstacles)
                if best is None:
                    continue
                center, quality, support, depth, contacts, inward = best
                score = (2.0 * quality
                         + min(support / 20.0, 1.0)
                         + 0.5 * (depth - self.depths[0]) / max(np.ptp(self.depths), 1e-6)
                         - np.linalg.norm(center - centroid) / max(size, 1e-3)
                         - 0.2 * R[2, 2])                     # mild preference for top-down
                poses.append((score, center, R, contacts, inward))

        poses.sort(key=lambda p: p[0], reverse=True)
        if self.quality_metric == "ferrari_canny":
            radius = max(0.5 * size, 0.02)
            ranked = []
            for score, center, R, contacts, inward in poses[:self.n_evaluate]:
                eps = robust_ferrari_canny(contacts, inward, centroid, self.friction_coef,
                                           self._rng, n_samples=self.n_perturb,
                                           torque_scale=radius)
                if eps > 0.0:
                    ranked.append((eps, center, R))
            ranked.sort(key=lambda p: p[0], reverse=True)
        else:
            ranked = [(s, c, R) for s, c, R, _, _ in poses]
        return [self._make_candidate(c, R, s) for s, c, R in self._diverse(ranked)]

    @staticmethod
    def _local_frame(pts, normals, i, radius: float = 0.012):
        """(normal, curvature axis, binormal) at point i, from neighbourhood normals."""
        nb = np.linalg.norm(pts - pts[i], axis=1) < radius
        if nb.sum() < 5:
            return None
        n_nb = normals[nb]
        _, vecs = np.linalg.eigh(n_nb.T @ n_nb)
        normal = vecs[:, 2] * np.sign(vecs[:, 2] @ normals[i] or 1.0)
        axis = vecs[:, 0]                                 # least normal variation: curvature axis
        axis -= normal * (axis @ normal)
        axis /= np.linalg.norm(axis)
        return normal, axis, np.cross(normal, axis)

    def _orientations(self, normal, axis, binormal):
        """Hand frames (columns x, y=closing, z=approach) around the local frame."""
        base = np.column_stack([-axis, binormal, -normal])
        for rot_axis in (axis, binormal):
            for a in self.angles:
                if rot_axis is binormal and a == 0.0:
                    continue                                    # duplicate of the first family
                R = _rotation(rot_axis, a) @ base
                if R[2, 2] <= 0.1:                              # not from below
                    yield R

    def _search_hand(self, R, seed_pt, pts, normals, obstacles):
        """Deepest collision-free hand pose for orientation R, or None.

        Returns (pad centre, antipodal quality, points between pads, depth, contacts, inward normals).
        """
        m, sweep = self.collision_margin, self.approach_dist
        coll = pts if len(obstacles) == 0 else np.vstack([pts, obstacles])
        q = (coll - seed_pt) @ R
        near = (np.abs(q[:, 0]) < _PALM_HALF[0] + m) & (np.linalg.norm(q, axis=1) < 0.25)
        q = q[near]
        qx, qy, qz = q[:, 0], q[:, 1], q[:, 2]

        # (depth, point) and (shift, point) grids
        zh = qz[None, :] - self.depths[:, None] + _PAD_CENTER_Z
        ay = np.abs(qy[None, :] - self.shifts[:, None])
        palm_xy = (np.abs(qx) < _PALM_HALF[0] + m)[None, :] & (ay < _PALM_HALF[1] + m)
        palm_z = (zh > _PALM_Z[0] - sweep) & (zh < _PALM_Z[1] + m)
        fing_xy = (np.abs(qx) < _FINGER_HALF_X + m)[None, :] & (ay > _FINGER_Y[0]) & (ay < _FINGER_Y[1] + m)
        fing_z = (zh > _FINGER_Z[0] - sweep) & (zh < _FINGER_Z[1] + m)
        hit = (palm_xy[:, None, :] & palm_z[None, :, :]) | (fing_xy[:, None, :] & fing_z[None, :, :])
        collides = hit.any(axis=2)                    # (S, D)

        valid = ~collides
        if self.table_z is not None:
            valid &= self._above_table_grid(R, seed_pt)
        if not valid.any():
            return None

        # deepest valid depth, middle valid finger placement (as GPD)
        d_idx = np.flatnonzero(valid.any(axis=0))[-1]
        s_ok = np.flatnonzero(valid[:, d_idx])
        s_idx = s_ok[len(s_ok) // 2]
        depth, shift = self.depths[d_idx], self.shifts[s_idx]

        obj = (pts - seed_pt) @ R
        between = ((np.abs(obj[:, 0]) < _PAD_HALF_X) & (np.abs(obj[:, 1] - shift) < _FINGER_Y[0])
                   & (obj[:, 2] - depth + _PAD_CENTER_Z > _PAD_Z[0])
                   & (obj[:, 2] - depth + _PAD_CENTER_Z < _PAD_Z[1]))
        n_between = int(between.sum())
        if n_between < self.min_points:
            return None

        y, n_between_pts = obj[between, 1], normals[between]
        quality = self._antipodal_quality(y, n_between_pts @ R[:, 1])
        if quality < self.min_quality:
            return None
        center = seed_pt + R[:, 1] * shift + R[:, 2] * depth
        contacts, inward = self._contacts(pts[between], n_between_pts, y, R[:, 1])
        return center, quality, n_between, depth, contacts, inward

    @staticmethod
    def _contacts(p, n, y, closing, band: float = 0.004):
        """Contacts at both extremes along the closing axis, with inward normals of the surface there."""
        band = min(band, (y.max() - y.min()) / 3.0)   # thin walls: keep the two sides apart
        mid = p.mean(axis=0) - closing * y.mean()
        out_p, out_n = [], []
        for side, extreme in ((y >= y.max() - band, y.max()), (y <= y.min() + band, y.min())):
            normal = -np.mean(n[side], axis=0)
            out_p.append(mid + closing * extreme)
            out_n.append(normal / max(np.linalg.norm(normal), 1e-9))
        return np.array(out_p), np.array(out_n)

    @staticmethod
    def _antipodal_quality(y, n_y, band: float = 0.004) -> float:
        """How squarely the touched surfaces face the fingers (1 = antipodal).

        Medians keep blended normals at edges from dominating.
        """
        hi = n_y[y >= y.max() - band]
        lo = n_y[y <= y.min() + band]
        return float(min(np.median(hi), np.median(-lo)))

    def _above_table_grid(self, R, seed_pt) -> np.ndarray:
        """(S, D) mask: hand at grasp and pre-grasp stays above the table."""
        floor = self.table_z + self.table_clearance
        corners = np.array([[sx * _PALM_HALF[0], sy * _PALM_HALF[1], z]
                            for sx in (-1, 1) for sy in (-1, 1) for z in _PALM_Z]
                           + [[sx * _FINGER_HALF_X, sy * _FINGER_Y[1], _FINGER_Z[1]]
                              for sx in (-1, 1) for sy in (-1, 1)])
        cz = corners @ R[2] - _PAD_CENTER_Z * R[2, 2]
        lowest = seed_pt[2] + cz.min() + self.shifts[:, None] * R[2, 1] + self.depths[None, :] * R[2, 2]
        lowest_pre = lowest - self.approach_dist * R[2, 2] if R[2, 2] > 0 else lowest
        return (lowest >= floor) & (lowest_pre >= floor)

    def _diverse(self, poses, min_dist: float = 0.01, min_angle: float = np.radians(20)):
        """Greedy non-maximum suppression over (centre, approach, closing axis)."""
        kept = []
        cos_min = np.cos(min_angle)
        for s, c, R in poses:
            if all(np.linalg.norm(c - kc) > min_dist
                   or R[:, 2] @ kR[:, 2] < cos_min
                   or abs(R[:, 1] @ kR[:, 1]) < cos_min
                   for _, kc, kR in kept):
                kept.append((s, c, R))
                if len(kept) >= self.max_candidates:
                    break
        return kept

    def _make_candidate(self, center: np.ndarray, R: np.ndarray, score: float) -> GraspCandidate:
        approach_dir = R[:, 2]
        grasp_pos = center - approach_dir * GRASP_CONTACT_OFFSET
        return GraspCandidate(
            grasp_type=GraspType.SIX_DOF,
            grasp_quat=GraspPlanner._rotmat_to_quat(R),
            approach_pos=grasp_pos - approach_dir * self.approach_dist,
            grasp_pos=grasp_pos,
            lift_pos=grasp_pos + np.array([0.0, 0.0, self.lift_height]),
            score=float(score),
        )

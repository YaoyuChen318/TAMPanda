"""IK-based grasping using GraspPlanner for geometry-aware grasp candidates.

GraspPlanner features:
  - Reads object geometry (half-extents) to compute valid grasp widths
  - Generates and ranks multiple candidates (top-down X/Y, front approach)
  - Accounts for table clearance and gripper opening limits
  - Aligns finger axis with block orientation to minimise required opening
"""

import argparse
import time

import numpy as np

from tampanda import ControllerStatus, GraspPlanner
from tampanda.planners.grasp_planner import GraspType
from tampanda.symbolic.domains.tabletop import GridDomain, StateManager
from tampanda.symbolic.domains.tabletop.env_builder import make_symbolic_builder

_HOME_QPOS = np.array([0, 0, 0, -1.57079, 0, 1.57079, -0.7853, 0.04])
_HOME_CTRL = np.array([0, 0, 0, -1.57079, 0, 1.57079, -0.7853, 255])


def _pick_candidate(candidates):
    """Select best candidate for tall thin cylinders.

    GraspPlanner scores top-down highest (+20), but for tall cylinders the
    top-down grasp point lands inside the cylinder body (contact at body
    centre ≠ top surface) and the round surface makes it mechanically
    unstable.  A front (side) approach contacts at the correct height and
    grips around the cylinder properly.
    """
    front = next((c for c in candidates if c.grasp_type == GraspType.FRONT), None)
    return front if front is not None else (candidates[0] if candidates else None)


def _straight_path_free(env, q_from, q_to, step=0.1):
    """Collision-check the straight joint-space move that move_to_compensated executes."""
    n = max(1, int(np.ceil(np.max(np.abs(q_to - q_from)) / step)))
    return all(env.is_collision_free(q_from + (q_to - q_from) * (i / n)) for i in range(1, n + 1))


def _solve_ik(env, ik, pos, quat, dt, max_checks=50):
    """IK goal for a straight joint-space move, or None.

    No planner here, so with FrantikIK take the closest solution whose straight
    path is collision-free (else the closest, e.g. a grasp touching the object).
    """
    ik.set_target_position(pos, quat)
    if not hasattr(ik, "solve_all"):
        return ik.configuration.q[:7].copy() if ik.converge_ik(dt) else None

    q_now = env.data.qpos[:7].copy()
    ik.update_configuration(env.data.qpos)
    solutions = ik.solve_all()
    if len(solutions) == 0:
        return None
    goal, checked = solutions[0], []
    for q in solutions:
        # skip near-duplicates from the dense q7 sampling
        if any(np.linalg.norm(q - c) < 0.2 for c in checked):
            continue
        checked.append(q)
        if _straight_path_free(env, q_now, q):
            goal = q
            break
        if len(checked) >= max_checks:
            break
    q_full = env.data.qpos.copy()
    q_full[:7] = goal
    ik.update_configuration(q_full)
    return goal


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ik", choices=["mink", "frantik"], default="mink",
                    help="IK solver: mink (numerical) or frantik (analytical)")
    args = ap.parse_args()

    env = make_symbolic_builder().build_env(rate=200.0, ik_solver=args.ik)
    ik = env.get_ik()

    grid = GridDomain(
        model=env.model,
        cell_size=0.04,
        working_area=(0.4, 0.3),
        table_body_name="simple_table",
        table_geom_name="simple_table_surface",
        grid_offset_x=0.05,
        grid_offset_y=0.25,
    )
    state_manager = StateManager(grid, env)
    grasp_planner = GraspPlanner(table_z=grid.table_height)

    with env.launch_viewer() as viewer:
        targets   = []
        target    = None
        candidate = None

        dt = env.rate.dt
        while viewer.is_running():
            if target is None and len(targets) == 0:
                state_manager.sample_random_state(n_cylinders=3)
                env.data.qpos[:8] = _HOME_QPOS
                env.data.ctrl[:8] = _HOME_CTRL
                env.reset_velocities()
                env.forward()
                env.rest(2.0)
                targets = sorted(state_manager.ground_state()["cylinders"].keys())

            env.step()

            if env.sim_time > 0.0 and target is None and len(targets) > 0:
                target     = targets[0]
                target_pos = env.get_object_position(target)
                half_size  = env.get_object_half_size(target)
                block_quat = env.get_object_orientation(target)
                candidates = grasp_planner.generate_candidates(target_pos, half_size, block_quat)
                candidate  = _pick_candidate(candidates)
                step = 0
                if candidate:
                    print(f"Grasp candidate: {candidate.grasp_type.value}  score={candidate.score:.0f}")
                else:
                    print(f"No valid grasp candidate for {target}, skipping.")

            if target is not None and candidate is not None:
                if env.controller.get_status() == ControllerStatus.IDLE:
                    step += 1
                    print(f"Time: {env.sim_time:.2f}, Step: {step}")

                    if step == 1:
                        print(f"Approaching {target}...")
                        goal_q = _solve_ik(env, ik, candidate.approach_pos, candidate.grasp_quat, dt)
                        converged = goal_q is not None
                        print("IK result:", "Converged" if converged else "Not converged")
                        if converged:
                            env.move_to_compensated(goal_q)
                        else:
                            print(f"IK failed, skipping {target}.")
                            candidate = None

                    if step == 2:
                        print("Moving to grasp pose...")
                        goal_q = _solve_ik(env, ik, candidate.grasp_pos, candidate.grasp_quat, dt)
                        converged = goal_q is not None
                        print("IK result:", "Converged" if converged else "Not converged")
                        if converged:
                            env.move_to_compensated(goal_q)
                        else:
                            print(f"IK failed, skipping {target}.")
                            candidate = None

                    if step == 3:
                        print("Closing gripper...")
                        env.controller.close_gripper()

                    if step == 4:
                        print("Lifting object...")
                        env.attach_object_to_ee(target)
                        goal_q = _solve_ik(env, ik, candidate.lift_pos, candidate.grasp_quat, dt)
                        converged = goal_q is not None
                        print("IK result:", "Converged" if converged else "Not converged")
                        if converged:
                            env.move_to_compensated(goal_q)
                        else:
                            print(f"IK failed, releasing and skipping {target}.")
                            env.detach_object()
                            env.controller.open_gripper()
                            candidate = None

                    if step == 5:
                        print("Moving to dropoff...")
                        dropoff_pos, dropoff_quat = env.get_dropoff_pose()
                        goal_q = _solve_ik(env, ik, dropoff_pos, dropoff_quat, dt)
                        converged = goal_q is not None
                        print("IK result:", "Converged" if converged else "Not converged")
                        if converged:
                            env.move_to_compensated(goal_q)
                        else:
                            print(f"IK failed, releasing and skipping {target}.")
                            env.detach_object()
                            env.controller.open_gripper()
                            candidate = None

                    if step == 6:
                        print("Opening gripper...")
                        env.detach_object()
                        env.controller.open_gripper()

                    if step == 7:
                        print(f"Completed {target}!")
                        target    = None
                        candidate = None
                        step = 0
                        if len(targets) > 1:
                            targets = targets[1:]
                        else:
                            targets = []
                            time.sleep(2.0)

                if candidate is not None and (step == 3 or step == 6):
                    env.rest(2.0)

            elif target is not None and candidate is None:
                # No valid grasp found — skip to next target
                target = None
                if len(targets) > 1:
                    targets = targets[1:]
                else:
                    targets = []

            env.controller.step()


if __name__ == "__main__":
    main()

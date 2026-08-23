"""Shared planning utilities used by tiptop_run, websocket_server, and tiptop_h5_run."""

import json
import logging
import time
from pathlib import Path

import numpy as np
from curobo.wrap.reacher.ik_solver import IKSolver
from curobo.wrap.reacher.motion_gen import MotionGen
from cutamp.algorithm import run_cutamp
from cutamp.config import TAMPConfiguration
from cutamp.constraint_checker import ConstraintChecker
from cutamp.cost_reduction import CostReducer
from cutamp.envs import TAMPEnvironment
from cutamp.scripts.utils import default_constraint_to_mult, default_constraint_to_tol
from cutamp.task_planning.constraints import StablePlacement
from cutamp.task_planning.costs import GraspCost
from jaxtyping import Float

from tiptop.trajectory_blending import arm_joint_limits, blend_cutamp_plan, resolve_blend_config
from tiptop.utils import NumpyEncoder

_log = logging.getLogger(__name__)


def save_tiptop_plan(serialized_plan: dict, output_path: Path) -> None:
    """Save a serialized TiPToP plan to a JSON file."""
    with open(output_path, "w") as f:
        json.dump(serialized_plan, f, cls=NumpyEncoder, indent=2)


def load_tiptop_plan(path: Path) -> dict:
    """Load a serialized TiPToP plan from a JSON file."""
    with open(path) as f:
        plan = json.load(f)
    plan["q_init"] = np.array(plan["q_init"], dtype=np.float32)
    for step in plan["steps"]:
        if step["type"] == "trajectory":
            step["positions"] = np.array(step["positions"], dtype=np.float32)
            step["velocities"] = np.array(step["velocities"], dtype=np.float32)
            if "cost" in step:  # optional, schema >= 1.1.0
                step["cost"] = {k: np.array(v, dtype=np.float32) for k, v in step["cost"].items()}
    return plan


def build_tamp_config(
    num_particles: int,
    max_planning_time: float,
    opt_steps: int,
    robot_type: str,
    time_dilation_factor: float,
    collision_activation_distance: float = 0.0,
    enable_visualizer: bool = False,
    traj_length_norm: float = 2.0,
    grasp_orientation_cost: bool = False,
    grasp_center_cost: bool = False,
    arm_mode: str = "single",
    dual_task: str = "parallel",
    max_motion_refine_attempts: int | None = 32,
    transit_apex_height: float = 0.0,
    transit_apex_min_dist: float = 0.10,
) -> TAMPConfiguration:
    """Build a TAMPConfiguration with TiPToP defaults.

    See https://github.com/tiptop-robot/cuTAMP/blob/main/cutamp/config.py for
    documentation of each TAMPConfiguration parameter.

    ``arm_mode``/``dual_task`` opt into cuTAMP's simultaneous dual-arm planning (only valid with
    ``robot_type == "bimanual_yam_dual"`` -- cuTAMP's own ``validate_tamp_config`` enforces the two
    travel together). Every other embodiment leaves these at their single-arm defaults.
    """
    return TAMPConfiguration(
        num_particles=num_particles,
        max_loop_dur=max_planning_time,
        num_opt_steps=opt_steps,
        m2t2_grasps=True,
        prop_satisfying_break=0.1,
        robot=robot_type,
        arm_mode=arm_mode,
        dual_task=dual_task,
        curobo_plan=True,
        max_motion_refine_attempts=max_motion_refine_attempts,
        warmup_ik=False,
        warmup_motion_gen=False,
        num_initial_plans=10,
        cache_subgraphs=True,
        world_activation_distance=collision_activation_distance,
        movable_activation_distance=0.01,
        time_dilation_factor=time_dilation_factor,
        placement_check="obb",
        placement_shrink_dist=0.01,
        enable_visualizer=enable_visualizer,
        coll_sphere_radius=0.008,
        # Cost-sensitive task planning that minimizes joint-space distance traveled: each
        # move(q1, tau, q2) action is charged ||q1 - q2||_p between its start/end configurations,
        # a lower bound on the shortest collision-free path length. p=2 (default) is the Euclidean
        # straight-line distance; p=inf is the max-joint-displacement (infinity-norm) the TiPToP
        # paper minimizes, opted into per config via `traj_length_norm: "inf"` in cfg/tamp
        # tamp_overrides (resolve_traj_length_norm). See cuTAMP trajectory_length / TrajectoryLength.
        traj_length_norm=traj_length_norm,
        # Gate for the grasp orientation-change soft cost (weight set in run_planning). Enabled from
        # cfg/tamp when `grasp_pose_change_weight` is present; see resolve_grasp_orientation_cost.
        grasp_orientation_cost=grasp_orientation_cost,
        # Gate for the off-center grasp soft cost (weight set in run_planning). Enabled from cfg/tamp
        # when `grasp_center_weight` is present; see resolve_grasp_center_cost.
        grasp_center_cost=grasp_center_cost,
        # Explicit apex waypoint in each Pick/Place free-space transit: planned as
        # retract -> apex -> pre-grasp so the end-effector lifts, traverses and descends instead of
        # sweeping low across the table. Off (0.0) unless a cfg/tamp yml sets `transit_apex_height`
        # in tamp_overrides; see resolve_transit_apex and cuTAMP's TAMPConfiguration.
        transit_apex_height=transit_apex_height,
        transit_apex_min_dist=transit_apex_min_dist,
    )


def run_planning(
    env: TAMPEnvironment,
    config: TAMPConfiguration,
    q_init: np.ndarray,
    ik_solver: IKSolver,
    grasps: dict,
    motion_gen: MotionGen,
    all_surfaces: list,
    experiment_dir: Path | None = None,
    cost_overrides: dict | None = None,
    q_return: np.ndarray | list | None = None,
) -> tuple[list | None, float, str | None]:
    """Run cuTAMP planning and return (plan, planning_time_seconds, failure_reason).

    Returns (None, elapsed, failure_reason) if cuTAMP fails to find a plan.

    ``cost_overrides`` is the config's ``tamp_overrides`` dict; it is used here only to resolve the
    trajectory-blending settings (``blend_trajectory`` etc. -- see resolve_blend_config). Blending is
    off unless the config opts in.

    ``q_return`` overrides where the plan's closing GoToInitial drives to, which otherwise is the
    ``q_init`` it started from. Only a caller concatenating plans needs it -- see
    ``tiptop_run.plan_clear_then_task``, whose second plan starts mid-episode.
    """
    constraint_to_tol = default_constraint_to_tol.copy()
    constraint_to_mult = default_constraint_to_mult.copy()
    # Loosen tolerances slightly to enable finding a plan practically
    for surface in all_surfaces:
        constraint_to_tol[StablePlacement.type][f"{surface.name}_in_xy"] = 1e-2
        constraint_to_tol[StablePlacement.type][f"{surface.name}_support"] = 1e-2
        constraint_to_mult[StablePlacement.type][f"{surface.name}_support"] = 1.0
    # Opt-in grasp orientation-change cost (cfg/tamp `grasp_pose_change_weight` in tamp_overrides):
    # weights cuTAMP's GraspCost = geodesic angle between each grasp's EE orientation and the robot's
    # initial EE orientation, steering the planner toward grasps that reorient the wrist least. Absent
    # / zero -> the multiplier is never set, so the reducer drops the (still-cheap) computed value and
    # cuTAMP behavior is unchanged. Assigned as a fresh dict so we don't mutate the shared default.
    # `grasp_center_weight` works the same way for the off-center grasp cost: grasp_center_offset is
    # the horizontal distance in METERS from the object's centroid to the grasp TCP, so it needs a much
    # larger weight than grasp_rot_change (radians, <= pi) to matter. 20-50 is a starting range, not a
    # calibration: measured over saved runs the candidate grasps on one object span roughly 3-5 cm, so
    # w=30 separates them by ~1 cost unit, the order traj_length varies over. What settles it is one
    # real run -- the per-term weighted values in `best_cost_breakdown`
    # (<exp_dir>/optimization/opt_*.json). An order of magnitude below the other terms means raise it;
    # the largest term means lower it. Note the charge is raw meters, so one weight is a tiebreaker on
    # a small object and decisive on a large one.
    grasp_weights = {}
    grasp_weight = (cost_overrides or {}).get("grasp_pose_change_weight")
    if grasp_weight:
        grasp_weights["grasp_rot_change"] = float(grasp_weight)
    center_weight = (cost_overrides or {}).get("grasp_center_weight")
    if center_weight:
        grasp_weights["grasp_center_offset"] = float(center_weight)
    if grasp_weights:
        # Fresh dict: default_constraint_to_mult.copy() is shallow, so mutating the inner dict leaks.
        constraint_to_mult[GraspCost.type] = grasp_weights
        _log.info("Grasp soft costs active: " + ", ".join(f"{k} weight={v}" for k, v in grasp_weights.items()))
    cost_reducer = CostReducer(constraint_to_mult)
    constraint_checker = ConstraintChecker(constraint_to_tol)

    start = time.perf_counter()
    cutamp_plan, _, failure_reason = run_cutamp(
        env,
        config,
        cost_reducer,
        constraint_checker,
        q_init=q_init,
        ik_solver=ik_solver,
        grasps=grasps,
        motion_gen=motion_gen,
        experiment_dir=experiment_dir,
        q_return=q_return,
    )
    elapsed = time.perf_counter() - start
    _log.info(f"cuTAMP planning took: {elapsed:.2f}s")

    if cutamp_plan is None:
        _log.error(f"cuTAMP failed to find a plan: {failure_reason}")
    else:
        _log.info(f"Found plan with {len(cutamp_plan)} steps")
        # Optionally blend + re-time consecutive trajectory segments into continuous strokes so the
        # arm only stops at gripper events (opt-in via `blend_trajectory` in tamp_overrides; see
        # trajectory_blending). Done here, before both serialize_plan and execute_cutamp_plan, so the
        # saved and executed plans are the identical (possibly blended) object.
        blend_config = resolve_blend_config(cost_overrides)
        if blend_config.enabled:
            dof = next(
                (s["plan"].position.shape[1] for s in cutamp_plan if s.get("type") == "trajectory"), None
            )
            if dof is not None:
                vel_limit, acc_limit = arm_joint_limits(motion_gen, dof)
                cutamp_plan = _apply_blend(cutamp_plan, blend_config, vel_limit, acc_limit)

    return cutamp_plan, elapsed, failure_reason


def _apply_blend(cutamp_plan, blend_config, vel_limit, acc_limit):
    """Dispatch trajectory blending on ``blend_config.mode`` (see resolve_blend_config).

    ``spline`` (default) uses the analytic time law in ``trajectory_blending``. ``flow`` samples a full
    human-like stroke per operation from the conditional flow-matching model (``flow_blending``), so
    generated data reproduces the distribution of teleoperator styles; it loads from
    ``blend_config.model_path``, and any setup failure (missing/corrupt checkpoint, import error) is logged
    and falls back to the analytic spline blend, so a plan is never lost to a model problem.
    """
    if blend_config.mode == "flow":
        try:
            from tiptop.flow_blending import flow_blend_cutamp_plan
            from tiptop.networks.flow_timing import FlowModel

            model = FlowModel(blend_config.model_path)
            return flow_blend_cutamp_plan(
                cutamp_plan, blend_config, model, vel_limit=vel_limit, acc_limit=acc_limit
            )
        except Exception:
            _log.exception("Flow blending unavailable; falling back to the analytic spline blend")
    return blend_cutamp_plan(cutamp_plan, blend_config, vel_limit=vel_limit, acc_limit=acc_limit)


def _per_timestep_cost(velocity, position=None, trace_cfg=None) -> dict:
    """Per-timestep trajectory-cost arrays for plotting / validation.

    Derived from the joint velocities of a single trajectory segment. Mirrors the
    cuRobo ``UniformVelocityCost``: squared joint speed ``e_t = sum_dof(v_t**2)``, its
    squared deviation from the per-segment (trajopt-horizon) mean ``(e_t - mean_t e)**2``,
    and the joint speed ``||v_t||``. The mean is taken over the segment to match how
    cuRobo computes the cost per trajopt horizon. ``velocity`` is a torch tensor [T, dof].

    When ``position`` (torch tensor [T, dof]) is given and ``trace_cfg`` selects them, also emits the
    per-segment cuRobo motion-manifold cost traces the optimizer saw -- the raw (weight-independent)
    cost each term contributes, broadcast over the segment's T timesteps (see resolve_trace_cfg):

      - ``vae_manifold``  (DROID Mahalanobis distance; curobo cost/vae_manifold_cost.py)
      - ``joint_density`` (per-joint W1 to DROID; curobo cost/joint_density_cost.py)
      - ``rnd_novelty``   (raw RND novelty; curobo cost/rnd_novelty_cost.py)

    ``trace_cfg`` carries ``source_dt`` (the trajopt base_dt the manifold costs finite-difference at,
    NOT the plan's playback dt) and ``n_joints``, plus a per-term sub-dict for each trace to emit.
    Each trace is best effort: a missing artifact/load error is logged and that key is omitted so
    plotting still works.
    """
    speed_sq = (velocity * velocity).sum(dim=-1)  # [T]
    speed = speed_sq.sqrt()  # [T] joint speed ||v_t||
    uniform_velocity = (speed_sq - speed_sq.mean()).square()  # [T] (cost shape, weight = 1)
    out = {
        "speed": speed.cpu().numpy(),
        "uniform_velocity": uniform_velocity.cpu().numpy(),
        "dof_speed_sq": speed_sq.cpu().numpy(),
    }
    if trace_cfg is None or position is None:
        return out

    source_dt = float(trace_cfg.get("source_dt", 0.15))
    n_joints = int(trace_cfg.get("n_joints", 7))
    if "vae" in trace_cfg:
        try:
            from curobo.rollout.cost.vae_manifold_cost import DEFAULT_VAE_MANIFOLD_CKPT, trajectory_score_trace

            out["vae_manifold"] = trajectory_score_trace(
                position, source_dt,
                checkpoint_path=trace_cfg["vae"].get("checkpoint_path") or DEFAULT_VAE_MANIFOLD_CKPT,
                n_joints=n_joints,
            )
        except Exception as exc:  # missing artifact / load error -> skip the trace
            _log.warning(f"VAE-manifold cost trace skipped: {exc}")
    if "joint_density" in trace_cfg:
        try:
            from curobo.rollout.cost.joint_density_cost import trajectory_density_trace

            out["joint_density"] = trajectory_density_trace(
                position, n_joints=n_joints, huber_delta=float(trace_cfg["joint_density"].get("huber_delta", 0.05))
            )
        except Exception as exc:
            _log.warning(f"Joint-density cost trace skipped: {exc}")
    if "rnd_novelty" in trace_cfg:
        try:
            from curobo.rollout.cost.rnd_novelty_cost import trajectory_novelty_trace

            out["rnd_novelty"] = trajectory_novelty_trace(position, source_dt, n_joints=n_joints)
        except Exception as exc:
            _log.warning(f"RND-novelty cost trace skipped: {exc}")
    return out


def serialize_plan(cutamp_plan: list[dict], q_init: Float[np.ndarray, "d"], trace_cfg: dict | None = None) -> dict:
    """Serialize a cuTAMP plan to a dict.

    Schema versioning follows semver: bump minor for new optional fields, major for breaking changes.
    If the schema changes, update load_tiptop_plan accordingly.

    ``trace_cfg`` (optional) selects which cuRobo motion-manifold cost traces to record per trajectory
    segment (``vae_manifold`` / ``joint_density`` / ``rnd_novelty``) -- built by
    motion_planning.resolve_trace_cfg from the cfg/tamp cost overrides, so only the costs actually
    active in the run are logged. See _per_timestep_cost.
    """
    steps = []
    for step in cutamp_plan:
        if step["type"] == "trajectory":
            steps.append(
                {
                    "type": "trajectory",
                    "label": step["label"],
                    "positions": step["plan"].position.cpu().numpy(),
                    "velocities": step["plan"].velocity.cpu().numpy(),
                    "dt": step["dt"],
                    "cost": _per_timestep_cost(
                        step["plan"].velocity, position=step["plan"].position, trace_cfg=trace_cfg,
                    ),
                }
            )
        elif step["type"] == "gripper":
            entry = {"type": "gripper", "label": step["label"], "action": step["action"]}
            # Present only on cuTAMP's dual-arm path (motion_solver.py::gripper_step): "arms" names
            # every hand this step actuates (plural -> simultaneous, e.g. PickBoth/PlaceBoth); "arm"
            # is set too when exactly one hand acts (PickGiver/PlaceTaker, or one side of a
            # Handover), for consumers that only care about a single hand. Single-arm cuTAMP steps
            # never carry either key, so this is purely additive for existing plans.
            if "arm" in step:
                entry["arm"] = step["arm"]
            if "arms" in step:
                entry["arms"] = list(step["arms"])
            steps.append(entry)
    return {"version": "1.4.0", "q_init": q_init, "steps": steps}

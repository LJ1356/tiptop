"""Shared planning utilities used by tiptop_run, websocket_server, and tiptop_h5_run."""

import json
import logging
import time
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np
from curobo.wrap.reacher.ik_solver import IKSolver
from curobo.wrap.reacher.motion_gen import MotionGen
from cutamp.algorithm import run_cutamp
from cutamp.config import TAMPConfiguration
from cutamp.constraint_checker import ConstraintChecker
from cutamp.cost_reduction import CostReducer
from cutamp.envs import TAMPEnvironment
from cutamp.scripts.utils import default_constraint_to_mult, default_constraint_to_tol
from cutamp.tamp_domain import get_initial_state
from cutamp.task_planning import PlanSkeleton, State
from cutamp.task_planning.constraints import StablePlacement
from cutamp.task_planning.costs import GraspCost
from cutamp.task_planning.search import FABRICABLE_TYPES
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
    q_home: Sequence[float] | None = None,
) -> TAMPConfiguration:
    """Build a TAMPConfiguration with TiPToP defaults.

    See https://github.com/tiptop-robot/cuTAMP/blob/main/cutamp/config.py for
    documentation of each TAMPConfiguration parameter.
    """
    return TAMPConfiguration(
        num_particles=num_particles,
        max_loop_dur=max_planning_time,
        num_opt_steps=opt_steps,
        m2t2_grasps=True,
        prop_satisfying_break=0.1,
        robot=robot_type,
        curobo_plan=True,
        max_motion_refine_attempts=32,
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
        # Where a plan parks the arm after its last operator. Pass the robot's real home (cfg/tiptop
        # `robot.q_home`); left None, cuTAMP ends every plan back at the configuration it planned
        # FROM, which on a rollout resuming from a teleop hand-off is wherever the human left the
        # arm -- unreachable often enough to fail the whole plan (see cutamp motion_solver).
        q_home=tuple(q_home) if q_home is not None else None,
    )


def environment_initial_state(env: TAMPEnvironment) -> State:
    """The symbolic initial state cuTAMP will derive for ``env``.

    Mirrors ``TAMPWorld.initial_state``, which reads the same ``env.type_to_objects`` through
    ``get_objects_by_type``. Reproduced here so a task plan can be checked against the environment
    before paying for the world (and the GPU) that cuTAMP builds.

    Note this state is a function of the object NAMES alone -- it says every movable has not been
    picked up and the hand is empty, never where anything is. Two perception passes over the same
    scene therefore give the same initial state whatever moved in between.
    """
    return get_initial_state(
        movables=[obj.name for obj in env.type_to_objects.get("Movable", [])],
        surfaces=[obj.name for obj in env.type_to_objects.get("Surface", [])],
        sticks=[obj.name for obj in env.type_to_objects.get("Stick", [])],
        buttons=[obj.name for obj in env.type_to_objects.get("Button", [])],
    )


def skeleton_reuse_rejection(skeleton: PlanSkeleton, initial_state: State, goal_state: State) -> str | None:
    """Why ``skeleton`` cannot be reused for this problem, or None if it can be.

    A ground operator carries only its lifted operator and object-name strings, so a skeleton found
    against one perception pass is re-runnable against another -- as long as it still describes a
    valid solution here. Checked exactly as ``breadth_first_search`` would have: every operator's
    preconditions hold in turn, and the goal is a subset of the resulting state.

    The object-name check is redundant with the precondition walk for every operator whose
    preconditions mention all its arguments, but it is what turns "an object the skeleton needs was
    not detected this time" into a clear rejection here rather than an index error deep inside
    particle initialization. Configurations and trajectories (FABRICABLE_TYPES) are exempt: the
    search invents those symbols, so they are never in the initial state -- same rule BFS applies to
    goal literals.
    """
    if not skeleton:
        return "the cached task plan is empty"
    literals_by_type: dict[str, set[str]] = defaultdict(set)
    for atom in initial_state:
        for param, value in zip(atom.fluent.parameters, atom.values):
            literals_by_type[param.type].add(value)
    for op in skeleton:
        missing = sorted(
            {
                f"{value} ({param.type})"
                for param, value in zip(op.operator.parameters, op.values)
                if param.type not in FABRICABLE_TYPES and value not in literals_by_type[param.type]
            }
        )
        if missing:
            return f"{op.name} refers to object(s) not in this scene: {', '.join(missing)}"

    state = initial_state
    for op in skeleton:
        if not op.preconditions.issubset(state):
            unmet = sorted(str(atom) for atom in op.preconditions - state)
            return f"preconditions of {op.name} are not met: {', '.join(unmet)}"
        state = op.apply(state)
    if not goal_state.issubset(state):
        unmet = sorted(str(atom) for atom in goal_state - state)
        return f"it does not reach this goal, missing: {', '.join(unmet)}"
    return None


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
    reuse_plan_skeleton: PlanSkeleton | None = None,
    plan_out: dict | None = None,
) -> tuple[list | None, float, str | None]:
    """Run cuTAMP planning and return (plan, planning_time_seconds, failure_reason).

    Returns (None, elapsed, failure_reason) if cuTAMP fails to find a plan.

    ``cost_overrides`` is the config's ``tamp_overrides`` dict; it is used here only to resolve the
    trajectory-blending settings (``blend_trajectory`` etc. -- see resolve_blend_config). Blending is
    off unless the config opts in.

    ``reuse_plan_skeleton`` is a task plan from an earlier call (see ``plan_out``) to reuse instead
    of searching for one: grasps, placements and trajectories are all still solved from scratch
    against this scene, only the symbolic search is skipped. It is rejected outright if it no longer
    solves this problem (skeleton_reuse_rejection), and if it is accepted but yields no plan, this
    falls back to a full search rather than returning empty-handed. Either way ``elapsed`` covers
    every cuTAMP call made.

    ``plan_out``, if given, gets {"plan_skeleton": ..., "reused": bool} for the returned plan.
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
    grasp_weight = (cost_overrides or {}).get("grasp_pose_change_weight")
    if grasp_weight:
        constraint_to_mult[GraspCost.type] = {"grasp_rot_change": float(grasp_weight)}
        _log.info(f"Grasp orientation-change cost active: grasp_rot_change weight={float(grasp_weight)}")
    cost_reducer = CostReducer(constraint_to_mult)
    constraint_checker = ConstraintChecker(constraint_to_tol)

    def solve(skeleton):
        cutamp_out: dict = {}
        plan, _, reason = run_cutamp(
            env,
            config,
            cost_reducer,
            constraint_checker,
            q_init=q_init,
            ik_solver=ik_solver,
            grasps=grasps,
            motion_gen=motion_gen,
            experiment_dir=experiment_dir,
            reuse_plan_skeleton=skeleton,
            plan_out=cutamp_out,
        )
        return plan, reason, cutamp_out.get("plan_skeleton")

    start = time.perf_counter()
    reused = False
    if reuse_plan_skeleton is not None:
        rejection = skeleton_reuse_rejection(reuse_plan_skeleton, environment_initial_state(env), env.goal_state)
        if rejection:
            _log.info(f"Not reusing the previous task plan ({rejection}); planning the task from scratch")
            reuse_plan_skeleton = None
        else:
            _log.info(f"Reusing the previous task plan: {[op.name for op in reuse_plan_skeleton]}")

    cutamp_plan, failure_reason, final_skeleton = solve(reuse_plan_skeleton)
    if cutamp_plan is not None:
        reused = reuse_plan_skeleton is not None
    elif reuse_plan_skeleton is not None:
        # The task plan still applies symbolically, but this scene admits no grasp/placement/motion
        # for it -- the objects have moved. A different skeleton may well work, so search after all.
        _log.warning(f"Reused task plan produced no motion plan ({failure_reason}); falling back to a full task search")
        cutamp_plan, failure_reason, final_skeleton = solve(None)
    elapsed = time.perf_counter() - start
    _log.info(f"cuTAMP planning took: {elapsed:.2f}s")
    if plan_out is not None:
        plan_out["plan_skeleton"] = final_skeleton
        plan_out["reused"] = reused

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
            steps.append({"type": "gripper", "label": step["label"], "action": step["action"]})
    return {"version": "1.3.0", "q_init": q_init, "steps": steps}

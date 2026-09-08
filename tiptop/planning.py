"""Shared planning utilities used by tiptop_run, websocket_server, and tiptop_h5_run."""

import dataclasses
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
from cutamp.particle_initialization import NoGraspsError
from cutamp.utils.support import NoSupportRegion
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
    grasp_center_cost: bool = False,
    arm_mode: str = "single",
    dual_task: str = "parallel",
    max_motion_refine_attempts: int | None = 32,
    transit_apex_height: float = 0.0,
    transit_apex_min_dist: float = 0.10,
    q_home: Sequence[float] | None = None,
    posture_selection: dict | None = None,
    require_m2t2_grasps: bool = False,
    placement: dict | None = None,
) -> TAMPConfiguration:
    """Build a TAMPConfiguration with TiPToP defaults.

    See https://github.com/tiptop-robot/cuTAMP/blob/main/cutamp/config.py for
    documentation of each TAMPConfiguration parameter.

    ``placement`` overrides the placement-region keys, from ``resolve_placement_support``. Empty or
    absent leaves the bounding-box region every config has always used.

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
        # Where an object may be placed on a surface. The default is the surface's oriented bounding
        # box with the object's bottom at the box's top, which is the height of the surface's highest
        # vertex -- a box's lid, a plate's rim. A cfg/tamp yml opts into the fitted support region
        # instead with `placement_support: true`; see resolve_placement_support, which supplies every
        # key in `placement`, and cutamp/utils/support.py for what it fits.
        **({"placement_check": "obb", "placement_shrink_dist": 0.01} | (placement or {})),
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
        # Where a plan parks the arm after its last operator. Pass the robot's real home (cfg/tiptop
        # `robot.q_home`); left None, cuTAMP ends every plan back at the configuration it planned
        # FROM, which on a rollout resuming from a teleop hand-off is wherever the human left the
        # arm -- unreachable often enough to fail the whole plan (see cutamp motion_solver).
        q_home=tuple(q_home) if q_home is not None else None,
        # Teleop-posture IK branch selection: solve each endpoint's IK with return_seeds=k and keep
        # the branch whose q1 - q3 (the FR3 shoulder null-space coordinate) best fits this lab's
        # teleop band, instead of cuRobo's top seed. Off unless a cfg/tamp yml sets
        # `posture_selection_seeds`; see resolve_posture_selection and cuTAMP's TAMPConfiguration.
        **(posture_selection or {}),
        # Fail rather than substitute collision-sphere heuristic grasps when perception proposed
        # nothing for an object that must be picked. Off unless a cfg/tamp yml sets
        # `require_m2t2_grasps`; see resolve_require_m2t2_grasps and cuTAMP's TAMPConfiguration.
        require_m2t2_grasps=require_m2t2_grasps,
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
    return_home: bool = True,
    q_return: np.ndarray | list | None = None,
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

    ``experiment_dir`` holds one `attempt_N` subdirectory per cuTAMP call (see attempt_dir), so the
    reuse attempt and the fallback search each get their own logs instead of colliding.

    ``plan_out``, if given, gets {"plan_skeleton": ..., "reused": bool} for the returned plan.

    ``return_home`` is False for a plan that is one LEG of a longer episode -- a HITL phase with more
    phases to come, say. cuTAMP then leaves off the final drive back to ``q_home``, so the arm stops
    at the retract above whatever it just placed and the next leg (or the human taking over) carries
    on from there instead of from home. See TAMPConfiguration.return_home.

    ``q_return`` overrides where the plan's closing GoToInitial drives to, which otherwise is
    ``config.q_home`` (or the ``q_init`` it started from when that is unset). Only a caller
    concatenating plans needs it -- see ``tiptop_run.plan_clear_then_task``, whose second plan
    starts mid-episode.

    ``return_home``, ``config.q_home`` and ``q_return`` all answer the same question -- where the
    plan ends -- at three different scopes, so they are ordered rather than exclusive, matching
    cuTAMP's own ``run_cutamp``/``solve_curobo``:

      * ``return_home=False`` drops the closing drive entirely, so NEITHER of the other two is
        consulted; the plan stops at the retract above what it last placed.
      * otherwise ``q_return`` (per-CALL) wins when given -- goal clearing plans two legs against one
        shared config and the task leg has to end where the EPISODE started, not where the clearing
        leg handed over;
      * otherwise ``config.q_home`` (per-RUN, from build_tamp_config), or ``q_init`` when that is
        None, which is cuTAMP's original behaviour.

    Nothing warns when two of them are set: the ordering is here so a caller that passes both gets
    a defined answer rather than whichever the code happened to read last.
    """
    if not return_home:
        # `config` is built once per session and shared, so this leg gets its own copy rather than
        # mutating the one every other leg is about to plan with. TAMPConfiguration is frozen, and
        # everything expensive (motion_gen, ik_solver) is passed in separately -- nothing is rebuilt.
        config = dataclasses.replace(config, return_home=False)
    # Deep enough to own the per-type dicts: `.copy()` is shallow, so writing a surface's tolerance
    # into the inner dict below would mutate cuTAMP's module-level default for the whole process.
    constraint_to_tol = {k: dict(v) for k, v in default_constraint_to_tol.items()}
    constraint_to_mult = {k: dict(v) for k, v in default_constraint_to_mult.items()}
    # Loosen tolerances slightly to enable finding a plan practically
    for surface in all_surfaces:
        constraint_to_tol[StablePlacement.type][f"{surface.name}_in_xy"] = 1e-2
        constraint_to_tol[StablePlacement.type][f"{surface.name}_support"] = 1e-2
        constraint_to_mult[StablePlacement.type][f"{surface.name}_support"] = 1.0
        # Emitted only where a support region holds the object at ONE orientation (see
        # cutamp.utils.support.Footprint). |sin(yaw error)|, so this is ~4 degrees -- tight, because
        # the region was fitted for that orientation and a rectangle turned out of it sweeps
        # straight over the edge it was fitted to clear.
        constraint_to_tol[StablePlacement.type][f"{surface.name}_yaw"] = 7e-2
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

    def attempt_dir() -> Path | None:
        """Pick this cuTAMP call's own `attempt_N` subdirectory of ``experiment_dir``.

        cuTAMP's ExperimentLogger refuses to overwrite anything it has already written, so two calls
        sharing one directory die on the second one's `optimization/opt_0001.json` (each call
        restarts its own optimization counter). Picking the first free N off disk, rather than
        counting in memory, also keeps two runs that land on the same directory apart.
        """
        if experiment_dir is None:
            return None  # let cuTAMP name the experiment itself
        n = 0
        while (experiment_dir / f"attempt_{n}").exists():
            n += 1
        return experiment_dir / f"attempt_{n}"

    starved: list[str] = []  # set by solve() when an object had no M2T2 grasps; see below

    def solve(skeleton):
        cutamp_out: dict = {}
        try:
            plan, _, reason = run_cutamp(
                env,
                config,
                cost_reducer,
                constraint_checker,
                q_init=q_init,
                ik_solver=ik_solver,
                grasps=grasps,
                motion_gen=motion_gen,
                experiment_dir=attempt_dir(),
                reuse_plan_skeleton=skeleton,
                plan_out=cutamp_out,
                # Read from the closure so the reuse attempt and the fallback search below both end
                # at the same configuration; cuTAMP ignores it when config.return_home is off.
                q_return=q_return,
            )
        except (NoGraspsError, NoSupportRegion) as exc:
            # NoGraspsError: `require_m2t2_grasps` refused to substitute heuristic collision-sphere
            # grasps for an object perception proposed nothing for. NoSupportRegion: no level patch
            # of a goal surface is big enough to hold the object (placement_check="support"), which
            # is the honest answer when the only thing the bounding box offered was the top of a
            # box's lid. Both are PLANNING failures, not crashes: reported the same way as any other,
            # so the reset path drops the offending object and retries and the task path fails the
            # episode cleanly instead of unwinding the session.
            #
            # Recorded so the reuse fallback below can be SKIPPED. Both are facts about this scene's
            # perception, not about the skeleton, so a full task search would hit the same object or
            # surface and fail identically -- for the price of a whole search.
            starved.append(str(exc))
            return None, str(exc), None
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
    elif reuse_plan_skeleton is not None and not starved:
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

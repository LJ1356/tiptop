#!/usr/bin/env python3
"""Websocket server for tiptop perception + planning pipeline.

This server exposes the full tiptop pipeline (detection, segmentation, grasp generation,
and motion planning) over websocket, allowing clients running different Python versions
to query for trajectory plans.

Usage:
    pixi run python -m tiptop.websocket_server --port 8765
"""

import asyncio
import http
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path

import aiohttp
import msgpack_numpy
import numpy as np
import rerun as rr
import tyro
import websockets.asyncio.server as ws_server
import websockets.frames
from curobo.geom.types import WorldConfig
from curobo.wrap.reacher.ik_solver import IKSolver
from curobo.wrap.reacher.motion_gen import MotionGen

from tiptop.config import tiptop_cfg
from tiptop.goal_clearing import resolve_clear_goal_surfaces
from tiptop.motion_planning import (
    build_curobo_solvers,
    apply_perception_overrides,
    resolve_grasp_center_cost,
    resolve_grasp_orientation_cost,
    resolve_max_motion_refine_attempts,
    resolve_time_dilation_factor,
    resolve_traj_length_norm,
    resolve_transit_apex,
    resolve_trace_cfg,
)
from tiptop.perception.cameras import Frame
from tiptop.planning import build_tamp_config, run_planning, save_tiptop_plan, serialize_plan
from tiptop.recording import save_run_metadata, save_run_outputs
from tiptop.tiptop_run import Observation, plan_clear_then_task, run_perception
from tiptop.utils import NumpyEncoder, add_file_handler, check_cutamp_version, get_robot_rerun, print_tiptop_banner, remove_file_handler, setup_logging

_log = logging.getLogger(__name__)


def _load_curobo_overrides(spec: str | None) -> dict:
    """Parse cuRobo cost overrides from a JSON file path OR an inline JSON string.

    Returns {} when ``spec`` is falsy. These are the same override keys the parameter sweep and
    the tiptop-viz UI use (vae_manifold_weight, rnd_novelty_weight, joint_density_weight, smooth_weight,
    self_collision_weight, time_dilation_factor[_literal], ...) -- see motion_planning.apply_cost_overrides.
    Applying them at
    server build time makes every plan this server produces use those tamp parameters, so e.g. a
    data-gen job can generate a dataset with a chosen manifold/novelty/smoothness regime.
    """
    if not spec:
        return {}
    p = Path(spec)
    text = p.read_text() if p.exists() else spec
    overrides = json.loads(text)
    if not isinstance(overrides, dict):
        raise ValueError(f"cuRobo overrides must be a JSON object, got {type(overrides).__name__}")
    return overrides


class TiptopPlanningServer:
    """Websocket server for tiptop perception + planning pipeline.

    Follows the same pattern as openpi's WebsocketPolicyServer.
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8765,
        num_particles: int = 256,
        max_planning_time: float = 60.0,
        rerun_mode: str = "disabled",
        include_workspace: bool = False,
        overlap_perception: bool = True,
        max_concurrent_perception: int = 8,
        curobo_overrides: str | None = None,
    ) -> None:
        if rerun_mode not in {"stream", "save", "disabled"}:
            raise ValueError(f"Invalid rerun mode: {rerun_mode}")

        self._host = host
        self._port = port
        self._rerun_mode = rerun_mode
        self._include_workspace = include_workspace
        self._cfg = tiptop_cfg()

        # cuRobo cost/tamp-parameter overrides (empty by default -> stock gradient_trajopt.yml +
        # tiptop.yml behavior). Applied at solver build time so every plan uses these tamp params.
        self._curobo_overrides = _load_curobo_overrides(curobo_overrides)
        # Perception knobs travel in the same dict; applied before any perception so the grasp
        # candidates cuTAMP receives reflect them. See apply_perception_overrides.
        for _key, (_old, _new) in apply_perception_overrides(self._cfg, self._curobo_overrides).items():
            _log.info(f"Perception override: {_key} {_old} -> {_new}")
        time_dilation_factor = resolve_time_dilation_factor(
            self._curobo_overrides, self._cfg.robot.time_dilation_factor
        )
        if self._curobo_overrides:
            _log.info(f"cuRobo cost overrides active: {self._curobo_overrides}")
            _log.info(f"Resolved time_dilation_factor={time_dilation_factor}")

        self._metadata = {
            "server": "tiptop",
            "version": "0.1.0",
            "num_particles": num_particles,
            "max_planning_time": max_planning_time,
            "curobo_overrides": self._curobo_overrides,
            "time_dilation_factor": time_dilation_factor,
        }

        # Motion planning components (initialized during warmup)
        self._ik_solver: IKSolver | None = None
        self._motion_gen: MotionGen | None = None
        self._initial_world_cfg: WorldConfig | None = None
        _apex_height, _apex_min_dist = resolve_transit_apex(self._curobo_overrides)
        if _apex_height > 0:
            _log.info(f"Transit apex active: {_apex_height}m (min transit distance {_apex_min_dist}m)")
        self._config = build_tamp_config(
            num_particles=num_particles,
            max_planning_time=max_planning_time,
            opt_steps=500,
            robot_type=self._cfg.robot.type,
            time_dilation_factor=time_dilation_factor,
            traj_length_norm=resolve_traj_length_norm(self._curobo_overrides),
            grasp_orientation_cost=resolve_grasp_orientation_cost(self._curobo_overrides),
            grasp_center_cost=resolve_grasp_center_cost(self._curobo_overrides),
            # Only meaningful for robot_type == "bimanual_yam_dual" (see tiptop_yam_dual.yml); every
            # other config leaves robot.arm_mode/dual_task unset and gets cuTAMP's single-arm defaults.
            arm_mode=self._cfg.robot.get("arm_mode", "single"),
            dual_task=self._cfg.robot.get("dual_task", "parallel"),
            max_motion_refine_attempts=resolve_max_motion_refine_attempts(self._curobo_overrides),
            # Apex waypoint in each Pick/Place transit (0 = off), a TAMP-config knob like
            # traj_length_norm rather than a cuRobo cost weight. See resolve_transit_apex.
            transit_apex_height=_apex_height,
            transit_apex_min_dist=_apex_min_dist,
        )
        self._output_dir = Path("tiptop_server_outputs")
        # Concurrency model. The slow part of a plan is I/O-bound perception (Gemini / SAM2 / M2T2
        # over HTTP, ~4-7s); the cuRobo/GPU solve is only ~1.5s. Running multiple cuRobo solves
        # concurrently (esp. across processes) trips a power transient on high-TDP GPUs, so we keep
        # the GPU solve single-flight under _gpu_lock while letting concurrent requests overlap
        # their perception. _perception_sem bounds that overlap so we don't stampede the Gemini API
        # / M2T2 server. _pipeline_lock restores whole-pipeline serialization when overlap is off.
        self._overlap_perception = overlap_perception
        self._gpu_lock = asyncio.Lock()
        self._pipeline_lock = asyncio.Lock()
        self._perception_sem = asyncio.Semaphore(max(1, max_concurrent_perception))
        self._run_counter = 0

    def _reset_motion_planning(self) -> None:
        """Reset collision world to initial state to clear stale cached state between runs."""
        self._ik_solver.update_world(self._initial_world_cfg)
        self._motion_gen.update_world(self._initial_world_cfg)
        self._motion_gen.reset(reset_seed=False)

    @staticmethod
    def _health_check(connection: ws_server.ServerConnection, request: ws_server.Request) -> ws_server.Response | None:
        """Handle health check requests."""
        if request.path == "/health":
            return connection.respond(http.HTTPStatus.OK, "OK\n")
        return None

    def serve_forever(self) -> None:
        """Run the server forever (blocking)."""
        asyncio.run(self.run())

    async def run(self) -> None:
        """Async main loop."""
        # Warm up motion planning on startup
        _log.info("Setting up motion planning...")
        self._ik_solver, self._motion_gen, self._initial_world_cfg = build_curobo_solvers(
            self._config.num_particles,
            self._config.coll_n_spheres,
            include_workspace=self._include_workspace,
            cost_overrides=self._curobo_overrides,
        )

        _log.info(f"Starting TiptopPlanningServer on ws://{self._host}:{self._port}")
        async with ws_server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            process_request=self._health_check,
        ) as server:
            await server.serve_forever()

    async def _handler(self, websocket: ws_server.ServerConnection) -> None:
        """Handle a websocket connection."""
        _log.info(f"Connection from {websocket.remote_address} opened")
        packer = msgpack_numpy.Packer()

        # Send metadata on connection
        await websocket.send(packer.pack(self._metadata))

        while True:
            try:
                start_time = time.monotonic()

                # Receive observation from client
                raw_data = await websocket.recv()
                obs = msgpack_numpy.unpackb(raw_data)

                _log.info(f"Received planning request: task='{obs.get('task', 'unknown')}'")

                # In overlap mode, concurrent requests interleave their I/O-bound perception and
                # serialize only the GPU critical section inside _run_pipeline (_gpu_lock). With
                # overlap disabled, fall back to serializing the whole pipeline (legacy behavior).
                infer_start = time.monotonic()
                if self._overlap_perception:
                    result = await self._run_pipeline(obs)
                else:
                    async with self._pipeline_lock:
                        result = await self._run_pipeline(obs)
                infer_time = time.monotonic() - infer_start

                # Add timing info
                result["server_timing"] = {
                    "infer_ms": infer_time * 1000,
                    "total_ms": (time.monotonic() - start_time) * 1000,
                }

                # Send result back as JSON
                await websocket.send(json.dumps(result, cls=NumpyEncoder))
                _log.info(f"Sent planning result: success={result['success']}, time={infer_time:.2f}s")

            except websockets.ConnectionClosed:
                _log.info(f"Connection from {websocket.remote_address} closed")
                break
            except Exception as e:
                _log.exception("Error processing request")
                await websocket.send(str(e))
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise

    async def _run_pipeline(self, obs: dict) -> dict:
        """Run the full perception + planning pipeline.

        Args:
            obs: Dictionary containing:
                - rgb: uint8 array (H, W, 3)
                - depth: float32 array (H, W) in meters
                - intrinsics: float32 array (3, 3)
                - world_from_cam: float32 array (4, 4)
                - task: str
                - q_init: float32 array (7,) joint positions

        Returns:
            Dictionary containing:
                - success: bool
                - plan: list of plan steps (trajectory or gripper actions)
                - error: str or None
                - save_dir: absolute path to the per-run output directory (logs, metadata, plan, rerun .rrd)
        """
        now = datetime.now()
        # Unique per-run suffix: with overlap on, concurrent requests can share the same wall-clock
        # second, and a second-resolution directory name would collide (clobbering each other's
        # outputs). The counter increments on the single event-loop thread, so it's race-free.
        self._run_counter += 1
        timestamp = f"{now.strftime('%Y-%m-%d_%H-%M-%S')}_{self._run_counter:04d}"
        iso_timestamp = now.isoformat(timespec="seconds")
        save_dir = self._output_dir / timestamp
        save_dir.mkdir(parents=True, exist_ok=True)
        file_handler = add_file_handler(save_dir / "tiptop_run.log")

        env = None
        task_instruction = None
        observation = None
        processed_scene = None
        grounded_atoms = None
        perception_duration = None
        planning_duration = None
        failure_reason = None
        cutamp_plan = None
        try:
            # Extract and preprocess observation
            rgb = obs["rgb"].astype(np.uint8)
            depth = obs["depth"].copy().astype(np.float32)
            K = obs["intrinsics"].astype(np.float32)
            world_from_cam = obs["world_from_cam"].astype(np.float32)
            task_instruction = obs["task"]
            q_init = obs["q_init"]

            # Preprocess depth: remove invalid values and truncate range
            depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
            depth[depth < 0] = 0.0
            depth[depth > self._cfg.perception.depth_trunc_m] = 0.0

            frame = Frame(serial="static", timestamp=0.0, rgb=rgb, intrinsics=K, depth=depth)
            observation = Observation(frame=frame, world_from_cam=world_from_cam, q_init=q_init)

            # Initialize rerun if enabled (idempotent)
            if self._rerun_mode != "disabled":
                rr.init("tiptop_server", recording_id=timestamp, spawn=self._rerun_mode == "stream")
                if self._rerun_mode == "save":
                    rrd_path = save_dir / "tiptop.rrd"
                    rr.save(rrd_path)
                    _log.info(f"Saving Rerun stream to {rrd_path}")
                rerun_robot = get_robot_rerun()

            _log.info(f"Processing: RGB shape={rgb.shape}, depth shape={depth.shape}, task='{task_instruction}'")
            if self._rerun_mode != "disabled":
                rerun_robot.set_joint_positions(q_init)

            connector = aiohttp.TCPConnector(limit=10, force_close=True)
            timeout = aiohttp.ClientTimeout(total=120.0)
            # Bound how many perceptions run at once so concurrent requests don't stampede the
            # Gemini API / M2T2 server; the cuRobo solve is serialized separately below.
            async with self._perception_sem:
                async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
                    perception_start = time.monotonic()
                    env, all_surfaces, processed_scene, grounded_atoms = await run_perception(
                        session,
                        observation,
                        task_instruction,
                        save_dir,
                        depth_estimator=None,
                        include_workspace=self._include_workspace,
                    )
                    perception_duration = time.monotonic() - perception_start
            # Log camera intrinsics and pose to rerun if enabled
            if self._rerun_mode != "disabled":
                rr.log("cam", rr.Pinhole(image_from_camera=K))
                rr.log(
                    "cam",
                    rr.Transform3D(translation=world_from_cam[:3, 3], mat3x3=world_from_cam[:3, :3], axis_length=0.05),
                )

            # Run cuTAMP planning. Serialize the GPU critical section (collision-world reset + the
            # cuRobo solve) under _gpu_lock: concurrent cuRobo replays are what trip the power
            # transient, so GPU work stays single-flight in this process even while other requests'
            # perception overlaps. The reset clears stale cuTAMP world state from the previous solve
            # and must stay paired with the solve under the same lock.
            _log.info("Running cuTAMP planning...")
            async with self._gpu_lock:
                self._reset_motion_planning()
                if resolve_clear_goal_surfaces(self._curobo_overrides):
                    # Same knob and same two-phase planner as the TAMP datagen path: clear whatever
                    # occupies a goal surface, then plan the task against that, and return the two
                    # concatenated. The client executes one plan either way. See tiptop.goal_clearing.
                    cutamp_plan, planning_duration, failure_reason, _ = await asyncio.to_thread(
                        plan_clear_then_task,
                        self._config,
                        processed_scene,
                        q_init,
                        grounded_atoms,
                        save_dir,
                        ik_solver=self._ik_solver,
                        motion_gen=self._motion_gen,
                        cost_overrides=self._curobo_overrides,
                    )
                else:
                    cutamp_plan, planning_duration, failure_reason = await asyncio.to_thread(
                        run_planning,
                        env,
                        self._config,
                        q_init,
                        self._ik_solver,
                        processed_scene.grasps,
                        self._motion_gen,
                        all_surfaces,
                        cost_overrides=self._curobo_overrides,
                    )

            if cutamp_plan is None:
                return {
                    "success": False,
                    "plan": None,
                    "error": f"cuTAMP failed to find a plan: {failure_reason}",
                    "save_dir": str(save_dir.resolve()),
                }

            serialized_plan = serialize_plan(cutamp_plan, q_init, trace_cfg=resolve_trace_cfg(self._curobo_overrides))
            plan_path = save_dir / "tiptop_plan.json"
            save_tiptop_plan(serialized_plan, plan_path)
            _log.info(f"Saved outputs to {save_dir}")

            return {
                "success": True,
                "plan": serialized_plan,
                "error": None,
                "save_dir": str(save_dir.resolve()),
            }

        except Exception as e:
            _log.exception("Pipeline error")
            if not failure_reason:
                failure_reason = str(e)
            return {
                "success": False,
                "plan": None,
                "error": str(e),
                "save_dir": str(save_dir.resolve()),
            }
        finally:
            if env is not None and processed_scene is not None:
                save_run_outputs(save_dir, env, processed_scene.grasps)
            save_run_metadata(
                save_dir=save_dir,
                timestamp=iso_timestamp,
                task_instruction=task_instruction,
                q_at_capture=observation.q_init if observation is not None else None,
                world_from_cam=observation.world_from_cam if observation is not None else None,
                perception_duration=perception_duration,
                grounded_atoms=grounded_atoms,
                planning_success=cutamp_plan is not None,
                planning_failure_reason=failure_reason,
                planning_duration=planning_duration,
            )
            if file_handler is not None:
                remove_file_handler(file_handler)


def _run_server(
    host: str = "0.0.0.0",
    port: int = 8765,
    num_particles: int = 256,
    max_planning_time: float = 60.0,
    rerun_mode: str = "disabled",
    include_workspace: bool = False,
    overlap_perception: bool = True,
    max_concurrent_perception: int = 8,
    curobo_overrides: str | None = None,
) -> None:
    """Tiptop websocket planning server.

    Args:
        host: Host to bind to.
        port: Port to bind to.
        num_particles: Number of particles for cuTAMP.
        max_planning_time: Max planning time in seconds.
        rerun_mode: Rerun visualization mode. 'stream' spawns the Rerun viewer; 'save' writes .rrd files to disk; 'disabled' skips all Rerun logging.
        include_workspace: If True, include real-robot workspace cuboids in the collision world.
        overlap_perception: If True (default), concurrent planning requests overlap their I/O-bound
            perception while the cuRobo/GPU solve stays serialized; lets one server feed many
            parallel sim envs without running concurrent cuRobo (the cause of GPU power-transient
            crashes). Set False to serialize the whole pipeline (legacy single-flight behavior).
        max_concurrent_perception: Max perceptions running at once when overlap is on; bounds load
            on the Gemini API / M2T2 grasp server.
        curobo_overrides: cuRobo cost/tamp-parameter overrides as a JSON file path OR an inline JSON
            object string (e.g. '{"vae_manifold_weight": 5000, "rnd_novelty_weight": 100000,
            "time_dilation_factor_literal": 0.5}'). Applied at solver build time so every plan this
            server produces uses those tamp parameters. Keys match motion_planning.apply_cost_overrides.
            Absent -> stock gradient_trajopt.yml / tiptop.yml defaults.
    """
    print_tiptop_banner()
    check_cutamp_version()
    setup_logging()
    logging.getLogger("websockets.server").setLevel(logging.INFO)

    server = TiptopPlanningServer(
        host=host,
        port=port,
        num_particles=num_particles,
        max_planning_time=max_planning_time,
        rerun_mode=rerun_mode,
        include_workspace=include_workspace,
        overlap_perception=overlap_perception,
        max_concurrent_perception=max_concurrent_perception,
        curobo_overrides=curobo_overrides,
    )
    exit_code = 1
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        _log.info("Shutting down server (Ctrl+C)")
        exit_code = 0
    except Exception:
        _log.exception("Server failed")
    finally:
        if rerun_mode != "disabled":
            rr.disconnect()
        # Force exit to avoid segfault during GPU resource cleanup (CUDA/Warp/cuRobo destructors)
        os._exit(exit_code)


def entrypoint():
    tyro.cli(_run_server)


if __name__ == "__main__":
    entrypoint()

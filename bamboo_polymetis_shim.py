"""ZMQ shim: bamboo wire protocol -> polymetis (arm) + direct Modbus RTU (Robotiq gripper).

Runs on the NUC where polymetis is local (gRPC 50051). Listens on:
  control_port (5555): execute_trajectory, get_robot_state, ...
  gripper_port (5559): open_gripper, close_gripper (Robotiq 2F-85 via /dev/ttyUSB0)

Wire format mirrors bamboo-franka-client 0.1.1 (msgpack-encoded REQ/REP).

----------------------------------------------------------------------------------
FIX (gripper proprioception): RobotiqDriver._read_status() previously read the gripper
position as `r.registers[1] & 0xFF`. That byte is **gPR** (the Position *Request* echo =
the last commanded target), NOT the actual position. So `width` always reported exactly
the commanded value (0.085 open / 0.000 closed) even while holding an object -- useless as
proprioception. The actual measured position **gPO** is the high byte of register 2.

Robotiq 2F-85 input/status block at 0x07D0 (3 x 16-bit regs, big-endian bytes):
  registers[0] = [ GRIPPER STATUS (gACT/gGTO/gSTA/gOBJ) | reserved ]
  registers[1] = [ gFLT                                  | gPR  (commanded echo) ]
  registers[2] = [ gPO  (ACTUAL position)                | gCU  (motor current)  ]

Changed: gPO now reads (r.registers[2] >> 8) & 0xFF, and gCU (current) is also exposed.

Two unrelated corrections vs. the previous file (these were syntax/name errors that would
stop the file importing/running -- likely copy artifacts; verify against your copy):
  * `_gripper_handler` open_gripper: the `try:` was under-indented -> fixed.
  * `start_joint_velocity`: `_torch.flt32` -> `_torch.float32`.
----------------------------------------------------------------------------------
EXECUTION MODE (execute_trajectory): by default the control handler streams every cuRobo waypoint
through joint impedance so the arm tracks the planned path -- including the DROID-manifold
jerkiness that vae_manifold_weight induces. Pass --endpoint-exec to revert to the legacy behavior,
which collapses a trajectory to its FINAL waypoint and lets polymetis min-jerk there over the
summed duration (intermediate waypoints discarded, so the executed arm path is NOT the planned one
-- fine in free workspace). Under --endpoint-exec, anyone recording plan-derived commanded actions
should treat the plan waypoints, not the executed motion, as the command source.

The state_port (5557) serves get_robot_state from a cache filled by a ~100 Hz background
poller, so encoders stay readable while _control_handler is parked inside a blocking
move_to_joint_positions (the control socket cannot answer during motion).
----------------------------------------------------------------------------------
GRIPPER STATE CACHE: get_gripper_state is likewise served from a ~50 Hz background poller
inside RobotiqDriver (start_polling / get_cached_state), NOT a live Modbus read on the
request path. A live read stalls on the post-arm-motion RTU retries exactly during the
open/close ramp, starving the LeRobot gripper sampler so the recorded gripper obs collapses
to a single-frame jump. The poller and move writes share the one RTU link via _io_lock
(pymodbus RTU is not thread-safe).
----------------------------------------------------------------------------------
"""
from __future__ import annotations
import argparse, logging, sys, threading, time

import msgpack
import numpy as np
import zmq

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("bamboo-shim")

# --- polymetis: optional gripper (Robotiq). We treat Robotiq as a network-controlled gripper
# already exposed by polymetis's GripperInterface or, if not, fall back to a TCP/Modbus call.
# For DROID lab, Robotiq is wired through droid_server / polymetis GripperInterface :)

def _import_polymetis():
    sys.path.insert(0, "/home/prism-droid-nuc/droid/droid/fairo/polymetis/polymetis/python")
    from polymetis import RobotInterface
    try:
        from polymetis import GripperInterface
    except Exception:
        GripperInterface = None
    return RobotInterface, GripperInterface


# === Robotiq 2F-85 Modbus RTU driver ===
# Reference: Robotiq Universal Controller Manual; register map 0x03E8 (output) / 0x07D0 (input)
class RobotiqDriver:
    """Minimal driver for Robotiq 2F-85 over Modbus RTU.

    Modbus RTU frames over /dev/ttyUSB0 occasionally get corrupted around Franka arm
    activity (RS-485 noise / timing), so every register access retries with a short
    settle + serial reconnect. The link recovers within seconds, so a few retries turn
    transient failures into successes instead of aborting the caller.
    """

    OPEN_POS = 0      # 0 = fully open (~85mm)
    CLOSED_POS = 255  # 255 = fully closed (0mm)

    def __init__(self, port: str = "/dev/ttyUSB0", slave: int = 9):
        from pymodbus.client.sync import ModbusSerialClient
        self.slave = slave
        self._last_state: dict | None = None  # last successful get_state, reused on transient read failure
        # A single RTU serial link, now touched by two threads: the gripper handler (move writes +
        # bootstrap reads) and the background _poll_loop (continuous status reads). pymodbus RTU is
        # NOT thread-safe -- interleaved transactions corrupt the frame -- so every register access
        # goes through _io_lock. _state_lock guards the cached-state handoff to get_cached_state.
        self._io_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._poll_stop = threading.Event()
        self._poll_thread: threading.Thread | None = None
        self.client = ModbusSerialClient(
            method="rtu", port=port,
            baudrate=115200, stopbits=1, bytesize=8, parity="N", timeout=1.0,
        )
        if not self.client.connect():
            raise RuntimeError(f"Cannot open Robotiq on {port}")
        self.activate()
        # Continuously refresh the cache in the background so get_cached_state() -- the path the
        # LeRobot gripper sampler and the client-side settle poll both hit -- never does a live
        # Modbus read. Without this, those reads stall on the post-arm-motion RTU retries exactly
        # during the open/close ramp, so the recorded gripper obs collapses to a single-frame jump.
        self.start_polling()

    def _reconnect(self) -> None:
        """Reopen the serial link if it dropped; best-effort."""
        try:
            if not self.client.is_socket_open():
                self.client.connect()
        except Exception:
            pass

    def _write_action(self, act_req: int, pos: int, speed: int, force: int,
                      retries: int = 4, retry_delay: float = 0.15) -> bool:
        # reg[0]=ACT_REQ<<8, reg[1]=pos, reg[2]=(speed<<8)|force
        regs = [(act_req & 0xFF) << 8, pos & 0xFF, ((speed & 0xFF) << 8) | (force & 0xFF)]
        last = None
        with self._io_lock:  # serialize against the background poller's reads on the shared RTU link
            for attempt in range(retries):
                try:
                    r = self.client.write_registers(0x03E8, regs, unit=self.slave)
                    if r is not None and not r.isError():
                        if attempt:
                            log.info(f"Robotiq write succeeded on attempt {attempt + 1}")
                        return True
                    last = r
                except Exception as e:
                    last = e
                # Transient RTU error (common right after arm motion). Settle, reconnect, retry.
                time.sleep(retry_delay)
                self._reconnect()
        log.warning(f"Robotiq Modbus write failed after {retries} attempts: {last}")
        return False

    def _read_status(self, retries: int = 3, retry_delay: float = 0.1):
        with self._io_lock:  # serialize against move writes on the shared RTU link
            for _ in range(retries):
                try:
                    r = self.client.read_input_registers(0x07D0, 3, unit=self.slave)
                    if r is not None and not r.isError():
                        # Robotiq input/status registers (big-endian bytes within each register):
                        #   registers[0] = [ GRIPPER STATUS (high) | reserved (low) ]
                        #   registers[1] = [ gFLT (high)           | gPR  (low)  ]  gPR = commanded echo
                        #   registers[2] = [ gPO  (high)           | gCU  (low)  ]  gPO = ACTUAL position
                        return {
                            "gOBJ": (r.registers[0] >> 14) & 0x03,
                            "gSTA": (r.registers[0] >> 12) & 0x03,
                            "gGTO": (r.registers[0] >> 11) & 0x01,
                            "gACT": (r.registers[0] >> 8) & 0x01,
                            # FIX: actual position gPO is the HIGH byte of register 2 (byte 4),
                            # NOT `registers[1] & 0xFF` (which is gPR, the commanded-position echo).
                            "gPO":  (r.registers[2] >> 8) & 0xFF,   # actual position (0-255)
                            "gCU":  r.registers[2] & 0xFF,          # motor current (0-255), ~0.1 units
                        }
                except Exception:
                    pass
                time.sleep(retry_delay)
                self._reconnect()
        return None

    def activate(self, timeout: float = 10.0):
        # Clear first, then ACT
        self._write_action(0x00, 0, 0, 0)
        time.sleep(0.2)
        self._write_action(0x01, 0, 0, 0)
        # Wait until gSTA == 3 (activation complete). A Robotiq does a multi-second referencing
        # stroke after power-up, so allow a generous budget. Use single-shot reads here so one
        # slow/failed read doesn't consume the whole window (the retrying _read_status can block
        # ~3s per call on a flaky link).
        t0 = time.time()
        while time.time() - t0 < timeout:
            st = self._read_status(retries=1)
            if st and st["gSTA"] == 3:
                log.info(f"Robotiq activated. status={st}")
                return True
            time.sleep(0.2)
        log.warning("Robotiq activation timeout; continuing anyway.")
        return False

    def move(self, pos: int, speed: int = 128, force: int = 100, blocking: bool = True, timeout: float = 3.0) -> dict:
        pos = max(0, min(255, int(pos)))
        speed = max(0, min(255, int(speed)))
        force = max(0, min(255, int(force)))
        ok = self._write_action(0x09, pos, speed, force)  # rACT|rGTO
        if not ok:
            return {"success": False, "error": "Modbus write failed"}
        if blocking:
            t0 = time.time()
            while time.time() - t0 < timeout:
                st = self._read_status()
                if st and st["gOBJ"] in (1, 2, 3):  # object detected (1/2) or motion complete (3)
                    break
                time.sleep(0.05)
        return {"success": True}

    def width_to_pos(self, width_m: float) -> int:
        # Robotiq 2F-85: 0.0..0.085 m maps to 255..0 (inverse)
        width_clamped = max(0.0, min(0.085, float(width_m)))
        return int(round(255 - (width_clamped / 0.085) * 255))

    def pos_to_width(self, pos: int) -> float:
        return float(255 - max(0, min(255, int(pos)))) / 255.0 * 0.085

    def get_state(self) -> dict:
        st = self._read_status()
        if st is None:
            # Transient Modbus read failure. Do NOT synthesize a fully-closed reading (the old
            # gPO=255 fallback -> width 0.0), which would land in recorded gripper proprioception
            # as a false grasp. Reuse the last good sample; if there has never been one, flag the
            # reply stale rather than fabricate a position.
            if self._last_state is not None:
                return dict(self._last_state)
            return {
                "width": self.pos_to_width(self.OPEN_POS),
                "is_grasped": False,
                "is_moving": False,
                "current": 0.0,
                "stale": True,
            }
        width = self.pos_to_width(st["gPO"])  # derived from the ACTUAL position (gPO)
        is_grasped = st["gOBJ"] in (1, 2)
        is_moving = st["gOBJ"] == 0 and st["gGTO"] == 1
        # `current` is the motor current (~grasp force proxy); harmless extra field for clients
        # that ignore it (e.g. bamboo client reads only width/is_grasped/is_moving).
        state = {
            "width": width,
            "is_grasped": is_grasped,
            "is_moving": is_moving,
            "current": float(st.get("gCU", 0)) * 0.1,
        }
        with self._state_lock:
            self._last_state = state
        return state

    def start_polling(self, period_s: float = 0.02) -> None:
        """Spin a background thread refreshing the cached state at ~1/period_s Hz (idempotent)."""
        if self._poll_thread is not None and self._poll_thread.is_alive():
            return
        self._poll_stop.clear()
        self._poll_thread = threading.Thread(
            target=self._poll_loop, args=(period_s,), name="robotiq-state-poller", daemon=True
        )
        self._poll_thread.start()

    def _poll_loop(self, period_s: float) -> None:
        while not self._poll_stop.is_set():
            t0 = time.time()
            try:
                self.get_state()  # live read under _io_lock; updates _last_state
            except Exception:
                pass  # keep the last good sample; never let a read error kill the poller
            time.sleep(max(0.0, period_s - (time.time() - t0)))

    def get_cached_state(self) -> dict:
        """Return the most recent polled state without touching the serial link.

        This is the read path for clients (LeRobot sampler, settle poll). It answers from the
        background poller's cache so those reads stay fast and dense through an open/close ramp,
        instead of stalling on the post-arm-motion RTU retries a live read would incur. Falls back
        to a one-off live read only to bootstrap before the poller's first sample lands.
        """
        with self._state_lock:
            cached = self._last_state
        if cached is not None:
            return dict(cached)
        return self.get_state()

    def close(self):
        self._poll_stop.set()
        if self._poll_thread is not None:
            self._poll_thread.join(timeout=1.0)
        try:
            self.client.close()
        except Exception:
            pass


def _np_from(a):
    return np.asarray(a, dtype=np.float32)


def _connect_robot(ip, port, robotiq_port: str | None = "/dev/ttyUSB0"):
    RobotInterface, _ = _import_polymetis()
    log.info(f"Connecting RobotInterface to {ip}:{port} ...")
    robot = RobotInterface(ip_address=ip, port=port, enforce_version=False)
    log.info(f"Connected. q0 = {robot.get_joint_positions().tolist()}")

    gripper = None
    if robotiq_port:
        try:
            gripper = RobotiqDriver(port=robotiq_port)
            log.info(f"RobotiqDriver connected on {robotiq_port}.")
        except Exception as e:
            log.warning(f"RobotiqDriver unavailable on {robotiq_port}: {e}; using STUB gripper (motions are no-ops).")
            gripper = None
    return robot, gripper


def _control_handler(socket: zmq.Socket, robot, gripper, robotiq_only: bool, dense_exec: bool = True):
    log.info(f"Control loop listening... (dense_exec={dense_exec})")
    while True:
        replied = False
        try:
            req_raw = socket.recv()
            req = msgpack.unpackb(req_raw, raw=False)
            cmd = (req or {}).get("command", "")
            log.info(f"control <= {cmd}")
            if cmd == "get_robot_state":
                q = robot.get_joint_positions().tolist()
                dq = robot.get_joint_velocities().tolist()
                ee_pos, ee_quat = robot.get_ee_pose()
                pose = np.eye(4)
                pose[:3, 3] = ee_pos.numpy()
                from scipy.spatial.transform import Rotation as R
                pose[:3, :3] = R.from_quat(ee_quat.numpy()).as_matrix()
                # bamboo expects O_T_EE as 16-float COLUMN-MAJOR
                o_t_ee = pose.T.flatten().tolist()
                data = {
                    "q": q,
                    "dq": dq,
                    "tau_J": [0.0] * 7,
                    "O_T_EE": o_t_ee,
                    "time_sec": time.time(),
                }
                resp = {"success": True, "data": data}
            elif cmd == "execute_trajectory":
                data = req.get("data", {}) or {}
                waypoints = data.get("waypoints", [])
                default_duration = float(data.get("default_duration", 0.5))
                if not waypoints:
                    resp = {"success": False, "error": "empty waypoints"}
                elif dense_exec and len(waypoints) > 1:
                    # Dense execution: stream EVERY cuRobo waypoint through a joint-impedance
                    # controller so the arm tracks the planned path -- including the DROID-manifold
                    # jerkiness that vae_manifold_weight induces. Contrast with the endpoint path
                    # below, which min-jerks to the final waypoint and discards the plan's shape.
                    import torch
                    try:
                        robot.start_joint_impedance()
                        streamed = 0
                        # Pace to an ABSOLUTE per-waypoint deadline rather than sleeping the full
                        # duration after each setpoint. update_desired_joint_positions is a ~5 ms gRPC
                        # round trip; sleeping d ON TOP of it adds that overhead to every ~20 ms
                        # waypoint, so trajectories ran a near-constant ~1.25x slower than commanded
                        # (measured 1.21-1.26x on real episodes). Sleeping to a running deadline absorbs
                        # the call time instead, so the arm tracks the planned duration.
                        t_next = time.monotonic()
                        for wp in waypoints:
                            q = wp.get("q_goal") or []
                            if len(q) != 7:
                                raise ValueError(f"waypoint {streamed} q_goal len {len(q)} != 7")
                            d = float(wp.get("duration", default_duration))
                            robot.update_desired_joint_positions(torch.tensor(q, dtype=torch.float32))
                            t_next += d if d > 0 else default_duration
                            time.sleep(max(0.0, t_next - time.monotonic()))
                            streamed += 1
                        # Leave the arm holding the final setpoint for the next command.
                        robot.terminate_current_policy()
                        log.info(f"  trajectory: streamed {streamed} dense waypoints (joint impedance)")
                        resp = {"success": True}
                    except Exception as e:
                        log.exception("dense trajectory exec failed")
                        try:
                            robot.terminate_current_policy()  # don't leave a half-run policy active
                        except Exception:
                            pass
                        resp = {"success": False, "error": str(e)}
                else:
                    # Endpoint execution (legacy, --endpoint-exec): sum the per-waypoint durations to
                    # get total trajectory time, then send only the FINAL waypoint to polymetis, which
                    # min-jerks to the goal in `total_dur` seconds. cuRobo's intermediate waypoints are
                    # lost -- fine in free workspace, but it erases the planned trajectory shape (the
                    # default dense path above tracks it instead).
                    total_dur = 0.0
                    for wp in waypoints:
                        d = float(wp.get("duration", default_duration))
                        total_dur += d if d > 0 else default_duration
                    # Polymetis warns if time_to_go < ~0.5s. Enforce a sensible minimum.
                    total_dur = max(total_dur, 1.0)
                    final = waypoints[-1]
                    q_goal = final.get("q_goal") or []
                    if len(q_goal) != 7:
                        resp = {"success": False, "error": f"final waypoint q_goal len {len(q_goal)} != 7"}
                    else:
                        log.info(f"  trajectory: {len(waypoints)} wp -> final goal in {total_dur:.2f}s")
                        try:
                            robot.move_to_joint_positions(positions=_np_from(q_goal), time_to_go=total_dur)
                            resp = {"success": True}
                        except Exception as e:
                            log.exception("trajectory exec failed")
                            resp = {"success": False, "error": str(e)}
            elif cmd == "start_joint_impedance":
                try:
                    robot.start_cartesian_impedance()  # PATCH: match DROID env.step (cartesian impedance + joint pos updates)
                    resp = {"success": True}
                except Exception as e:
                    log.exception("start_joint_impedance failed")
                    resp = {"success": False, "error": str(e)}
            elif cmd == "update_joint_positions":
                try:
                    pos = req.get("positions", [])
                    if len(pos) != 7:
                        resp = {"success": False, "error": f"positions len {len(pos)} != 7"}
                    else:
                        import torch
                        robot.update_desired_joint_positions(torch.tensor(pos, dtype=torch.float32))
                        resp = {"success": True}
                except Exception as e:
                    resp = {"success": False, "error": str(e)}
            elif cmd == "start_joint_velocity":
                try:
                    import torch as _torch
                    # NOTE: corrected `_torch.flt32` -> `_torch.float32` (the former is not a valid dtype).
                    robot.start_joint_velocity_control(joint_vel_desired=_torch.zeros(7, dtype=_torch.float32))
                    resp = {"success": True}
                except Exception as e:
                    log.exception("start_joint_velocity_control failed")
                    resp = {"success": False, "error": str(e)}
            elif cmd == "update_joint_velocity":
                try:
                    vel = req.get("vel", [])
                    if len(vel) != 7:
                        resp = {"success": False, "error": f"vel len {len(vel)} != 7"}
                    else:
                        import torch
                        robot.update_desired_joint_velocities(torch.tensor(vel, dtype=torch.float32))
                        resp = {"success": True}
                except Exception as e:
                    resp = {"success": False, "error": str(e)}
            elif cmd == "stop_velocity":
                try:
                    robot.terminate_current_policy()
                    resp = {"success": True}
                except Exception as e:
                    resp = {"success": False, "error": str(e)}
            elif cmd in ("open_gripper", "close_gripper") and robotiq_only:
                # On the control socket, Franka-hand commands; we don't expose Franka hand here.
                resp = {"success": False, "error": "Franka-hand commands on control socket are not implemented (Robotiq-only setup)."}
            else:
                resp = {"success": False, "error": f"Unknown command on control socket: {cmd}"}
            replied = True
            socket.send(msgpack.packb(resp))
        except Exception as e:
            log.exception("control handler error")
            # Only reply on error if no reply already went out. A second send on a REP socket
            # that already sent raises EFSM (swallowed here) and wedges the socket so every later
            # recv() throws and this thread busy-loops on errors.
            if not replied:
                try:
                    socket.send(msgpack.packb({"success": False, "error": f"shim error: {e}"}))
                except Exception:
                    pass


def _gripper_handler(socket: zmq.Socket, gripper):
    log.info("Gripper loop listening...")
    # State stub used when real gripper is unavailable — keeps TiPToP planning alive.
    STUB_WIDTH = 0.085  # max-open width (Robotiq 2F-85)
    stub_state = {"width": STUB_WIDTH, "is_grasped": False, "is_moving": False}
    while True:
        replied = False
        try:
            req_raw = socket.recv()
            req = msgpack.unpackb(req_raw, raw=False)
            cmd = (req or {}).get("command", "")
            log.info(f"gripper <= {cmd}")
            if gripper is None:
                # STUB mode: state queries succeed; motion commands return success but warn.
                if cmd in ("get_gripper_state", "get_state"):
                    resp = {"success": True, "state": dict(stub_state)}
                elif cmd == "open_gripper":
                    log.warning("STUB: open_gripper requested but no real gripper attached (no-op).")
                    stub_state["width"] = float(req.get("width", STUB_WIDTH))
                    resp = {"success": True}
                elif cmd == "close_gripper":
                    log.warning("STUB: close_gripper requested but no real gripper attached (no-op).")
                    stub_state["width"] = 0.0
                    resp = {"success": True}
                else:
                    resp = {"success": False, "error": f"Unknown gripper command: {cmd}"}
                replied = True
                socket.send(msgpack.packb(resp))
                continue
            elif cmd == "open_gripper":
                # bamboo client passes width in metres, speed/force normalised 0..1
                width = float(req.get("width", 0.085))
                speed01 = float(req.get("speed", 0.5))
                force01 = float(req.get("force", 0.5))
                blocking = bool(req.get("blocking", True))
                try:
                    pos = gripper.width_to_pos(width)
                    speed = int(max(0, min(255, speed01 * 255)))
                    force = int(max(0, min(255, force01 * 255)))
                    resp = gripper.move(pos, speed=speed, force=force, blocking=blocking)
                except Exception as e:
                    resp = {"success": False, "error": str(e)}
            elif cmd == "close_gripper":
                speed01 = float(req.get("speed", 0.5))
                force01 = float(req.get("force", 0.5))
                blocking = bool(req.get("blocking", True))
                try:
                    speed = int(max(0, min(255, speed01 * 255)))
                    force = int(max(0, min(255, force01 * 255)))
                    resp = gripper.move(gripper.CLOSED_POS, speed=speed, force=force, blocking=blocking)
                except Exception as e:
                    resp = {"success": False, "error": str(e)}
            elif cmd in ("get_gripper_state", "get_state"):
                try:
                    # Serve from the background poller's cache (no live Modbus read on the request
                    # path) so sampler/settle reads stay dense through the open/close ramp.
                    st = gripper.get_cached_state()
                    resp = {
                        "success": True,
                        "state": {
                            "width": float(st["width"]),
                            "is_grasped": bool(st["is_grasped"]),
                            "is_moving": bool(st["is_moving"]),
                            "current": float(st.get("current", 0.0)),
                        },
                    }
                    log.info(
                        f"  gripper state: width={float(st['width']):.4f} "
                        f"is_grasped={bool(st['is_grasped'])} is_moving={bool(st['is_moving'])} "
                        f"current={float(st.get('current', 0.0)):.2f} stale={bool(st.get('stale', False))}"
                    )
                except Exception as e:
                    resp = {"success": False, "error": str(e)}
            else:
                resp = {"success": False, "error": f"Unknown gripper command: {cmd}"}
            replied = True
            socket.send(msgpack.packb(resp))
        except Exception as e:
            log.exception("gripper handler error")
            # Only reply on error if no reply already went out (a double send wedges the REP socket).
            if not replied:
                try:
                    socket.send(msgpack.packb({"success": False, "error": f"shim error: {e}"}))
                except Exception:
                    pass


class _StateCache:
    """Thread-safe holder for the most recent robot-state sample (poller -> state handler)."""

    def __init__(self):
        self._lock = threading.Lock()
        self._data: dict | None = None

    def set(self, data: dict) -> None:
        with self._lock:
            self._data = data

    def get(self) -> dict | None:
        with self._lock:
            return self._data


def _state_poller(robot, cache: _StateCache, period_s: float = 0.01) -> None:
    """~100 Hz background loop caching q/dq/ee-pose so the state socket never touches the robot.

    polymetis get_* are independent gRPC reads, so they succeed even while _control_handler is
    parked inside a blocking move_to_joint_positions on the control socket. A transient read error
    keeps the last good sample rather than clearing the cache.
    """
    from scipy.spatial.transform import Rotation as R

    while True:
        t0 = time.time()
        try:
            q = robot.get_joint_positions().tolist()
            dq = robot.get_joint_velocities().tolist()
            ee_pos, ee_quat = robot.get_ee_pose()
            pose = np.eye(4)
            pose[:3, 3] = ee_pos.numpy()
            pose[:3, :3] = R.from_quat(ee_quat.numpy()).as_matrix()
            cache.set({
                "q": q,
                "dq": dq,
                "tau_J": [0.0] * 7,
                # bamboo expects O_T_EE as 16-float COLUMN-MAJOR (same shape as _control_handler).
                "O_T_EE": pose.T.flatten().tolist(),
                "time_sec": time.time(),
            })
        except Exception:
            pass  # keep last good sample; gRPC hiccups during motion are transient
        time.sleep(max(0.0, period_s - (time.time() - t0)))


def _state_handler(socket: zmq.Socket, cache: _StateCache) -> None:
    """Answer get_robot_state/ping instantly from ``cache`` on its own thread (never blocks)."""
    log.info("State loop listening...")
    while True:
        replied = False
        try:
            req = msgpack.unpackb(socket.recv(), raw=False)
            cmd = (req or {}).get("command", "")
            if cmd == "get_robot_state":
                data = cache.get()
                resp = {"success": True, "data": data} if data is not None else {
                    "success": False, "error": "state cache not warmed yet"
                }
            elif cmd == "ping":
                resp = {"success": True}
            else:
                resp = {"success": False, "error": f"Unknown command on state socket: {cmd}"}
            replied = True
            socket.send(msgpack.packb(resp))
        except Exception as e:
            log.exception("state handler error")
            # Only reply on error if no reply already went out (a double send wedges the REP socket).
            if not replied:
                try:
                    socket.send(msgpack.packb({"success": False, "error": f"shim error: {e}"}))
                except Exception:
                    pass


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bind", default="0.0.0.0")
    p.add_argument("--control-port", type=int, default=5555)
    p.add_argument("--gripper-port", type=int, default=5559)
    p.add_argument("--state-port", type=int, default=5557)
    p.add_argument("--polymetis-ip", default="localhost")
    p.add_argument("--polymetis-port", type=int, default=50051)
    p.add_argument("--robotiq-port", default="/dev/ttyUSB0", help="serial device (set '' to stub)")
    p.add_argument(
        "--endpoint-exec",
        action="store_true",
        help="revert to the legacy endpoint behavior: min-jerk to each segment's FINAL waypoint and "
        "discard the intermediate ones. The default streams cuRobo's dense waypoints through joint "
        "impedance so the arm tracks the planned path (incl. DROID-manifold jerkiness from "
        "vae_manifold_weight).",
    )
    args = p.parse_args()
    dense_exec = not args.endpoint_exec

    robot, gripper = _connect_robot(args.polymetis_ip, args.polymetis_port, args.robotiq_port or None)

    ctx = zmq.Context.instance()
    ctrl_sock = ctx.socket(zmq.REP); ctrl_sock.bind(f"tcp://{args.bind}:{args.control_port}")
    grip_sock = ctx.socket(zmq.REP); grip_sock.bind(f"tcp://{args.bind}:{args.gripper_port}")
    state_sock = ctx.socket(zmq.REP); state_sock.bind(f"tcp://{args.bind}:{args.state_port}")
    log.info(
        f"Bamboo-polymetis shim ready: control=tcp://{args.bind}:{args.control_port} "
        f"gripper=tcp://{args.bind}:{args.gripper_port} state=tcp://{args.bind}:{args.state_port}"
    )

    cache = _StateCache()
    poller = threading.Thread(target=_state_poller, args=(robot, cache), daemon=True)
    t1 = threading.Thread(target=_control_handler, args=(ctrl_sock, robot, gripper, True, dense_exec), daemon=True)
    t2 = threading.Thread(target=_gripper_handler, args=(grip_sock, gripper), daemon=True)
    t3 = threading.Thread(target=_state_handler, args=(state_sock, cache), daemon=True)
    poller.start(); t1.start(); t2.start(); t3.start()
    t1.join(); t2.join(); t3.join()


if __name__ == "__main__":
    main()
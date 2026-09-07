"""ZMQ client for the bimanual YAM arm server (``tiptop/yam_arm_server.py``).

Implements the same surface tiptop already consumes from ``BambooFrankaClient``, so
``execute_plan``, ``motion_planning.go_to_q`` and ``lerobot_capture`` drive a YAM without knowing
it is one. Exhaustively, that surface is:

``get_joint_positions`` · ``execute_joint_impedance_path`` · ``open_gripper`` · ``close_gripper`` ·
``get_gripper_state`` · ``close`` · the ``control_socket`` attribute (tiptop's queued-trajectory
path talks to the shim directly over it) · the ``gripper_socket`` attribute (its presence is how
``execute_plan._can_overlap_gripper`` decides the gripper can actuate while the arm keeps moving).

**Arm selection.** One client owns both arms. Which one ``get_joint_positions`` reads and
``execute_joint_impedance_path`` drives follows ``tiptop_cfg().robot.type`` — i.e. whichever arm
:func:`tiptop.yam.active_arm` currently names. The server is told on every change, because tiptop's
queued-trajectory path sends bare ``execute_trajectory`` requests with no arm field.
"""

import logging
import threading

import msgpack
import numpy as np
import zmq
from jaxtyping import Float

from tiptop.config import tiptop_cfg
from tiptop.yam import ARM_DOF, NEUTRAL_Q, arm_of, other_arm

_log = logging.getLogger(__name__)

# Jaw travel at full open, metres. Must match yam_arm_server.GRIPPER_STROKE_M — the server reports
# a width in metres and this is the constant it divides by.
GRIPPER_STROKE_M = 0.096

DEFAULT_CONTROL_PORT = 5555
DEFAULT_GRIPPER_PORT = 5559
DEFAULT_STATE_PORT = 5557

# A trajectory request blocks for the whole motion, so its receive timeout has to cover the longest
# plausible segment plus the server's own slack. Everything else is a fast cache read.
_RPC_TIMEOUT_MS = 15_000
_TRAJ_TIMEOUT_MS = 180_000


class YamClientError(RuntimeError):
    """The arm server refused a request or could not be reached."""


class YamClient:
    """Talks to ``yam_arm_server.py`` over three REP sockets (control / gripper / state)."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        control_port: int = DEFAULT_CONTROL_PORT,
        gripper_port: int = DEFAULT_GRIPPER_PORT,
        state_port: int = DEFAULT_STATE_PORT,
    ):
        self.host = host
        self.control_port = control_port
        self.gripper_port = gripper_port
        self.state_port = state_port

        self._ctx = zmq.Context()
        # ZMQ REQ sockets are neither thread-safe nor tolerant of interleaved traffic, and during
        # execution the gripper runs on a background thread (execute_plan._spawn_gripper_command)
        # while the main thread drives the arm. One lock per socket keeps each strictly
        # request/reply; without them an overlapped gripper open racing a state read wedges the
        # socket in EFSM and the rollout dies mid-motion.
        self._control_lock = threading.Lock()
        self._gripper_lock = threading.Lock()
        self._sockets: dict[str, zmq.Socket] = {
            "control": self._connect(control_port, _RPC_TIMEOUT_MS),
            "gripper": self._connect(gripper_port, _RPC_TIMEOUT_MS),
        }

        self._active_arm: str | None = None
        self._test_connection()
        # bimanual_yam_dual has no single "active arm" (see the `arm` property below) -- both arms
        # are always addressed explicitly, and the dual execution path opens its own dedicated
        # control socket (execute_plan._DualQueuedArm) rather than routing through this client's
        # active-arm server state. Skip the sync so construction succeeds in that embodiment too.
        if tiptop_cfg().robot.type != "bimanual_yam_dual":
            self.sync_active_arm()

    # -- plumbing ------------------------------------------------------------------------------
    def _connect(self, port: int, timeout_ms: int) -> zmq.Socket:
        sock = self._ctx.socket(zmq.REQ)
        sock.setsockopt(zmq.RCVTIMEO, timeout_ms)
        sock.setsockopt(zmq.LINGER, 0)
        sock.connect(f"tcp://{self.host}:{port}")
        return sock

    def _rpc(self, which: str, request: dict, timeout_ms: int | None = None) -> dict:
        """One request/reply on the named socket, rebuilding it if the exchange breaks.

        A REQ socket is a strict send/recv state machine: a timed-out recv, or an exception between
        the two, leaves it stuck expecting the reply that never came, and EVERY later call on it
        fails. This client is a process-wide cached singleton held for the whole warm session, so
        without recovery one interrupted round trip — a preempt during a long trajectory, a
        momentarily unresponsive server — would poison every subsequent rollout. Rebuilding is
        cheap and correct: the request is lost, the caller sees an error, and the next call works.
        """
        lock, port, default_timeout = self._socket_config(which)
        with lock:
            sock = self._sockets[which]
            if timeout_ms is not None:
                sock.setsockopt(zmq.RCVTIMEO, timeout_ms)
            try:
                sock.send(msgpack.packb(request, use_bin_type=True))
                reply = msgpack.unpackb(sock.recv(), raw=False) or {}
                return reply
            except zmq.Again as e:
                self._reconnect(which)
                raise YamClientError(
                    f"no reply from the YAM arm server at tcp://{self.host}:{port} for "
                    f"{request.get('command')!r}. Is yam_arm_server.py running?"
                ) from e
            except BaseException:
                # Includes KeyboardInterrupt: a preempt landing between send and recv is exactly the
                # case that strands the socket mid-exchange.
                self._reconnect(which)
                raise
            finally:
                if timeout_ms is not None and self._sockets[which] is sock:
                    sock.setsockopt(zmq.RCVTIMEO, default_timeout)

    def _socket_config(self, which: str) -> tuple[threading.Lock, int, int]:
        if which == "control":
            return self._control_lock, self.control_port, _RPC_TIMEOUT_MS
        return self._gripper_lock, self.gripper_port, _RPC_TIMEOUT_MS

    def _reconnect(self, which: str) -> None:
        """Replace a socket left mid-exchange. Caller must hold that socket's lock."""
        _, port, timeout = self._socket_config(which)
        old = self._sockets.get(which)
        if old is not None:
            try:
                old.close(linger=0)
            except Exception:
                pass
        self._sockets[which] = self._connect(port, timeout)
        _log.warning("Rebuilt the YAM %s socket after a broken exchange", which)

    @property
    def control_socket(self) -> zmq.Socket:
        """The live control socket. A property because a broken exchange replaces it (see _rpc);
        ``execute_plan._QueuedArm`` grabs this once, which is why it builds its own client."""
        return self._sockets["control"]

    @property
    def gripper_socket(self) -> zmq.Socket:
        """Present so ``execute_plan._can_overlap_gripper`` sees a dedicated gripper channel."""
        return self._sockets["gripper"]

    def _control(self, request: dict, timeout_ms: int | None = None) -> dict:
        return self._rpc("control", request, timeout_ms)

    def _gripper(self, request: dict, timeout_ms: int | None = None) -> dict:
        return self._rpc("gripper", request, timeout_ms)

    def _test_connection(self) -> None:
        res = self._control({"command": "ping"}, timeout_ms=5000)
        if not res.get("success"):
            raise YamClientError(f"YAM arm server ping failed: {res.get('error')}")

    # -- arm selection -------------------------------------------------------------------------
    @property
    def arm(self) -> str:
        """The arm the active embodiment names (``robot.type: bimanual_yam_<arm>``).

        Raises for ``bimanual_yam_dual`` (via ``arm_of``) -- there is no single active arm in that
        embodiment, every call addresses left/right explicitly (see ``get_arm_state``,
        ``get_dual_joint_positions``, and the dual execution path in
        ``execute_plan._DualQueuedArm``).
        """
        return arm_of(tiptop_cfg().robot.type)

    def sync_active_arm(self) -> str:
        """Push the active arm to the server if it has changed.

        Only needed for requests that do NOT name an arm — which, from tiptop, means exactly
        ``execute_plan._QueuedArm``: it talks to the shim protocol directly and is deliberately
        embodiment-agnostic, so the server has to know which arm "the arm" is. Every request this
        client sends names its arm explicitly, so nothing else depends on it.

        Call from the MAIN thread only. It is a control-socket round trip, and the control socket is
        the one that blocks for the length of a trajectory; issuing it from the gripper thread (which
        runs concurrently with arm motion, see ``execute_plan._spawn_gripper_command``) would queue a
        request behind a motion that has not finished. That is why the gripper and state paths below
        use the plain :attr:`arm` property, which reads the config and touches no socket.
        """
        arm = self.arm
        if arm != self._active_arm:
            res = self._control({"command": "set_active_arm", "arm": arm})
            if not res.get("success"):
                raise YamClientError(f"could not select the {arm} arm: {res.get('error')}")
            self._active_arm = arm
        return arm

    # -- state ---------------------------------------------------------------------------------
    def get_joint_positions(self) -> list[float]:
        """The ACTIVE arm's 6 joint positions, in radians.

        6 numbers, not 7: cuRobo's YAM configs expose 6 active DOF (the fingers are locked), so this
        is what ``go_to_q``, ``capture_live_observation`` and cuTAMP's ``q_init`` all expect.
        """
        arm = self.arm
        res = self._control({"command": "get_robot_state", "arm": arm})
        if not res.get("success"):
            raise YamClientError(f"get_robot_state failed: {res.get('error')}")
        return [float(x) for x in res["data"]["q"][:ARM_DOF]]

    def get_arm_state(self, arm: str | None = None) -> dict:
        """One arm's measured ``{q, dq, gripper, gripper_vel, gripper_eff, time_sec}``."""
        res = self._control({"command": "get_robot_state", "arm": arm or self.arm})
        if not res.get("success"):
            raise YamClientError(f"get_robot_state failed: {res.get('error')}")
        return res["data"]

    def get_dual_joint_positions(self) -> list[float]:
        """Both arms' 12 joint positions, left then right -- the ``bimanual_yam_dual`` ``q_init``.

        Two sequential ``get_arm_state`` round trips, each naming its arm explicitly (unlike
        ``get_joint_positions``, this never touches the ``arm`` property, so it works in dual mode
        where that property has nothing to return).
        """
        left = self.get_arm_state("left")["q"][:ARM_DOF]
        right = self.get_arm_state("right")["q"][:ARM_DOF]
        return [float(x) for x in (*left, *right)]

    def get_gripper_state(self, arm: str | None = None) -> dict:
        """``{"success", "state": {"width" (m), "normalized", "is_grasped", "is_moving", ...}}``.

        Width is in metres because that is the unit tiptop's gripper plumbing speaks
        (``lerobot_capture._read_gripper_width`` / ``GRIPPER_MAX_WIDTH``).
        """
        return self._gripper({"command": "get_gripper_state", "arm": arm or self.arm})

    # -- motion --------------------------------------------------------------------------------
    def execute_joint_impedance_path(
        self,
        joint_confs: Float[np.ndarray, "n 6"],
        joint_vels: Float[np.ndarray, "n 6"] | None = None,
        durations: list | None = None,
        default_duration: float = 0.5,
        kp: np.ndarray | None = None,
        kd: np.ndarray | None = None,
    ) -> dict:
        """Stream a dense joint trajectory to the active arm and block until it finishes.

        ``kp``/``kd`` are accepted for signature compatibility with ``BambooFrankaClient`` and
        ignored: i2rt owns the YAM's gains (``i2rt/robots/config/yam.yml``) and re-tuning them
        per-trajectory from here would be a good way to make the arm behave differently between the
        planner's model and reality.
        """
        if kp is not None or kd is not None:
            _log.debug("YamClient ignores per-path kp/kd; i2rt owns the YAM's gains")
        arm = self.sync_active_arm()
        confs = np.asarray(joint_confs, dtype=np.float64).reshape(len(joint_confs), -1)[:, :ARM_DOF]
        vels = (
            np.asarray(joint_vels, dtype=np.float64).reshape(len(joint_vels), -1)[:, :ARM_DOF]
            if joint_vels is not None
            else np.zeros_like(confs)
        )
        if durations is None:
            durations = [default_duration] * len(confs)
        waypoints = [
            {"q_goal": q.tolist(), "velocity": v.tolist(), "duration": float(d)}
            for q, v, d in zip(confs, vels, durations)
        ]
        total = float(sum(durations))
        return self._control(
            {
                "command": "execute_trajectory",
                "arm": arm,
                "data": {
                    "arm": arm,
                    "waypoints": waypoints,
                    "default_duration": default_duration,
                    "queue": False,
                },
            },
            timeout_ms=int(total * 1500) + _TRAJ_TIMEOUT_MS,
        )

    def move_to(self, q_goal, duration: float = 3.0, arm: str | None = None) -> dict:
        """Interpolate one arm to ``q_goal`` (6 joints, or 7 to set the gripper too).

        The one path allowed to travel a long way in one call — ``execute_joint_impedance_path``
        refuses a large first step, because a planned trajectory should always start from the
        measured state. Used to park the idle arm and to recover a stranded one.
        """
        return self._control(
            {
                "command": "move_to",
                "arm": arm or self.arm,
                "q_goal": [float(x) for x in np.asarray(q_goal, dtype=np.float64).reshape(-1)],
                "duration": float(duration),
            },
            timeout_ms=int(duration * 1500) + _RPC_TIMEOUT_MS,
        )

    def park_idle_arm(self, duration: float = 4.0, tolerance: float = 0.05) -> dict:
        """Put the NON-active arm at :data:`~tiptop.yam.NEUTRAL_Q`, where cuRobo believes it is.

        The active arm's cuRobo config locks the other arm at that posture and collision-checks it
        there. If the parked arm is actually somewhere else, every plan is validated against a robot
        that does not exist — so this runs before each arm's plan, and no-ops when the arm is
        already in place so a bimanual episode does not shuffle the idle arm twice per rollout.
        """
        idle = other_arm(self.arm)
        try:
            measured = np.asarray(self.get_arm_state(idle)["q"], dtype=np.float64)[:ARM_DOF]
        except YamClientError as e:
            return {"success": False, "error": str(e)}
        error = float(np.abs(measured - np.asarray(NEUTRAL_Q)).max())
        if error <= tolerance:
            _log.info(f"{idle} arm already parked at the neutral posture (max error {error:.3f} rad)")
            return {"success": True, "moved": False}
        _log.info(f"Parking the {idle} arm at the neutral posture (max error {error:.3f} rad)")
        res = self.move_to(NEUTRAL_Q, duration=duration, arm=idle)
        return {**res, "moved": True}

    def free_drive(self, hold_gripper: bool = True, arm: str | None = None) -> dict:
        """Make one arm backdrivable so it can be pushed around by hand.

        The gripper keeps holding by default, so anything clamped in the jaws stays clamped — unlike
        the server's ``relax``, which drops it. :meth:`hold` puts the arm back under PD control.
        """
        return self._control(
            {"command": "free_drive", "arm": arm or self.arm, "hold_gripper": bool(hold_gripper)},
            timeout_ms=_RPC_TIMEOUT_MS,
        )

    def hold(self, arm: str | None = None) -> dict:
        """Re-engage PD control, latching wherever the arm currently is. Undoes :meth:`free_drive`."""
        return self._control({"command": "hold", "arm": arm or self.arm}, timeout_ms=_RPC_TIMEOUT_MS)

    # -- gripper -------------------------------------------------------------------------------
    def open_gripper(
        self, speed: float = 1.0, force: float = 0.1, blocking: bool = True, arm: str | None = None
    ) -> dict:
        return self._gripper(
            {"command": "open_gripper", "arm": arm or self.arm,
             "speed": float(speed), "force": float(force), "blocking": bool(blocking)},
            timeout_ms=_RPC_TIMEOUT_MS,
        )

    def close_gripper(
        self, speed: float = 1.0, force: float = 0.8, blocking: bool = True, arm: str | None = None
    ) -> dict:
        return self._gripper(
            {"command": "close_gripper", "arm": arm or self.arm,
             "speed": float(speed), "force": float(force), "blocking": bool(blocking)},
            timeout_ms=_RPC_TIMEOUT_MS,
        )

    # -- teardown ------------------------------------------------------------------------------
    def close(self) -> None:
        for sock in list(self._sockets.values()):
            try:
                sock.close(linger=0)
            except Exception:
                pass
        self._sockets.clear()
        try:
            self._ctx.term()
        except Exception:
            pass

    def __enter__(self) -> "YamClient":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class YamStateReader:
    """Read-only connection to the arm server's STATE port, for the LeRobot capture.

    Its own socket for the same reason the Franka has one: the control socket is parked inside a
    blocking trajectory for the length of every plan segment, while the capture samples measured
    state at 30 Hz throughout. The state port serves a background cache, so it answers mid-motion.

    Unlike :class:`YamClient` this returns BOTH arms in one round trip — a bimanual episode records
    all 14 numbers whichever arm is moving.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = DEFAULT_STATE_PORT, timeout_ms: int = 300):
        self.host = host
        self.port = port
        self._timeout_ms = timeout_ms
        self._ctx = zmq.Context()
        self._sock: zmq.Socket | None = None
        self._connect()

    def _connect(self) -> None:
        """(Re)create the REQ socket. A timed-out recv leaves REQ unusable, so recover by reconnecting."""
        if self._sock is not None:
            self._sock.close(linger=0)
        self._sock = self._ctx.socket(zmq.REQ)
        self._sock.setsockopt(zmq.RCVTIMEO, self._timeout_ms)
        self._sock.setsockopt(zmq.LINGER, 0)
        self._sock.connect(f"tcp://{self.host}:{self.port}")

    def read(self) -> dict | None:
        """Both arms' measured state, or None if the server did not answer in time."""
        try:
            self._sock.send(msgpack.packb({"command": "get_robot_state"}, use_bin_type=True))
            reply = msgpack.unpackb(self._sock.recv(), raw=False)
        except Exception:
            self._connect()
            return None
        if not (isinstance(reply, dict) and reply.get("success")):
            return None
        return reply.get("data")

    def close(self) -> None:
        if self._sock is not None:
            self._sock.close(linger=0)
        self._ctx.term()


def build_yam_client() -> YamClient:
    """A fresh YamClient with its own sockets, from the active tiptop config."""
    cfg = tiptop_cfg()
    robot = cfg.robot
    return YamClient(
        host=robot.get("host", "127.0.0.1"),
        control_port=int(robot.get("port", DEFAULT_CONTROL_PORT)),
        gripper_port=int(robot.get("gripper_port", DEFAULT_GRIPPER_PORT)),
        state_port=int(robot.get("state_port", DEFAULT_STATE_PORT)),
    )

"""Bimanual YAM hardware support for TiPToP.

cuTAMP plans a single kinematic chain, so the ACTIVE ARM is part of the embodiment: ``robot.type``
is ``bimanual_yam_left`` or ``bimanual_yam_right``, and the arm that is not planning is locked in
the cuRobo config at :data:`NEUTRAL_Q` and collision-checked as an obstacle (see
``cutamp/robots/assets/bimanual_yam_left.yml``'s ``lock_joints``).

A sequential-bimanual rollout therefore switches ``robot.type`` between the two plans of one
episode. :func:`active_arm` is that switch — it is the single source of truth that
:func:`~tiptop.workspace.workspace_cuboids`, :func:`~tiptop.utils.get_robot_rerun`,
:func:`~tiptop.planning.build_tamp_config` and :class:`~tiptop.yam.yam_client.YamClient` all read.
"""

import numpy as np

ARMS = ("left", "right")

# Elbow-up rest posture. MUST match cutamp.robots.bimanual_yam.bimanual_yam_neutral_joint_positions,
# the `lock_joints` values in both cuRobo configs, and yam_arm_server.NEUTRAL_Q. When one arm plans,
# the other is collision-checked at exactly this configuration, so the parked arm must physically be
# here or the planner's collision model does not describe the robot.
NEUTRAL_Q = (0.0, np.pi / 4, np.pi / 2, 0.0, 0.0, 0.0)

# How far an idle arm may sit from NEUTRAL_Q before the other arm's plans are untrustworthy. The
# planning arm's cuRobo config collision-checks it at exactly that posture, so a parked arm that has
# drifted makes every plan a statement about a robot that is not there.
IDLE_ARM_TOLERANCE_RAD = 0.10

ARM_DOF = 6  # revolute joints per arm; the gripper is tracked separately
BIMANUAL_ARM_DOF = 2 * ARM_DOF  # the 12 joint columns of a bimanual robot_state.npz


def yam_robot_type(arm: str) -> str:
    """``"left"`` -> ``"bimanual_yam_left"``."""
    if arm not in ARMS:
        raise ValueError(f"arm must be one of {ARMS}, got {arm!r}")
    return f"bimanual_yam_{arm}"


def arm_of(robot_type: str) -> str:
    """``"bimanual_yam_left"`` -> ``"left"``. Raises for a non-YAM robot type."""
    if not robot_type.startswith("bimanual_yam_"):
        raise ValueError(f"{robot_type!r} is not a bimanual YAM robot type")
    arm = robot_type.rsplit("_", 1)[1]
    if arm not in ARMS:
        raise ValueError(f"{robot_type!r} does not name a known arm; expected one of {ARMS}")
    return arm


def other_arm(arm: str) -> str:
    return "right" if arm == "left" else "left"


def is_yam(robot_type: str) -> bool:
    return robot_type.startswith("bimanual_yam_")


def active_arm(arm: str):
    """Make ``arm`` the planning and execution arm for the duration of the block.

    Thin wrapper over :func:`tiptop.config.as_robot_type` — every consumer of ``robot.type`` follows,
    including :class:`~tiptop.yam.yam_client.YamClient`, which reads it to decide which arm to drive.
    """
    from tiptop.config import as_robot_type

    return as_robot_type(yam_robot_type(arm))


def step_arms(step: dict) -> list[str]:
    """Which hand(s) a gripper step names — raw cuTAMP dual-path step OR a serialized plan step.

    ``arms`` (plural, from ``motion_solver.py::gripper_step`` / ``tiptop.planning.serialize_plan``)
    is the source of truth when present; ``arm`` (singular, set whenever exactly one hand acts) is a
    fallback for a step that somehow carries only that. No name at all defaults to BOTH arms —
    conservative for a physical gripper action, since actuating an arm that did not need it only
    closes an already-open (or opens an already-closed) gripper, not something a defensive skip
    would prevent. Shared by :mod:`tiptop.execute_plan` (dual execution dispatch) and
    :mod:`tiptop.yam.capture` (per-arm gripper channels when flattening a dual plan).
    """
    arms = step.get("arms")
    if arms:
        return list(arms)
    arm = step.get("arm")
    return [arm] if arm else list(ARMS)

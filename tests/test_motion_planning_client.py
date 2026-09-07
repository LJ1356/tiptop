"""``go_to_q`` must not close the shared robot client.

``get_robot_client()`` is a process-wide cached singleton (``utils.get_bamboo_client`` is ``@cache``d)
that a warm ``tiptop-run`` session holds for its whole life. ``go_to_q`` used to ``client.close()``
after every motion, which terminated the shared ZMQ context. Only the CONTROL path recovers from
that -- a control send raises, ``_recreate_control_socket`` finds the terminated context refuses new
sockets, and rebuilds context + gripper socket -- while ``_send_robotiq_command`` has no such
recovery. So any gripper call landing between the motion and the next control call raised
``ZMQError(ENOTSOCK)``, which is why "return home, then open the gripper" silently failed at the
start of every episode and hard-failed the scene reset.
"""

import numpy as np
import pytest
import torch

from tiptop import motion_planning
from tiptop.config import tiptop_cfg


class _StubClient:
    """Stands in for BambooFrankaClient. Records whether anyone closed it."""

    def __init__(self, q):
        self.q = list(q)
        self.closed = False
        self.executed = 0

    def get_joint_positions(self):
        return self.q

    def execute_joint_impedance_path(self, joint_confs, joint_vels, durations):
        self.executed += 1
        return {"success": True}

    def close(self):
        self.closed = True


class _StubPlan:
    def __init__(self, dof, n=5):
        self.position = torch.zeros((n, dof))
        self.velocity = torch.zeros((n, dof))


class _StubResult:
    def __init__(self, dof):
        self.success = torch.tensor([True])
        self.interpolated_plan = _StubPlan(dof)
        self.interpolation_dt = 0.02
        self.status = None


class _StubMotionGen:
    def __init__(self, dof):
        self.dof = dof

    def plan_single_js(self, start, goal, plan_config):
        return _StubResult(self.dof)


@pytest.fixture
def dof():
    return int(tiptop_cfg().robot.dof)


def test_go_to_q_leaves_the_shared_client_open(monkeypatch, dof):
    client = _StubClient([0.0] * dof)
    monkeypatch.setattr(motion_planning, "get_robot_client", lambda: client)

    # Far enough from the start that the distance check does not short-circuit.
    target = np.full(dof, 0.5, dtype=np.float64)
    motion_planning.go_to_q(target, time_dilation_factor=1.0, motion_gen=_StubMotionGen(dof))

    assert client.executed == 1, "the trajectory should still be executed"
    assert not client.closed, (
        "go_to_q closed the shared client: that terminates the process-wide ZMQ context and leaves "
        "the gripper socket dead until some control call happens to rebuild it"
    )


def test_go_to_q_is_a_noop_when_already_at_the_target(monkeypatch, dof):
    client = _StubClient([0.0] * dof)
    monkeypatch.setattr(motion_planning, "get_robot_client", lambda: client)

    motion_planning.go_to_q(np.zeros(dof), time_dilation_factor=1.0, motion_gen=_StubMotionGen(dof))

    assert client.executed == 0 and not client.closed

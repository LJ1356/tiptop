from dataclasses import replace
from functools import cache

from curobo.geom.types import Cuboid
from curobo.types.base import TensorDeviceType
from cutamp.envs.utils import unit_quat
from cutamp.utils.rerun_utils import log_curobo_mesh_to_rerun

from tiptop.config import tiptop_cfg
from tiptop.utils import YAM_ROBOT_TYPES, get_robot_rerun

tensor_args = TensorDeviceType()

# The workspace to motion plan against for collisions


def fr3_workspace() -> tuple[Cuboid, ...]:
    # PATCH: minimal workspace — single far-away ceiling so cuRobo has at least one obstacle
    # to satisfy its sphere-collision init (it errors on empty obstacles).
    ceiling = Cuboid(
        "ceiling",
        dims=[3.0, 3.0, 0.05],
        pose=[0.0, 0.0, 1.50, *unit_quat],
        color=[200, 200, 200],
    )
    return (ceiling,)


def bimanual_yam_workspace() -> tuple[Cuboid, ...]:
    # The table and the objects come from perception, so there is no fixed furniture to model here.
    # Like fr3_workspace this is a single far-away ceiling, present because cuRobo's sphere-collision
    # init errors on an empty obstacle set. It sits high enough to clear the base column.
    #
    # If the real cell has structure perception cannot see -- a rig above the table, a wall the arm
    # could reach behind -- add it here: this is the only place a YAM plan learns about geometry that
    # is not in the point cloud.
    ceiling = Cuboid("ceiling", dims=[3.0, 3.0, 0.05], pose=[0.0, 0.0, 1.75, *unit_quat], color=[200, 200, 200])
    return (ceiling,)


def ur5_workspace() -> tuple[Cuboid, ...]:
    pedestal_table = Cuboid(
        "pedestal_table",
        dims=[0.48, 0.87, 0.02],
        pose=[-0.05, 0, -0.01, *unit_quat],
        color=[222, 184, 135],
    )
    # ceiling is more for avoiding crazy large motions
    ceiling = Cuboid("ceiling", dims=[1.5, 2.0, 0.01], pose=[0.5, 0.0, 1.0, *unit_quat], color=[255, 255, 255])

    # humans work to the left and right of the UR5...
    wall_left = Cuboid("wall_left", dims=[1.5, 0.1, 1.0], pose=[0.0, 0.8, 0.2, *unit_quat], color=[225, 225, 225])
    wall_right = Cuboid("wall_right", dims=[1.5, 0.1, 1.0], pose=[0.0, -0.8, 0.2, *unit_quat], color=[225, 225, 225])
    obstacles = (pedestal_table, ceiling, wall_left, wall_right)
    return obstacles


@cache
def workspace_cuboids() -> tuple[Cuboid, ...]:
    """Return workspace cuboids for the configured robot, with names prefixed by 'workspace_' to avoid collisions with objects discovered by actual perception."""
    cfg = tiptop_cfg()
    if cfg.robot.type == "fr3_robotiq":
        cuboids = fr3_workspace()
    elif cfg.robot.type == "panda":
        cuboids = fr3_workspace()
    elif cfg.robot.type == "fr3":
        cuboids = fr3_workspace()
    elif cfg.robot.type == "panda_robotiq":
        cuboids = fr3_workspace()
    elif cfg.robot.type == "ur5":
        cuboids = ur5_workspace()
    elif cfg.robot.type in YAM_ROBOT_TYPES:
        cuboids = bimanual_yam_workspace()
    else:
        raise ValueError(f"Unknown robot type: {cfg.robot.type}")
    return tuple(replace(c, name=f"workspace_{c.name}") for c in cuboids)


if __name__ == "__main__":
    import rerun as rr

    rr.init("robot_workspace", spawn=True)
    get_robot_rerun()
    for obj in workspace_cuboids():
        log_curobo_mesh_to_rerun(f"world/{obj.name}", obj.get_mesh(), static_transform=True)

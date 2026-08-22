import numpy as np
import pytest
import quaternion as qt
from motion_stack.utils.joint_state import JState
from motion_stack.utils.pose import Pose
from motion_stack.utils.time import Time

from gogo_operator.operator import OperatorController
from gogo_operator.state import RobotRegistry
from gogo_operator.tui import OperatorTUI


@pytest.mark.asyncio
async def test_all_tui_screens_build_from_registry_state() -> None:
    registry = RobotRegistry()
    registry.observe_joints("leg1", {"joint1_1": JState("joint1_1", position=0.25)})
    registry.observe_ik("leg1", Pose(Time(0), np.zeros(3), qt.one))
    controller = OperatorController(registry)
    controller.select_robot("leg1", True)
    tui = OperatorTUI(controller, "test")
    snapshot = registry.snapshot()

    tui._build_main(snapshot)
    tui._build_monitor(snapshot)
    tui._build_robot_select(snapshot)
    tui._build_joint_select(snapshot)
    tui._build_wheel_select(snapshot)
    tui._build_ik_select(snapshot)

    assert len(tui.body) > 0

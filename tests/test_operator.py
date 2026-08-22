import pytest
from motion_stack.utils.joint_state import JState

from gogo_operator.operator import OperatorController
from gogo_operator.state import RobotRegistry


@pytest.mark.asyncio
async def test_joint_and_wheel_selections_are_mutually_exclusive() -> None:
    registry = RobotRegistry()
    registry.observe_joints("leg1", {"joint1_1": JState("joint1_1", position=0.0)})
    controller = OperatorController(registry)
    controller.select_robot("leg1", True)
    reference = ("leg1", "joint1_1")

    controller.set_joint_selection("joint", reference, True)
    controller.set_joint_selection("wheel", reference, "inverted")

    assert reference not in controller.selected_joints
    assert reference in controller.inverted_wheels


@pytest.mark.asyncio
async def test_conflicting_duplicate_joint_directions_are_skipped() -> None:
    registry = RobotRegistry()
    for namespace in ("left", "right"):
        registry.observe_joints(namespace, {"wheel": JState("wheel", position=0.0)})
    controller = OperatorController(registry)
    controller.selected_robots.update({"left", "right"})

    target = controller._joint_target(
        {("left", "wheel")}, {("right", "wheel")}, 0.2, -0.2
    )

    assert target == {}
    assert "conflicting directions" in controller.logs[-1]

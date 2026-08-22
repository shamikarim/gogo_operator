import asyncio

import pytest
from motion_stack.utils.joint_state import JState

from gogo_operator.state import RobotRegistry, display_namespace


@pytest.mark.asyncio
async def test_registry_tracks_namespaced_joint_samples() -> None:
    registry = RobotRegistry()
    limb_id = registry.discover("/leg1/")
    registry.observe_joints("leg1", {"joint1_1": JState("joint1_1", position=0.25)})
    await asyncio.sleep(0)

    snapshot = registry.snapshot()
    robot = snapshot.get("leg1")
    assert robot is not None
    assert robot.limb_id == limb_id
    assert robot.joints[0].name == "joint1_1"
    assert robot.joints[0].position == 0.25
    assert display_namespace(robot.namespace) == "/leg1"


@pytest.mark.asyncio
async def test_registry_updates_are_exposed_as_a_base_sub() -> None:
    registry = RobotRegistry()
    update = registry.updates.wait_for_new()
    registry.discover("")

    snapshot = await asyncio.wait_for(update, timeout=1)
    assert snapshot.robots[0].namespace == ""
    assert display_namespace("") == "/"

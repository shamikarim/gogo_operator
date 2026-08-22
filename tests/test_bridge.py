import asyncio

import pytest
from asyncio_for_robotics import BaseSub
from motion_stack.lvl1.core import JStateBatch
from motion_stack.utils.joint_state import JState
from motion_stack.utils.pose import Pose
from motion_stack.utils.pose_state import MultiPose

from gogo_operator.bridge import DynamicJointBridge
from gogo_operator.state import RobotRegistry


class FakeHook:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class FakeJointPublisher(FakeHook):
    def __init__(self) -> None:
        super().__init__()
        self.filter: set[str] = set()


class FakeBridge(DynamicJointBridge):
    def __init__(
        self,
        registry: RobotRegistry,
        sensor_output: BaseSub[JStateBatch],
        command_input: BaseSub[JStateBatch],
        ik_sensor_output: BaseSub[MultiPose],
        ik_command_input: BaseSub[MultiPose],
    ) -> None:
        super().__init__(
            registry,
            sensor_output,
            command_input,
            ik_sensor_output,
            ik_command_input,
        )
        self.feeds: dict[str, BaseSub[JStateBatch]] = {}
        self.publishers: dict[str, FakeJointPublisher] = {}

    def start_discovery(self) -> None:
        pass

    def stop_discovery(self) -> None:
        pass

    def _make_joint_hooks(
        self,
        namespace: str,
        sensor_output: BaseSub[JStateBatch],
        command_input: BaseSub[JStateBatch],
    ):
        publisher = FakeJointPublisher()
        self.feeds[namespace] = sensor_output
        self.publishers[namespace] = publisher
        return FakeHook(), publisher

    def _make_pose_hooks(
        self,
        namespace: str,
        sensor_output: BaseSub[Pose],
        command_input: BaseSub[Pose],
    ):
        return FakeHook(), FakeHook()


@pytest.mark.asyncio
async def test_bridge_forwards_samples_and_updates_namespace_filter() -> None:
    registry = RobotRegistry()
    sensor_output: BaseSub[JStateBatch] = BaseSub()
    command_input: BaseSub[JStateBatch] = BaseSub()
    ik_sensor_output: BaseSub[MultiPose] = BaseSub()
    ik_command_input: BaseSub[MultiPose] = BaseSub()
    bridge = FakeBridge(
        registry,
        sensor_output,
        command_input,
        ik_sensor_output,
        ik_command_input,
    )
    bridge.add_robot("leg1")

    forwarded = sensor_output.wait_for_new()
    bridge.feeds["leg1"].input_data({"joint1_1": JState("joint1_1", position=0.5)})
    states = await asyncio.wait_for(forwarded, timeout=1)

    assert states["joint1_1"].position == 0.5
    assert registry.joint_names("leg1") == {"joint1_1"}
    assert bridge.publishers["leg1"].filter == {"joint1_1"}
    bridge.close()
    assert registry.namespaces == ()

import asyncio
from abc import ABC, abstractmethod
from contextlib import suppress
from dataclasses import dataclass
from functools import partial
from typing import Any, Optional, Protocol, Tuple

from asyncio_for_robotics import BaseSub
from motion_stack.lvl1.core import JStateBatch
from motion_stack.utils.pose import Pose
from motion_stack.utils.pose_state import MultiPose

from .state import RobotRegistry, normalize_namespace


class _Closeable(Protocol):
    def close(self) -> None: ...


class _JointPublisher(_Closeable, Protocol):
    @property
    def filter(self) -> set[str]: ...

    @filter.setter
    def filter(self, value: set[str]) -> None: ...


@dataclass
class _RobotConnection:
    limb_id: int
    joint_feed: BaseSub[JStateBatch]
    joint_subscriber: _Closeable
    joint_publisher: _JointPublisher
    joint_filter: set[str]
    pose_feed: BaseSub[Pose]
    pose_subscriber: _Closeable
    pose_command: BaseSub[Pose]
    pose_publisher: _Closeable


class DynamicJointBridge(ABC):
    """Shared robot discovery and BaseSub wiring for transport backends."""

    _EMPTY_FILTER = {"__gogo_operator_no_joint__"}

    def __init__(
        self,
        registry: RobotRegistry,
        sensor_output: BaseSub[JStateBatch],
        command_input: BaseSub[JStateBatch],
        ik_sensor_output: BaseSub[MultiPose],
        ik_command_input: BaseSub[MultiPose],
        *,
        timeout: float = 5.0,
        expire_stale: bool = True,
    ) -> None:
        self.registry = registry
        self.sensor_output = sensor_output
        self.command_input = command_input
        self.ik_sensor_output = ik_sensor_output
        self.ik_command_input = ik_command_input
        self.timeout = timeout
        self.expire_stale = expire_stale
        self._connections: dict[str, _RobotConnection] = {}
        self._started = False
        self._closed = False

        self.ik_command_input.asap_callback.append(self._dispatch_ik_commands)

    @property
    def namespaces(self) -> Tuple[str, ...]:
        return tuple(sorted(self._connections))

    def add_robot(self, namespace: str) -> None:
        namespace = normalize_namespace(namespace)
        if namespace in self._connections or self._closed:
            return

        limb_id = self.registry.discover(namespace)
        joint_feed: BaseSub[JStateBatch] = BaseSub(scope=None)
        joint_callback = partial(self._forward_joints, namespace)
        joint_feed.asap_callback.append(joint_callback)

        pose_feed: BaseSub[Pose] = BaseSub(scope=None)
        pose_callback = partial(self._forward_pose, namespace, limb_id)
        pose_feed.asap_callback.append(pose_callback)
        pose_command: BaseSub[Pose] = BaseSub(scope=None)

        created_hooks: list[_Closeable] = []
        try:
            joint_subscriber, joint_publisher = self._make_joint_hooks(
                namespace, joint_feed, self.command_input
            )
            created_hooks.extend([joint_subscriber, joint_publisher])
            pose_subscriber, pose_publisher = self._make_pose_hooks(
                namespace, pose_feed, pose_command
            )
            created_hooks.extend([pose_subscriber, pose_publisher])
        except Exception:
            for hook in reversed(created_hooks):
                with suppress(Exception):
                    hook.close()
            with suppress(ValueError):
                joint_feed.asap_callback.remove(joint_callback)
            with suppress(ValueError):
                pose_feed.asap_callback.remove(pose_callback)
            joint_feed.close()
            pose_feed.close()
            pose_command.close()
            self.registry.remove(namespace)
            raise

        joint_filter = set(self._EMPTY_FILTER)
        joint_publisher.filter = joint_filter
        self._connections[namespace] = _RobotConnection(
            limb_id=limb_id,
            joint_feed=joint_feed,
            joint_subscriber=joint_subscriber,
            joint_publisher=joint_publisher,
            joint_filter=joint_filter,
            pose_feed=pose_feed,
            pose_subscriber=pose_subscriber,
            pose_command=pose_command,
            pose_publisher=pose_publisher,
        )

    def remove_robot(self, namespace: str) -> None:
        namespace = normalize_namespace(namespace)
        connection = self._connections.pop(namespace, None)
        if connection is None:
            return

        resources = [
            connection.joint_subscriber,
            connection.joint_publisher,
            connection.pose_subscriber,
            connection.pose_publisher,
            connection.joint_feed,
            connection.pose_feed,
            connection.pose_command,
        ]
        for resource in resources:
            with suppress(Exception):
                resource.close()
        self.registry.remove(namespace)

    async def run(self) -> None:
        self.start_discovery()
        self._started = True
        try:
            while True:
                await asyncio.sleep(max(0.1, min(1.0, self.timeout / 2)))
                if self.expire_stale:
                    self._remove_stale_robots()
        finally:
            self.close()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True

        with suppress(ValueError):
            self.ik_command_input.asap_callback.remove(self._dispatch_ik_commands)
        self.stop_discovery()
        for namespace in tuple(self._connections):
            self.remove_robot(namespace)

    def _forward_joints(self, namespace: str, states: JStateBatch) -> None:
        self.registry.observe_joints(namespace, states)
        connection = self._connections.get(namespace)
        if connection is not None:
            connection.joint_filter.clear()
            connection.joint_filter.update(self.registry.joint_names(namespace))
        self.sensor_output.input_data(states)

    def _forward_pose(self, namespace: str, limb_id: int, pose: Pose) -> None:
        self.registry.observe_ik(namespace, pose)
        self.ik_sensor_output.input_data({limb_id: pose})

    def _dispatch_ik_commands(self, poses: MultiPose) -> None:
        by_limb = {
            connection.limb_id: connection for connection in self._connections.values()
        }
        for limb_id, pose in poses.items():
            connection = by_limb.get(limb_id)
            if connection is not None:
                connection.pose_command.input_data(pose)

    def _remove_stale_robots(self) -> None:
        now = asyncio.get_running_loop().time()
        stale = []
        for namespace in self._connections:
            last_seen = self.registry.last_seen(namespace)
            if last_seen is not None and now - last_seen > self.timeout:
                stale.append(namespace)
        for namespace in stale:
            self.remove_robot(namespace)

    @abstractmethod
    def start_discovery(self) -> None:
        pass

    @abstractmethod
    def stop_discovery(self) -> None:
        pass

    @abstractmethod
    def _make_joint_hooks(
        self,
        namespace: str,
        sensor_output: BaseSub[JStateBatch],
        command_input: BaseSub[JStateBatch],
    ) -> tuple[_Closeable, _JointPublisher]:
        pass

    @abstractmethod
    def _make_pose_hooks(
        self,
        namespace: str,
        sensor_output: BaseSub[Pose],
        command_input: BaseSub[Pose],
    ) -> tuple[_Closeable, _Closeable]:
        pass

import asyncio
from contextlib import suppress
from typing import Any, Optional

from asyncio_for_robotics import BaseSub
from motion_stack.lvl1.core import JStateBatch
from motion_stack.utils.pose import Pose
from motion_stack.utils.pose_state import MultiPose

from .bridge import DynamicJointBridge, _Closeable, _JointPublisher
from .state import RobotRegistry


class _PyzerosJointPublisher:
    """Own both the Motion Stack callback hook and its pyzeros publisher."""

    def __init__(self, hook: Any) -> None:
        self._hook = hook

    @property
    def filter(self) -> set[str]:
        return self._hook.filter

    @filter.setter
    def filter(self, value: set[str]) -> None:
        self._hook.filter = value

    def close(self) -> None:
        try:
            self._hook.close()
        finally:
            self._hook.pyz_pub.close()


def _topic(namespace: str, name: str, *, leading_slash: bool = False) -> str:
    namespace = namespace.strip("/")
    name = name.strip("/")
    value = f"{namespace}/{name}" if namespace else name
    return f"/{value}" if leading_slash else value


class ZenohBackend(DynamicJointBridge):
    """Native Zenoh discovery and Motion Stack 2 bridge hooks."""

    def __init__(
        self,
        registry: RobotRegistry,
        sensor_output: BaseSub[JStateBatch],
        command_input: BaseSub[JStateBatch],
        ik_sensor_output: BaseSub[MultiPose],
        ik_command_input: BaseSub[MultiPose],
        *,
        timeout: float = 5.0,
        session: Optional[Any] = None,
    ) -> None:
        super().__init__(
            registry,
            sensor_output,
            command_input,
            ik_sensor_output,
            ik_command_input,
            timeout=timeout,
            expire_stale=True,
        )
        self.session = session
        self._discovery_sub = None

    def start_discovery(self) -> None:
        import asyncio_for_robotics.zenoh as afor_zenoh

        self._discovery_sub = afor_zenoh.Sub(
            "**/joint_read", session=self.session, scope=None
        )
        self._discovery_sub.asap_callback.append(self._sample_received)

    def stop_discovery(self) -> None:
        if self._discovery_sub is None:
            return
        with suppress(ValueError):
            self._discovery_sub.asap_callback.remove(self._sample_received)
        self._discovery_sub.close()
        self._discovery_sub = None

    def _sample_received(self, sample: Any) -> None:
        key = str(sample.key_expr).strip("/")
        if "/" in key:
            namespace, topic = key.rsplit("/", 1)
        else:
            namespace, topic = "", key
        if topic == "joint_read":
            self.add_robot(namespace)

    def _make_joint_hooks(
        self,
        namespace: str,
        sensor_output: BaseSub[JStateBatch],
        command_input: BaseSub[JStateBatch],
    ) -> tuple[_Closeable, _JointPublisher]:
        from ms_zenoh_bridge.lvl1_zenoh import PublisherHookJSB, SubscriberHookJSB

        subscriber = SubscriberHookJSB(
            sensor_output,
            _topic(namespace, "joint_read"),
            session=self.session,
            scope=None,
        )
        publisher = PublisherHookJSB(
            command_input,
            _topic(namespace, "joint_set"),
            session=self.session,
            scope=None,
        )
        return subscriber, publisher

    def _make_pose_hooks(
        self,
        namespace: str,
        sensor_output: BaseSub[Pose],
        command_input: BaseSub[Pose],
    ) -> tuple[_Closeable, _Closeable]:
        from ms_zenoh_bridge.lvl2.lvl2_zenoh import (
            PublisherHookPose,
            SubscriberHookPose,
        )

        subscriber = SubscriberHookPose(
            sensor_output,
            _topic(namespace, "tip_pos"),
            session=self.session,
            scope=None,
        )
        publisher = PublisherHookPose(
            command_input,
            _topic(namespace, "set_ik_target"),
            session=self.session,
            scope=None,
        )
        return subscriber, publisher


class PyzerosBackend(DynamicJointBridge):
    """ROS 2-compatible pyzeros backend using Zenoh liveliness discovery."""

    def __init__(
        self,
        registry: RobotRegistry,
        sensor_output: BaseSub[JStateBatch],
        command_input: BaseSub[JStateBatch],
        ik_sensor_output: BaseSub[MultiPose],
        ik_command_input: BaseSub[MultiPose],
        *,
        timeout: float = 5.0,
        session: Optional[Any] = None,
    ) -> None:
        super().__init__(
            registry,
            sensor_output,
            command_input,
            ik_sensor_output,
            ik_command_input,
            timeout=timeout,
            expire_stale=False,
        )
        self.session = session
        self._token_watcher = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def start_discovery(self) -> None:
        import asyncio_for_robotics.zenoh as afor_zenoh

        self._loop = asyncio.get_running_loop()
        self._token_watcher = (
            afor_zenoh.auto_session()
            .liveliness()
            .declare_subscriber(
                "@ros2_lv/**/MP/**/sensor_msgs::msg::dds_::JointState_/**",
                self._token_received_threadsafe,
                history=True,
            )
        )

    def stop_discovery(self) -> None:
        if self._token_watcher is None:
            return
        self._token_watcher.undeclare()
        self._token_watcher = None

    def _token_received_threadsafe(self, sample: Any) -> None:
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._token_received, sample)

    def _token_received(self, sample: Any) -> None:
        import zenoh

        namespace = self._namespace_from_liveliness(str(sample.key_expr))
        if namespace is None:
            return
        if sample.kind == zenoh.SampleKind.PUT:
            self.add_robot(namespace)
        elif sample.kind == zenoh.SampleKind.DELETE:
            self.remove_robot(namespace)

    @staticmethod
    def _namespace_from_liveliness(key: str) -> Optional[str]:
        if "%joint_read/" not in key:
            return None
        for segment in key.split("/"):
            if segment.endswith("%joint_read"):
                return segment.removesuffix("%joint_read").replace("%", "/").strip("/")
        return None

    def _make_joint_hooks(
        self,
        namespace: str,
        sensor_output: BaseSub[JStateBatch],
        command_input: BaseSub[JStateBatch],
    ) -> tuple[_Closeable, _JointPublisher]:
        from ms_pyzeros_bridge.lvl1_pyzeros import PublisherHookJSB, SubscriberHookJSB

        subscriber = SubscriberHookJSB(
            sensor_output,
            _topic(namespace, "joint_read", leading_slash=True),
            session=self.session,
            scope=None,
        )
        publisher = PublisherHookJSB(
            command_input,
            _topic(namespace, "joint_set", leading_slash=True),
            session=self.session,
            scope=None,
        )
        return subscriber, _PyzerosJointPublisher(publisher)

    def _make_pose_hooks(
        self,
        namespace: str,
        sensor_output: BaseSub[Pose],
        command_input: BaseSub[Pose],
    ) -> tuple[_Closeable, _Closeable]:
        from ms_pyzeros_bridge.lvl2.lvl2_pyzeros import (
            PublisherHookPose,
            SubscriberHookPose,
        )

        subscriber = SubscriberHookPose(
            sensor_output,
            _topic(namespace, "tip_pos", leading_slash=True),
            session=self.session,
            scope=None,
        )
        publisher = PublisherHookPose(
            command_input,
            _topic(namespace, "set_ik_target", leading_slash=True),
            session=self.session,
            scope=None,
        )
        return subscriber, publisher

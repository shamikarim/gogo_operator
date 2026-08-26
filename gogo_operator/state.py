import re
from dataclasses import dataclass
from time import monotonic
from typing import Dict, Optional, Tuple

from asyncio_for_robotics import BaseSub
from motion_stack.lvl1.core import JStateBatch
from motion_stack.utils.joint_state import JState
from motion_stack.utils.pose import Pose

_NUMBER_PART = re.compile(r"(\d+)")


def _natural_sort_key(value: str) -> tuple[tuple[int, int, str], ...]:
    parts = []
    for part in _NUMBER_PART.split(value):
        if part.isdigit():
            number = str(int(part))
            parts.append((1, len(number), number))
        else:
            parts.append((0, 0, part.casefold()))
    return tuple(parts)


def normalize_namespace(namespace: str) -> str:
    """Return the canonical form used for robot namespaces."""
    return namespace.strip("/")


def display_namespace(namespace: str) -> str:
    """Return a readable namespace, including the root namespace."""
    namespace = normalize_namespace(namespace)
    return f"/{namespace}" if namespace else "/"


@dataclass(frozen=True)
class JointSnapshot:
    name: str
    position: Optional[float]
    velocity: Optional[float]
    effort: Optional[float]
    age: float


@dataclass(frozen=True)
class RobotSnapshot:
    namespace: str
    limb_id: int
    joints: Tuple[JointSnapshot, ...]
    sample_age: Optional[float]
    ik_ready: bool
    ik_sample_age: Optional[float]


@dataclass(frozen=True)
class FleetSnapshot:
    robots: Tuple[RobotSnapshot, ...]

    def get(self, namespace: str) -> Optional[RobotSnapshot]:
        namespace = normalize_namespace(namespace)
        return next(
            (robot for robot in self.robots if robot.namespace == namespace), None
        )


@dataclass
class _RobotState:
    namespace: str
    limb_id: int
    joints: Dict[str, JState]
    joint_last_seen: Dict[str, float]
    last_seen: Optional[float] = None
    ik_pose: Optional[Pose] = None
    ik_last_seen: Optional[float] = None


class RobotRegistry:
    """Transport-independent live view of robots and their sensor data."""

    def __init__(self) -> None:
        self.updates: BaseSub[FleetSnapshot] = BaseSub()
        self._robots: Dict[str, _RobotState] = {}
        self._next_limb_id = 1
        self.structural_version = 0

    @property
    def namespaces(self) -> Tuple[str, ...]:
        return tuple(sorted(self._robots))

    def discover(self, namespace: str) -> int:
        namespace = normalize_namespace(namespace)
        current = self._robots.get(namespace)
        if current is not None:
            return current.limb_id

        limb_id = self._next_limb_id
        self._next_limb_id += 1
        self._robots[namespace] = _RobotState(
            namespace, limb_id, {}, {}, last_seen=monotonic()
        )
        self.structural_version += 1
        self._publish()
        return limb_id

    def remove(self, namespace: str) -> bool:
        namespace = normalize_namespace(namespace)
        if self._robots.pop(namespace, None) is None:
            return False
        self.structural_version += 1
        self._publish()
        return True

    def observe_joints(self, namespace: str, states: JStateBatch) -> None:
        namespace = normalize_namespace(namespace)
        if namespace not in self._robots:
            self.discover(namespace)

        robot = self._robots[namespace]
        now = monotonic()
        previous_names = set(robot.joints)
        for name, state in states.items():
            robot.joints[name] = state.copy()
            robot.joint_last_seen[name] = now
        robot.last_seen = now

        if set(robot.joints) != previous_names:
            self.structural_version += 1
        self._publish()

    def observe_ik(self, namespace: str, pose: Pose) -> None:
        namespace = normalize_namespace(namespace)
        if namespace not in self._robots:
            self.discover(namespace)

        robot = self._robots[namespace]
        was_ready = robot.ik_pose is not None
        robot.ik_pose = pose.copy()
        robot.ik_last_seen = monotonic()
        if not was_ready:
            self.structural_version += 1
        self._publish()

    def limb_id(self, namespace: str) -> Optional[int]:
        robot = self._robots.get(normalize_namespace(namespace))
        return None if robot is None else robot.limb_id

    def joint_names(self, namespace: str) -> frozenset[str]:
        robot = self._robots.get(normalize_namespace(namespace))
        return frozenset() if robot is None else frozenset(robot.joints)

    def last_seen(self, namespace: str) -> Optional[float]:
        robot = self._robots.get(normalize_namespace(namespace))
        return None if robot is None else robot.last_seen

    def snapshot(self) -> FleetSnapshot:
        now = monotonic()
        robots = []
        for robot in sorted(self._robots.values(), key=lambda item: item.namespace):
            joints = tuple(
                JointSnapshot(
                    name=name,
                    position=state.position,
                    velocity=state.velocity,
                    effort=state.effort,
                    age=max(0.0, now - robot.joint_last_seen[name]),
                )
                for name, state in sorted(
                    robot.joints.items(), key=lambda item: _natural_sort_key(item[0])
                )
            )
            robots.append(
                RobotSnapshot(
                    namespace=robot.namespace,
                    limb_id=robot.limb_id,
                    joints=joints,
                    sample_age=(
                        None
                        if robot.last_seen is None
                        else max(0.0, now - robot.last_seen)
                    ),
                    ik_ready=robot.ik_pose is not None,
                    ik_sample_age=(
                        None
                        if robot.ik_last_seen is None
                        else max(0.0, now - robot.ik_last_seen)
                    ),
                )
            )
        return FleetSnapshot(tuple(robots))

    def _publish(self) -> None:
        self.updates.input_data(self.snapshot())

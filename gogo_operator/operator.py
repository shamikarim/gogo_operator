import asyncio
import time
from collections import deque
from contextlib import suppress
from typing import Iterable, Literal, Optional, Tuple, cast

import numpy as np
import quaternion as qt
from asyncio_for_robotics import BaseSub
from gogo_keyboard import codes
from gogo_keyboard.keyboard import Key, KeySub
from motion_stack.lvl1.core import JStateBatch
from motion_stack.lvl1.joint_api import AsyncJointSyncer, make_delta_time
from motion_stack.lvl2.ik_api import AsyncIkSyncer
from motion_stack.utils.math import Flo3
from motion_stack.utils.pose import VelPose
from motion_stack.utils.pose_state import MultiPose
from motion_stack.utils.time import Time

from .state import RobotRegistry, normalize_namespace

SelectionState = Literal[False, True, "inverted"]
JointRef = Tuple[str, str]


class OperatorController:
    """Transport-free Motion Stack 2 operator state and controls."""

    def __init__(self, registry: RobotRegistry) -> None:
        self.registry = registry
        self.sensor_input: BaseSub[JStateBatch] = BaseSub()
        self.command_output: BaseSub[JStateBatch] = BaseSub()
        self.ik_sensor_input: BaseSub[MultiPose] = BaseSub()
        self.ik_command_output: BaseSub[MultiPose] = BaseSub()

        self.joint_syncer = AsyncJointSyncer(interpolation_delta=np.deg2rad(7))
        self.wheel_syncer = AsyncJointSyncer(interpolation_delta=np.deg2rad(15))
        self.ik_syncer = AsyncIkSyncer()

        self.sensor_input.asap_callback.extend(
            [
                self.joint_syncer.sensor_input.input_data,
                self.wheel_syncer.sensor_input.input_data,
            ]
        )
        self.joint_syncer.command_output.asap_callback.append(
            self.command_output.input_data
        )
        self.wheel_syncer.command_output.asap_callback.append(
            self.command_output.input_data
        )
        self.ik_sensor_input.asap_callback.append(
            self.ik_syncer.sensor_input.input_data
        )
        self.ik_syncer.command_output.asap_callback.append(
            self.ik_command_output.input_data
        )
        self.registry.updates.asap_callback.append(self._registry_updated)

        self.current_mode = "main"
        self.selected_robots: set[str] = set()
        self.selected_joints: set[JointRef] = set()
        self.inverted_joints: set[JointRef] = set()
        self.selected_wheels: set[JointRef] = set()
        self.inverted_wheels: set[JointRef] = set()
        self.selected_ik_robots: set[str] = set()

        self.joint_speed = 0.15
        self.wheel_speed = 0.2
        self.translation_speed = 70.0
        self.rotation_speed = float(np.deg2rad(7))
        self.ik_end_effector_frame = False

        self.logs: deque[str] = deque(maxlen=7)
        self.revision = 0
        self.stop_event = asyncio.Event()
        self._ik_key_states: set[int] = set()
        self._closed = False

    def start(self, task_group) -> None:
        task_group.create_task(self.joint_syncer.run(), name="operator_joint_syncer")
        task_group.create_task(self.wheel_syncer.run(), name="operator_wheel_syncer")
        task_group.create_task(self.ik_syncer.run(), name="operator_ik_syncer")

    def request_stop(self) -> None:
        self.stop_all()
        self.stop_event.set()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.stop_all()
        with suppress(ValueError):
            self.registry.updates.asap_callback.remove(self._registry_updated)

    def set_mode(self, mode: str) -> None:
        if mode not in {
            "main",
            "monitor",
            "robot_select",
            "joint_select",
            "wheel_select",
            "ik_select",
        }:
            raise ValueError(f"Unknown operator mode: {mode}")
        if mode != self.current_mode:
            self.stop_all()
        if mode != "ik_select":
            self._ik_key_states.clear()
        self.current_mode = mode
        if mode != "main":
            self.ensure_robot_selection()
        self.revision += 1

    def ensure_robot_selection(self) -> None:
        if not self.selected_robots:
            self.selected_robots.update(self.registry.namespaces)

    def select_robot(self, namespace: str, selected: bool) -> None:
        namespace = normalize_namespace(namespace)
        if selected and namespace in self.registry.namespaces:
            self.selected_robots.add(namespace)
        else:
            self.selected_robots.discard(namespace)
            self._discard_robot_selections(namespace)
        self.revision += 1
        self.add_log(
            "I", f"Selected robots: {self._display_names(self.selected_robots)}"
        )

    def set_joint_selection(
        self,
        kind: Literal["joint", "wheel"],
        reference: JointRef,
        state: SelectionState,
    ) -> None:
        namespace, joint = reference
        reference = (normalize_namespace(namespace), joint)
        if kind == "joint":
            direct, inverted = self.selected_joints, self.inverted_joints
            other_direct, other_inverted = self.selected_wheels, self.inverted_wheels
        else:
            direct, inverted = self.selected_wheels, self.inverted_wheels
            other_direct, other_inverted = self.selected_joints, self.inverted_joints

        direct.discard(reference)
        inverted.discard(reference)
        if state is True:
            direct.add(reference)
        elif state == "inverted":
            inverted.add(reference)
        if state is not False:
            other_direct.discard(reference)
            other_inverted.discard(reference)
        self.revision += 1

    def select_ik_robot(self, namespace: str, selected: bool) -> None:
        namespace = normalize_namespace(namespace)
        if selected and namespace in self.selected_robots:
            self.selected_ik_robots.add(namespace)
        else:
            self.selected_ik_robots.discard(namespace)
        self.revision += 1

    def move_joints(self, speed: float) -> bool:
        target = self._joint_target(
            self.selected_joints, self.inverted_joints, speed, -speed
        )
        if not target or not self._ready(self.joint_syncer, target):
            return False
        self.joint_syncer.clear()
        self.joint_syncer.speed_safe(target, make_delta_time())
        self.add_log("I", f"Joint speed command: {target}")
        return True

    def zero_joints(self) -> bool:
        references = self.selected_joints | self.inverted_joints
        target = self._joint_target(references, set(), 0.0, 0.0)
        if not target or not self._ready(self.joint_syncer, target):
            return False
        self.joint_syncer.clear()
        self.joint_syncer.lerp(target)
        self.add_log("I", f"Zeroing joints: {sorted(target)}")
        return True

    def move_wheels(self, forward: float, turn: float = 0.0) -> bool:
        target = self._joint_target(
            self.selected_wheels,
            self.inverted_wheels,
            forward + turn,
            -forward + turn,
        )
        if not target or not self._ready(self.wheel_syncer, target):
            return False
        self.wheel_syncer.clear()
        self.wheel_syncer.speed_safe(target, make_delta_time())
        self.add_log("I", f"Wheel speed command: {target}")
        return True

    def move_ik(self) -> bool:
        if not self._ik_key_states:
            self._cancel(self.ik_syncer)
            return False

        limb_ids = {
            limb_id
            for namespace in self.selected_ik_robots
            if (limb_id := self.registry.limb_id(namespace)) is not None
        }
        ready, missing = self.ik_syncer.ready(limb_ids)
        if not limb_ids or not ready:
            if missing:
                self.add_log(
                    "W", f"IK feedback missing for local limbs: {sorted(missing)}"
                )
            return False

        linear = cast(Flo3, np.zeros(3, dtype=float))
        rotation = cast(Flo3, np.zeros(3, dtype=float))
        keys = self._ik_key_states
        linear[0] = self.translation_speed * (
            int(codes.KEY_UP in keys) - int(codes.KEY_DOWN in keys)
        )
        linear[1] = self.translation_speed * (
            int(codes.KEY_LEFT in keys) - int(codes.KEY_RIGHT in keys)
        )
        linear[2] = self.translation_speed * (
            int(codes.KEY_I in keys) - int(codes.KEY_K in keys)
        )
        rotation[0] = self.rotation_speed * (
            int(codes.KEY_E in keys) - int(codes.KEY_Q in keys)
        )
        rotation[1] = self.rotation_speed * (
            int(codes.KEY_W in keys) - int(codes.KEY_S in keys)
        )
        rotation[2] = self.rotation_speed * (
            int(codes.KEY_A in keys) - int(codes.KEY_D in keys)
        )

        if not np.any(linear) and not np.any(rotation):
            self._cancel(self.ik_syncer)
            return False

        velocities = {}
        for limb_id in limb_ids:
            limb_linear = linear.copy()
            limb_rotation = rotation.copy()
            if not self.ik_end_effector_frame:
                pose = self.ik_syncer.sensor[limb_id]
                inverse = pose.quat.conjugate()
                limb_linear = cast(Flo3, qt.rotate_vectors(inverse, limb_linear))
                delta_rotation = qt.from_rotation_vector(limb_rotation)
                relative_rotation = inverse * delta_rotation * pose.quat
                limb_rotation = cast(Flo3, qt.as_rotation_vector(relative_rotation))
            velocities[limb_id] = VelPose(
                Time(time.time_ns()), limb_linear, limb_rotation
            )

        self.ik_syncer.speed_safe(velocities, make_delta_time())
        return True

    def stop_all(self) -> None:
        self._cancel(self.joint_syncer)
        self._cancel(self.wheel_syncer)
        self._cancel(self.ik_syncer)

    async def keyboard_loop(self) -> None:
        key_sub = KeySub(termination_callback=self.request_stop)
        async for key in key_sub.listen_reliable():
            self.handle_key(key)

    def handle_key(self, key: Key) -> None:
        modifier = key.modifiers & ~(codes.MODIFIER_NUM | codes.MODIFIER_CAPS)
        if key.is_pressed:
            if (
                modifier & (codes.MODIFIER_LCTRL | codes.MODIFIER_RCTRL)
                and key.code == codes.KEY_C
            ):
                self.request_stop()
                return
            if key.code == codes.KEY_ESCAPE:
                self.set_mode("main")
                return
            if key.code == codes.KEY_SPACE:
                self.stop_all()
                return

            if self.current_mode == "main":
                mode_keys = {
                    codes.KEY_R: "robot_select",
                    codes.KEY_J: "joint_select",
                    codes.KEY_W: "wheel_select",
                    codes.KEY_U: "ik_select",
                    codes.KEY_M: "monitor",
                }
                if key.code in mode_keys:
                    self.set_mode(mode_keys[key.code])
                return

            if self.current_mode == "joint_select":
                if key.code == codes.KEY_W:
                    self.move_joints(self.joint_speed)
                elif key.code == codes.KEY_S:
                    self.move_joints(-self.joint_speed)
                elif key.code == codes.KEY_Z:
                    self.zero_joints()
                return

            if self.current_mode == "wheel_select":
                wheel_commands = {
                    codes.KEY_W: (self.wheel_speed, 0.0),
                    codes.KEY_S: (-self.wheel_speed, 0.0),
                    codes.KEY_A: (0.0, -self.wheel_speed),
                    codes.KEY_D: (0.0, self.wheel_speed),
                }
                if key.code in wheel_commands:
                    self.move_wheels(*wheel_commands[key.code])
                return

            if self.current_mode == "ik_select" and key.code in self._ik_control_keys:
                self._ik_key_states.add(key.code)
                self.move_ik()
        else:
            if self.current_mode == "joint_select" and key.code in {
                codes.KEY_W,
                codes.KEY_S,
            }:
                self._cancel(self.joint_syncer)
            elif self.current_mode == "wheel_select" and key.code in {
                codes.KEY_W,
                codes.KEY_S,
                codes.KEY_A,
                codes.KEY_D,
            }:
                self._cancel(self.wheel_syncer)
            elif self.current_mode == "ik_select" and key.code in self._ik_control_keys:
                self._ik_key_states.discard(key.code)
                self.move_ik()

    @property
    def _ik_control_keys(self) -> frozenset[int]:
        return frozenset(
            {
                codes.KEY_UP,
                codes.KEY_DOWN,
                codes.KEY_LEFT,
                codes.KEY_RIGHT,
                codes.KEY_I,
                codes.KEY_K,
                codes.KEY_Q,
                codes.KEY_E,
                codes.KEY_W,
                codes.KEY_S,
                codes.KEY_A,
                codes.KEY_D,
            }
        )

    def add_log(self, level: str, message: str) -> None:
        self.logs.append(f"{time.strftime('%H:%M:%S')} [{level}] {message}")

    def _joint_target(
        self,
        direct: Iterable[JointRef],
        inverted: Iterable[JointRef],
        direct_value: float,
        inverted_value: float,
    ) -> dict[str, float]:
        target: dict[str, float] = {}
        conflicts: set[str] = set()
        for references, value in ((direct, direct_value), (inverted, inverted_value)):
            for namespace, joint in references:
                if namespace not in self.selected_robots:
                    continue
                if joint not in self.registry.joint_names(namespace):
                    continue
                previous = target.get(joint)
                if previous is not None and previous != value:
                    conflicts.add(joint)
                else:
                    target[joint] = value
        for joint in conflicts:
            target.pop(joint, None)
        if conflicts:
            self.add_log(
                "W",
                "Skipped duplicate joint names with conflicting directions: "
                + ", ".join(sorted(conflicts)),
            )
        return target

    def _ready(self, syncer: AsyncJointSyncer, target: dict[str, float]) -> bool:
        ready, missing = syncer.ready(target)
        if not ready:
            self.add_log("W", f"Joint feedback missing for: {sorted(missing)}")
        return ready

    def _registry_updated(self, _) -> None:
        current = set(self.registry.namespaces)
        removed = self.selected_robots - current
        for namespace in removed:
            self._discard_robot_selections(namespace)
        self.selected_robots.intersection_update(current)
        if removed:
            self.revision += 1

    def _discard_robot_selections(self, namespace: str) -> None:
        self.selected_joints = {
            ref for ref in self.selected_joints if ref[0] != namespace
        }
        self.inverted_joints = {
            ref for ref in self.inverted_joints if ref[0] != namespace
        }
        self.selected_wheels = {
            ref for ref in self.selected_wheels if ref[0] != namespace
        }
        self.inverted_wheels = {
            ref for ref in self.inverted_wheels if ref[0] != namespace
        }
        self.selected_ik_robots.discard(namespace)

    @staticmethod
    def _cancel(syncer) -> None:
        if not syncer.last_future.done():
            syncer.last_future.cancel()

    @staticmethod
    def _display_names(namespaces: Iterable[str]) -> str:
        names = [f"/{name}" if name else "/" for name in sorted(namespaces)]
        return ", ".join(names) if names else "none"

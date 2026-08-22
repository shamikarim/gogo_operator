import asyncio
import math
from typing import Any, Callable, Literal, Optional

import urwid

from .operator import JointRef, OperatorController
from .state import FleetSnapshot, RobotSnapshot, display_namespace


class TriStateCheckbox(urwid.CheckBox):
    states = {
        False: urwid.SelectableIcon("[ ]", 1),
        True: urwid.SelectableIcon("[X]", 1),
        "mixed": urwid.SelectableIcon("[R]", 1),
    }

    def __init__(self, label: str, state: bool | Literal["mixed"] = False) -> None:
        super().__init__(label, state=state, has_mixed=True)


class OperatorTUI:
    """Urwid interface running on the application's asyncio event loop."""

    ROBOT_COLORS = [
        "dark red",
        "dark green",
        "brown",
        "dark blue",
        "dark magenta",
        "dark cyan",
        "light gray",
        "light red",
        "light green",
        "yellow",
        "light blue",
        "light magenta",
        "light cyan",
        "white",
    ]
    JOINTS_PER_COLUMN = 3
    MAX_JOINT_COLUMNS = 4

    def __init__(self, controller: OperatorController, transport: str) -> None:
        self.controller = controller
        self.transport = transport
        self.mode_header = urwid.Text("")
        self.robot_header = urwid.Text("", align="right")
        header = urwid.Columns(
            [("weight", 2, self.mode_header), ("weight", 3, self.robot_header)],
            dividechars=1,
        )
        self.body = urwid.SimpleFocusListWalker([])
        self.listbox = urwid.ListBox(self.body)
        self.logs = urwid.Text("")
        self.frame = urwid.Frame(
            self.listbox,
            header=header,
            footer=urwid.LineBox(self.logs, title=" Logs "),
        )
        event_loop = urwid.AsyncioEventLoop(loop=asyncio.get_running_loop())
        self.loop = urwid.MainLoop(
            self.frame,
            self._palette(),
            event_loop=event_loop,
            unhandled_input=self._unhandled_input,
        )
        self._last_signature: Optional[tuple] = None
        self._rendered_mode: Optional[str] = None

    async def run(self) -> None:
        with self.loop.start():
            while not self.controller.stop_event.is_set():
                self.refresh()
                try:
                    await asyncio.wait_for(self.controller.stop_event.wait(), 0.2)
                except asyncio.TimeoutError:
                    pass

    def refresh(self, force: bool = False) -> None:
        snapshot = self.controller.registry.snapshot()
        self.mode_header.set_text(
            f"Gogo Operator 🦍  |  {self.transport}  |  "
            f"{self.controller.current_mode}"
        )
        selected = self.controller.selected_robots
        marks = []
        for robot in snapshot.robots:
            attr = "selected" if robot.namespace in selected else "unselected"
            marks.extend([(attr, display_namespace(robot.namespace)), " "])
        self.robot_header.set_text(marks or "waiting for robots...")
        self.logs.set_text("\n".join(self.controller.logs))

        signature = self._signature(snapshot)
        mode = self.controller.current_mode
        if force or signature != self._last_signature:
            focus_position = self.body.focus if self.body else None
            preserve_focus = mode == self._rendered_mode
            self._last_signature = signature
            builder = getattr(self, f"_build_{mode}")
            builder(snapshot)
            if preserve_focus and focus_position is not None and self.body:
                self.body.set_focus(min(focus_position, len(self.body) - 1))
            self._rendered_mode = mode
        self.loop.draw_screen()

    def _signature(self, snapshot: FleetSnapshot) -> tuple:
        structure = tuple(
            (
                robot.namespace,
                tuple(joint.name for joint in robot.joints),
                robot.ik_ready,
            )
            for robot in snapshot.robots
        )
        if self.controller.current_mode == "monitor":
            values = tuple(
                (
                    robot.namespace,
                    tuple(
                        (
                            joint.name,
                            self._rounded(joint.position),
                            self._rounded(joint.velocity),
                            self._rounded(joint.effort),
                            round(joint.age, 1),
                        )
                        for joint in robot.joints
                    ),
                )
                for robot in snapshot.robots
            )
        else:
            values = ()
        return structure, values, self.controller.current_mode, self.controller.revision

    def _build_main(self, snapshot: FleetSnapshot) -> None:
        self.body.clear()
        self.body.extend(
            [
                self._button("Monitor robots and joints  [M]", "monitor"),
                self._button("Robot selection           [R]", "robot_select"),
                self._button("Joint control             [J]", "joint_select"),
                self._button("Wheel control             [W]", "wheel_select"),
                self._button("IK control                [U]", "ik_select"),
                urwid.Divider(),
                urwid.Text(
                    "Motion keys are read from the Gorilla keyboard window. "
                    "Space stops all motion; Esc returns here."
                ),
                urwid.Divider(),
                urwid.Text(self._fleet_summary(snapshot)),
            ]
        )

    def _build_monitor(self, snapshot: FleetSnapshot) -> None:
        self.body.clear()
        if not snapshot.robots:
            self.body.append(urwid.Text("Waiting for joint_read publishers..."))
        for robot in snapshot.robots:
            self.body.append(urwid.Text(("title", self._robot_title(robot))))
            self.body.append(
                urwid.Columns(
                    [
                        ("weight", 3, urwid.Text("joint")),
                        ("weight", 1, urwid.Text("position")),
                        ("weight", 1, urwid.Text("velocity")),
                        ("weight", 1, urwid.Text("effort")),
                        (9, urwid.Text("age")),
                    ]
                )
            )
            for joint in robot.joints:
                self.body.append(
                    urwid.Columns(
                        [
                            ("weight", 3, urwid.Text(joint.name)),
                            (
                                "weight",
                                1,
                                urwid.Text(self._angle(joint.position)),
                            ),
                            ("weight", 1, urwid.Text(self._number(joint.velocity))),
                            ("weight", 1, urwid.Text(self._number(joint.effort))),
                            (9, urwid.Text(f"{joint.age:5.1f} s")),
                        ]
                    )
                )
            self.body.append(urwid.Divider())
        self.body.append(self._back_button())

    def _build_robot_select(self, snapshot: FleetSnapshot) -> None:
        self.body.clear()
        if not snapshot.robots:
            self.body.append(urwid.Text("Waiting for robot discovery..."))
        for robot in snapshot.robots:
            label = self._robot_title(robot)
            checkbox = urwid.CheckBox(
                label,
                state=robot.namespace in self.controller.selected_robots,
            )
            urwid.connect_signal(
                checkbox,
                "change",
                lambda _, state, namespace=robot.namespace: self._select_robot(
                    namespace, state
                ),
            )
            self.body.append(urwid.AttrMap(checkbox, None, focus_map="focus"))
        self.body.extend(
            [
                urwid.Divider(),
                self._action_button("Select all", self._select_all_robots),
                self._action_button("Clear all", self._clear_robots),
                self._back_button(),
            ]
        )

    def _build_joint_select(self, snapshot: FleetSnapshot) -> None:
        self._build_joint_picker(snapshot, "joint")

    def _build_wheel_select(self, snapshot: FleetSnapshot) -> None:
        self._build_joint_picker(snapshot, "wheel")

    def _build_joint_picker(
        self, snapshot: FleetSnapshot, kind: Literal["joint", "wheel"]
    ) -> None:
        self.body.clear()
        title = "Joint" if kind == "joint" else "Wheel"
        self.body.append(
            urwid.Text(
                f"{title} selection: [X] normal, [R] reversed. "
                "Space/Enter cycles the state."
            )
        )
        self.body.append(self._speed_picker(kind))
        self.body.append(urwid.Divider())

        robots = [
            robot
            for robot in snapshot.robots
            if robot.namespace in self.controller.selected_robots
        ]
        for color_index, robot in enumerate(robots):
            base_attr = f"robot{color_index % len(self.ROBOT_COLORS)}"
            focus_attr = f"{base_attr}_focus"
            self.body.append(urwid.Text((base_attr, self._robot_title(robot))))

            joints_per_column = max(
                self.JOINTS_PER_COLUMN,
                math.ceil(len(robot.joints) / self.MAX_JOINT_COLUMNS),
            )
            columns = []
            for start in range(0, len(robot.joints), joints_per_column):
                items = []
                for joint in robot.joints[start : start + joints_per_column]:
                    reference = (robot.namespace, joint.name)
                    state = self._selection_state(kind, reference)
                    checkbox = TriStateCheckbox(joint.name, state=state)
                    urwid.connect_signal(
                        checkbox,
                        "change",
                        lambda _, new_state, ref=reference: (
                            self.controller.set_joint_selection(
                                kind,
                                ref,
                                "inverted" if new_state == "mixed" else new_state,
                            )
                        ),
                    )
                    items.append(
                        urwid.AttrMap(
                            checkbox,
                            base_attr,
                            focus_map=focus_attr,
                        )
                    )
                columns.append(("weight", 1, urwid.Pile(items)))
            if columns:
                self.body.append(urwid.Columns(columns, dividechars=1))
            self.body.append(urwid.Divider())
        if not robots:
            self.body.append(urwid.Text("Select at least one discovered robot first."))
        self.body.extend(
            [
                self._action_button(
                    f"Clear {title.lower()} selection",
                    lambda: self._clear_joint_selection(kind),
                ),
                self._back_button(),
            ]
        )

    def _build_ik_select(self, snapshot: FleetSnapshot) -> None:
        self.body.clear()
        self.body.extend(
            [
                urwid.Text(
                    "IK controls: arrows X/Y, I/K Z, Q/E roll, W/S pitch, A/D yaw."
                ),
                self._ik_frame_picker(),
                urwid.Divider(),
            ]
        )
        shown = False
        for robot in snapshot.robots:
            if robot.namespace not in self.controller.selected_robots:
                continue
            shown = True
            if not robot.ik_ready:
                self.body.append(
                    urwid.AttrMap(
                        urwid.Text(
                            f"[ ] {display_namespace(robot.namespace)} (waiting for tip_pos)"
                        ),
                        "disabled",
                    )
                )
                continue
            checkbox = urwid.CheckBox(
                display_namespace(robot.namespace),
                state=robot.namespace in self.controller.selected_ik_robots,
            )
            urwid.connect_signal(
                checkbox,
                "change",
                lambda _, state, namespace=robot.namespace: (
                    self.controller.select_ik_robot(namespace, state)
                ),
            )
            self.body.append(urwid.AttrMap(checkbox, None, focus_map="focus"))
        if not shown:
            self.body.append(urwid.Text("Select at least one discovered robot first."))
        self.body.extend([urwid.Divider(), self._back_button()])

    def _speed_picker(self, kind: str) -> urwid.Widget:
        if kind == "joint":
            attribute = "joint_speed"
            levels = [("Low", 0.05), ("Medium", 0.15), ("High", 0.3)]
        else:
            attribute = "wheel_speed"
            levels = [("Low", 0.1), ("Medium", 0.2), ("High", 0.5)]
        group = []
        buttons = []
        for label, value in levels:
            button = urwid.RadioButton(
                group,
                label,
                state=math.isclose(getattr(self.controller, attribute), value),
            )
            urwid.connect_signal(
                button,
                "change",
                lambda _, state, val=value, attr=attribute: (
                    setattr(self.controller, attr, val) if state else None
                ),
            )
            buttons.append(button)
        return urwid.GridFlow(buttons, 16, 1, 0, "left")

    def _ik_frame_picker(self) -> urwid.Widget:
        group = []
        base = urwid.RadioButton(
            group,
            "Base frame",
            state=not self.controller.ik_end_effector_frame,
        )
        end = urwid.RadioButton(
            group,
            "End-effector frame",
            state=self.controller.ik_end_effector_frame,
        )
        urwid.connect_signal(
            base,
            "change",
            lambda _, state: (
                setattr(self.controller, "ik_end_effector_frame", False)
                if state
                else None
            ),
        )
        urwid.connect_signal(
            end,
            "change",
            lambda _, state: (
                setattr(self.controller, "ik_end_effector_frame", True)
                if state
                else None
            ),
        )
        return urwid.GridFlow([base, end], 24, 1, 0, "left")

    def _selection_state(
        self, kind: Literal["joint", "wheel"], reference: JointRef
    ) -> bool | Literal["mixed"]:
        if kind == "joint":
            direct = self.controller.selected_joints
            inverted = self.controller.inverted_joints
        else:
            direct = self.controller.selected_wheels
            inverted = self.controller.inverted_wheels
        if reference in direct:
            return True
        if reference in inverted:
            return "mixed"
        return False

    def _select_robot(self, namespace: str, state: bool) -> None:
        self.controller.select_robot(namespace, state)
        self.refresh(force=True)

    def _select_all_robots(self) -> None:
        self.controller.selected_robots.update(self.controller.registry.namespaces)
        self.controller.revision += 1
        self.refresh(force=True)

    def _clear_robots(self) -> None:
        for namespace in tuple(self.controller.selected_robots):
            self.controller.select_robot(namespace, False)
        self.refresh(force=True)

    def _clear_joint_selection(self, kind: Literal["joint", "wheel"]) -> None:
        if kind == "joint":
            self.controller.selected_joints.clear()
            self.controller.inverted_joints.clear()
        else:
            self.controller.selected_wheels.clear()
            self.controller.inverted_wheels.clear()
        self.controller.revision += 1
        self.refresh(force=True)

    def _button(self, label: str, mode: str) -> urwid.Widget:
        return self._action_button(label, lambda: self._set_mode(mode))

    def _back_button(self) -> urwid.Widget:
        return self._button("Back to main", "main")

    def _action_button(self, label: str, callback: Callable[[], Any]) -> urwid.Widget:
        button = urwid.Button(label, on_press=lambda _: callback())
        return urwid.AttrMap(button, None, focus_map="focus")

    def _set_mode(self, mode: str) -> None:
        self.controller.set_mode(mode)
        self.refresh(force=True)

    def _unhandled_input(self, key: Any) -> None:
        if key in {"q", "Q", "ctrl c"}:
            self.controller.request_stop()
        elif key == "esc":
            self._set_mode("main")

    @staticmethod
    def _fleet_summary(snapshot: FleetSnapshot) -> str:
        robot_count = len(snapshot.robots)
        joint_count = sum(len(robot.joints) for robot in snapshot.robots)
        ik_count = sum(robot.ik_ready for robot in snapshot.robots)
        return f"Detected: {robot_count} robot(s), {joint_count} joint(s), {ik_count} IK endpoint(s)"

    @staticmethod
    def _robot_title(robot: RobotSnapshot) -> str:
        age = (
            "no samples"
            if robot.sample_age is None
            else f"{robot.sample_age:.1f} s ago"
        )
        ik = "IK ready" if robot.ik_ready else "no IK"
        return (
            f"{display_namespace(robot.namespace)}  |  "
            f"{len(robot.joints)} joints  |  {age}  |  {ik}"
        )

    @staticmethod
    def _number(value: Optional[float]) -> str:
        return "--" if value is None else f"{value:.3f}"

    @staticmethod
    def _angle(value: Optional[float]) -> str:
        return "--" if value is None else f"{math.degrees(value):.1f} deg"

    @staticmethod
    def _rounded(value: Optional[float]) -> Optional[float]:
        return None if value is None else round(value, 3)

    @classmethod
    def _palette(cls) -> list[tuple[str, str, str]]:
        palette = [
            ("disabled", "dark gray", ""),
            ("focus", "black", "light gray"),
            ("selected", "light green", ""),
            ("unselected", "light red", ""),
            ("title", "light cyan,bold", ""),
        ]
        for index, color in enumerate(cls.ROBOT_COLORS):
            palette.append((f"robot{index}", color, ""))
            palette.append((f"robot{index}_focus", color, ""))
        return palette

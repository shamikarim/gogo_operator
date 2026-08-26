from typing import cast

import numpy as np
import pytest
import quaternion as qt
import urwid
from motion_stack.utils.joint_state import JState
from motion_stack.utils.math import Flo3
from motion_stack.utils.pose import Pose
from motion_stack.utils.time import Time

from gogo_operator.operator import OperatorController
from gogo_operator.state import RobotRegistry
from gogo_operator.tui import OperatorTUI, TriStateCheckbox


@pytest.mark.asyncio
async def test_all_tui_screens_build_from_registry_state() -> None:
    registry = RobotRegistry()
    registry.observe_joints("leg1", {"joint1_1": JState("joint1_1", position=0.25)})
    registry.observe_ik(
        "leg1", Pose(Time(0), cast(Flo3, np.zeros(3)), qt.one)
    )
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


@pytest.mark.asyncio
async def test_joint_picker_groups_colors_by_natural_name_prefix() -> None:
    registry = RobotRegistry()
    registry.observe_joints(
        "leg1",
        {
            "wheel_1": JState("wheel_1", position=0.0),
            "arm_10": JState("arm_10", position=0.0),
            "panel_1": JState("panel_1", position=0.0),
            "arm_2": JState("arm_2", position=0.0),
            "wheel_0": JState("wheel_0", position=0.0),
            "arm_1": JState("arm_1", position=0.0),
            "panel_0": JState("panel_0", position=0.0),
        },
    )
    controller = OperatorController(registry)
    controller.select_robot("leg1", True)
    tui = OperatorTUI(controller, "test")

    tui._build_joint_select(registry.snapshot())

    joint_grid = next(
        widget for widget in tui.body if isinstance(widget, urwid.Columns)
    )
    assert len(joint_grid.contents) == 3
    labels = []
    colors = []
    for column, _ in joint_grid.contents:
        pile = cast(urwid.Pile, column)
        for entry in pile.contents:
            joint = cast(tuple[urwid.Widget, object], entry)[0]
            assert isinstance(joint, urwid.AttrMap)
            checkbox = cast(TriStateCheckbox, joint.original_widget)
            labels.append(str(checkbox.label))
            colors.append(joint.get_attr_map()[None])
    assert labels == [
        "arm_1",
        "arm_2",
        "arm_10",
        "panel_0",
        "panel_1",
        "wheel_0",
        "wheel_1",
    ]
    assert colors == [
        "robot0",
        "robot0",
        "robot0",
        "robot1",
        "robot1",
        "robot2",
        "robot2",
    ]


def test_right_click_toggles_reverse_selection() -> None:
    checkbox = TriStateCheckbox("joint")
    changes: list[object] = []
    urwid.connect_signal(
        checkbox, "change", lambda _, state: changes.append(state)
    )

    assert checkbox.mouse_event((20,), "mouse press", 3, 0, 0, True)
    assert checkbox.get_state() == "mixed"
    assert checkbox.mouse_event((20,), "mouse press", 3, 0, 0, True)
    assert checkbox.get_state() is False
    checkbox.set_state(True)
    assert checkbox.mouse_event((20,), "mouse press", 3, 0, 0, True)
    assert checkbox.get_state() is False
    assert changes == ["mixed", False, True, False]


def test_left_click_only_toggles_normal_selection() -> None:
    checkbox = TriStateCheckbox("joint")

    assert checkbox.mouse_event((20,), "mouse press", 1, 0, 0, True)
    assert checkbox.get_state() is True
    assert checkbox.mouse_event((20,), "mouse press", 1, 0, 0, True)
    assert checkbox.get_state() is False
    checkbox.set_state("mixed")
    assert checkbox.mouse_event((20,), "mouse press", 1, 0, 0, True)
    assert checkbox.get_state() is True


@pytest.mark.parametrize("mode", ["joint_select", "wheel_select"])
@pytest.mark.asyncio
async def test_selection_click_preserves_nested_focus(mode: str) -> None:
    registry = RobotRegistry()
    registry.observe_joints(
        "leg1",
        {
            f"joint_{index}": JState(f"joint_{index}", position=0.0)
            for index in range(4)
        },
    )
    controller = OperatorController(registry)
    controller.select_robot("leg1", True)
    controller.set_mode(mode)
    tui = OperatorTUI(controller, "test")
    tui.loop.draw_screen = lambda: None
    tui.refresh()

    joint_grid = next(
        widget for widget in tui.body if isinstance(widget, urwid.Columns)
    )
    joint_grid.focus_position = 1
    second_column = cast(urwid.Pile, joint_grid.contents[1][0])
    checkbox_entry = cast(tuple[urwid.Widget, object], second_column.contents[0])
    checkbox_map = cast(urwid.AttrMap, checkbox_entry[0])
    checkbox = cast(TriStateCheckbox, checkbox_map.original_widget)

    checkbox.mouse_event((20,), "mouse press", 1, 0, 0, True)
    tui.refresh()

    refreshed_grid = next(
        widget for widget in tui.body if isinstance(widget, urwid.Columns)
    )
    assert refreshed_grid is joint_grid
    assert joint_grid.focus_position == 1
    assert checkbox.get_state() is True


@pytest.mark.asyncio
async def test_wheel_picker_caps_large_joint_sets_at_four_columns() -> None:
    registry = RobotRegistry()
    registry.observe_joints(
        "leg1",
        {
            f"joint_{index}": JState(f"joint_{index}", position=0.0)
            for index in range(124)
        },
    )
    controller = OperatorController(registry)
    controller.select_robot("leg1", True)
    tui = OperatorTUI(controller, "test")

    tui._build_wheel_select(registry.snapshot())

    joint_grid = next(
        widget for widget in tui.body if isinstance(widget, urwid.Columns)
    )
    assert len(joint_grid.contents) == OperatorTUI.MAX_JOINT_COLUMNS
    assert all(
        len(cast(urwid.Pile, column).contents) == 31
        for column, _ in joint_grid.contents
    )


@pytest.mark.asyncio
async def test_main_screen_omits_redundant_motion_stack_title() -> None:
    controller = OperatorController(RobotRegistry())
    tui = OperatorTUI(controller, "test")

    tui._build_main(controller.registry.snapshot())

    text = " ".join(
        str(widget.text) for widget in tui.body if isinstance(widget, urwid.Text)
    )
    assert "Motion Stack 2 Operator" not in text


@pytest.mark.asyncio
async def test_live_monitor_refresh_preserves_scroll_position() -> None:
    registry = RobotRegistry()
    registry.observe_joints(
        "leg1",
        {
            f"joint_{index}": JState(f"joint_{index}", position=0.0)
            for index in range(20)
        },
    )
    controller = OperatorController(registry)
    controller.set_mode("monitor")
    tui = OperatorTUI(controller, "test")
    tui.loop.draw_screen = lambda: None
    tui.refresh()

    tui.body.set_focus(10)
    tui.listbox.shift_focus((80, 20), 5)
    registry.observe_joints(
        "leg1",
        {
            f"joint_{index}": JState(f"joint_{index}", position=1.0)
            for index in range(20)
        },
    )

    tui.refresh()

    assert tui.body.focus == 10
    assert tui.listbox.get_focus_offset_inset((80, 20)) == (5, 0)

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
from gogo_operator.tui import OperatorTUI


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
async def test_joint_picker_uses_colored_three_item_columns() -> None:
    registry = RobotRegistry()
    registry.observe_joints(
        "leg1",
        {
            f"joint_{index}": JState(f"joint_{index}", position=0.0)
            for index in range(7)
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
    first_column = cast(urwid.Pile, joint_grid.contents[0][0])
    assert isinstance(first_column, urwid.Pile)
    first_entry = cast(tuple[urwid.Widget, object], first_column.contents[0])
    first_joint = first_entry[0]
    assert isinstance(first_joint, urwid.AttrMap)
    assert first_joint.get_attr_map() == {None: "robot0"}


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

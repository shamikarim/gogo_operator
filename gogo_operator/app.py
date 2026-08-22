import asyncio
from contextlib import nullcontext
from typing import Any, Optional

import asyncio_for_robotics as afor

from .backends import PyzerosBackend, ZenohBackend
from .bridge import DynamicJointBridge
from .operator import OperatorController
from .state import RobotRegistry
from .tui import OperatorTUI


class OperatorApp:
    """Composable operator application for built-in or custom bridges."""

    def __init__(
        self,
        controller: OperatorController,
        bridge: DynamicJointBridge,
        *,
        transport_name: str,
        keyboard: bool = True,
    ) -> None:
        self.controller = controller
        self.bridge = bridge
        self.transport_name = transport_name
        self.keyboard = keyboard

    async def run(self) -> None:
        scope = afor.Scope.current()
        scope.exit_stack.callback(self.controller.close)
        scope.exit_stack.callback(self.bridge.close)
        self.controller.start(scope.task_group)
        scope.task_group.create_task(self.bridge.run(), name="operator_bridge")
        if self.keyboard:
            scope.task_group.create_task(
                self.controller.keyboard_loop(), name="operator_keyboard"
            )

        tui = OperatorTUI(self.controller, self.transport_name)
        await tui.run()


def build_app(
    transport: str,
    *,
    timeout: float = 5.0,
    keyboard: bool = True,
    session: Optional[Any] = None,
) -> OperatorApp:
    """Build an app while allowing callers to supply an existing session."""
    registry = RobotRegistry()
    controller = OperatorController(registry)
    backend_type = {"pyzeros": PyzerosBackend, "zenoh": ZenohBackend}.get(transport)
    if backend_type is None:
        raise ValueError(f"Unsupported transport: {transport}")
    bridge = backend_type(
        registry,
        controller.sensor_input,
        controller.command_output,
        controller.ik_sensor_input,
        controller.ik_command_output,
        timeout=timeout,
        session=session,
    )
    return OperatorApp(
        controller,
        bridge,
        transport_name=transport,
        keyboard=keyboard,
    )


@afor.scoped
async def run_operator(
    transport: str,
    *,
    timeout: float = 5.0,
    keyboard: bool = True,
    session: Optional[Any] = None,
) -> None:
    """Run the operator inside an afor scope."""
    app = build_app(transport, timeout=timeout, keyboard=keyboard, session=session)
    await app.run()


def transport_context(transport: str, node_name: str):
    if transport == "zenoh":
        import asyncio_for_robotics.zenoh as afor_zenoh

        return afor_zenoh.auto_context()
    if transport != "pyzeros":
        return nullcontext()
    try:
        import pyzeros
    except ImportError as error:
        raise RuntimeError(
            "The pyzeros transport requires pyzeros and ms-pyzeros-bridge."
        ) from error
    return pyzeros.auto_context(node=node_name)


def run_sync(
    transport: str,
    *,
    timeout: float = 5.0,
    keyboard: bool = True,
    node_name: str = "gogo_operator",
) -> None:
    """Synchronous entry point that creates the pyzeros context when needed."""
    with transport_context(transport, node_name) as session:
        asyncio.run(
            run_operator(
                transport,
                timeout=timeout,
                keyboard=keyboard,
                session=session,
            )
        )

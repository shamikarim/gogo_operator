# Gogo Operator

`gogo-operator` is an asyncio/Urwid operator for Motion Stack 2. It has no
`rclpy` dependency: robot discovery and messages are provided by either the
native Zenoh bridge or the ROS 2-compatible pyzeros bridge.

The operator discovers every `*/joint_read` publisher, keeps the robot
namespace attached to its joints, monitors joint values and `tip_pos` IK
feedback, and routes commands back to the matching namespace. The shared
operator logic only sees Motion Stack types and `BaseSub` streams.

## Run

Install the workspace and start the default pyzeros backend:

```console
pixi install
pixi run operator
```

Use native Zenoh instead:

```console
pixi run operator-zenoh
```

Useful CLI options:

```console
gogo-operator --help
gogo-operator --transport zenoh --timeout 8
gogo-operator --transport pyzeros --node-name my_operator
gogo-operator --no-keyboard
```

The terminal window is used for menus and selection. Held motion keys are read
from the separate Gorilla keyboard window so key-up events reliably stop
motion. `Space` stops every active command and `Esc` returns to the main menu.

Controls:

- Main menu: `M` monitor, `R` robots, `J` joints, `W` wheels, `U` IK.
- Joint mode: hold `W`/`S`; `Z` sends selected joints to zero.
- Wheel mode: hold `W`/`S` for forward/backward and `A`/`D` for turning.
- IK mode: arrows move X/Y, `I`/`K` move Z, and `Q`/`E`, `W`/`S`,
  `A`/`D` rotate around X, Y, and Z.

## Use as a package

The built-in app accepts an existing transport session, which lets another
async application own the session lifecycle:

```python
import asyncio_for_robotics as afor
from gogo_operator import build_app


@afor.scoped
async def main(zenoh_session):
    app = build_app("zenoh", session=zenoh_session)
    await app.run()
```

For a new transport, subclass `DynamicJointBridge` and implement discovery plus
the four existing Motion Stack bridge hooks. `OperatorController`,
`RobotRegistry`, and `OperatorApp` can then be reused unchanged.

## Discovery behavior

- Native Zenoh discovers `**/joint_read` keys and removes a robot when no
  samples arrive within `--timeout`.
- pyzeros follows ROS 2 Zenoh liveliness tokens, including DELETE events, so an
  idle but still-live robot is not removed just because its joint values stop
  changing.
- Joint commands are filtered per namespace. If two selected namespaces expose
  the same joint name with conflicting directions, that joint is skipped and a
  warning is shown instead of sending an ambiguous command.

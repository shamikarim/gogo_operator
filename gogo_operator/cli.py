import argparse
from contextlib import suppress
from typing import Optional, Sequence

from .app import run_sync


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Transport-independent Motion Stack 2 terminal operator"
    )
    parser.add_argument(
        "--transport",
        choices=("pyzeros", "zenoh"),
        default="pyzeros",
        help="Bridge used for discovery and commands (default: pyzeros)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=5.0,
        help="Zenoh robot timeout in seconds (default: 5)",
    )
    parser.add_argument(
        "--node-name",
        default="gogo_operator",
        help="pyzeros node name (default: gogo_operator)",
    )
    parser.add_argument(
        "--no-keyboard",
        action="store_true",
        help="Do not open the separate Gorilla keyboard window",
    )
    args = parser.parse_args(argv)
    if args.timeout <= 0:
        parser.error("--timeout must be greater than zero")
    return args


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    with suppress(KeyboardInterrupt):
        run_sync(
            args.transport,
            timeout=args.timeout,
            keyboard=not args.no_keyboard,
            node_name=args.node_name,
        )


if __name__ == "__main__":
    main()

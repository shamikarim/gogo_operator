import pytest

from gogo_operator.backends import PyzerosBackend
from gogo_operator.cli import parse_args


def test_cli_defaults_to_pyzeros() -> None:
    args = parse_args([])
    assert args.transport == "pyzeros"
    assert args.timeout == 5.0


def test_cli_rejects_non_positive_timeout() -> None:
    with pytest.raises(SystemExit):
        parse_args(["--timeout", "0"])


def test_pyzeros_liveliness_namespace_parser() -> None:
    key = "@ros2_lv/0/MP/leg1%joint_read/reader"
    assert PyzerosBackend._namespace_from_liveliness(key) == "leg1"

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from deep.device import resolve_device


def test_resolve_device_cpu() -> None:
    info = resolve_device("cpu")
    assert info["requested"] == "cpu"
    assert info["resolved"] == "cpu"


def test_resolve_device_auto_returns_known_value() -> None:
    info = resolve_device("auto")
    assert info["requested"] == "auto"
    assert info["resolved"] in {"cpu", "cuda"} or info["resolved"].startswith("cuda:")

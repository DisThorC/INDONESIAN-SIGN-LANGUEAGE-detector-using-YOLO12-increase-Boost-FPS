"""
Bridge script to allow launching with: python src/realtime.py [args]
Delegates to the top-level realtime.py main().
"""
from __future__ import annotations

import importlib.util
import pathlib
import sys
from typing import Callable


def _load_top_level_main() -> Callable[[], None]:
    try:
        from realtime import main as entry  # type: ignore
        return entry
    except Exception:
        root = pathlib.Path(__file__).resolve().parents[1]
        rt = root / "realtime.py"
        if not rt.exists():
            raise FileNotFoundError(f"Top-level realtime.py not found at {rt}")
        spec = importlib.util.spec_from_file_location("realtime", rt)
        assert spec and spec.loader
        mod = importlib.util.module_from_spec(spec)
        sys.modules["realtime"] = mod
        spec.loader.exec_module(mod)
        assert hasattr(mod, "main") and callable(getattr(mod, "main"))
        return getattr(mod, "main")


essential_explanation = "Run as: python src/realtime.py [--flags]; this wraps top-level realtime.py"


def main() -> None:
    entry = _load_top_level_main()
    entry()


if __name__ == "__main__":
    main()

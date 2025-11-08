"""
Package wrapper for realtime webcam inference.

Allows running:
  - python -m src.yolo12.realtime [args]
  - python .\src\yolo12\realtime.py [args]
Delegates to the top-level realtime.py in the project root to avoid code duplication.
"""
from __future__ import annotations

import importlib.util
import pathlib
import sys
from typing import Callable


def _load_top_level_realtime_main() -> Callable[[], None]:
    """Import the project's top-level realtime.py and return its main() function.

    This keeps the single source of truth in the root file, while enabling package-style execution.
    """
    try:
        # Try normal import first (project root on sys.path)
        from realtime import main as entry  # type: ignore
        return entry
    except Exception:
        # Fallback: import by file path
        root = pathlib.Path(__file__).resolve().parents[2]  # <project_root>
        rt_path = root / "realtime.py"
        if not rt_path.exists():
            raise FileNotFoundError(f"Top-level realtime.py not found at: {rt_path}")
        spec = importlib.util.spec_from_file_location("realtime", rt_path)
        if spec is None or spec.loader is None:
            raise ImportError("Failed to load spec for top-level realtime.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules["realtime"] = mod
        spec.loader.exec_module(mod)
        if not hasattr(mod, "main") or not callable(getattr(mod, "main")):
            raise AttributeError("realtime.py does not define a callable main()")
        return getattr(mod, "main")


def main() -> None:
    entry = _load_top_level_realtime_main()
    entry()


if __name__ == "__main__":
    main()

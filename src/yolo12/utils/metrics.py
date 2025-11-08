from __future__ import annotations

from typing import Any, Dict


def summarize_ultralytics_metrics(metrics_obj: Any) -> Dict[str, float]:
    """Extract a minimal dict of metrics from Ultralytics validation results."""
    out: Dict[str, float] = {}
    try:
        box = getattr(metrics_obj, "box", None)
        if box is not None:
            out.update(
                {
                    "map50": float(getattr(box, "map50", float("nan"))),
                    "map": float(getattr(box, "map", float("nan"))),
                    "precision": float(getattr(box, "mp", float("nan"))),
                    "recall": float(getattr(box, "mr", float("nan"))),
                }
            )
    except Exception:
        pass
    return out

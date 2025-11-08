import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from ultralytics import YOLO


def create_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Extended evaluation: mAP, per-class AP, confusion matrix")
    p.add_argument("--weights", type=str, required=True, help="Path to trained weights .pt")
    p.add_argument("--data", type=str, default="configs/data.yaml", help="Path to data.yaml")
    p.add_argument("--split", type=str, default="val", choices=["train", "val", "test"], help="Dataset split")
    p.add_argument("--device", type=str, default="cpu", help="Device: cpu|cuda")
    p.add_argument("--save-json", type=str, default="", help="Optional path to dump metrics JSON")
    p.add_argument("--show-matrix", action="store_true", help="Print confusion matrix inline")
    return p


def safe_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return 0.0


def evaluate(args: argparse.Namespace) -> Dict[str, Any]:
    model = YOLO(args.weights)
    print(f"[yolo12] Validating on {args.split} split using {args.data}")
    results = model.val(data=args.data, split=args.split, device=args.device)

    out: Dict[str, Any] = {}
    # Aggregate high-level metrics
    try:
        box = results.box  # type: ignore[attr-defined]
        out.update({
            "mAP50": safe_float(getattr(box, "map50", 0)),
            "mAP50_95": safe_float(getattr(box, "map", 0)),
            "precision": safe_float(getattr(box, "mp", 0)),
            "recall": safe_float(getattr(box, "mr", 0)),
        })
    except Exception:
        pass

    # Per-class AP (if available)
    try:
        # Ultralytics metrics may expose maps list (per-class mAP@0.5:0.95)
        maps: List[float] = getattr(results, "maps", [])  # type: ignore[attr-defined]
        names: Dict[int, str] = results.names  # type: ignore[attr-defined]
        per_class = {names[i]: safe_float(maps[i]) for i in range(min(len(names), len(maps)))}
        out["per_class_mAP50_95"] = per_class
    except Exception:
        out["per_class_mAP50_95"] = {}

    # Confusion matrix (boxes) if exposed
    try:
        cm = getattr(results, "confusion_matrix", None)  # type: ignore[attr-defined]
        if cm is not None and hasattr(cm, "matrix"):
            matrix = cm.matrix  # type: ignore[attr-defined]
            if isinstance(matrix, np.ndarray):
                names: Dict[int, str] = results.names  # type: ignore[attr-defined]
                out["confusion_matrix"] = {
                    "labels": [names[i] for i in range(len(names))],
                    "matrix": matrix.tolist(),
                }
    except Exception:
        pass

    # Display summary
    print("[yolo12] Metrics summary:")
    for k in ["mAP50", "mAP50_95", "precision", "recall"]:
        if k in out:
            print(f"  {k}: {out[k]:.4f}")
    if out.get("per_class_mAP50_95"):
        print("[yolo12] Per-class mAP@50:95 (top 10):")
        sorted_items = sorted(out["per_class_mAP50_95"].items(), key=lambda x: x[1], reverse=True)[:10]
        for name, val in sorted_items:
            print(f"  {name}: {val:.4f}")
    if args.show_matrix and out.get("confusion_matrix"):
        labels = out["confusion_matrix"]["labels"]
        matrix = out["confusion_matrix"]["matrix"]
        print("[yolo12] Confusion matrix (rows=GT, cols=Pred):")
        # Print a trimmed matrix if large
        for i, row in enumerate(matrix):
            print(f"  {labels[i]:15s} | " + " ".join(f"{int(v):3d}" for v in row))

    if args.save_json:
        out_path = Path(args.save_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print(f"[yolo12] Saved metrics JSON -> {out_path}")
    return out


def main() -> None:
    parser = create_parser()
    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()

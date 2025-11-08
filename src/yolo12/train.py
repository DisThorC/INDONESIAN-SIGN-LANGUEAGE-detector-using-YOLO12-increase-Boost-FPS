import argparse
import os
from pathlib import Path
from typing import Any, Dict

import yaml
from ultralytics import YOLO


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def create_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train Ultralytics YOLO for ISL")
    p.add_argument("--data", type=str, default="configs/data.yaml", help="Path to data.yaml")
    p.add_argument("--params", type=str, default="configs/params.yaml", help="Path to params.yaml")
    p.add_argument("--model", type=str, default=None, help="Model weights or architecture, e.g., yolov8n.pt")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--imgsz", type=int, default=None)
    p.add_argument("--batch", type=int, default=None)
    p.add_argument("--device", type=str, default=None, help='"cpu" or "cuda"')
    p.add_argument("--workers", type=int, default=None, help="DataLoader workers (Windows: try 0)")
    p.add_argument("--resume", action="store_true", help="Resume training from last checkpoint")
    p.add_argument("--name", type=str, default=None, help="Run name")
    p.add_argument("--project", type=str, default=None, help="Project directory")
    return p


def train(args: argparse.Namespace) -> None:
    params = load_yaml(args.params)

    # CLI overrides
    for k in ["model", "epochs", "imgsz", "batch", "device", "workers", "name", "project"]:
        v = getattr(args, k)
        if v is not None:
            params[k] = v

    model_path = params.get("model", "yolov8n.pt")
    # If a local weights path is configured but doesn't exist, fall back to a small base model
    mp = Path(str(model_path))
    if ("/" in str(model_path) or "\\" in str(model_path)) and not mp.exists():
        print(f"[yolo12] Warning: model not found at '{model_path}'. Falling back to 'yolov8n.pt'.")
        model_path = "yolov8n.pt"
    model = YOLO(model_path)

    project = params.get("project", "runs/train")
    name = params.get("name", "exp")

    train_kwargs = {
        "data": args.data,
        "epochs": int(params.get("epochs", 50)),
        "imgsz": int(params.get("imgsz", 640)),
        "batch": int(params.get("batch", 16)),
        "device": str(params.get("device", "cpu")),
        "project": project,
        "name": name,
        "exist_ok": bool(params.get("exist_ok", True)),
        "patience": int(params.get("patience", 50)),
    "workers": int(params.get("workers", 4)),
        "seed": int(params.get("seed", 42)),
        "save": bool(params.get("save", True)),
        "val": bool(params.get("val", True)),
        "resume": bool(args.resume or params.get("resume", False)),
        "optimizer": params.get("optimizer", "auto"),
    }

    os.makedirs(project, exist_ok=True)
    print(f"[yolo12] Training with args: {train_kwargs}")
    results = model.train(**train_kwargs)
    print("[yolo12] Training complete.")
    save_dir = Path(results.save_dir) if hasattr(results, "save_dir") else Path(project) / name
    print(f"[yolo12] Results saved to: {save_dir}")


def main() -> None:
    parser = create_parser()
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()

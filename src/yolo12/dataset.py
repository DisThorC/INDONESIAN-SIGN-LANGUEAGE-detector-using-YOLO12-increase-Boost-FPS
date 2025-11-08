from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, List

import yaml


REQUIRED_SUBDIRS = [
    "images/train",
    "images/val",
    "images/test",
    "labels/train",
    "labels/val",
    "labels/test",
]


def validate_dataset_structure(base: str | Path) -> bool:
    base = Path(base)
    missing = [p for p in REQUIRED_SUBDIRS if not (base / p).exists()]
    if missing:
        print(f"[yolo12] Missing dataset folders under {base}:")
        for m in missing:
            print(" -", m)
        return False
    return True


def write_data_yaml(dataset_root: str | Path, class_names: Iterable[str], out_path: str | Path) -> Path:
    dataset_root = Path(dataset_root)
    out_path = Path(out_path)
    content = {
        "path": str(dataset_root).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": list(class_names),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(content, f, sort_keys=False, allow_unicode=True)
    print(f"[yolo12] Wrote data.yaml to {out_path}")
    return out_path

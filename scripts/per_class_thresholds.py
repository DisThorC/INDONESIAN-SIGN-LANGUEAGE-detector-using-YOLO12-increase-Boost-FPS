import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

# Types
Box = Tuple[float, float, float, float]
Pred = Tuple[Box, float, int]
GT = Tuple[Box, int]


def iou_xyxy(a: Box, b: Box) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    iw = max(0.0, inter_x2 - inter_x1)
    ih = max(0.0, inter_y2 - inter_y1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, (ax2 - ax1)) * max(0.0, (ay2 - ay1))
    area_b = max(0.0, (bx2 - bx1)) * max(0.0, (by2 - by1))
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def xywhn_to_xyxy(x: float, y: float, w: float, h: float, img_w: int, img_h: int) -> Box:
    cx = x * img_w
    cy = y * img_h
    ww = w * img_w
    hh = h * img_h
    x1 = max(0.0, cx - ww / 2.0)
    y1 = max(0.0, cy - hh / 2.0)
    x2 = min(float(img_w - 1), cx + ww / 2.0)
    y2 = min(float(img_h - 1), cy + hh / 2.0)
    return (x1, y1, x2, y2)


def load_gt_labels(labels_dir: Path, image_path: Path, img_w: int, img_h: int) -> List[GT]:
    stem = image_path.stem
    label_path = labels_dir / f"{stem}.txt"
    gts: List[GT] = []
    if not label_path.exists():
        return gts
    try:
        with open(label_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5 and len(parts) != 6:
                    # YOLO format: cls x y w h [conf]
                    continue
                cls = int(float(parts[0]))
                x, y, w, h = map(float, parts[1:5])
                gts.append((xywhn_to_xyxy(x, y, w, h, img_w, img_h), cls))
    except Exception:
        pass
    return gts


def match_tp_fp(preds: List[Pred], gts: List[GT], iou_thresh: float = 0.5) -> Tuple[List[float], List[float], Dict[int, int]]:
    """
    Greedy match predictions to GTs per class.
    Returns:
      tp_conf: confidences for matched true positives (all classes pooled)
      fp_conf: confidences for false positives (all classes pooled)
      fn_by_cls: remaining unmatched GT count per class
    """
    used = [False] * len(gts)
    tp_conf: List[float] = []
    fp_conf: List[float] = []
    fn_by_cls: Dict[int, int] = {}

    # sort preds by conf desc for stable matching
    order = sorted(range(len(preds)), key=lambda i: preds[i][1], reverse=True)
    for i in order:
        box_p, conf, cls_p = preds[i]
        best_j = -1
        best_iou = 0.0
        for j, (box_g, cls_g) in enumerate(gts):
            if used[j] or cls_g != cls_p:
                continue
            iou = iou_xyxy(box_p, box_g)
            if iou >= iou_thresh and iou > best_iou:
                best_iou = iou
                best_j = j
        if best_j >= 0:
            used[best_j] = True
            tp_conf.append(float(conf))
        else:
            fp_conf.append(float(conf))

    # count FN by class
    for k, u in enumerate(used):
        if not u:
            _, cls_g = gts[k]
            fn_by_cls[cls_g] = fn_by_cls.get(cls_g, 0) + 1
    return tp_conf, fp_conf, fn_by_cls


def compute_per_class_thresholds(
    weights: Path,
    data_yaml: Path,
    imgsz: int = 640,
    iou_match: float = 0.5,
    candidates: List[float] = None,
    device: str = "auto",
    optimize: str = "f1",
    target_precision: float = 0.95,
    dump_curves: bool = False,
) -> Dict[str, object]:
    if candidates is None:
        candidates = [round(x, 2) for x in np.linspace(0.2, 0.9, 15)]

    # Load dataset paths
    import yaml
    with open(data_yaml, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    base = Path(cfg.get("path", "."))
    val_images = base / cfg.get("val", "valid/images")
    val_labels = base / "valid/labels"

    # Model
    model = YOLO(str(weights))
    names = model.model.names if hasattr(model.model, "names") else {}

    # Collect per-class stats
    tp_by_cls: Dict[int, List[float]] = {}
    fp_by_cls: Dict[int, List[float]] = {}
    fn_total_by_cls: Dict[int, int] = {}
    img_paths = list(val_images.glob("**/*.*"))
    img_paths = [p for p in img_paths if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]
    for img_path in img_paths:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        gts = load_gt_labels(val_labels, img_path, w, h)
        # run prediction at very low conf to get as many candidates as possible
        res = model.predict(
            source=img,
            imgsz=imgsz,
            conf=0.001,
            iou=0.65,
            device=device,
            half=(device == "cuda"),
            max_det=300,
            verbose=False,
        )[0]
        preds: List[Pred] = []
        if res.boxes is not None and len(res.boxes) > 0:
            xyxy = res.boxes.xyxy.cpu().numpy()
            confs = res.boxes.conf.cpu().numpy()
            clss = res.boxes.cls.cpu().numpy()
            for i in range(len(xyxy)):
                preds.append((tuple(map(float, xyxy[i])), float(confs[i]), int(clss[i])))

        # match
        tps, fps, fn_by_cls = match_tp_fp(preds, gts, iou_thresh=iou_match)
        # distribute tp/fp by class from preds matches
        # rebuild per-class using matched info by re-matching to collect per cls splits
        # Simpler: split by class label of prediction; tps and fps arrays hold conf only. We'll do class-specific by recomputing quickly:
        # create dict predictions by class
        by_cls: Dict[int, List[Tuple[Box, float]]] = {}
        for b, c, k in [(p[0], p[1], p[2]) for p in preds]:
            by_cls.setdefault(k, []).append((b, c))
        # For each class, run class-specific matching for detailed tp/fp conf lists
        for c, items in by_cls.items():
            # prepare filtered lists
            preds_c: List[Pred] = [(b, conf, c) for (b, conf) in items]
            gts_c: List[GT] = [(b, cc) for (b, cc) in gts if cc == c]
            tpc, fpc, fnc = match_tp_fp(preds_c, gts_c, iou_thresh=iou_match)
            if tpc:
                tp_by_cls.setdefault(c, []).extend(tpc)
            if fpc:
                fp_by_cls.setdefault(c, []).extend(fpc)
            if fnc:
                fn_total_by_cls[c] = fn_total_by_cls.get(c, 0) + sum(fnc.values())

    # Compute thresholds via sweep per class
    thresholds_by_id: Dict[int, float] = {}
    metrics_by_id: Dict[int, Dict[str, float]] = {}
    curves_by_id: Dict[int, Dict[str, List[float]]] = {}
    for c in set(list(tp_by_cls.keys()) + list(fp_by_cls.keys()) + list(fn_total_by_cls.keys())):
        tp_conf = np.array(tp_by_cls.get(c, []), dtype=float)
        fp_conf = np.array(fp_by_cls.get(c, []), dtype=float)
        fn = int(fn_total_by_cls.get(c, 0))
        if tp_conf.size + fp_conf.size + fn == 0:
            continue
        if dump_curves:
            precs: List[float] = []
            recs: List[float] = []
            f1s: List[float] = []
            for th in candidates:
                tp = int((tp_conf >= th).sum())
                fp = int((fp_conf >= th).sum())
                fn_eff = fn + int((tp_conf < th).sum())
                p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                r = tp / (tp + fn_eff) if (tp + fn_eff) > 0 else 0.0
                f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
                precs.append(float(p))
                recs.append(float(r))
                f1s.append(float(f1))
            curves_by_id[c] = {
                "thresholds": [float(t) for t in candidates],
                "precision": precs,
                "recall": recs,
                "f1": f1s,
            }
        if optimize == "precision":
            # choose minimal threshold that achieves >= target_precision
            chosen = None
            best_p = 0.0
            best_r = 0.0
            for th in sorted(candidates):
                tp = int((tp_conf >= th).sum())
                fp = int((fp_conf >= th).sum())
                fn_eff = fn + int((tp_conf < th).sum())
                p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                r = tp / (tp + fn_eff) if (tp + fn_eff) > 0 else 0.0
                if p >= target_precision:
                    chosen = float(th)
                    best_p, best_r = float(p), float(r)
                    break
            if chosen is None:
                # fallback to maximizing F1
                best_f1 = -1.0
                chosen = 0.45
                for th in candidates:
                    tp = int((tp_conf >= th).sum())
                    fp = int((fp_conf >= th).sum())
                    fn_eff = fn + int((tp_conf < th).sum())
                    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                    r = tp / (tp + fn_eff) if (tp + fn_eff) > 0 else 0.0
                    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
                    if f1 > best_f1:
                        best_f1 = f1
                        chosen = float(th)
                        best_p, best_r = float(p), float(r)
            thresholds_by_id[c] = float(chosen)
            metrics_by_id[c] = {"precision": best_p, "recall": best_r}
        else:
            best_f1 = -1.0
            best_th = 0.45
            best_p = 0.0
            best_r = 0.0
            for th in candidates:
                tp = int((tp_conf >= th).sum())
                fp = int((fp_conf >= th).sum())
                # fn: GT missing + TPs below threshold
                fn_eff = fn + int((tp_conf < th).sum())
                p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                r = tp / (tp + fn_eff) if (tp + fn_eff) > 0 else 0.0
                f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
                if f1 > best_f1:
                    best_f1, best_th, best_p, best_r = f1, float(th), float(p), float(r)
            thresholds_by_id[c] = float(best_th)
            metrics_by_id[c] = {"f1": best_f1 if best_f1 >= 0 else 0.0, "precision": best_p, "recall": best_r}

    # Build name maps
    id_to_name: Dict[int, str] = {}
    if isinstance(names, dict):
        for k, v in names.items():
            try:
                id_to_name[int(k)] = str(v)
            except Exception:
                pass

    name_thresholds = {id_to_name.get(i, str(i)): th for i, th in thresholds_by_id.items()}

    return {
        "thresholds_by_id": thresholds_by_id,
        "thresholds_by_name": name_thresholds,
        "metrics_by_id": metrics_by_id,
        "class_names": id_to_name,
        **({"curves_by_id": curves_by_id} if dump_curves else {}),
    }



def main():
    ap = argparse.ArgumentParser(description="Recommend per-class confidence thresholds using validation set")
    ap.add_argument("--weights", type=str, default=str(Path("runs/train/exp_y12s_50e_640/weights/best.pt")), help="Path to model weights .pt")
    ap.add_argument("--data", type=str, default="configs/data.yaml", help="Path to data.yaml")
    ap.add_argument("--imgsz", type=int, default=640, help="Inference size for evaluation")
    ap.add_argument("--iou-match", type=float, default=0.5, help="IoU threshold for TP/FP matching")
    ap.add_argument("--device", type=str, default="auto", help="cpu|cuda|auto")
    ap.add_argument("--out", type=str, default="runs/analysis/per_class_thresholds.json", help="Where to save JSON map")
    ap.add_argument("--optimize", type=str, default="f1", choices=["f1", "precision"], help="Optimize threshold for: F1 or target precision")
    ap.add_argument("--target-precision", type=float, default=0.95, help="Target precision if --optimize precision")
    ap.add_argument("--dump-pr-curves", action="store_true", help="Include per-class precision/recall/F1 values for each threshold candidate in JSON output")
    args = ap.parse_args()

    res = compute_per_class_thresholds(
        weights=Path(args.weights),
        data_yaml=Path(args.data),
        imgsz=int(args.imgsz),
        iou_match=float(args.iou_match),
        device=str(args.device),
        optimize=str(args.optimize),
        target_precision=float(args.target_precision),
        dump_curves=bool(args.dump_pr_curves),
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    print(f"[per-class] Saved thresholds -> {out_path}")


if __name__ == "__main__":
    main()

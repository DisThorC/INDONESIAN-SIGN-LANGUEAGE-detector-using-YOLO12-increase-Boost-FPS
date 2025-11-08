from collections import deque
from typing import Deque, Dict, List, Optional, Tuple


def iou(a: List[float], b: List[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter + 1e-6
    return float(inter / union) if union > 0 else 0.0


class SimpleTracker:
    """
    Lightweight IOU-based tracker with optional temporal smoothing of class/conf.

    update(dets) accepts a list of (bbox[x1,y1,x2,y2], conf, cls) and returns mapping:
        track_id -> (bbox, conf, cls)
    """

    def __init__(
        self,
        iou_thresh: float = 0.3,
        max_miss: int = 30,
        smooth_window: int = 0,
        semantic_prior: Optional[Dict[Tuple[int, int], float]] = None,
        semantic_threshold: float = -1.0,
        default_semantic_prob: float = 0.05,
    ):
        self.iou_thresh = float(iou_thresh)
        self.max_miss = int(max_miss)
        self.next_id = 1
        self.tracks: Dict[int, Dict] = {}
        self.smooth_window = int(max(0, smooth_window))
        # Semantic smoothing (Markov style transition prior)
        self.semantic_prior = semantic_prior or None
        self.semantic_threshold = float(semantic_threshold)
        self.default_semantic_prob = float(default_semantic_prob)

    def _assign(self, dets: List[Tuple[List[float], float, int]]):
        assigned: Dict[int, Tuple[List[float], float, int]] = {}
        used = set()
        # Greedy IOU matching
        for tid, t in list(self.tracks.items()):
            best_iou = -1.0
            best_j = -1
            for j, d in enumerate(dets):
                if j in used:
                    continue
                i = iou(t["bbox"], d[0])
                if i > best_iou:
                    best_iou = i
                    best_j = j
            if best_j >= 0 and best_iou >= self.iou_thresh:
                b, conf, cls = dets[best_j]
                t.update({"bbox": b, "miss": 0})
                # history for smoothing
                if self.smooth_window > 0:
                    t.setdefault("cls_hist", deque(maxlen=self.smooth_window)).append(int(cls))
                    t.setdefault("conf_hist", deque(maxlen=self.smooth_window)).append(float(conf))
                    voted_cls = self._vote_cls(t["cls_hist"])  # type: ignore[arg-type]
                    avg_conf = float(sum(t["conf_hist"]) / len(t["conf_hist"]))  # type: ignore[arg-type]
                    # Semantic smoothing decision
                    prev_cls = t.get("cls", voted_cls)
                    final_cls = voted_cls
                    if (
                        self.semantic_prior is not None
                        and self.semantic_threshold >= 0.0
                        and prev_cls is not None
                        and voted_cls != prev_cls
                    ):
                        prob = self.semantic_prior.get((int(prev_cls), int(voted_cls)), self.default_semantic_prob)
                        # If transition probability below threshold, keep previous class
                        if prob < self.semantic_threshold:
                            final_cls = int(prev_cls)
                    cls = int(final_cls)
                    conf = avg_conf
                else:
                    t["cls"] = int(cls)
                    t["conf"] = float(conf)
                # Update track's current class/conf after semantic smoothing
                t["cls"] = int(cls)
                t["conf"] = float(conf)
                assigned[tid] = (b, conf, cls)
                used.add(best_j)
            else:
                t["miss"] = t.get("miss", 0) + 1
                if t["miss"] > self.max_miss:
                    del self.tracks[tid]

        # New tracks for unmatched detections
        for j, d in enumerate(dets):
            if j in used:
                continue
            b, conf, cls = d
            tid = self.next_id
            self.next_id += 1
            self.tracks[tid] = {
                "bbox": b,
                "cls": int(cls),
                "conf": float(conf),
                "miss": 0,
                "cls_hist": deque([int(cls)], maxlen=self.smooth_window) if self.smooth_window > 0 else None,
                "conf_hist": deque([float(conf)], maxlen=self.smooth_window) if self.smooth_window > 0 else None,
            }
            assigned[tid] = (b, float(conf), int(cls))

        return assigned

    @staticmethod
    def _vote_cls(hist: Deque[int]) -> int:
        # Majority vote; on ties keep most recent
        if not hist:
            return 0
        counts: Dict[int, int] = {}
        for c in hist:
            counts[c] = counts.get(c, 0) + 1
        max_cnt = max(counts.values())
        candidates = [c for c, v in counts.items() if v == max_cnt]
        # pick most recent among candidates
        for c in reversed(hist):
            if c in candidates:
                return c
        return hist[-1]

    def update(self, dets: List[Tuple[List[float], float, int]]):
        return self._assign(dets)

    def set_smooth_window(self, n: int) -> None:
        """Dynamically change smoothing window size.

        Rebuilds internal deques with the new maxlen while preserving the most recent items.
        """
        n = int(max(0, n))
        if n == self.smooth_window:
            return
        self.smooth_window = n
        # Rebuild histories to the new window size
        for t in self.tracks.values():
            cls_hist = t.get("cls_hist")
            conf_hist = t.get("conf_hist")
            if n == 0:
                t["cls_hist"] = None
                t["conf_hist"] = None
                # keep last assigned instantaneous values
                continue
            # Create new deques with limited last n items
            if cls_hist is not None:
                from collections import deque

                new_cls = deque(list(cls_hist)[-n:], maxlen=n)
                t["cls_hist"] = new_cls
            else:
                from collections import deque

                t["cls_hist"] = deque(maxlen=n)
            if conf_hist is not None:
                from collections import deque

                new_conf = deque(list(conf_hist)[-n:], maxlen=n)
                t["conf_hist"] = new_conf
            else:
                from collections import deque

                t["conf_hist"] = deque(maxlen=n)

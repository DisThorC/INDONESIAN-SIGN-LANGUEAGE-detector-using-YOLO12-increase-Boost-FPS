"""
Sequence recognition stub utilities.

Goal: Aggregate per-frame detections into stable gesture tokens with durations.
This is a placeholder module for future temporal modeling (e.g., HMM/LSTM/TCN).
"""
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class Token:
    cls: int
    start_t: float
    end_t: float

    @property
    def duration(self) -> float:
        return self.end_t - self.start_t


class SequenceAggregator:
    """Aggregate frame-wise class predictions into tokens by dwell time.

    push(cls_id, t) with the top predicted class id and its timestamp.
    When a class changes and prior class met min_dwell seconds, a Token is emitted.
    """

    def __init__(self, min_dwell: float = 0.6):
        self.min_dwell = float(min_dwell)
        self._cur_cls = None
        self._cur_start = 0.0
        self._last_t = 0.0
        self.tokens: List[Token] = []

    def push(self, cls_id: int, t: float) -> None:
        if self._cur_cls is None:
            self._cur_cls = cls_id
            self._cur_start = t
            self._last_t = t
            return
        if cls_id == self._cur_cls:
            self._last_t = t
            return
        # class changed
        if (self._last_t - self._cur_start) >= self.min_dwell:
            self.tokens.append(Token(self._cur_cls, self._cur_start, self._last_t))
        self._cur_cls = cls_id
        self._cur_start = t
        self._last_t = t

    def flush(self) -> List[Token]:
        # emit the last token if it meets dwell time
        if self._cur_cls is not None and (self._last_t - self._cur_start) >= self.min_dwell:
            self.tokens.append(Token(self._cur_cls, self._cur_start, self._last_t))
        out = self.tokens
        self.tokens = []
        self._cur_cls = None
        return out


def demo_from_pairs(pairs: List[Tuple[int, float]], min_dwell: float = 0.6) -> List[Token]:
    agg = SequenceAggregator(min_dwell=min_dwell)
    for cls_id, t in pairs:
        agg.push(cls_id, t)
    return agg.flush()

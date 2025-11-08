from __future__ import annotations

from typing import Dict, Iterable, List, Tuple


class MarkovDecoder:
    """
    Simple first-order Markov decoder over discrete classes.
    Given a prior transition probability P(next|prev) and (optional) emission weights,
    it decodes the best sequence using a greedy or short beam approach.

    For online smoothing, we provide `step(prev, candidates)` to pick the next class
    based on transition priors and candidate scores.
    """

    def __init__(self, transition: Dict[Tuple[int, int], float], default_prob: float = 0.05):
        self.trans = transition
        self.default = float(default_prob)

    def prob(self, a: int, b: int) -> float:
        return float(self.trans.get((int(a), int(b)), self.default))

    def step(self, prev: int, candidates: Iterable[Tuple[int, float]]) -> int:
        """Pick next class maximizing P(b|prev) * score(b).

        candidates: iterable of (class_id, score) where score in [0,1] or any positive weight.
        Returns selected class id.
        """
        best_cls = None
        best_val = -1.0
        for c, s in candidates:
            val = self.prob(prev, int(c)) * float(s)
            if val > best_val:
                best_val = val
                best_cls = int(c)
        # fallback: if all 0, pick highest score
        if best_cls is None:
            for c, s in candidates:
                if best_cls is None or s > best_val:
                    best_cls = int(c)
                    best_val = float(s)
        return int(best_cls) if best_cls is not None else int(prev)

    def decode(self, seq: List[int]) -> List[int]:
        """Offline smoothing: given raw sequence of class ids, apply Markov prior greedily.
        Output sequence of same length (smoothed).
        """
        if not seq:
            return []
        out = [int(seq[0])]
        for c in seq[1:]:
            prev = out[-1]
            # compare staying vs switching to c
            stay = self.prob(prev, prev)
            go = self.prob(prev, int(c))
            out.append(prev if stay >= go else int(c))
        return out

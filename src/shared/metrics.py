"""Local re-implementations of the HECKTOR 2026 leaderboard metrics, so we can
validate faithfully offline.

  - Segmentation: aggregated Dice (DSCagg) — pooled across the whole set, the
    official ranking metric (NOT per-case mean Dice).
  - Lesion-level F1 (IoU>=0.30 grouping) — the 2022/HNTS node-detection metric;
    useful even if 2026 ranks segmentation by DSCagg, since node detection
    drives the TN-staging signal.
  - Survival: Harrell C-index and integrated Brier score.
"""
from __future__ import annotations
import numpy as np
from scipy import ndimage


# ----------------------------------------------------------------------------
# Aggregated Dice
# ----------------------------------------------------------------------------
class AggregatedDice:
    """DSCagg = 2*sum_i|A_i ∩ B_i| / sum_i(|A_i|+|B_i|), accumulated over cases."""

    def __init__(self, classes=(1, 2)):
        self.classes = classes
        self.inter = {c: 0 for c in classes}
        self.denom = {c: 0 for c in classes}

    def update(self, pred: np.ndarray, gt: np.ndarray):
        for c in self.classes:
            p, g = pred == c, gt == c
            self.inter[c] += int(np.logical_and(p, g).sum())
            self.denom[c] += int(p.sum() + g.sum())

    def result(self) -> dict:
        out = {}
        for c in self.classes:
            out[c] = (2.0 * self.inter[c] / self.denom[c]) if self.denom[c] else 1.0
        out["mean"] = float(np.mean([out[c] for c in self.classes]))
        return out


# ----------------------------------------------------------------------------
# Lesion-level F1 with IoU>=thr grouping (union-find over overlapping CCs)
# ----------------------------------------------------------------------------
class _UF:
    def __init__(self, n): self.p = list(range(n))
    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]; a = self.p[a]
        return a
    def union(self, a, b): self.p[self.find(a)] = self.find(b)


class LesionF1:
    """Group overlapping predicted + GT connected components; a group is a TP
    if its composite IoU >= ``thr`` (default 0.30), else its GT components are
    FN and predicted components are FP. Aggregated F1 = 2TP/(2TP+FP+FN)."""

    def __init__(self, label=2, thr=0.30):
        self.label, self.thr = label, thr
        self.tp = self.fp = self.fn = 0

    def update(self, pred: np.ndarray, gt: np.ndarray):
        g_cc, ng = ndimage.label(gt == self.label)
        p_cc, npd = ndimage.label(pred == self.label)
        if ng == 0 and npd == 0:
            return
        # nodes: 0..ng-1 are GT, ng..ng+npd-1 are pred
        uf = _UF(ng + npd)
        overlap = (g_cc > 0) & (p_cc > 0)
        for gi, pi in set(zip(g_cc[overlap].tolist(), p_cc[overlap].tolist())):
            uf.union(gi - 1, ng + pi - 1)
        groups: dict[int, list[int]] = {}
        for n in range(ng + npd):
            groups.setdefault(uf.find(n), []).append(n)
        for members in groups.values():
            g_ids = [m + 1 for m in members if m < ng]
            p_ids = [m - ng + 1 for m in members if m >= ng]
            gmask = np.isin(g_cc, g_ids) if g_ids else np.zeros_like(g_cc, bool)
            pmask = np.isin(p_cc, p_ids) if p_ids else np.zeros_like(p_cc, bool)
            if not g_ids:          # pure prediction -> false positive(s)
                self.fp += 1; continue
            if not p_ids:          # missed GT -> false negative(s)
                self.fn += 1; continue
            inter = np.logical_and(gmask, pmask).sum()
            union = np.logical_or(gmask, pmask).sum()
            iou = inter / union if union else 0.0
            if iou >= self.thr:
                self.tp += 1
            else:                  # sub-threshold match: both wrong
                self.fp += 1; self.fn += 1

    def result(self) -> dict:
        denom = 2 * self.tp + self.fp + self.fn
        f1 = (2 * self.tp / denom) if denom else 1.0
        return {"f1": f1, "tp": self.tp, "fp": self.fp, "fn": self.fn}


# ----------------------------------------------------------------------------
# Survival
# ----------------------------------------------------------------------------
def c_index(time, event, risk) -> float:
    """Harrell C-index. ``risk`` higher = higher risk (concordant w/ sksurv)."""
    from sksurv.metrics import concordance_index_censored
    event = np.asarray(event).astype(bool)
    return float(concordance_index_censored(event, np.asarray(time), np.asarray(risk))[0])

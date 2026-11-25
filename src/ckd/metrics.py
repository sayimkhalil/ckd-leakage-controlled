"""Discrimination, calibration and uncertainty metrics."""
from __future__ import annotations

import numpy as np
from scipy.stats import norm
from sklearn.metrics import (brier_score_loss, confusion_matrix, f1_score,
                             precision_score, roc_auc_score)

from .config import N_BOOT_AUC, SEED, THRESHOLD


def bootstrap_auc_ci(y, p, n_boot: int = N_BOOT_AUC, seed: int = SEED,
                     alpha: float = 0.05):
    """Nonparametric percentile bootstrap of test-set predictions."""
    y, p = np.asarray(y), np.asarray(p)
    rng = np.random.default_rng(seed)
    aucs = []
    n = len(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y[idx], p[idx]))
    lo, hi = np.percentile(aucs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def wilson_ci(k: int, n: int, alpha: float = 0.05):
    """Wilson score interval for a binomial proportion (Wilson, 1927)."""
    z = norm.ppf(1 - alpha / 2)
    phat = k / n
    denom = 1 + z ** 2 / n
    centre = (phat + z ** 2 / (2 * n)) / denom
    half = z * np.sqrt(phat * (1 - phat) / n + z ** 2 / (4 * n ** 2)) / denom
    return float(centre - half), float(centre + half)


def evaluate(y, p, threshold: float = THRESHOLD) -> dict:
    y, p = np.asarray(y), np.asarray(p)
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    n = len(y)
    acc = (tp + tn) / n
    auc = roc_auc_score(y, p)
    auc_lo, auc_hi = bootstrap_auc_ci(y, p)
    acc_lo, acc_hi = wilson_ci(int(tp + tn), n)
    return {
        "n": n, "auc": auc, "auc_lo": auc_lo, "auc_hi": auc_hi,
        "accuracy": acc, "acc_lo": acc_lo, "acc_hi": acc_hi,
        "sensitivity": tp / (tp + fn) if tp + fn else np.nan,
        "specificity": tn / (tn + fp) if tn + fp else np.nan,
        "precision": precision_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "brier": brier_score_loss(y, p),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
    }

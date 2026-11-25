"""Shared fitting, cross-validation, SHAP and plotting helpers."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402
from sklearn.model_selection import StratifiedKFold, cross_val_score  # noqa: E402

from .config import N_FOLDS, SEED  # noqa: E402
from .metrics import evaluate  # noqa: E402

plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 300,
                     "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False})


def cv_auc(pipe, X, y):
    """Stratified five-fold CV AUC on the training partition. The whole
    pipeline (including SMOTE) is refit inside each fold."""
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    s = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc")
    return float(s.mean()), float(s.std())


def fit_evaluate(pipe, X_tr, y_tr, X_te, y_te, with_cv: bool = True):
    pipe.fit(X_tr, y_tr)
    p = pipe.predict_proba(X_te)[:, 1]
    res = evaluate(y_te, p)
    if with_cv:
        res["cv_auc_mean"], res["cv_auc_sd"] = cv_auc(pipe, X_tr, y_tr)
    return res, p


# SHAP -----------------------------------------------------------------------
def _as_2d(values):
    if isinstance(values, list):          # older SHAP: one array per class
        values = values[1]
    values = np.asarray(values)
    if values.ndim == 3:                  # (n, features, classes)
        values = values[..., 1]
    return values


def shap_importance(pipe, X_tr, X_te, model: str) -> pd.Series:
    """Mean |SHAP| per selected feature on the test set.

    The explainer acts on the transformed (imputed, scaled, selected) design
    matrix so that features keep their clinical identity. The LR background
    is the transformed training matrix (before SMOTE)."""
    front = pipe[:2]                       # prep + reduce, already fit
    names = list(front.get_feature_names_out())
    Xtr_t = front.transform(X_tr)
    Xte_t = front.transform(X_te)
    clf = pipe.named_steps["clf"]
    if model == "LightGBM":
        vals = shap.TreeExplainer(clf).shap_values(Xte_t)
    else:
        masker = shap.maskers.Independent(Xtr_t, max_samples=len(Xtr_t))
        vals = shap.LinearExplainer(clf, masker).shap_values(Xte_t)
    vals = _as_2d(vals)
    return pd.Series(np.abs(vals).mean(axis=0), index=names).sort_values(
        ascending=False)


def shap_agreement(imp_a: pd.Series, imp_b: pd.Series):
    shared = imp_a.index.intersection(imp_b.index)
    rho, p = spearmanr(imp_a[shared], imp_b[shared])
    return float(rho), float(p), len(shared)


# Plots ----------------------------------------------------------------------
def plot_roc(curves: dict, path, title=None):
    fig, ax = plt.subplots(figsize=(5, 5))
    for label, (y, p, auc) in curves.items():
        fpr, tpr, _ = roc_curve(y, p)
        ax.plot(fpr, tpr, lw=1.6, label=f"{label} (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("False positive rate (1 − specificity)")
    ax.set_ylabel("True positive rate (sensitivity)")
    if title:
        ax.set_title(title)
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_calibration(curves: dict, path, n_bins: int = 10):
    fig, ax = plt.subplots(figsize=(5, 5))
    for label, (y, p, brier) in curves.items():
        frac, mean_pred = calibration_curve(y, p, n_bins=n_bins,
                                            strategy="uniform")
        ax.plot(mean_pred, frac, "o-", ms=4, lw=1.4,
                label=f"{label} (Brier = {brier:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="Perfect calibration")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed fraction of CKD")
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_shap_bar(imp: pd.Series, labels: dict, path, title=None):
    imp = imp.sort_values()
    fig, ax = plt.subplots(figsize=(6, 0.35 * len(imp) + 1))
    ax.barh([labels.get(i, i) for i in imp.index], imp.values,
            color="#3b6ea5")
    ax.set_xlabel("Mean |SHAP value|")
    if title:
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_auc_bars(df: pd.DataFrame, path):
    """df columns: configuration, model, auc, auc_lo, auc_hi."""
    configs = list(dict.fromkeys(df["configuration"]))
    models = list(dict.fromkeys(df["model"]))
    x = np.arange(len(configs))
    w = 0.8 / len(models)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for i, m in enumerate(models):
        sub = df[df["model"] == m].set_index("configuration").loc[configs]
        err = np.vstack([sub["auc"] - sub["auc_lo"],
                         sub["auc_hi"] - sub["auc"]])
        ax.bar(x + (i - (len(models) - 1) / 2) * w, sub["auc"], w,
               yerr=err, capsize=3, label=m)
    ax.set_xticks(x, configs, fontsize=8)
    ax.set_ylabel("Test-set AUC (95% CI)")
    lo = max(0.0, float(df["auc_lo"].min()) - 0.02)
    ax.set_ylim(lo, 1.005)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)

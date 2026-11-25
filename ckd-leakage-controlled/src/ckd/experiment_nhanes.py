"""NHANES 2017-2018 external validation: Table 4 and Figures 7-8.

The identical leakage-controlled protocol is applied to two feature sets:
  * controlled   - pre-renal-lab cardiometabolic and demographic features only
  * leakage-prone - controlled features + the six label-defining or
                    label-correlated renal variables
"""
from __future__ import annotations

import json

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import FIGURES, SEED, TABLES, TEST_SIZE, ensure_dirs
from .data_nhanes import (CONTROLLED_FEATURES, LABELS, LEAKAGE_FEATURES,
                          load_nhanes)
from .evaluation import (fit_evaluate, plot_roc, plot_shap_bar,
                         shap_agreement, shap_importance)
from .pipelines import make_pipeline

MODELS = ["LR", "LightGBM"]
ARMS = {"Controlled": CONTROLLED_FEATURES,
        "Leakage-prone": CONTROLLED_FEATURES + LEAKAGE_FEATURES}


def run(cohort: pd.DataFrame | None = None) -> dict:
    ensure_dirs()
    df = load_nhanes() if cohort is None else cohort
    y = df["ckd"].astype(int)
    idx_tr, idx_te = train_test_split(df.index, test_size=TEST_SIZE,
                                      stratify=y, random_state=SEED)
    summary = {
        "n": int(len(df)), "n_ckd": int(y.sum()),
        "prevalence": float(y.mean()),
        "pct_egfr_lt60": float((df["egfr"] < 60).mean()),
        "pct_uacr_ge30": float((df["uacr"] >= 30).mean()),
        "n_test": int(len(idx_te)), "n_test_ckd": int(y[idx_te].sum()),
    }

    rows, roc, imp = [], {}, {}
    for arm, feats in ARMS.items():
        X = df[feats]
        X_tr, X_te = X.loc[idx_tr], X.loc[idx_te]
        y_tr, y_te = y.loc[idx_tr], y.loc[idx_te]
        for m in MODELS:
            pipe = make_pipeline(m, "FS", feats, [], smote=True)
            res, p = fit_evaluate(pipe, X_tr, y_tr, X_te, y_te)
            rows.append({"configuration": arm, "model": m, **res,
                         "selected_features": ";".join(
                             pipe[:2].get_feature_names_out())})
            roc[f"{arm} — {m}"] = (y_te, p, res["auc"])
            if arm == "Controlled":
                imp[m] = shap_importance(pipe, X_tr, X_te, m)

    t4 = pd.DataFrame(rows)
    t4.to_csv(TABLES / "table4_nhanes_external_validation.csv", index=False)
    plot_roc(roc, FIGURES / "figure7_nhanes_roc.png")
    plot_shap_bar(imp["LightGBM"], LABELS,
                  FIGURES / "figure8_nhanes_shap_lightgbm.png")
    pd.DataFrame(imp).to_csv(TABLES / "shap_mean_abs_nhanes_controlled.csv")

    auc = t4.set_index(["configuration", "model"])["auc"]
    summary["auc_inflation"] = {
        m: float(auc[("Leakage-prone", m)] - auc[("Controlled", m)])
        for m in MODELS}
    rho, pval, k = shap_agreement(imp["LR"], imp["LightGBM"])
    summary["shap_spearman_controlled"] = {"rho": rho, "p": pval,
                                           "n_features": k}
    with open(TABLES / "summary_nhanes.json", "w") as f:
        json.dump(summary, f, indent=2)
    return summary


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))

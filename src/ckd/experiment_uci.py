"""UCI Chronic Kidney Disease analysis: Tables 1-3 and Figures 1-6."""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split

from .config import (FIGURES, K_FEATURES, N_BOOT_STABILITY, SEED, TABLES,
                     TEST_SIZE, ensure_dirs)
from .data_uci import CATEGORICAL, LABELS, NUMERIC, load_uci
from .evaluation import (fit_evaluate, plot_auc_bars, plot_calibration,
                         plot_roc, plot_shap_bar, shap_agreement,
                         shap_importance)
from .metrics import evaluate
from .pipelines import (make_model_only, make_pipeline, make_preprocessor,
                        selected_feature_names)

MODELS = ["LR", "LightGBM"]


def split(X, y):
    return train_test_split(X, y, test_size=TEST_SIZE, stratify=y,
                            random_state=SEED)


# Table 1 --------------------------------------------------------------------
def feature_selection_table(X_tr, y_tr) -> pd.DataFrame:
    prep = make_preprocessor(NUMERIC, CATEGORICAL).fit(X_tr)
    Xt = prep.transform(X_tr)
    names = prep.get_feature_names_out()
    sel = SelectKBest(f_classif, k=K_FEATURES).fit(Xt, y_tr)
    scores = pd.Series(sel.scores_, index=names)

    rng = np.random.default_rng(SEED)
    counts = pd.Series(0, index=names, dtype=float)
    n = len(X_tr)
    for _ in range(N_BOOT_STABILITY):
        idx = rng.integers(0, n, n)
        Xb, yb = X_tr.iloc[idx], y_tr.iloc[idx]
        pb = make_preprocessor(NUMERIC, CATEGORICAL).fit(Xb)
        sb = SelectKBest(f_classif, k=K_FEATURES).fit(pb.transform(Xb), yb)
        counts[names[sb.get_support()]] += 1
    freq = counts / N_BOOT_STABILITY

    top = scores.sort_values(ascending=False).head(K_FEATURES).index
    return pd.DataFrame({"feature": [LABELS[c] for c in top],
                         "anova_f": scores[top].round(1).values,
                         "selection_frequency": freq[top].round(2).values})


# Leakage-prone comparators --------------------------------------------------
def leaky_preprocess_full(X, y):
    """Variant A/B front end: imputation, scaling and ANOVA selection are fit
    on the FULL dataset before partitioning (deliberate leakage)."""
    prep = make_preprocessor(NUMERIC, CATEGORICAL).fit(X)
    Xt = prep.transform(X)
    sel = SelectKBest(f_classif, k=K_FEATURES).fit(Xt, y)
    return sel.transform(Xt)


def run() -> dict:
    ensure_dirs()
    X, y = load_uci()
    X_tr, X_te, y_tr, y_te = split(X, y)
    summary = {"n": int(len(y)), "n_ckd": int(y.sum()),
               "n_test": int(len(y_te)), "n_test_ckd": int(y_te.sum()),
               "missingness": X.isna().mean().round(3).to_dict()}

    # Figure 1: class distribution
    fig, ax = plt.subplots(figsize=(4, 3.5))
    counts = y.value_counts().reindex([1, 0])
    ax.bar(["CKD", "Non-CKD"], counts.values, color=["#b03a2e", "#3b6ea5"])
    for i, v in enumerate(counts.values):
        ax.text(i, v + 4, str(v), ha="center")
    ax.set_ylabel("Patients")
    fig.tight_layout()
    fig.savefig(FIGURES / "figure1_class_distribution.png")
    plt.close(fig)

    # Table 1
    t1 = feature_selection_table(X_tr, y_tr)
    t1.to_csv(TABLES / "table1_feature_selection_stability.csv", index=False)

    # Table 2 + Figure 2: leakage-controlled, 4 combinations
    rows, roc, preds, pipes = [], {}, {}, {}
    for reducer in ["FS", "PCA"]:
        for m in MODELS:
            pipe = make_pipeline(m, reducer, NUMERIC, CATEGORICAL, smote=True)
            res, p = fit_evaluate(pipe, X_tr, y_tr, X_te, y_te)
            label = f"{m}, {'feature selection' if reducer == 'FS' else 'PCA'}"
            rows.append({"model_pipeline": label, **res})
            roc[label] = (y_te, p, res["auc"])
            preds[(m, reducer)] = p
            pipes[(m, reducer)] = pipe
            if reducer == "PCA":
                summary[f"pca_components_{m}"] = int(
                    pipe.named_steps["reduce"].n_components_)
    t2 = pd.DataFrame(rows)
    t2.to_csv(TABLES / "table2_controlled_performance.csv", index=False)
    plot_roc(roc, FIGURES / "figure2_roc_controlled.png")
    summary["selected_features"] = selected_feature_names(pipes[("LR", "FS")])

    # Table 3 + Figure 3: leakage-prone comparators and no-SMOTE
    t3 = []
    for m in MODELS:
        r = t2[t2["model_pipeline"] == f"{m}, feature selection"].iloc[0]
        t3.append({"configuration": "Controlled", "model": m,
                   **r.drop("model_pipeline").to_dict()})

    # Variant A: preprocessing on full data, same 80 real test patients
    Xa = leaky_preprocess_full(X, y)
    pos_tr = X.index.get_indexer(X_tr.index)
    pos_te = X.index.get_indexer(X_te.index)
    for m in MODELS:
        pipe = make_model_only(m, smote=True).fit(Xa[pos_tr], y_tr)
        p = pipe.predict_proba(Xa[pos_te])[:, 1]
        t3.append({"configuration": "Leakage A (preprocessing)", "model": m,
                   **evaluate(y_te, p)})

    # Variant B: SMOTE on the full dataset before partitioning
    Xb, yb = SMOTE(random_state=SEED).fit_resample(Xa, y)
    Xb_tr, Xb_te, yb_tr, yb_te = train_test_split(
        Xb, yb, test_size=TEST_SIZE, stratify=yb, random_state=SEED)
    for m in MODELS:
        pipe = make_model_only(m, smote=False).fit(Xb_tr, yb_tr)
        p = pipe.predict_proba(Xb_te)[:, 1]
        t3.append({"configuration": "Leakage B (SMOTE before split)",
                   "model": m, **evaluate(yb_te, p)})

    # No SMOTE (leakage-controlled otherwise)
    for m in MODELS:
        pipe = make_pipeline(m, "FS", NUMERIC, CATEGORICAL, smote=False)
        res, _ = fit_evaluate(pipe, X_tr, y_tr, X_te, y_te, with_cv=False)
        t3.append({"configuration": "No SMOTE", "model": m, **res})
    t3 = pd.DataFrame(t3)
    t3.to_csv(TABLES / "table3_leakage_comparators_and_smote.csv",
              index=False)
    plot_auc_bars(t3[t3["configuration"] != "No SMOTE"],
                  FIGURES / "figure3_auc_leakage_comparators.png")

    # Figures 4-5: SHAP, plus Spearman agreement
    imp = {m: shap_importance(pipes[(m, "FS")], X_tr, X_te, m)
           for m in MODELS}
    plot_shap_bar(imp["LightGBM"], LABELS, FIGURES / "figure4_shap_lightgbm.png")
    plot_shap_bar(imp["LR"], LABELS, FIGURES / "figure5_shap_lr.png")
    rho, pval, k = shap_agreement(imp["LR"], imp["LightGBM"])
    summary["shap_spearman"] = {"rho": rho, "p": pval, "n_features": k}
    pd.DataFrame({m: imp[m] for m in MODELS}).to_csv(
        TABLES / "shap_mean_abs_uci.csv")

    # Figure 6: calibration
    cal = {}
    for m in MODELS:
        p = preds[(m, "FS")]
        cal[m] = (y_te, p, evaluate(y_te, p)["brier"])
    plot_calibration(cal, FIGURES / "figure6_calibration.png", n_bins=5)

    with open(TABLES / "summary_uci.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    return summary


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, default=float))

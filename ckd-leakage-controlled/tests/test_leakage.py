"""Checks that the controlled pipeline never sees test data and that the
derived clinical variables are computed correctly. Run with ``pytest``."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ckd.data_nhanes import ckd_epi_2021  # noqa: E402
from ckd.metrics import wilson_ci  # noqa: E402
from ckd.pipelines import make_pipeline  # noqa: E402


def _toy(n=200, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, 12)),
                     columns=[f"x{i}" for i in range(12)])
    X.iloc[rng.integers(0, n, 30), 0] = np.nan
    y = pd.Series((X["x1"] + rng.normal(0, 0.5, n) > 0.3).astype(int))
    return X, y


def test_preprocessing_fit_on_training_only():
    X, y = _toy()
    X_tr, X_te, y_tr = X.iloc[:160], X.iloc[160:], y.iloc[:160]
    pipe = make_pipeline("LR", "FS", list(X.columns), []).fit(X_tr, y_tr)
    num = pipe.named_steps["prep"].named_transformers_["num"]
    # imputer median and scaler mean must equal the TRAINING statistics
    assert np.allclose(num.named_steps["impute"].statistics_,
                       X_tr.median().values)
    imputed = X_tr.fillna(X_tr.median())
    assert np.allclose(num.named_steps["scale"].mean_, imputed.mean().values)
    # and must differ from statistics computed on the full data
    assert not np.allclose(num.named_steps["scale"].mean_,
                           X.fillna(X.median()).mean().values)
    assert pipe.predict_proba(X_te).shape == (40, 2)


def test_smote_not_applied_at_prediction():
    X, y = _toy()
    pipe = make_pipeline("LR", "FS", list(X.columns), []).fit(X, y)
    assert pipe.predict_proba(X).shape[0] == len(X)  # no synthetic rows


def test_ckd_epi_2021_reference_values():
    # 50-year-old, Scr 1.0 mg/dL: published equation gives ~92 (M) and ~69 (F)
    m = ckd_epi_2021([1.0], [50], [False])[0]
    f = ckd_epi_2021([1.0], [50], [True])[0]
    assert m == pytest.approx(91.7, abs=0.5)
    assert f == pytest.approx(68.6, abs=0.5)


def test_wilson_interval():
    lo, hi = wilson_ci(79, 80)
    assert 0.93 < lo < 0.94 and 0.99 < hi <= 1.0

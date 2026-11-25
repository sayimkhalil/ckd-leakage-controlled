"""Load and encode the UCI Chronic Kidney Disease dataset (Rubini et al., 2015).

Source: https://doi.org/10.24432/C5G020 (UCI ML Repository, dataset id 336).

Place ``chronic_kidney_disease_full.arff`` in ``data/raw/`` (it is inside the
zip downloaded from the UCI page), or let the loader fetch it with the
``ucimlrepo`` package. Only *encoding* happens here; imputation and scaling are
fit inside the modelling pipelines on the training partition only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import DATA_RAW

COLUMNS = ["age", "bp", "sg", "al", "su", "rbc", "pc", "pcc", "ba", "bgr",
           "bu", "sc", "sod", "pot", "hemo", "pcv", "wc", "rc", "htn", "dm",
           "cad", "appet", "pe", "ane", "class"]

NUMERIC = ["age", "bp", "sg", "al", "su", "bgr", "bu", "sc", "sod", "pot",
           "hemo", "pcv", "wc", "rc"]          # sg, al, su kept as numeric ordinals
CATEGORICAL = ["rbc", "pc", "pcc", "ba", "htn", "dm", "cad", "appet", "pe",
               "ane"]

# value -> 1 ; the complementary value -> 0
POSITIVE_LEVEL = {"rbc": "abnormal", "pc": "abnormal", "pcc": "present",
                  "ba": "present", "htn": "yes", "dm": "yes", "cad": "yes",
                  "appet": "poor", "pe": "yes", "ane": "yes"}

LABELS = {
    "age": "Age", "bp": "Blood pressure", "sg": "Specific gravity",
    "al": "Albumin", "su": "Sugar", "rbc": "Red blood cells",
    "pc": "Pus cell", "pcc": "Pus cell clumps", "ba": "Bacteria",
    "bgr": "Blood glucose random", "bu": "Blood urea",
    "sc": "Serum creatinine", "sod": "Sodium", "pot": "Potassium",
    "hemo": "Hemoglobin", "pcv": "Packed cell volume",
    "wc": "White blood cell count", "rc": "Red blood cell count",
    "htn": "Hypertension", "dm": "Diabetes mellitus",
    "cad": "Coronary artery disease", "appet": "Appetite",
    "pe": "Pedal edema", "ane": "Anemia",
}


def _read_arff(path) -> pd.DataFrame:
    """Robust reader: the public ARFF has stray tabs, spaces and extra commas."""
    rows, in_data = [], False
    for line in open(path, encoding="utf-8", errors="ignore"):
        line = line.strip()
        if not line or line.startswith("%"):
            continue
        if line.lower().startswith("@data"):
            in_data = True
            continue
        if not in_data:
            continue
        vals = [v.strip().strip("'\"") for v in line.split(",")]
        vals = [v for v in vals if v != ""] if len(vals) > len(COLUMNS) else vals
        if len(vals) != len(COLUMNS):
            raise ValueError(f"Unexpected row length {len(vals)}: {line}")
        rows.append(vals)
    return pd.DataFrame(rows, columns=COLUMNS)


def _fetch_ucimlrepo() -> pd.DataFrame:
    from ucimlrepo import fetch_ucirepo  # optional dependency
    ds = fetch_ucirepo(id=336)
    df = pd.concat([ds.data.features, ds.data.targets], axis=1)
    df.columns = COLUMNS
    return df.astype(str)


def load_raw(path=None) -> pd.DataFrame:
    path = path or DATA_RAW / "chronic_kidney_disease_full.arff"
    if path.exists():
        return _read_arff(path)
    return _fetch_ucimlrepo()


def encode(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    df = df.copy()
    df = df.replace({"?": np.nan, "nan": np.nan, "None": np.nan, "": np.nan})
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].str.strip().str.replace("\t", "", regex=False)
    for c in NUMERIC:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in CATEGORICAL:
        df[c] = df[c].map(lambda v, c=c: np.nan if pd.isna(v)
                          else float(v == POSITIVE_LEVEL[c]))
    y = df.pop("class").str.replace(".", "", regex=False).map(
        {"ckd": 1, "notckd": 0}).astype(int)
    X = df[NUMERIC + CATEGORICAL]
    return X, y


def load_uci() -> tuple[pd.DataFrame, pd.Series]:
    return encode(load_raw())

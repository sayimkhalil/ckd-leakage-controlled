"""Assemble the NHANES 2017-2018 (cycle J) external-validation cohort.

Public-use files from the CDC National Center for Health Statistics:
https://wwwn.cdc.gov/nchs/nhanes/continuousnhanes/default.aspx?BeginYear=2017

The 11 ``.xpt`` files listed in ``FILES`` are downloaded into ``data/raw/nhanes``
on first use (or can be placed there manually), merged on ``SEQN``, restricted
to adults >= 18 years with non-missing serum creatinine and urine
albumin/creatinine ratio, and labelled by the KDIGO definition:

    CKD = eGFR < 60 mL/min/1.73 m^2  OR  uACR >= 30 mg/g

with eGFR from the race-free CKD-EPI 2021 creatinine equation.
"""
from __future__ import annotations

import urllib.request

import numpy as np
import pandas as pd

from .config import DATA_PROCESSED, DATA_RAW

NHANES_DIR = DATA_RAW / "nhanes"
FILES = ["DEMO_J", "BMX_J", "BPX_J", "BIOPRO_J", "CBC_J", "KIQ_U_J",
         "ALB_CR_J", "DIQ_J", "BPQ_J", "SMQ_J", "GHB_J"]
URL_PATTERNS = [
    "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/{f}.xpt",
    "https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/{f}.XPT",
]

# Feature sets ---------------------------------------------------------------
RACE_DUMMIES = ["race_mexican_american", "race_other_hispanic",
                "race_nh_white", "race_nh_black", "race_nh_asian",
                "race_other_multi"]
CONTROLLED_FEATURES = ["age", "female", "bmi", "sbp", "dbp", "hemoglobin",
                       "hba1c", "diabetes", "hypertension", "bp_medication",
                       "smoking_status", "income_poverty_ratio"] + RACE_DUMMIES
# Label-defining or label-correlated variables admitted only in the
# leakage-prone arm.
LEAKAGE_FEATURES = ["serum_creatinine", "egfr", "bun", "uacr",
                    "self_report_weak_kidneys", "self_report_dialysis"]

LABELS = {
    "age": "Age", "female": "Female sex", "bmi": "Body mass index",
    "sbp": "Systolic BP", "dbp": "Diastolic BP", "hemoglobin": "Hemoglobin",
    "hba1c": "Glycated hemoglobin (HbA1c)", "diabetes": "Diabetes (self-report)",
    "hypertension": "Hypertension (self-report)",
    "bp_medication": "Antihypertensive medication",
    "smoking_status": "Smoking status",
    "income_poverty_ratio": "Income-to-poverty ratio",
    "race_mexican_american": "Race: Mexican American",
    "race_other_hispanic": "Race: Other Hispanic",
    "race_nh_white": "Race: Non-Hispanic White",
    "race_nh_black": "Race: Non-Hispanic Black",
    "race_nh_asian": "Race: Non-Hispanic Asian",
    "race_other_multi": "Race: Other/multiracial",
    "serum_creatinine": "Serum creatinine", "egfr": "eGFR",
    "bun": "Blood urea nitrogen", "uacr": "Urine albumin/creatinine ratio",
    "self_report_weak_kidneys": "Self-report: weak/failing kidneys",
    "self_report_dialysis": "Self-report: dialysis",
}


# Download / read ------------------------------------------------------------
def download(force: bool = False) -> None:
    NHANES_DIR.mkdir(parents=True, exist_ok=True)
    for f in FILES:
        dest = NHANES_DIR / f"{f}.xpt"
        if dest.exists() and not force:
            continue
        last_err = None
        for pat in URL_PATTERNS:
            try:
                urllib.request.urlretrieve(pat.format(f=f), dest)
                last_err = None
                break
            except Exception as e:  # noqa: BLE001
                last_err = e
        if last_err is not None:
            raise RuntimeError(f"Could not download {f}: {last_err}. "
                               f"Download it manually into {NHANES_DIR}.")


def _read(f: str) -> pd.DataFrame:
    path = NHANES_DIR / f"{f}.xpt"
    if not path.exists():
        path = NHANES_DIR / f"{f}.XPT"
    return pd.read_sas(path, format="xport", encoding="utf-8")


# Derived variables ----------------------------------------------------------
def ckd_epi_2021(scr_mg_dl, age, female) -> np.ndarray:
    """Race-free CKD-EPI 2021 creatinine equation (Inker et al., NEJM 2021)."""
    scr = np.asarray(scr_mg_dl, dtype=float)
    age = np.asarray(age, dtype=float)
    female = np.asarray(female, dtype=bool)
    kappa = np.where(female, 0.7, 0.9)
    alpha = np.where(female, -0.241, -0.302)
    ratio = scr / kappa
    egfr = (142.0 * np.minimum(ratio, 1.0) ** alpha
            * np.maximum(ratio, 1.0) ** -1.200
            * 0.9938 ** age
            * np.where(female, 1.012, 1.0))
    return egfr


def _yes_no(s: pd.Series) -> pd.Series:
    """NHANES 1 = yes, 2 = no; 7/9 (refused/don't know) -> missing."""
    return s.map({1: 1.0, 2: 0.0})


def build_cohort(save: bool = True) -> pd.DataFrame:
    d = {f: _read(f) for f in FILES}
    df = d["DEMO_J"][["SEQN", "RIDAGEYR", "RIAGENDR", "RIDRETH3", "INDFMPIR"]]
    for f, cols in [("BMX_J", ["BMXBMI"]),
                    ("BPX_J", ["BPXSY1", "BPXSY2", "BPXSY3", "BPXSY4",
                               "BPXDI1", "BPXDI2", "BPXDI3", "BPXDI4"]),
                    ("BIOPRO_J", ["LBXSCR", "LBXSBU"]),
                    ("CBC_J", ["LBXHGB"]),
                    ("KIQ_U_J", ["KIQ022", "KIQ025"]),
                    ("ALB_CR_J", ["URDACT"]),
                    ("DIQ_J", ["DIQ010"]),
                    ("BPQ_J", ["BPQ020", "BPQ050A"]),
                    ("SMQ_J", ["SMQ020", "SMQ040"]),
                    ("GHB_J", ["LBXGH"])]:
        df = df.merge(d[f][["SEQN"] + cols], on="SEQN", how="left")

    out = pd.DataFrame({"SEQN": df["SEQN"].astype(int)})
    out["age"] = df["RIDAGEYR"]
    out["female"] = (df["RIAGENDR"] == 2).astype(float)
    out["bmi"] = df["BMXBMI"]
    dbp = df[["BPXDI1", "BPXDI2", "BPXDI3", "BPXDI4"]].replace(0, np.nan)
    out["sbp"] = df[["BPXSY1", "BPXSY2", "BPXSY3", "BPXSY4"]].mean(axis=1)
    out["dbp"] = dbp.mean(axis=1)
    out["hemoglobin"] = df["LBXHGB"]
    out["hba1c"] = df["LBXGH"]
    out["diabetes"] = df["DIQ010"].map({1: 1.0, 2: 0.0, 3: 0.0})  # borderline -> no
    out["hypertension"] = _yes_no(df["BPQ020"])
    out["bp_medication"] = (df["BPQ050A"] == 1).astype(float)
    smoked_100 = df["SMQ020"]
    now = df["SMQ040"]
    out["smoking_status"] = np.select(
        [smoked_100 == 2, (smoked_100 == 1) & (now == 3),
         (smoked_100 == 1) & now.isin([1, 2])], [0.0, 1.0, 2.0], np.nan)
    out["income_poverty_ratio"] = df["INDFMPIR"]
    race_map = {1: "race_mexican_american", 2: "race_other_hispanic",
                3: "race_nh_white", 4: "race_nh_black", 6: "race_nh_asian",
                7: "race_other_multi"}
    for code, name in race_map.items():
        out[name] = (df["RIDRETH3"] == code).astype(float)

    out["serum_creatinine"] = df["LBXSCR"]
    out["egfr"] = ckd_epi_2021(df["LBXSCR"], df["RIDAGEYR"],
                               df["RIAGENDR"] == 2)
    out["bun"] = df["LBXSBU"]
    out["uacr"] = df["URDACT"]
    out["self_report_weak_kidneys"] = _yes_no(df["KIQ022"])
    out["self_report_dialysis"] = _yes_no(df["KIQ025"])

    # Inclusion: adults with both label-defining measurements present.
    out = out[(out["age"] >= 18) & out["serum_creatinine"].notna()
              & out["uacr"].notna()].reset_index(drop=True)
    out["ckd"] = ((out["egfr"] < 60) | (out["uacr"] >= 30)).astype(int)

    if save:
        DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
        out.to_csv(DATA_PROCESSED / "nhanes_2017_2018_ckd_cohort.csv",
                   index=False)
    return out


def load_nhanes(download_if_missing: bool = True) -> pd.DataFrame:
    cached = DATA_PROCESSED / "nhanes_2017_2018_ckd_cohort.csv"
    if cached.exists():
        return pd.read_csv(cached)
    if download_if_missing:
        download()
    return build_cohort()

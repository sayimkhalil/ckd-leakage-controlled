# ckd-leakage-controlled

Analysis code for the manuscript

> **Application of machine learning in the prediction of chronic kidney disease: a leakage-controlled evaluation with external validation on real clinical data**

The code reproduces every table and figure in the manuscript:

| Output | Content | Script |
|---|---|---|
| Figure 1 | UCI class distribution | `experiment_uci.py` |
| Table 1 | ANOVA F-scores and selection stability (200 bootstraps) | `experiment_uci.py` |
| Table 2, Figure 2 | Leakage-controlled performance, 4 model–pipeline combinations | `experiment_uci.py` |
| Table 3, Figure 3 | Leakage-prone comparators (Variant A, Variant B) and no-SMOTE ablation | `experiment_uci.py` |
| Figures 4–5 | SHAP importance (LightGBM, Logistic Regression) and Spearman agreement | `experiment_uci.py` |
| Figure 6 | Calibration curves and Brier scores | `experiment_uci.py` |
| Table 4, Figure 7 | NHANES 2017–2018 controlled vs leakage-prone arms | `experiment_nhanes.py` |
| Figure 8 | NHANES SHAP importance (controlled LightGBM) | `experiment_nhanes.py` |

## Quick start

```bash
git clone https://github.com/sayimkhalil/ckd-leakage-controlled.git
cd ckd-leakage-controlled
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python run_all.py            # both cohorts, about 1–2 minutes on a laptop
pytest -q                    # leakage and formula checks
```

Results are written to `results/tables/` (CSV and JSON) and `results/figures/` (PNG, 300 dpi).
Run one cohort with `python run_all.py --uci` or `python run_all.py --nhanes`.

## Data

No data are redistributed in this repository. Both sources are public and de-identified.

**UCI Chronic Kidney Disease** (Rubini, Soundarapandian & Eswaran, 2015; doi:[10.24432/C5G020](https://doi.org/10.24432/C5G020)).
Download the zip from the UCI page and copy `chronic_kidney_disease_full.arff` to `data/raw/`.
If the file is absent the loader falls back to `ucimlrepo` (`pip install ucimlrepo`, dataset id 336).

**NHANES 2017–2018 (cycle J)** (CDC/NCHS public-use files, <https://wwwn.cdc.gov/nchs/nhanes/>).
The first run downloads the 11 files below into `data/raw/nhanes/`, merges them on `SEQN`, and caches the analysis cohort as `data/processed/nhanes_2017_2018_ckd_cohort.csv`. If your network blocks the download, fetch the `.xpt` files by hand from the CDC site and put them in that folder.

`DEMO_J, BMX_J, BPX_J, BIOPRO_J, CBC_J, KIQ_U_J, ALB_CR_J, DIQ_J, BPQ_J, SMQ_J, GHB_J`

## What "leakage-controlled" means here

All data-dependent steps sit inside one `imblearn` pipeline (`src/ckd/pipelines.py`):

```
median/mode imputation → standard scaling → ANOVA SelectKBest(k=10) or PCA(95% variance) → SMOTE → classifier
```

The pipeline is fit on the 80% training partition only. In stratified five-fold cross-validation the whole pipeline, SMOTE included, is refit inside each training fold. SMOTE runs only during `fit` and is never applied to the held-out test set. `tests/test_leakage.py` checks that the fitted imputation and scaling statistics equal the training-set statistics, not the full-data ones.

### Comparators

| Configuration | What leaks |
|---|---|
| **Variant A** (UCI) | Imputation, scaling and ANOVA selection fit on all 400 patients before the split; the same 80 real test patients are kept; SMOTE inside training only. |
| **Variant B** (UCI) | As Variant A, plus SMOTE on the full dataset before the split, so the test set (n = 100) contains synthetic samples. |
| **No SMOTE** (UCI) | Controlled pipeline with SMOTE removed (ablation, not leakage). |
| **Leakage-prone arm** (NHANES) | Feature-level leakage: serum creatinine, eGFR, blood urea nitrogen, urine albumin/creatinine ratio and two kidney self-report items (KIQ022 weak/failing kidneys, KIQ025 dialysis) are admitted as features. eGFR and uACR define the label. |

## NHANES cohort definition

* Adults aged ≥ 18 years with non-missing serum creatinine (`LBXSCR`) and urine albumin/creatinine ratio (`URDACT`).
* eGFR from the race-free CKD-EPI 2021 creatinine equation (Inker et al., *N Engl J Med* 2021).
* CKD (KDIGO 2024): eGFR < 60 mL/min/1.73 m² **or** uACR ≥ 30 mg/g, from a single measurement.
* Controlled features: age, sex, BMI, mean systolic and diastolic BP (`BPXSY1–4`, `BPXDI1–4`), hemoglobin, HbA1c, self-reported diabetes (`DIQ010`; borderline coded as no), self-reported hypertension (`BPQ020`), antihypertensive medication (`BPQ050A`), smoking status (never/former/current from `SMQ020`/`SMQ040`), income-to-poverty ratio, race/ethnicity (`RIDRETH3`, one-hot).
* NHANES codes 7/9 (refused/don't know) are treated as missing.

Survey weights are not applied: the aim is a protocol-level comparison of controlled and leakage-prone pipelines, not population prevalence estimation.

## Settings

| Item | Value |
|---|---|
| Random seed | 42 (split, SMOTE, CV, both classifiers) |
| Split | 80/20 stratified |
| Logistic Regression | L2, C = 1.0, lbfgs, max_iter = 1000 |
| LightGBM | 100 estimators, learning rate 0.1, 31 leaves, min 20 samples per leaf, no subsampling, no L1/L2 |
| AUC 95% CI | percentile bootstrap of test-set predictions, 2000 resamples |
| Accuracy 95% CI | Wilson score interval |
| Threshold | 0.5 |
| SHAP | `TreeExplainer` (LightGBM); `LinearExplainer` with the full transformed training matrix as background (LR); mean \|SHAP\| on the test set |

All settings are in `src/ckd/config.py`.

## Software

Python 3.12 with the versions pinned in `requirements.txt` (scikit-learn 1.5.2, LightGBM 4.5.0, imbalanced-learn 0.12.4, SHAP 0.54.1, SciPy 1.14.1, pandas 2.2.3). Newer library versions run the code but can change results in the third decimal place.

## Repository layout

```
ckd-leakage-controlled/
├── run_all.py                 entry point
├── requirements.txt
├── src/ckd/
│   ├── config.py              seeds, hyperparameters, paths
│   ├── data_uci.py            UCI loading and encoding
│   ├── data_nhanes.py         NHANES download, merge, eGFR, KDIGO label
│   ├── pipelines.py           leakage-controlled pipelines
│   ├── metrics.py             AUC bootstrap CI, Wilson CI, per-class metrics, Brier
│   ├── evaluation.py          CV, SHAP, plots
│   ├── experiment_uci.py      Tables 1–3, Figures 1–6
│   └── experiment_nhanes.py   Table 4, Figures 7–8
├── tests/test_leakage.py
├── data/                      raw/ and processed/ (git-ignored)
└── results/                   tables/ and figures/ (generated)
```

## Citation

If you use this code, please cite the article (full reference to be added on publication).

## License

MIT; see `LICENSE`. The datasets are subject to their providers' terms (UCI: CC BY 4.0; NHANES: public domain).

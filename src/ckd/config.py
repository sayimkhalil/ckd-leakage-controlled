"""Global configuration shared by every analysis in the repository."""
from pathlib import Path

SEED = 42                 # train-test split, SMOTE, CV and both classifiers
TEST_SIZE = 0.20          # 80/20 stratified split
N_FOLDS = 5               # stratified five-fold CV on the training partition
K_FEATURES = 10           # ANOVA SelectKBest
PCA_VARIANCE = 0.95       # components explaining 95% of training variance
N_BOOT_AUC = 2000         # bootstrap resamples for the AUC 95% CI
N_BOOT_STABILITY = 200    # bootstrap resamples for feature-selection stability
THRESHOLD = 0.5           # probability threshold for confusion-matrix metrics

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
TABLES = RESULTS / "tables"

LR_PARAMS = dict(penalty="l2", C=1.0, solver="lbfgs", max_iter=1000,
                 random_state=SEED)
LGBM_PARAMS = dict(n_estimators=100, learning_rate=0.1, num_leaves=31,
                   min_child_samples=20, subsample=1.0, colsample_bytree=1.0,
                   reg_alpha=0.0, reg_lambda=0.0, random_state=SEED,
                   verbose=-1)


def ensure_dirs() -> None:
    for d in (DATA_RAW, DATA_PROCESSED, FIGURES, TABLES):
        d.mkdir(parents=True, exist_ok=True)

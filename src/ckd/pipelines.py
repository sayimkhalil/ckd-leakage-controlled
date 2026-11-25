"""Leakage-controlled modelling pipelines.

Every data-dependent step (imputation, scaling, ANOVA selection, PCA, SMOTE)
lives inside one ``imblearn.pipeline.Pipeline``. Calling ``fit`` on the
training partition therefore fits every step on training data only, and in
cross-validation each step is refit inside each training fold. SMOTE is a
sampler, so imbalanced-learn applies it during ``fit`` only and never to data
passed to ``predict``/``predict_proba``.
"""
from __future__ import annotations

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .config import K_FEATURES, LGBM_PARAMS, LR_PARAMS, PCA_VARIANCE, SEED


def make_classifier(name: str):
    if name == "LR":
        return LogisticRegression(**LR_PARAMS)
    if name == "LightGBM":
        return LGBMClassifier(**LGBM_PARAMS)
    raise ValueError(name)


def make_preprocessor(numeric: list[str], categorical: list[str]):
    """Median imputation (numeric), mode imputation (categorical), z-scaling."""
    transformers = []
    if numeric:
        transformers.append(("num", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler())]), numeric))
    if categorical:
        transformers.append(("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("scale", StandardScaler())]), categorical))
    return ColumnTransformer(transformers, verbose_feature_names_out=False)


def make_reducer(kind: str):
    if kind == "FS":
        return SelectKBest(f_classif, k=K_FEATURES)
    if kind == "PCA":
        return PCA(n_components=PCA_VARIANCE, svd_solver="full",
                   random_state=SEED)
    raise ValueError(kind)


def make_pipeline(model: str, reducer: str, numeric: list[str],
                  categorical: list[str], smote: bool = True) -> Pipeline:
    steps = [("prep", make_preprocessor(numeric, categorical)),
             ("reduce", make_reducer(reducer))]
    if smote:
        steps.append(("smote", SMOTE(random_state=SEED)))
    steps.append(("clf", make_classifier(model)))
    return Pipeline(steps)


def make_model_only(model: str, smote: bool = True) -> Pipeline:
    """SMOTE + classifier, for leakage-prone variants whose preprocessing
    was (deliberately) fit outside the pipeline."""
    steps = [("smote", SMOTE(random_state=SEED))] if smote else []
    steps.append(("clf", make_classifier(model)))
    return Pipeline(steps)


def selected_feature_names(pipe: Pipeline) -> list[str]:
    names = pipe.named_steps["prep"].get_feature_names_out()
    mask = pipe.named_steps["reduce"].get_support()
    return [n for n, m in zip(names, mask) if m]

"""Model definitions and inference with the saved models."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from src.preprocessing import build_preprocessor

CLASSIFIERS = {
    "logistic_regression": LogisticRegression,
    "knn": KNeighborsClassifier,
    "decision_tree": DecisionTreeClassifier,
    "random_forest": RandomForestClassifier,
}


def build_model(name, params, data, cfg, scaling=None):
    """Return an unfitted Pipeline: preprocessing followed by the classifier `name` (a key of CLASSIFIERS).

    Keeping both steps in one object means the preprocessing is always fitted on exactly
    the rows the classifier is trained on, and the saved model accepts raw flows.

    Args:
        name: key of CLASSIFIERS.
        params: hyperparameters of the classifier.
        data: Datasets object (gives the feature column names).
        cfg: configuration dictionary.
        scaling: overrides cfg["preprocessing"]["scaling"] (used by the preprocessing study).
    """
    classifier = CLASSIFIERS[name](**params)
    if "random_state" in classifier.get_params():
        classifier.set_params(random_state=cfg["seed"])
    if name in ("random_forest", "knn"):
        classifier.set_params(n_jobs=-1)
    prep = build_preprocessor(data.numeric, data.categorical,
                              scaling or cfg["preprocessing"]["scaling"], cfg["preprocessing"]["min_frequency"])
    return Pipeline([("prep", prep), ("clf", classifier)])


def predict_flows(bundle, flows):
    """Apply the two-stage system to raw flow records.

    Stage 1: the binary model raises an alarm when P(attack) >= the saved threshold.
    Stage 2: flows with an alarm get an attack category from the multiclass model;
    the others are labelled Normal.

    Args:
        bundle: dictionary saved by main.py ("binary", "threshold", "multiclass", "features").
        flows: DataFrame with (at least) the feature columns of the dataset.
    Returns:
        DataFrame with the attack probability, the alarm decision and the predicted category.
    """
    X = flows[bundle["features"]]
    p_attack = bundle["binary"].predict_proba(X)[:, 1]
    alarm = p_attack >= bundle["threshold"]
    return pd.DataFrame({
        "P(attack)": p_attack.round(3),
        "alarm": alarm,
        "predicted category": np.where(alarm, bundle["multiclass"].predict(X), "Normal"),
    }, index=flows.index)

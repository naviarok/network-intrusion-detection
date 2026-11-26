"""Training logic: hyperparameter search on the validation set and threshold choice."""
import time
from dataclasses import replace

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import ParameterGrid

from src.model import build_model


def choose_threshold(y_true, p_attack, max_false_alarm_rate=0.05, step=0.01):
    """Lowest alarm threshold whose false alarm rate stays at or under the limit.

    The false alarm rate is the share of normal flows (y_true == 0) whose attack
    probability is >= the threshold. A lower threshold catches more attacks, so the
    lowest acceptable one gives the best detection rate under the limit.
    Returns None when even a threshold of 1.0 flags too many normal flows: this happens
    with a fully grown tree, whose probabilities are almost all exactly 0 or 1.
    """
    p_normal = np.asarray(p_attack)[np.asarray(y_true) == 0]
    for t in np.round(np.arange(step, 1 + step / 2, step), 2):  # step, 2*step, ..., 1.0
        if (p_normal >= t).mean() <= max_false_alarm_rate:
            return float(t)
    return None


def score_binary(model, val, features, cfg):
    """Validation score of a binary model: detection rate at its own alarm threshold.

    A model that cannot respect the false alarm limit gets a score of 0, so it is never selected.
    """
    p = model.predict_proba(val[features])[:, 1]
    t = choose_threshold(val["label"], p, cfg["threshold"]["max_false_alarm_rate"], cfg["threshold"]["step"])
    detection = 0.0 if t is None else (p[val["label"].to_numpy() == 1] >= t).mean()
    return {"score": detection, "threshold": t, "ROC AUC": roc_auc_score(val["label"], p)}


def score_multiclass(model, val, features, cfg):
    """Validation score of a category model: macro F1, the plain average of the per-class F1."""
    pred = model.predict(val[features])
    return {"score": f1_score(val["attack_cat"], pred, average="macro"),
            "accuracy": accuracy_score(val["attack_cat"], pred)}


def grid_search(search_space, data, target, score_fn, cfg):
    """Train every model and hyperparameter combination of `search_space` and keep the best.

    Each candidate is fitted on data.train and scored on data.val with `score_fn`
    (higher is better). The test set is never used here.

    Returns:
        results: one row per candidate (model, params, scores, fit time), best first.
        best_model: the fitted pipeline with the highest validation score.
    """
    results, best_model, best_score = [], None, -np.inf
    for name, grid in search_space.items():
        for params in ParameterGrid(grid):
            model = build_model(name, params, data, cfg)
            start = time.time()
            model.fit(data.train[data.features], data.train[target])
            scores = score_fn(model, data.val, data.features, cfg)
            results.append({"model": name, "params": params, **scores, "fit seconds": time.time() - start})
            print(f"  {name:20s} {str(params):75s} score={scores['score']:.4f}")
            if scores["score"] > best_score:
                best_model, best_score = model, scores["score"]
    results = pd.DataFrame(results).sort_values("score", ascending=False, ignore_index=True)
    return results, best_model


def train_binary(data, cfg):
    """Search for the best normal-vs-attack model; return (results, model, alarm threshold)."""
    results, model = grid_search(cfg["search"]["binary"], data, "label", score_binary, cfg)
    return results, model, results.loc[0, "threshold"]


def train_multiclass(data, cfg):
    """Search for the best attack-category model; return (results, model).

    The model only answers "which attack is it?" for flows the binary model has flagged,
    so it is trained and scored on attack flows only (9 categories). Normal traffic is the
    binary model's job, which keeps the two decisions consistent.
    """
    attacks = replace(data, train=data.train[data.train["label"] == 1], val=data.val[data.val["label"] == 1])
    return grid_search(cfg["search"]["multiclass"], attacks, "attack_cat", score_multiclass, cfg)

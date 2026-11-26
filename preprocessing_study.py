"""Step 2: measure how the preprocessing choices change the results.

Each experiment changes one choice and keeps everything else fixed (models from
`study` in the config, scored on the validation set):
  A. scaling of the numeric features: none / standard / log(1+x) + standard
  B. removing the duplicate rows of the train file or not
  C. keeping or dropping the TTL features (whose distribution differs in the test file)
B and C also report the test set, as an after-the-fact check of what validation could
not show; no setting of the final model is chosen from these test numbers.

    python preprocessing_study.py [--config configs/config.yaml]
"""
import argparse
import warnings
from dataclasses import replace

import pandas as pd
from sklearn.exceptions import ConvergenceWarning

from src.data import load_datasets
from src.evaluate import binary_scores, plot_grouped_bars
from src.model import build_model
from src.train import score_binary, score_multiclass
from src.utils import load_config, save_figure, save_table, section


def fit_and_score(name, data, cfg, target="label", scaling=None, on_test=False):
    """Fit one study model and return its validation (and optionally test) scores.

    As in main.py, the category model (target "attack_cat") only sees attack flows.
    """
    if target == "attack_cat":
        data = replace(data, train=data.train[data.train["label"] == 1], val=data.val[data.val["label"] == 1])
    model = build_model(name, cfg["study"]["models"][name], data, cfg, scaling)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(data.train[data.features], data.train[target])
    if target == "attack_cat":
        return {"val macro F1": score_multiclass(model, data.val, data.features, cfg)["score"]}

    val = score_binary(model, data.val, data.features, cfg)
    row = {"val detection rate": val["score"], "val ROC AUC": val["ROC AUC"], "threshold": val["threshold"],
           "converged": not any(issubclass(w.category, ConvergenceWarning) for w in caught)}
    if on_test:
        test = binary_scores(data.test["label"], model.predict_proba(data.test[data.features])[:, 1],
                             val["threshold"])
        row.update({"test detection rate": test["detection rate"], "test false alarm rate": test["false alarm rate"]})
    return row


def main():
    """Run experiments A, B and C and save their tables and figure."""
    parser = argparse.ArgumentParser(description="Compare preprocessing choices")
    parser.add_argument("--config", default="configs/config.yaml")
    cfg = load_config(parser.parse_args().config)
    figures, results = cfg["output"]["figures_dir"], cfg["output"]["results_dir"]
    pd.set_option("display.width", 200)
    data = load_datasets(cfg)

    section("A. Scaling of the numeric features (validation set)")
    rows = []
    for name in ["logistic_regression", "knn", "decision_tree"]:
        for scaling in cfg["study"]["scalings"]:
            print(f"  {name} / {scaling}")
            rows.append({"model": name, "scaling": scaling, **fit_and_score(name, data, cfg, scaling=scaling),
                         **fit_and_score(name, data, cfg, "attack_cat", scaling)})
    scaling_table = pd.DataFrame(rows).set_index(["model", "scaling"])
    print(scaling_table.round(3).to_string())
    save_table(scaling_table, results, "study_scaling.csv")
    save_figure(plot_grouped_bars(scaling_table["val detection rate"].unstack()[cfg["study"]["scalings"]],
                                  "detection rate at 5% false alarms", "Effect of the numeric scaling (validation)"),
                figures, "study_scaling.png")

    section("B. Duplicate rows of the train file")
    with_dups = load_datasets(cfg, remove_dups=False)
    rows = [{"model": name, "duplicates": label, **fit_and_score(name, d, cfg, on_test=True)}
            for name in ["logistic_regression", "random_forest"]
            for label, d in [("removed", data), ("kept", with_dups)]]
    dup_table = pd.DataFrame(rows).set_index(["model", "duplicates"])
    print(dup_table.round(3).to_string())
    save_table(dup_table, results, "study_duplicates.csv")

    section("C. TTL features")
    no_ttl = load_datasets(cfg, drop_features=cfg["study"]["ttl_features"])
    rows = [{"model": name, "TTL features": label, **fit_and_score(name, d, cfg, on_test=True)}
            for name in ["logistic_regression", "random_forest"]
            for label, d in [("kept", data), ("dropped", no_ttl)]]
    ttl_table = pd.DataFrame(rows).set_index(["model", "TTL features"])
    print(ttl_table.round(3).to_string())
    save_table(ttl_table, results, "study_ttl.csv")


if __name__ == "__main__":
    main()

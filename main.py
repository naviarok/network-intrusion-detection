"""Training entry point: runs the complete pipeline and saves the trained models.

Loads and de-duplicates the data, splits train / validation, searches the hyperparameter
space of every model for both stages of the system (1. normal vs attack, 2. category of
an attack), chooses the alarm threshold on validation, and saves everything to
models/best_model.pkl.

    python main.py [--config configs/config.yaml]
"""
import argparse

import pandas as pd

from src.data import load_datasets
from src.train import train_binary, train_multiclass
from src.utils import load_config, save_model, save_table, section


def main():
    parser = argparse.ArgumentParser(description="Train the UNSW-NB15 intrusion detection models")
    parser.add_argument("--config", default="configs/config.yaml")
    cfg = load_config(parser.parse_args().config)
    results_dir = cfg["output"]["results_dir"]
    pd.set_option("display.width", 250, "display.max_colwidth", 80)

    section("Data")
    data = load_datasets(cfg)
    print(f"train: {len(data.train)} rows | validation: {len(data.val)} rows | test (not used here): {len(data.test)}")
    print(f"{len(data.features)} features, scaling = {cfg['preprocessing']['scaling']}")

    section("Binary task: normal vs attack (score = detection rate at <= 5% false alarms)")
    binary_results, binary_model, threshold = train_binary(data, cfg)
    print(binary_results.round(4).to_string())
    save_table(binary_results, results_dir, "search_binary.csv")

    section("Multiclass task: category of the attack flows (score = macro F1)")
    multi_results, multi_model = train_multiclass(data, cfg)
    print(multi_results.round(4).to_string())
    save_table(multi_results, results_dir, "search_multiclass.csv")

    section("Selected models")
    print(f"binary:     {binary_results.loc[0, 'model']} {binary_results.loc[0, 'params']}, threshold {threshold}")
    print(f"multiclass: {multi_results.loc[0, 'model']} {multi_results.loc[0, 'params']}")
    bundle = {"binary": binary_model, "threshold": threshold, "multiclass": multi_model, "features": data.features}
    save_model(bundle, cfg["output"]["model_path"])
    print(f"saved to {cfg['output']['model_path']}")


if __name__ == "__main__":
    main()

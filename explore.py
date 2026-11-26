"""Step 1: explore the raw data. Prints the tables and saves the plots to figures/.

    python explore.py [--config configs/config.yaml]
"""
import argparse

import pandas as pd

from src.data import feature_columns, load_raw, remove_duplicates
from src.explore import (class_balance, data_checks, plot_class_balance, plot_log_transform, skew_summary,
                         ttl_attack_share, unseen_categories)
from src.utils import load_config, save_figure, save_table, section


def main():
    """Print the exploration tables and save the plots."""
    parser = argparse.ArgumentParser(description="Explore the UNSW-NB15 data")
    parser.add_argument("--config", default="configs/config.yaml")
    cfg = load_config(parser.parse_args().config)
    figures, results = cfg["output"]["figures_dir"], cfg["output"]["results_dir"]
    pd.set_option("display.width", 200)

    section("Data")
    train_file, test = load_raw(cfg)
    numeric, categorical = feature_columns(train_file)
    features = numeric + categorical
    print(f"train file: {train_file.shape}, test file: {test.shape}")
    print(f"{len(features)} features: {len(numeric)} numeric, {len(categorical)} categorical")
    print(f"share of attacks: train {train_file['label'].mean():.3f}, test {test['label'].mean():.3f}")

    section("Class balance")
    counts = class_balance(train_file, remove_duplicates(train_file, features), test)
    print(counts.to_string())
    save_table(counts, results, "class_balance.csv")
    save_figure(plot_class_balance(counts), figures, "class_balance.png")

    section("Data quality")
    print(data_checks(train_file, test, features).to_string())
    print(unseen_categories(train_file, test, categorical).to_string())

    section("Skewed numeric features")
    print(skew_summary(train_file, cfg["analysis"]["skewed_columns"]).to_string())
    save_figure(plot_log_transform(train_file), figures, "log_transform.png")

    section("Source TTL (sttl) versus label")
    print(ttl_attack_share(train_file).to_string())


if __name__ == "__main__":
    main()

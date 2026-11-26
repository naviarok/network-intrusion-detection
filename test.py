"""Independent test and inference with the saved models (never retrains anything).

Without arguments: evaluates models/best_model.pkl on the official test file, which was
never seen during training (detection, false alarms, attack categories, rare classes),
then shows predictions for a few test flows.
With --input: predicts every flow of your own CSV file (same columns as the dataset).

    python test.py [--config configs/config.yaml]
    python test.py --input my_flows.csv [--output predictions.csv]
"""
import argparse

import pandas as pd

from src.data import load_raw
from src.evaluate import (binary_scores, detection_by_category, false_alarm_rate_by, multiclass_report,
                          plot_confusion_and_roc, plot_false_alarms, plot_multiclass_confusion, ttl_shift)
from src.model import predict_flows
from src.utils import load_config, load_model, save_figure, save_table, section

def evaluate_on_test(bundle, cfg):
    """Print and save every test-set result used in the report.

    The train file is only read to compare its TTL distribution and class order with the test file.
    """
    figures, results, analysis = cfg["output"]["figures_dir"], cfg["output"]["results_dir"], cfg["analysis"]
    train_file, test = load_raw(cfg)
    t = bundle["threshold"]
    p_attack = bundle["binary"].predict_proba(test[bundle["features"]])[:, 1]
    alarm = p_attack >= t

    section("Binary detection on the test set")
    scores = pd.DataFrame({"threshold 0.5": binary_scores(test["label"], p_attack, 0.5),
                           f"threshold {t} (chosen)": binary_scores(test["label"], p_attack, t)}).T
    print(scores.round(3).to_string())
    save_table(scores, results, "test_binary.csv")
    save_figure(plot_confusion_and_roc(test["label"], p_attack, t), figures, "test_confusion_roc.png")

    section("Share of each category that raises an alarm")
    by_category = detection_by_category(test["attack_cat"], alarm)
    print(by_category.round(3).to_string())
    save_table(by_category, results, "test_detection_by_category.csv")

    section("False alarms")
    normal = (test["label"] == 0).to_numpy()
    print(f"{alarm[normal].sum()} false alarms out of {normal.sum()} normal flows")
    by_sttl = false_alarm_rate_by(test, alarm, "sttl", analysis["min_normal_flows"]["sttl"])
    by_service = false_alarm_rate_by(test, alarm, "service", analysis["min_normal_flows"]["service"])
    by_service = by_service.sort_values("normal flows", ascending=False)
    print(by_sttl.round(3).to_string())
    print(by_service.round(3).to_string())
    print(ttl_shift(train_file, test, analysis["ttl_values"]).round(1).to_string())
    save_figure(plot_false_alarms(by_sttl, by_service), figures, "test_false_alarms.png")

    section("Stage 2 alone: attack category of the test attacks")
    attacks = test[test["label"] == 1]
    pred = bundle["multiclass"].predict(attacks[bundle["features"]])
    report = multiclass_report(attacks["attack_cat"], pred)
    print(report.round(3).to_string())
    print("rare classes:\n" + report.loc[analysis["rare_classes"]].round(3).to_string())
    save_table(report, results, "test_multiclass.csv")
    labels = [c for c in train_file["attack_cat"].value_counts().index if c != "Normal"]
    save_figure(plot_multiclass_confusion(attacks["attack_cat"], pred, labels), figures,
                "test_multiclass_confusion.png")

    section("Complete system: alarm, then category (all test flows, 10 classes)")
    system = multiclass_report(test["attack_cat"], predict_flows(bundle, test)["predicted category"])
    print(system.round(3).to_string())
    save_table(system, results, "test_system.csv")

    section("Example: predictions for a few test flows")
    sample = test.sample(analysis["demo_flows"], random_state=cfg["seed"])
    print(pd.concat([sample[["proto", "service", "sbytes", "sttl"]], predict_flows(bundle, sample),
                     sample[["attack_cat"]].rename(columns={"attack_cat": "true category"})], axis=1).to_string())


def main():
    """Parse the command line, load the saved models, then evaluate them or predict --input."""
    parser = argparse.ArgumentParser(description="Test the saved models or predict new flows")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--input", help="CSV of new flows to classify (same columns as the dataset)")
    parser.add_argument("--output", default="predictions.csv", help="where to write predictions for --input")
    args = parser.parse_args()
    cfg = load_config(args.config)
    pd.set_option("display.width", 200)
    bundle = load_model(cfg["output"]["model_path"])

    if args.input:
        predictions = predict_flows(bundle, pd.read_csv(args.input))
        predictions.to_csv(args.output, index=False)
        print(predictions.to_string())
        print(f"\n{predictions['alarm'].sum()} alarms out of {len(predictions)} flows, saved to {args.output}")
    else:
        evaluate_on_test(bundle, cfg)


if __name__ == "__main__":
    main()

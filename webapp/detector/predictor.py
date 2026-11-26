"""Bridge between the website and the project's trained models (models/best_model.pkl)."""
import sys
from functools import lru_cache
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))  # makes the project's `src` package importable

from src.data import load_raw  # noqa: E402
from src.model import predict_flows  # noqa: E402
from src.utils import load_config, load_model  # noqa: E402


@lru_cache
def config():
    """Project configuration, with its relative paths made absolute (the server runs from webapp/)."""
    cfg = load_config(PROJECT_ROOT / "configs" / "config.yaml")
    cfg["data"]["raw_dir"] = str(PROJECT_ROOT / cfg["data"]["raw_dir"])
    return cfg


@lru_cache
def bundle():
    """The saved two-stage system, loaded once per server process."""
    return load_model(PROJECT_ROOT / config()["output"]["model_path"])


def category_choices():
    """Values of each text feature seen during training, as {column: [values]}.

    They come from the fitted one-hot encoder, so the <select> menus offer exactly
    the values the model knows.
    """
    encoder = bundle()["binary"].named_steps["prep"].named_transformers_["cat"]
    return {name: list(values) for name, values in zip(encoder.feature_names_in_, encoder.categories_)}


@lru_cache
def examples():
    """One real test flow per category that the model classifies correctly: {category: {feature: value}}.

    The model is wrong on part of the test flows (e.g. 16.8% of normal flows raise an
    alarm), so a random example could start the demo with a mistake. We draw up to 50
    random flows per category and keep the first correct one (or the first one, if none is).
    """
    _, test = load_raw(config())
    candidates = test.sample(frac=1, random_state=config()["seed"]).groupby("attack_cat").head(50)
    correct = predict_flows(bundle(), candidates)["predicted category"] == candidates["attack_cat"]
    chosen = pd.concat([candidates[correct], candidates]).groupby("attack_cat").head(1)
    flows = {row["attack_cat"]: row[bundle()["features"]].to_dict() for _, row in chosen.iterrows()}
    return dict(sorted(flows.items(), key=lambda item: (item[0] != "Normal", item[0])))


def predict(flow):
    """Run the two-stage system on one flow given as {feature: value}.

    Returns the attack probability (in %), whether an alarm is raised, the predicted
    category (Normal when there is no alarm) and the alarm threshold (in %).
    """
    row = predict_flows(bundle(), pd.DataFrame([flow])).iloc[0]
    return {
        "percent": round(100 * float(row["P(attack)"]), 1),
        "alarm": bool(row["alarm"]),
        "category": row["predicted category"],
        "threshold": round(100 * bundle()["threshold"]),
    }

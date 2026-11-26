"""Small helpers: configuration, saving/loading models, saving figures and tables."""
import os

import joblib
import matplotlib.pyplot as plt
import yaml

import src.theme  # noqa: F401  (applies the project's plot theme)


def load_config(path="configs/config.yaml"):
    """Read the YAML configuration file into a dictionary."""
    with open(path) as f:
        return yaml.safe_load(f)


def save_model(obj, path):
    """Save any Python object (here: the trained models and their threshold) to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(obj, path, compress=3)  # compression keeps the random forests small enough for GitHub


def load_model(path):
    """Load an object written by save_model."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found: run `python main.py` first to train the models.")
    return joblib.load(path)


def save_figure(fig, folder, name):
    """Save a matplotlib figure as a PNG in `folder` and close it."""
    os.makedirs(folder, exist_ok=True)
    fig.savefig(os.path.join(folder, name), dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_table(df, folder, name):
    """Save a result table as CSV so that the numbers in the report can be traced back."""
    os.makedirs(folder, exist_ok=True)
    df.to_csv(os.path.join(folder, name))


def section(title):
    """Print a visible section header in the console output."""
    print(f"\n=== {title} ===")

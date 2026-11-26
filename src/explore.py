"""Exploration of the raw data: the tables and plots that motivate the preprocessing choices."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.theme import MID, OUTLINE, SERIES

def class_balance(train_file, train_dedup, test):
    """Number of flows per attack category in the train file (before and after de-duplication) and test file."""
    return pd.DataFrame({"train file": train_file["attack_cat"].value_counts(),
                         "train, no duplicates": train_dedup["attack_cat"].value_counts(),
                         "test file": test["attack_cat"].value_counts()}).sort_values("train file", ascending=False)


def data_checks(train_file, test, features):
    """Missing values, duplicate rows, and identical flows that carry different labels."""
    dedup = train_file.drop_duplicates(subset=features + ["attack_cat"])
    n_categories = dedup.groupby(features)["attack_cat"].nunique()
    n_labels = dedup.groupby(features)["label"].nunique()
    return pd.Series({
        "missing values (train file)": int(train_file.isna().sum().sum()),
        "missing values (test file)": int(test.isna().sum().sum()),
        "duplicate rows (train file)": len(train_file) - len(dedup),
        "share of duplicates": round(1 - len(dedup) / len(train_file), 3),
        "flows seen with 2+ categories": int((n_categories > 1).sum()),
        "flows seen as both normal and attack": int((n_labels > 1).sum()),
    }, dtype=object).to_frame("value")


def unseen_categories(train_file, test, categorical):
    """For each text column: number of values in the train file, and values that only appear in test."""
    return pd.DataFrame({"values in train": [train_file[c].nunique() for c in categorical],
                         "only in test": [sorted(set(test[c]) - set(train_file[c])) for c in categorical]},
                        index=categorical)


def skew_summary(df, columns):
    """Quantiles of the most skewed numeric columns: the median is tiny compared with the maximum."""
    return df[columns].describe(percentiles=[0.5, 0.99]).T[["min", "50%", "99%", "max"]]


def ttl_attack_share(df, min_flows=500):
    """Normal and attack flows, and the share of attacks, for every common source TTL value."""
    table = pd.crosstab(df["sttl"], df["label"]).set_axis(["normal", "attack"], axis=1)
    table = table[table.sum(axis=1) > min_flows]
    table["attack %"] = (100 * table["attack"] / table.sum(axis=1)).round(1)
    return table


def plot_class_balance(counts):
    """Horizontal bars (log scale) of the class counts."""
    ax = counts.plot(kind="barh", logx=True, figsize=(8, 5), color=SERIES,
                     edgecolor=OUTLINE)
    ax.invert_yaxis()
    ax.set(xlabel="number of flows (log scale)", ylabel="", title="Flows per attack category")
    return ax.figure


def plot_log_transform(df, column="sbytes"):
    """Histogram of a column before and after log(1 + x)."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    axes[0].hist(df[column], bins=50, color=MID, edgecolor=OUTLINE, linewidth=0.3)
    axes[0].set(yscale="log", title=f"{column} (raw, log-scaled counts)")
    axes[1].hist(np.log1p(df[column]), bins=50, color=MID, edgecolor=OUTLINE, linewidth=0.3)
    axes[1].set(title=f"log(1 + {column})")
    fig.tight_layout()
    return fig

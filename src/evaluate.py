"""Evaluation metrics, error analysis and their plots."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (ConfusionMatrixDisplay, RocCurveDisplay, accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score)

from src.theme import DARK, GREEN_RAMP, LIGHT, MID, NEUTRAL, OUTLINE


def binary_scores(y_true, p_attack, t):
    """Detection metrics when every flow with P(attack) >= t raises an alarm."""
    pred = (np.asarray(p_attack) >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    return {
        "accuracy": accuracy_score(y_true, pred),
        "detection rate": recall_score(y_true, pred),  # attacks caught / all attacks
        "precision": precision_score(y_true, pred),    # real attacks / all alarms
        "false alarm rate": fp / (fp + tn),            # normal flows flagged / all normal flows
        "F1": f1_score(y_true, pred),
        "ROC AUC": roc_auc_score(y_true, p_attack),    # independent of t
    }


def detection_by_category(attack_cat, alarm):
    """Share of the flows of each category that raise an alarm (for Normal: the false alarm rate)."""
    table = pd.Series(np.asarray(alarm), index=attack_cat.index).groupby(attack_cat).agg(["count", "mean"])
    table.columns = ["flows", "share flagged"]
    return table.sort_values("flows", ascending=False)


def false_alarm_rate_by(df, alarm, column, min_flows=1):
    """False alarm rate of the normal flows of df, grouped by `column` (e.g. sttl or service)."""
    normal = (df["label"] == 0).to_numpy()
    flagged = pd.Series(np.asarray(alarm)[normal], index=df.index[normal])
    table = flagged.groupby(df.loc[normal, column]).agg(["count", "mean"])
    table.columns = ["normal flows", "false alarm rate"]
    return table[table["normal flows"] >= min_flows]


def ttl_shift(train, test, values=(0, 31, 62, 254)):
    """Share (%) of normal flows with each sttl value in the train and test files."""
    share = lambda df: 100 * df.loc[df["label"] == 0, "sttl"].value_counts(normalize=True)
    return pd.DataFrame({"% of normal train flows": share(train),
                         "% of normal test flows": share(test)}).reindex(list(values)).fillna(0)


def multiclass_report(y_true, pred):
    """Per-class precision, recall, F1 and support, plus the macro and weighted averages."""
    report = classification_report(y_true, pred, output_dict=True, zero_division=0)
    report.pop("accuracy")
    return pd.DataFrame(report).T.sort_values("support", ascending=False)


def plot_confusion_and_roc(y_true, p_attack, t):
    """Left: binary confusion matrix at threshold t. Right: ROC curve with the chosen operating point."""
    pred = (np.asarray(p_attack) >= t).astype(int)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    ConfusionMatrixDisplay.from_predictions(y_true, pred, display_labels=["normal", "attack"],
                                            values_format="d", cmap=GREEN_RAMP, ax=axes[0])
    axes[0].set_title(f"Confusion matrix (threshold {t})")
    RocCurveDisplay.from_predictions(y_true, p_attack, name="test set", ax=axes[1],
                                     curve_kwargs={"color": DARK, "linewidth": 2})
    scores = binary_scores(y_true, p_attack, t)
    axes[1].plot(scores["false alarm rate"], scores["detection rate"], "o", color=DARK, markersize=8, markeredgecolor="white", label=f"threshold {t}")
    axes[1].plot([0, 1], [0, 1], ":", color=NEUTRAL)
    axes[1].set(xlabel="false alarm rate", ylabel="detection rate", title="ROC curve")
    axes[1].legend(loc="lower right")
    fig.tight_layout()
    return fig


def plot_false_alarms(by_sttl, by_service):
    """Bar charts of the false alarm rate per sttl value and per service."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
    by_sttl["false alarm rate"].plot(kind="bar", rot=0, ax=axes[0], color=MID, edgecolor=OUTLINE)
    axes[0].set_title("False alarm rate by sttl")
    by_service["false alarm rate"].plot(kind="bar", rot=0, ax=axes[1], color=MID, edgecolor=OUTLINE)
    axes[1].set_title("False alarm rate by service")
    fig.tight_layout()
    return fig


def plot_grouped_bars(table, ylabel, title):
    """Grouped bar chart of a table: one group per row, one green shade per column."""
    ax = table.plot(kind="bar", rot=0, figsize=(8, 3.8), edgecolor=OUTLINE,
                    color=[LIGHT, MID, DARK][:table.shape[1]])  # light to dark follows the column order
    ax.set(ylabel=ylabel, xlabel="", title=title)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=table.shape[1], fontsize=8, frameon=False)
    return ax.figure


def plot_multiclass_confusion(y_true, pred, labels):
    """Confusion matrix normalized per true class: the diagonal is the recall of each category."""
    fig, ax = plt.subplots(figsize=(8, 7))
    ConfusionMatrixDisplay.from_predictions(y_true, pred, labels=labels, normalize="true", values_format=".2f",
                                            cmap=GREEN_RAMP, ax=ax, colorbar=False, xticks_rotation=45)
    ax.set_title("Attack categories: share of each true class (rows) per prediction")
    fig.tight_layout()
    return fig

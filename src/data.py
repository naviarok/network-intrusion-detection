"""Loading UNSW-NB15 and building the train / validation / test sets."""
import io
import os
import urllib.request
import zipfile
from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

CATEGORICAL = ["proto", "service", "state"]
NOT_FEATURES = ["id", "label", "attack_cat"]


@dataclass
class Datasets:
    """The three data splits and the names of the feature columns."""
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    numeric: list
    categorical: list

    @property
    def features(self):
        """All feature names: numeric first, then categorical."""
        return self.numeric + self.categorical


def download_data(cfg):
    """Download the official partition CSVs (about 12 MB zipped) into the raw data folder."""
    raw_dir, wanted = cfg["data"]["raw_dir"], [cfg["data"]["train_file"], cfg["data"]["test_file"]]
    os.makedirs(raw_dir, exist_ok=True)
    archive = zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(cfg["data"]["url"]).read()))
    for name in archive.namelist():
        if os.path.basename(name) in wanted:
            with open(os.path.join(raw_dir, os.path.basename(name)), "wb") as f:
                f.write(archive.read(name))


def load_raw(cfg):
    """Return the raw (train file, test file) DataFrames, downloading them first if needed."""
    paths = [os.path.join(cfg["data"]["raw_dir"], cfg["data"][key]) for key in ("train_file", "test_file")]
    if not all(os.path.exists(p) for p in paths):
        download_data(cfg)
    return pd.read_csv(paths[0]), pd.read_csv(paths[1])


def feature_columns(df, drop=()):
    """Return (numeric, categorical) feature names, leaving out `drop`, the targets and `id` (a row number)."""
    numeric = [c for c in df.columns if c not in NOT_FEATURES + CATEGORICAL + list(drop)]
    categorical = [c for c in CATEGORICAL if c not in drop]
    return numeric, categorical


def remove_duplicates(df, features):
    """Drop rows whose features and attack category repeat an earlier row.

    Copies would end up on both sides of the train/validation split and make the
    validation score look better than it is.
    """
    return df.drop_duplicates(subset=features + ["attack_cat"])


def load_datasets(cfg, remove_dups=None, drop_features=None):
    """Prepare the data used by every script.

    The train file is (optionally) de-duplicated, then split into train and validation,
    stratified on attack_cat so that rare categories such as Worms appear in both parts.
    The official test file is left untouched. The two keyword arguments override the
    config (used by the preprocessing study).
    """
    remove_dups = cfg["data"]["remove_duplicates"] if remove_dups is None else remove_dups
    drop = cfg["preprocessing"]["drop_features"] if drop_features is None else drop_features

    train_file, test = load_raw(cfg)
    numeric, categorical = feature_columns(train_file, drop)
    if remove_dups:
        # duplicates are judged on all 42 features, even when some are dropped from the model
        train_file = remove_duplicates(train_file, feature_columns(train_file)[0] + CATEGORICAL)
    train, val = train_test_split(train_file, test_size=cfg["data"]["val_size"],
                                  stratify=train_file["attack_cat"], random_state=cfg["seed"])
    return Datasets(train, val, test, numeric, categorical)

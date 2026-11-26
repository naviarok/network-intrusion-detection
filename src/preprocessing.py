"""Turns raw flow records into the numeric matrix the models are trained on."""
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler


def numeric_transform(scaling):
    """Transformation applied to the numeric columns.

    - "none": raw values (byte counts reach 10^7 while flags are 0/1).
    - "standard": subtract the mean, divide by the standard deviation.
    - "log_standard": log(1 + x) first, which compresses the very skewed counts, then standardize.
    All numeric columns are >= 0, so log(1 + x) is always defined.
    """
    if scaling == "none":
        return "passthrough"
    if scaling == "standard":
        return StandardScaler()
    if scaling == "log_standard":
        return make_pipeline(FunctionTransformer(np.log1p, feature_names_out="one-to-one"), StandardScaler())
    raise ValueError(f"unknown scaling {scaling!r}")


def build_preprocessor(numeric, categorical, scaling="log_standard", min_frequency=100):
    """Column-wise preprocessing: numeric_transform on numbers, one-hot encoding on text columns.

    Categories seen fewer than `min_frequency` times in the training rows share one
    "infrequent" column, and categories never seen in training (state ACC and CLO only
    appear in the test file) are sent to that column instead of raising an error.
    Everything is learned in fit(), which is only ever called on the training rows.
    """
    one_hot = OneHotEncoder(min_frequency=min_frequency, handle_unknown="infrequent_if_exist",
                            sparse_output=False)
    return ColumnTransformer([("num", numeric_transform(scaling), numeric), ("cat", one_hot, categorical)])

"""
preprocessing.py
================
Full reproducible preprocessing pipeline for the US Wildfire
fire_size_class classification task.

Designed to be imported from the project notebook:
    from preprocessing import build_pipeline, load_and_split

Authors: [your team names]
"""

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    LabelEncoder,
    OneHotEncoder,
    OrdinalEncoder,
    RobustScaler,
    StandardScaler,
)
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TARGET = "fire_size_class"

# Direct encodings of the target — catastrophic leakage if included
LEAKAGE_COLS = [
    "fire_size",       # perfectly correlated with class
    "fire_mag",        # unique value per class (B=0.1, C=1.0, ...)
    "fire_size_class", # the target itself
]

METADATA_COLS = [
    "Unnamed: 0.1", "Unnamed: 0",
    "fire_name",        # 53% missing, free-text
    "wstation_usaf", "wstation_wban",
    "wstation_byear", "wstation_eyear",
    "weather_file",
    "dstation_m",
]

# Note: disc_clean_date is KEPT — DateFeatureExtractor parses it
SPARSE_DATE_COLS = [
    "cont_clean_date", "disc_date_final", "cont_date_final",
    "putout_time", "disc_date_pre", "disc_pre_month",
]

WEATHER_SENTINEL = -1.0

WEATHER_COLS = [
    "Temp_pre_30", "Temp_pre_15", "Temp_pre_7", "Temp_cont",
    "Wind_pre_30", "Wind_pre_15", "Wind_pre_7", "Wind_cont",
    "Hum_pre_30",  "Hum_pre_15",  "Hum_pre_7",  "Hum_cont",
    "Prec_pre_30", "Prec_pre_15", "Prec_pre_7", "Prec_cont",
]

CONTINUOUS_COLS  = ["latitude", "longitude", "remoteness"]
CATEGORICAL_COLS = ["stat_cause_descr", "state"]
MONTH_COL        = "discovery_month"
VEGETATION_COL   = "Vegetation"
YEAR_COL         = "disc_pre_year"

MONTH_ORDER = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
]

CLASS_ORDER = ["B", "C", "D", "E", "F", "G"]


# ---------------------------------------------------------------------------
# Custom transformers
# ---------------------------------------------------------------------------

class SentinelToNaN(BaseEstimator, TransformerMixin):
    """
    Replace sentinel -1 values with np.nan before imputation.

    Justification: all 16 weather columns use -1 as a 'no data' code.
    25.7% of rows (14,235) are affected — all the same rows across all
    columns. Replacing with NaN before median imputation avoids
    contaminating fit-time statistics with the sentinel value.
    KNN imputation was rejected: with all weather cols missing on the
    same rows simultaneously, KNN has no complete neighbours and silently
    degrades to column median anyway, at 10x the compute cost.
    """

    def __init__(self, sentinel=WEATHER_SENTINEL):
        self.sentinel = sentinel

    def fit(self, X, y=None):
        self.is_fitted_ = True
        return self

    def transform(self, X):
        X = X.copy().astype(float)
        X[X == self.sentinel] = np.nan
        return X


class DateFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Parse disc_clean_date (MM/DD/YYYY) and extract cyclical time features.

    Output features:
      - month_num, day_of_year  : raw numeric
      - month_sin, month_cos    : cyclical encoding (Dec-Jan continuity)
      - doy_sin, doy_cos        : cyclical day-of-year encoding

    Cyclical encoding is preferred over raw integers: month 12 and month 1
    are adjacent in the calendar but 11 steps apart as raw integers.
    No fit-time state is used — zero leakage risk.
    """

    def fit(self, X, y=None):
        self.is_fitted_ = True
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            dates = pd.to_datetime(X.iloc[:, 0], format="mixed", errors="coerce")
        else:
            dates = pd.to_datetime(X, format="mixed", errors="coerce")

        month = dates.dt.month.fillna(6).astype(int)
        doy   = dates.dt.dayofyear.fillna(180).astype(int)

        result = pd.DataFrame({
            "month_num":   month,
            "day_of_year": doy,
            "month_sin":   np.sin(2 * np.pi * month / 12),
            "month_cos":   np.cos(2 * np.pi * month / 12),
            "doy_sin":     np.sin(2 * np.pi * doy / 366),
            "doy_cos":     np.cos(2 * np.pi * doy / 366),
        }, index=dates.index)

        return result.values

    def get_feature_names_out(self, input_features=None):
        return ["month_num", "day_of_year",
                "month_sin", "month_cos", "doy_sin", "doy_cos"]


class WeatherDeltaEngineer(BaseEstimator, TransformerMixin):
    """
    Engineer 8 trend features from the 30/15/7-day weather windows.

    For each variable (Temp, Wind, Hum, Prec):
      - <var>_trend_30_7 = value_7day - value_30day
      - <var>_trend_15_7 = value_7day - value_15day

    Positive trend = drying/warming in the week before the fire.
    Capturing the *change* in conditions is more predictive of fire
    severity than absolute values alone. Computed post-imputation,
    so no NaN propagation.
    """

    def fit(self, X, y=None):
        self._n_in = X.shape[1]
        self.is_fitted_ = True
        return self

    def transform(self, X):
        X = np.array(X)
        deltas = []
        for base in [0, 4, 8, 12]:   # Temp, Wind, Hum, Prec
            col_30 = X[:, base]
            col_15 = X[:, base + 1]
            col_7  = X[:, base + 2]
            deltas.append(col_7 - col_30)
            deltas.append(col_7 - col_15)
        return np.column_stack([X] + deltas)

    def get_feature_names_out(self, input_features=None):
        base_names = list(WEATHER_COLS)
        delta_names = []
        for var in ["Temp", "Wind", "Hum", "Prec"]:
            delta_names += [f"{var}_trend_30_7", f"{var}_trend_15_7"]
        return base_names + delta_names


# ---------------------------------------------------------------------------
# Data loading and splitting
# ---------------------------------------------------------------------------

def load_and_split(filepath, test_size=0.15, val_size=0.15, random_state=42):
    """
    Load CSV, drop leakage/metadata columns, and return a stratified
    three-way split: train / validation / test.

    Split rationale (55,367 records):
      70% train  (~38,757) — sufficient for CV within training set
      15% val    (~8,305)  — hyperparameter tuning reference
      15% test   (~8,305)  — held out, touched only for final evaluation

    Stratification on the target preserves the 66% class-B distribution
    across all three splits.
    """
    df = pd.read_csv(filepath)

    # Extract target BEFORE dropping leakage columns
    y = df[TARGET].copy()

    drop_cols = LEAKAGE_COLS + METADATA_COLS + SPARSE_DATE_COLS
    drop_cols = [c for c in drop_cols if c in df.columns]
    X = df.drop(columns=drop_cols)

    # Split 1: (train + val) vs test
    X_tv, X_test, y_tv, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # Split 2: train vs val
    val_frac_of_tv = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv,
        test_size=val_frac_of_tv, stratify=y_tv, random_state=random_state
    )

    print(f"Split  →  train: {len(X_train):,}  |  val: {len(X_val):,}  |  test: {len(X_test):,}")
    print(f"Class distribution (train):\n{y_train.value_counts(normalize=True).round(3)}\n")

    return X_train, X_val, X_test, y_train, y_val, y_test


# ---------------------------------------------------------------------------
# Pipeline builder
# ---------------------------------------------------------------------------

def build_pipeline(classifier=None):
    """
    Build a full leakage-free sklearn Pipeline.

    Structure:
        raw DataFrame
          └── ColumnTransformer
                ├── weather     : SentinelToNaN → median impute → WeatherDeltaEngineer → RobustScaler
                ├── continuous  : StandardScaler
                ├── categorical : OneHotEncoder (handle_unknown='ignore')
                ├── month       : OrdinalEncoder (Jan=0 … Dec=11)
                ├── vegetation  : OneHotEncoder (7 habitat codes)
                ├── year        : StandardScaler
                └── date        : DateFeatureExtractor (sin/cos cyclical features)
          └── classifier (optional)

    All fit-time statistics are learned on training data only.
    Pass classifier=None to return the preprocessor for feature inspection.
    """

    # Branch 1: Weather — RobustScaler chosen because Prec_pre_7 has 18.2%
    # outliers (IQR method) identified in EDA. RobustScaler uses median/IQR
    # and is unaffected by extreme precipitation spikes.
    weather_branch = Pipeline([
        ("sentinel_to_nan", SentinelToNaN(sentinel=WEATHER_SENTINEL)),
        ("median_impute",   SimpleImputer(strategy="median")),
        ("delta_engineer",  WeatherDeltaEngineer()),
        ("scaler",          RobustScaler()),
    ])

    # Branch 2: Continuous — no missing values, mild outliers
    continuous_branch = Pipeline([
        ("scaler", StandardScaler()),
    ])

    # Branch 3: Categorical — handle_unknown='ignore' means unseen categories
    # in val/test produce an all-zero row rather than raising an error
    categorical_branch = Pipeline([
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    # Branch 4: Month — ordinal preserves calendar order for tree models;
    # cyclical sin/cos in the date branch complements this for distance models
    month_branch = Pipeline([
        ("ordinal", OrdinalEncoder(
            categories=[MONTH_ORDER],
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )),
    ])

    # Branch 5: Vegetation — 7 nominal habitat codes, no ordinal relationship
    vegetation_branch = Pipeline([
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    # Branch 6: Year — captures any long-term trend across 1991-2015
    year_branch = Pipeline([
        ("scaler", StandardScaler()),
    ])

    # Branch 7: Date string → 6 cyclical time features
    date_branch = Pipeline([
        ("date_features", DateFeatureExtractor()),
    ])

    # remainder='drop' ensures no raw column passes through untransformed
    preprocessor = ColumnTransformer(
        transformers=[
            ("weather",     weather_branch,     WEATHER_COLS),
            ("continuous",  continuous_branch,  CONTINUOUS_COLS),
            ("categorical", categorical_branch, CATEGORICAL_COLS),
            ("month",       month_branch,       [MONTH_COL]),
            ("vegetation",  vegetation_branch,  [VEGETATION_COL]),
            ("year",        year_branch,        [YEAR_COL]),
            ("date",        date_branch,        ["disc_clean_date"]),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )

    steps = [("preprocessor", preprocessor)]
    if classifier is not None:
        steps.append(("classifier", classifier))

    return Pipeline(steps)


# ---------------------------------------------------------------------------
# Label encoder helper
# ---------------------------------------------------------------------------

def encode_labels(y_train, y_val=None, y_test=None):
    """
    Encode string labels (B–G) to integers 0–5 using a fixed mapping.
    Mapping is defined from CLASS_ORDER, not fitted on data — deterministic
    and leakage-free.
    """
    le = LabelEncoder()
    le.classes_ = np.array(CLASS_ORDER)

    y_train_enc = le.transform(y_train)
    results = [y_train_enc, le]

    if y_val is not None:
        results.insert(1, le.transform(y_val))
    if y_test is not None:
        results.insert(-1, le.transform(y_test))

    return results


# ---------------------------------------------------------------------------
# Smoke test — run: python src/preprocessing.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os
    DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "FW_Veg_Rem_Combined.csv")

    print("=" * 60)
    print("Wildfire preprocessing pipeline — smoke test")
    print("=" * 60)

    X_train, X_val, X_test, y_train, y_val, y_test = load_and_split(DATA_PATH)

    pipeline = build_pipeline(classifier=None)
    X_train_t = pipeline.fit_transform(X_train)
    X_val_t   = pipeline.transform(X_val)
    X_test_t  = pipeline.transform(X_test)

    print(f"X_train shape : {X_train_t.shape}")
    print(f"X_val   shape : {X_val_t.shape}")
    print(f"X_test  shape : {X_test_t.shape}")
    print(f"NaN anywhere  : {np.isnan(np.vstack([X_train_t, X_val_t, X_test_t])).any()}")
    print("\nSmoke test passed.")

"""Feature Engineering and Scaler Module.

Handles:
1. Cyclical time encoding (Day sin/cos, Year sin/cos).
2. Strict chronological splitting (70% Train, 15% Val, 15% Test).
3. Zero-leakage scaling (StandardScaler fit ONLY on training set).
4. Inverse transformation for metric evaluation in physical units.
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from src.config import (
    CLEANED_DATA_PATH,
    TARGET_COLS,
    ALL_FEATURE_COLS,
    TRAIN_SPLIT,
    VAL_SPLIT,
    TEST_SPLIT,
    RESULTS_DIR
)

SCALER_PATH = RESULTS_DIR / "scaler.pkl"
TARGET_SCALER_PATH = RESULTS_DIR / "target_scaler.pkl"


def add_cyclical_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds continuous sinusoidal representations for daily and annual cycles."""
    df_feat = df.copy()

    # Ensure index is datetime
    if not isinstance(df_feat.index, pd.DatetimeIndex):
        df_feat.index = pd.to_datetime(df_feat.index)

    # Convert timestamps to seconds since epoch
    timestamp_s = df_feat.index.map(pd.Timestamp.timestamp)

    # Daily cycle (seconds in a day = 24 * 3600 = 86400)
    day = 24 * 60 * 60
    df_feat["Day_sin"] = np.sin(timestamp_s * (2 * np.pi / day))
    df_feat["Day_cos"] = np.cos(timestamp_s * (2 * np.pi / day))

    # Annual cycle (seconds in a tropical year = 365.2425 * 86400 = 31556952)
    year = 365.2425 * day
    df_feat["Year_sin"] = np.sin(timestamp_s * (2 * np.pi / year))
    df_feat["Year_cos"] = np.cos(timestamp_s * (2 * np.pi / year))

    return df_feat


def add_calendar_features(df: pd.DataFrame, include_cyclical: bool = True) -> pd.DataFrame:
    """Creates explicit calendar features (hour, day_of_year, season) alongside optional cyclical time embeddings."""
    df_feat = df.copy()
    if not isinstance(df_feat.index, pd.DatetimeIndex):
        df_feat.index = pd.to_datetime(df_feat.index)

    # Raw calendar features
    df_feat["hour"] = df_feat.index.hour
    df_feat["day_of_year"] = df_feat.index.dayofyear
    # Season mapping: 1=Winter, 2=Spring, 3=Summer, 4=Autumn
    df_feat["season"] = (df_feat.index.month % 12 // 3) + 1

    if include_cyclical:
        df_feat = add_cyclical_time_features(df_feat)

    return df_feat


def add_lag_features(df: pd.DataFrame, target_cols=TARGET_COLS, lags=(1, 24)) -> pd.DataFrame:
    """Creates historical lag features for key target variables (e.g. t-1 hour, t-24 hour)."""
    df_feat = df.copy()
    for col in target_cols:
        if col in df_feat.columns:
            for lag in lags:
                col_clean = col.split()[0] # e.g. 'T' or 'p'
                lag_col_name = f"{col_clean}_lag{lag}"
                df_feat[lag_col_name] = df_feat[col].shift(lag)

    # Backfill initial lag rows to prevent NaNs
    df_feat = df_feat.bfill()
    return df_feat


def chronological_split(df: pd.DataFrame):
    """Splits data strictly chronologically: 70% Train, 15% Val, 15% Test."""
    n = len(df)
    train_end = int(n * TRAIN_SPLIT)
    val_end = int(n * (TRAIN_SPLIT + VAL_SPLIT))

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    print(f"[+] Train set: {train_df.index[0]} to {train_df.index[-1]} ({len(train_df):,} samples, {len(train_df)/n*100:.1f}%)")
    print(f"[+] Val set:   {val_df.index[0]} to {val_df.index[-1]} ({len(val_df):,} samples, {len(val_df)/n*100:.1f}%)")
    print(f"[+] Test set:  {test_df.index[0]} to {test_df.index[-1]} ({len(test_df):,} samples, {len(test_df)/n*100:.1f}%)")

    return train_df, val_df, test_df


class ZeroLeakagePreprocessor:
    """Standardizes features and targets strictly using training statistics."""

    def __init__(self, feature_cols=ALL_FEATURE_COLS, target_cols=TARGET_COLS):
        self.feature_cols = feature_cols
        self.target_cols = target_cols
        self.feature_scaler = StandardScaler()
        self.target_scaler = StandardScaler()
        self.is_fitted = False

    def fit(self, train_df: pd.DataFrame):
        """Fits scalers ONLY on the training split."""
        self.feature_scaler.fit(train_df[self.feature_cols])
        self.target_scaler.fit(train_df[self.target_cols])
        self.is_fitted = True

        # Save scalers for downstream evaluation
        with open(SCALER_PATH, "wb") as f:
            pickle.dump(self.feature_scaler, f)
        with open(TARGET_SCALER_PATH, "wb") as f:
            pickle.dump(self.target_scaler, f)
        print(f"[+] Scalers fitted and saved to {RESULTS_DIR}")

    def transform(self, df: pd.DataFrame):
        """Transforms features and targets using the fitted scalers."""
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fitted before transforming!")

        X_scaled = self.feature_scaler.transform(df[self.feature_cols])
        y_scaled = self.target_scaler.transform(df[self.target_cols])

        return X_scaled, y_scaled

    def inverse_transform_targets(self, y_scaled: np.ndarray) -> np.ndarray:
        """Inverse-transforms target predictions back to physical units (°C, mbar, %, m/s)."""
        return self.target_scaler.inverse_transform(y_scaled)


def build_preprocessed_splits(data_path: Path = CLEANED_DATA_PATH):
    """End-to-end pipeline: load -> add cyclical features -> chronological split -> zero-leakage scale."""
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    df_with_features = add_cyclical_time_features(df)

    train_df, val_df, test_df = chronological_split(df_with_features)

    preprocessor = ZeroLeakagePreprocessor()
    preprocessor.fit(train_df)

    X_train, y_train = preprocessor.transform(train_df)
    X_val, y_val = preprocessor.transform(val_df)
    X_test, y_test = preprocessor.transform(test_df)

    return (X_train, y_train), (X_val, y_val), (X_test, y_test), preprocessor, (train_df, val_df, test_df)


if __name__ == "__main__":
    build_preprocessed_splits()

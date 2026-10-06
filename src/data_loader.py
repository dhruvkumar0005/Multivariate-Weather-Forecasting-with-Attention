"""Data Loading, Auditing, and Cleaning Module.

Handles:
1. Raw CSV loading and datetime parsing.
2. Anomaly identification and sensor artifact replacement (e.g. wv = -9999 m/s).
3. Hourly aggregation (resampling to 1-hour intervals).
4. Missing value interpolation.
5. Exporting cleaned hourly dataset for downstream modeling.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from src.config import (
    RAW_DATA_PATH,
    CLEANED_DATA_PATH,
    DATETIME_COL,
    TARGET_COLS,
    PHYSICAL_BOUNDS
)


def load_raw_data(filepath: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Loads the raw Jena Climate dataset and sets the datetime index."""
    if not filepath.exists():
        raise FileNotFoundError(f"Raw dataset not found at {filepath}")

    print(f"[*] Loading raw dataset from: {filepath}")
    df = pd.read_csv(filepath)
    print(f"[+] Loaded {len(df):,} rows with {df.shape[1]} columns.")

    # Parse datetime (format: DD.MM.YYYY HH:MM:SS)
    print(f"[*] Parsing '{DATETIME_COL}' column...")
    df[DATETIME_COL] = pd.to_datetime(df[DATETIME_COL], format="%d.%m.%Y %H:%M:%S")
    df = df.sort_values(DATETIME_COL).reset_index(drop=True)
    df.set_index(DATETIME_COL, inplace=True)

    return df


def audit_dataset(df: pd.DataFrame) -> dict:
    """Inspects dataset statistics, checks duplicates, nulls, and sensor anomalies."""
    print("\n--- DATASET INTEGRITY AUDIT ---")
    audit_results = {}

    # Check duplicates
    duplicate_count = df.index.duplicated().sum()
    audit_results["duplicate_timestamps"] = int(duplicate_count)
    print(f"[-] Duplicate timestamps: {duplicate_count}")

    # Check missing values
    missing_counts = df.isna().sum().to_dict()
    audit_results["missing_values"] = missing_counts
    total_missing = sum(missing_counts.values())
    print(f"[-] Total missing cells in raw data: {total_missing}")

    # Check known sensor artifacts
    # The Jena dataset records faulty sensor values as -9999 for wind velocity
    faulty_wv = (df["wv (m/s)"] < 0).sum()
    faulty_max_wv = (df["max. wv (m/s)"] < 0).sum() if "max. wv (m/s)" in df.columns else 0
    audit_results["faulty_wv_count"] = int(faulty_wv)
    audit_results["faulty_max_wv_count"] = int(faulty_max_wv)
    print(f"[-] Faulty wind speed readings (wv < 0 m/s): {faulty_wv}")
    print(f"[-] Faulty max wind speed readings (max. wv < 0 m/s): {faulty_max_wv}")

    # Inspect physical bounds for key target variables
    print("\n--- Summary of Target Variables (Raw) ---")
    for col in TARGET_COLS:
        min_val = df[col].min()
        max_val = df[col].max()
        mean_val = df[col].mean()
        print(f"    {col:<12} | Min: {min_val:8.2f} | Max: {max_val:8.2f} | Mean: {mean_val:8.2f}")

    return audit_results


def clean_and_resample(df: pd.DataFrame) -> pd.DataFrame:
    """Cleans sensor anomalies, resamples from 10-minute to 1-hour resolution, and interpolates gaps."""
    print("\n--- CLEANING & RESAMPLING PIPELINE ---")
    cleaned_df = df.copy()

    # 1. Clean faulty wind velocity readings (replace -9999 or any negative value with NaN)
    for col in ["wv (m/s)", "max. wv (m/s)"]:
        if col in cleaned_df.columns:
            invalid_mask = cleaned_df[col] < 0
            if invalid_mask.any():
                print(f"[!] Replacing {invalid_mask.sum()} negative values in '{col}' with NaN")
                cleaned_df.loc[invalid_mask, col] = np.nan

    # 2. Resample to hourly intervals (mean aggregation)
    print("[*] Resampling from 10-minute intervals to hourly ('1h') mean...")
    hourly_df = cleaned_df.resample("1h").mean()
    print(f"[+] Resampled shape: {hourly_df.shape[0]:,} hourly rows (was {len(cleaned_df):,})")

    # 3. Handle any missing values arising from gaps
    null_count_before = hourly_df.isna().sum().sum()
    if null_count_before > 0:
        print(f"[!] Found {null_count_before} NaN values after resampling. Applying linear interpolation...")
        hourly_df = hourly_df.interpolate(method="time").bfill().ffill()
    null_count_after = hourly_df.isna().sum().sum()
    print(f"[+] Remaining NaN values after interpolation: {null_count_after}")

    # 4. Verify monotonic chronological ordering
    assert hourly_df.index.is_monotonic_increasing, "Index must be strictly increasing in time!"
    assert not hourly_df.index.has_duplicates, "Hourly index must have zero duplicate timestamps!"

    return hourly_df


def prepare_and_save_data(raw_path: Path = RAW_DATA_PATH, save_path: Path = CLEANED_DATA_PATH) -> pd.DataFrame:
    """Executes the full loading, auditing, cleaning, and saving workflow."""
    raw_df = load_raw_data(raw_path)
    audit_dataset(raw_df)
    clean_df = clean_and_resample(raw_df)

    print(f"\n[*] Saving cleaned hourly dataset to: {save_path}")
    clean_df.to_csv(save_path)
    print(f"[+] Successfully saved! File size: {save_path.stat().st_size / (1024 * 1024):.2f} MB")

    return clean_df


if __name__ == "__main__":
    prepare_and_save_data()

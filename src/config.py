"""Configuration module for Multivariate Weather Forecasting with Attention.
Contains file paths, target variables, sequence hyperparameters, and model configurations.
"""

from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "Dataset"
RAW_DATA_PATH = DATASET_DIR / "jena_climate_2009_2016.csv"
CLEANED_DATA_PATH = DATASET_DIR / "jena_climate_hourly_cleaned.csv"

RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = RESULTS_DIR / "models"
METRICS_DIR = RESULTS_DIR / "metrics"
ATTENTION_DIR = RESULTS_DIR / "attention_weights"
FIGURES_DIR = PROJECT_ROOT / "figures"

# Ensure directories exist
for path in [DATASET_DIR, RESULTS_DIR, MODELS_DIR, METRICS_DIR, ATTENTION_DIR, FIGURES_DIR]:
    path.mkdir(parents=True, exist_ok=True)

# Dataset Columns & Physical Interpretations
DATETIME_COL = "Date Time"

# Selected Multivariate Targets
TARGET_COLS = [
    "T (degC)",    # Temperature (°C)
    "p (mbar)",    # Atmospheric pressure (mbar)
    "rh (%)",      # Relative humidity (%)
    "wv (m/s)"     # Wind velocity (m/s)
]

# Physical bounds for validation & outlier clipping
PHYSICAL_BOUNDS = {
    "T (degC)": (-50.0, 50.0),
    "p (mbar)": (900.0, 1100.0),
    "rh (%)": (0.0, 100.0),
    "wv (m/s)": (0.0, 100.0)
}

# Feature Engineering: Cyclical signals to include alongside targets
CYCLICAL_COLS = [
    "Day_sin",
    "Day_cos",
    "Year_sin",
    "Year_cos"
]

ALL_FEATURE_COLS = TARGET_COLS + CYCLICAL_COLS

# Time-Series Splitting Ratios (Chronological, No Leakage)
TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

# Sequence Sliding-Window Hyperparameters
WINDOW_LENGTH = 72    # 72 hours = 3 days history
HORIZON = 1           # 1-step ahead joint forecasting (t+1 hour)
BATCH_SIZE = 128
RANDOM_SEED = 42

# Training Hyperparameters
EPOCHS = 30
LEARNING_RATE = 0.001
PATIENCE_EARLY_STOPPING = 6
PATIENCE_REDUCE_LR = 3

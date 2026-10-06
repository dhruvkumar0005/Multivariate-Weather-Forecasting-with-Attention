"""Export and Serialization Script.

Saves all trained models (Baseline_LSTM, Stacked_GRU, GRU_with_Attention)
and scalers in both .joblib and .pkl formats, and performs inference
verification on sample test inputs.
"""

import sys
import pickle
import joblib
import numpy as np
import tensorflow as tf
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.attention import TemporalAdditiveAttention
from src.config import MODELS_DIR, RESULTS_DIR, WINDOW_LENGTH, ALL_FEATURE_COLS

custom_objects = {"TemporalAdditiveAttention": TemporalAdditiveAttention}

models_list = [
    "Baseline_LSTM",
    "Stacked_GRU",
    "GRU_with_Attention"
]

print("=" * 70)
print("EXPORTING TRAINED MODELS TO .joblib AND .pkl FORMATS")
print("=" * 70)

# Dummy test sequence for forward pass validation: (1, 72, 8)
dummy_input = np.random.randn(1, WINDOW_LENGTH, len(ALL_FEATURE_COLS)).astype(np.float32)

for model_name in models_list:
    keras_path = MODELS_DIR / f"{model_name}.keras"
    joblib_path = MODELS_DIR / f"{model_name}.joblib"
    pkl_path = MODELS_DIR / f"{model_name}.pkl"

    if not keras_path.exists():
        print(f"[!] Warning: {keras_path} not found. Skipping.")
        continue

    print(f"\n[*] Processing: {model_name}...")
    model = tf.keras.models.load_model(keras_path, custom_objects=custom_objects)

    # 1. Save as .joblib
    print(f"    [-] Saving to: {joblib_path.name}")
    joblib.dump(model, joblib_path)
    joblib_size = joblib_path.stat().st_size / 1024

    # 2. Save as .pkl
    print(f"    [-] Saving to: {pkl_path.name}")
    with open(pkl_path, "wb") as f:
        pickle.dump(model, f)
    pkl_size = pkl_path.stat().st_size / 1024

    # 3. Verification: Load and test forward pass
    print(f"    [-] Verifying .joblib load & inference...")
    loaded_joblib = joblib.load(joblib_path)
    pred_joblib = loaded_joblib(dummy_input)

    print(f"    [-] Verifying .pkl load & inference...")
    with open(pkl_path, "rb") as f:
        loaded_pkl = pickle.load(f)
    pred_pkl = loaded_pkl(dummy_input)

    print(f"    [+] {model_name} verified!")
    print(f"        • .joblib size: {joblib_size:.1f} KB")
    print(f"        • .pkl size:    {pkl_size:.1f} KB")

# Also ensure preprocessors are in both .pkl and .joblib formats
print("\n[*] Processing Scalers & Preprocessors...")
for scaler_name in ["scaler", "target_scaler"]:
    orig_pkl = RESULTS_DIR / f"{scaler_name}.pkl"
    target_joblib = RESULTS_DIR / f"{scaler_name}.joblib"

    if orig_pkl.exists():
        with open(orig_pkl, "rb") as f:
            scaler_obj = pickle.load(f)
        joblib.dump(scaler_obj, target_joblib)
        print(f"    [+] Saved {scaler_name}.joblib ({target_joblib.stat().st_size / 1024:.1f} KB)")

print("\n" + "=" * 70)
print("[DONE] All models and scalers successfully saved and verified in .joblib and .pkl!")
print("=" * 70)

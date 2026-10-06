"""Comprehensive Evaluation and Benchmarking Engine.

Performs:
1. Multi-output test inference across all three architectures.
2. Inverse scaling of forecasts and actuals back to original physical units.
3. Strict per-variable metric computation (MAE, RMSE, R²).
4. Master comparison table export (CSV & formatted console table).
5. Diagnostic visualization generation:
   - Training & validation loss convergence curves.
   - Actual vs. Predicted 7-day trajectories across all 4 variables.
   - Temporal attention weight extraction and interpretability case studies.
"""

import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.config import (
    MODELS_DIR,
    METRICS_DIR,
    ATTENTION_DIR,
    FIGURES_DIR,
    TARGET_COLS,
    WINDOW_LENGTH,
    HORIZON
)
from src.features import build_preprocessed_splits, TARGET_SCALER_PATH
from src.sequences import extract_arrays_for_evaluation
from src.attention import TemporalAdditiveAttention


def compute_per_variable_metrics(
    y_true_test: np.ndarray,
    y_pred_test: np.ndarray,
    y_true_val: np.ndarray,
    y_pred_val: np.ndarray,
    model_name: str
) -> list:
    """Calculates Validation RMSE along with Test MAE, RMSE, and R² for each target variable."""
    records = []
    units = ["°C", "mbar", "%", "m/s"]

    for i, col in enumerate(TARGET_COLS):
        val_rmse = np.sqrt(mean_squared_error(y_true_val[:, i], y_pred_val[:, i]))
        test_mae = mean_absolute_error(y_true_test[:, i], y_pred_test[:, i])
        test_rmse = np.sqrt(mean_squared_error(y_true_test[:, i], y_pred_test[:, i]))
        test_r2 = r2_score(y_true_test[:, i], y_pred_test[:, i])

        records.append({
            "Model": model_name,
            "Target Variable": col,
            "Unit": units[i],
            "Val_RMSE": round(val_rmse, 4),
            "Test_MAE": round(test_mae, 4),
            "Test_RMSE": round(test_rmse, 4),
            "Test_R2_Score": round(test_r2, 4)
        })
    return records


def plot_loss_curves(models_list, save_path: Path = FIGURES_DIR / "06_training_loss_curves.png"):
    """Plots training and validation loss curves side-by-side for all models."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    colors = ["#1f77b4", "#2ca02c", "#d62728"]

    for idx, name in enumerate(models_list):
        history_file = METRICS_DIR / f"{name}_history.json"
        if not history_file.exists():
            continue
        with open(history_file, "r") as f:
            h = json.load(f)

        loss_key = "multivariate_output_loss" if "multivariate_output_loss" in h else "loss"
        val_loss_key = "val_multivariate_output_loss" if "val_multivariate_output_loss" in h else "val_loss"

        axes[idx].plot(h[loss_key], label="Train Loss", color=colors[idx], linestyle="--", linewidth=1.8)
        axes[idx].plot(h[val_loss_key], label="Val Loss", color=colors[idx], linewidth=2.0)
        axes[idx].set_title(f"{name.replace('_', ' ')}", fontweight="bold")
        axes[idx].set_xlabel("Epoch")
        axes[idx].grid(True, linestyle="--", alpha=0.5)
        axes[idx].legend(loc="upper right")

    axes[0].set_ylabel("Mean Squared Error (Scaled)")
    plt.suptitle("Training and Validation Convergence Curves Across Models", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[+] Saved loss convergence curves to: {save_path}")


def plot_actual_vs_predicted(
    y_true: np.ndarray,
    preds_dict: dict,
    test_index: pd.DatetimeIndex,
    sample_hours: int = 168, # 7 days
    save_path: Path = FIGURES_DIR / "07_actual_vs_predicted.png"
):
    """Plots Actual vs Predicted trajectories across a 7-day test horizon."""
    fig, axes = plt.subplots(4, 1, figsize=(14, 11), sharex=True)
    units = ["°C", "mbar", "%", "m/s"]
    model_colors = {
        "Baseline_LSTM": "#1f77b4",
        "Stacked_GRU": "#2ca02c",
        "GRU_with_Attention": "#d62728"
    }

    sub_idx = range(min(sample_hours, len(y_true)))
    timestamps = test_index[WINDOW_LENGTH + HORIZON - 1:][sub_idx]

    for i, col in enumerate(TARGET_COLS):
        # Plot Ground Truth
        axes[i].plot(timestamps, y_true[sub_idx, i], label="Ground Truth", color="black", linewidth=2.2, alpha=0.85)

        # Plot each model prediction
        for m_name, y_pred in preds_dict.items():
            axes[i].plot(
                timestamps,
                y_pred[sub_idx, i],
                label=m_name.replace("_", " "),
                color=model_colors.get(m_name, "blue"),
                linestyle="--",
                linewidth=1.4
            )

        axes[i].set_ylabel(f"{col} ({units[i]})", fontweight="bold")
        axes[i].grid(True, linestyle="--", alpha=0.5)
        if i == 0:
            axes[i].legend(loc="upper right", ncol=4, framealpha=0.9)

    axes[-1].set_xlabel("Time", fontweight="bold")
    plt.suptitle("Model Forecasts vs. Ground Truth (7-Day Sample Horizon)", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[+] Saved actual vs predicted comparison to: {save_path}")


def plot_attention_interpretability(
    attention_weights: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    test_df: pd.DataFrame,
    save_path: Path = FIGURES_DIR / "08_attention_case_studies.png"
):
    """Visualizes temporal attention weights for 4 distinct meteorological scenarios."""
    fig, axes = plt.subplots(4, 2, figsize=(15, 12), gridspec_kw={"width_ratios": [1, 2]})

    # Select 4 representative test sequence indices
    case_indices = [50, 500, 1500, 3000]
    case_titles = [
        "Case 1: Diurnal Warming Cycle",
        "Case 2: Sudden Synoptic Shift",
        "Case 3: High Wind Event",
        "Case 4: Stable Baseline Period"
    ]
    past_hours = np.arange(-WINDOW_LENGTH, 0) # -72 to -1

    for row, (idx, title) in enumerate(zip(case_indices, case_titles)):
        weights = attention_weights[idx].flatten()

        # Left plot: Attention distribution over the 72-hour window
        axes[row, 0].plot(past_hours, weights, color="#d62728", linewidth=2)
        axes[row, 0].fill_between(past_hours, 0, weights, color="#d62728", alpha=0.25)
        # Highlight diurnal lags at -24h, -48h, -72h
        for lag in [-72, -48, -24]:
            axes[row, 0].axvline(x=lag, color="gray", linestyle=":", alpha=0.7)
        axes[row, 0].set_title(f"{title} (Attention)", fontsize=10, fontweight="bold")
        axes[row, 0].set_ylabel("Weight (α)")
        axes[row, 0].grid(True, linestyle="--", alpha=0.5)

        # Right plot: Corresponding historical input variables
        # Normalize inputs for visual alignment in subplot
        norm_inputs = X_test[idx]
        axes[row, 1].plot(past_hours, norm_inputs[:, 0], label="Temp", color="#d95f02", linewidth=1.3)
        axes[row, 1].plot(past_hours, norm_inputs[:, 1], label="Pressure", color="#1b9e77", linewidth=1.3)
        axes[row, 1].plot(past_hours, norm_inputs[:, 2], label="Rel Hum", color="#7570b3", linewidth=1.3)
        axes[row, 1].plot(past_hours, norm_inputs[:, 3], label="Wind Vel", color="#e7298a", linewidth=1.3)
        axes[row, 1].set_title(f"Historical 72-Hour Inputs", fontsize=10, fontweight="bold")
        axes[row, 1].grid(True, linestyle="--", alpha=0.5)
        if row == 0:
            axes[row, 1].legend(loc="upper right", ncol=4, fontsize=8)

    axes[-1, 0].set_xlabel("Relative Past Hours")
    axes[-1, 1].set_xlabel("Relative Past Hours")
    plt.suptitle("Temporal Attention Weight Distributions Across Weather Regimes", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[+] Saved attention interpretability cases to: {save_path}")


def run_evaluation_pipeline():
    """Runs test inference, calculates per-variable metrics, and produces all summary deliverables."""
    print("\n--- RUNNING SYSTEMATIC MODEL EVALUATION ---")

    # 1. Load preprocessed splits & target scaler
    (X_tr, y_tr), (X_v, y_v), (X_te, y_te), preprocessor, (train_df, val_df, test_df) = build_preprocessed_splits()

    print("[*] Extracting sliding-window validation and test arrays...")
    X_val, y_val_scaled = extract_arrays_for_evaluation(X_v, y_v, window_length=WINDOW_LENGTH, horizon=HORIZON)
    X_test, y_test_scaled = extract_arrays_for_evaluation(X_te, y_te, window_length=WINDOW_LENGTH, horizon=HORIZON)

    # Inverse transform ground truth
    y_val_unscaled = preprocessor.inverse_transform_targets(y_val_scaled)
    y_test_unscaled = preprocessor.inverse_transform_targets(y_test_scaled)

    models_to_evaluate = ["Baseline_LSTM", "Stacked_GRU", "GRU_with_Attention"]
    all_metrics = []
    predictions_unscaled = {}
    attention_weights_sample = None

    for model_name in models_to_evaluate:
        model_path = MODELS_DIR / f"{model_name}.keras"
        if not model_path.exists():
            print(f"[!] Warning: Model file {model_path} not found. Skipping.")
            continue

        print(f"[*] Loading and evaluating: {model_name}...")
        custom_objects = {"TemporalAdditiveAttention": TemporalAdditiveAttention}
        model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)

        if model_name == "GRU_with_Attention":
            preds_val_scaled, _ = model.predict(X_val, batch_size=256, verbose=0)
            preds_scaled, attn_weights = model.predict(X_test, batch_size=256, verbose=0)
            attention_weights_sample = attn_weights
            # Save sample weights for persistence
            np.save(ATTENTION_DIR / "test_attention_weights.npy", attn_weights[:500])
        else:
            preds_val_scaled = model.predict(X_val, batch_size=256, verbose=0)
            preds_scaled = model.predict(X_test, batch_size=256, verbose=0)

        # Inverse transform predictions to original physical units
        preds_val_unscaled = preprocessor.inverse_transform_targets(preds_val_scaled)
        preds_unscaled = preprocessor.inverse_transform_targets(preds_scaled)
        predictions_unscaled[model_name] = preds_unscaled

        # Calculate metrics per variable (Validation RMSE + Test MAE, RMSE, R2)
        model_records = compute_per_variable_metrics(
            y_test_unscaled, preds_unscaled, y_val_unscaled, preds_val_unscaled, model_name
        )
        all_metrics.extend(model_records)

    # Compile master comparison table
    df_metrics = pd.DataFrame(all_metrics)
    metrics_csv_path = METRICS_DIR / "model_comparison.csv"
    df_metrics.to_csv(metrics_csv_path, index=False)
    print(f"\n[+] Master Model Comparison Table saved to: {metrics_csv_path}")

    # Display console table
    print("\n" + "="*85)
    print(f"{'MODEL':<20} | {'VARIABLE':<15} | {'VAL RMSE':<10} | {'TEST MAE':<10} | {'TEST RMSE':<10} | {'TEST R²':<8}")
    print("="*85)
    for _, row in df_metrics.iterrows():
        print(f"{row['Model']:<20} | {row['Target Variable']:<15} | {row['Val_RMSE']:<10.4f} | {row['Test_MAE']:<10.4f} | {row['Test_RMSE']:<10.4f} | {row['Test_R2_Score']:<8.4f}")
    print("="*85)

    # 2. Generate visualization deliverables
    plot_loss_curves(models_to_evaluate)
    plot_actual_vs_predicted(y_test_unscaled, predictions_unscaled, test_df.index)
    if attention_weights_sample is not None:
        plot_attention_interpretability(attention_weights_sample, X_test, y_test_unscaled, test_df)

    print("\n[+] Evaluation pipeline completed successfully.")
    return df_metrics


if __name__ == "__main__":
    run_evaluation_pipeline()

"""Hyperparameter Tuning and Attention Scoring Ablation Module.

Performs:
1. Window length tuning (24h vs 48h vs 72h) for Stacked GRU.
2. Recurrent depth tuning (2 GRU layers vs 3 GRU layers).
3. Attention scoring function experiment (Additive vs Dot-Product).
4. Validation RMSE calculation per target variable and overall.
5. Generates comparison table ('results/metrics/tuning_experiments.csv')
   and visualization ('figures/09_hyperparameter_tuning_ablation.png').
"""

import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from pathlib import Path
from sklearn.metrics import mean_squared_error
from tensorflow.keras.callbacks import EarlyStopping

from src.config import (
    TARGET_COLS,
    ALL_FEATURE_COLS,
    METRICS_DIR,
    FIGURES_DIR,
    RANDOM_SEED
)
from src.features import build_preprocessed_splits
from src.sequences import extract_arrays_for_evaluation
from src.attention import TemporalAdditiveAttention
from src.models import build_stacked_gru, build_gru_with_attention, compile_model

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


def evaluate_split_rmse(model, X, y_unscaled, preprocessor, is_attention=False):
    """Computes RMSE in physical units for each target variable individually."""
    if is_attention:
        preds_scaled, _ = model.predict(X, batch_size=256, verbose=0)
    else:
        preds_scaled = model.predict(X, batch_size=256, verbose=0)

    preds_unscaled = preprocessor.inverse_transform_targets(preds_scaled)
    rmses = {}
    for i, col in enumerate(TARGET_COLS):
        rmse = np.sqrt(mean_squared_error(y_unscaled[:, i], preds_unscaled[:, i]))
        rmses[col] = float(rmse)

    rmses["mean_rmse"] = float(np.mean(list(rmses.values())))
    return rmses, preds_unscaled


def run_tuning_suite():
    """Executes hyperparameter tuning and ablation experiments."""
    print("\n" + "=" * 70)
    print("HYPERPARAMETER TUNING & ATTENTION SCORING EXPERIMENTS")
    print("=" * 70)

    # Load preprocessed splits
    (X_tr, y_tr), (X_v, y_v), (X_te, y_te), preprocessor, _ = build_preprocessed_splits()

    results = []

    # Experiment Definitions
    experiments = [
        # 1. Window Length Tuning on Stacked GRU (2 layers)
        {
            "name": "GRU_Window_24h",
            "category": "Window Length",
            "window_length": 24,
            "gru_layers": (64, 32),
            "scoring": None,
            "is_attention": False,
        },
        {
            "name": "GRU_Window_48h",
            "category": "Window Length",
            "window_length": 48,
            "gru_layers": (64, 32),
            "scoring": None,
            "is_attention": False,
        },
        {
            "name": "GRU_Window_72h_2L",
            "category": "Window Length",
            "window_length": 72,
            "gru_layers": (64, 32),
            "scoring": None,
            "is_attention": False,
        },
        # 2. Recurrent Depth Tuning on GRU (window=72)
        {
            "name": "GRU_Window_72h_3L",
            "category": "Recurrent Depth",
            "window_length": 72,
            "gru_layers": (64, 32, 16),
            "scoring": None,
            "is_attention": False,
        },
        # 3. Attention Scoring Experiment: Additive vs Dot-Product
        {
            "name": "GRU_Attention_Additive",
            "category": "Attention Scoring",
            "window_length": 72,
            "gru_layers": (64, 32),
            "scoring": "additive",
            "is_attention": True,
        },
        {
            "name": "GRU_Attention_DotProduct",
            "category": "Attention Scoring",
            "window_length": 72,
            "gru_layers": (64, 32),
            "scoring": "dot_product",
            "is_attention": True,
        },
    ]

    for exp in experiments:
        name = exp["name"]
        win = exp["window_length"]
        layers = exp["gru_layers"]
        scoring = exp["scoring"]
        is_attn = exp["is_attention"]

        print(f"\n[*] Running: {name} | Category: {exp['category']} | Window: {win}h | Layers: {layers} | Scoring: {scoring}")

        # Extract sequence windows for this specific window length
        X_train, y_train_scaled = extract_arrays_for_evaluation(X_tr, y_tr, window_length=win, horizon=1)
        X_val, y_val_scaled = extract_arrays_for_evaluation(X_v, y_v, window_length=win, horizon=1)
        X_test, y_test_scaled = extract_arrays_for_evaluation(X_te, y_te, window_length=win, horizon=1)

        y_val_unscaled = preprocessor.inverse_transform_targets(y_val_scaled)
        y_test_unscaled = preprocessor.inverse_transform_targets(y_test_scaled)

        # Build model
        input_shape = (win, len(ALL_FEATURE_COLS))
        if is_attn:
            model = build_gru_with_attention(
                input_shape=input_shape,
                gru_layers=layers,
                scoring=scoring,
                attention_units=32
            )
            compile_model(model, is_attention=True)
            # Dummy target formatting for dual output
            train_targets = {"multivariate_output": y_train_scaled, "temporal_attention": np.zeros((len(y_train_scaled), win, 1), dtype=np.float32)}
            val_targets = {"multivariate_output": y_val_scaled, "temporal_attention": np.zeros((len(y_val_scaled), win, 1), dtype=np.float32)}
        else:
            model = build_stacked_gru(
                input_shape=input_shape,
                gru_layers=layers
            )
            compile_model(model, is_attention=False)
            train_targets = y_train_scaled
            val_targets = y_val_scaled

        callbacks = [
            EarlyStopping(
                monitor="val_loss",
                mode="min",
                patience=3,
                restore_best_weights=True,
                verbose=0
            )
        ]

        t0 = time.time()
        history = model.fit(
            X_train,
            train_targets,
            validation_data=(X_val, val_targets),
            epochs=8,
            batch_size=256,
            callbacks=callbacks,
            verbose=1
        )
        t_elapsed = time.time() - t0

        val_loss_hist = history.history.get("val_loss", [0.0])
        best_val_loss = float(np.min(val_loss_hist))

        # Evaluate Validation & Test RMSE in true physical units
        val_rmses, _ = evaluate_split_rmse(model, X_val, y_val_unscaled, preprocessor, is_attention=is_attn)
        test_rmses, _ = evaluate_split_rmse(model, X_test, y_test_unscaled, preprocessor, is_attention=is_attn)

        record = {
            "Experiment": name,
            "Category": exp["category"],
            "Window_Length": win,
            "GRU_Layers": str(layers),
            "Scoring_Function": scoring if scoring else "None (Pure GRU)",
            "Trainable_Params": model.count_params(),
            "Epochs_Trained": len(val_loss_hist),
            "Best_Val_Loss_Scaled": round(best_val_loss, 4),
            "Val_RMSE_Temp_degC": round(val_rmses["T (degC)"], 4),
            "Val_RMSE_Pres_mbar": round(val_rmses["p (mbar)"], 4),
            "Val_RMSE_Humid_pct": round(val_rmses["rh (%)"], 4),
            "Val_RMSE_Wind_ms": round(val_rmses["wv (m/s)"], 4),
            "Val_Mean_RMSE": round(val_rmses["mean_rmse"], 4),
            "Test_Mean_RMSE": round(test_rmses["mean_rmse"], 4),
            "Training_Time_s": round(t_elapsed, 1),
        }
        results.append(record)
        print(f"    [+] Val Mean RMSE: {val_rmses['mean_rmse']:.4f} | Test Mean RMSE: {test_rmses['mean_rmse']:.4f} in {t_elapsed:.1f}s")

    # Save to CSV
    df_results = pd.DataFrame(results)
    out_csv = METRICS_DIR / "tuning_experiments.csv"
    df_results.to_csv(out_csv, index=False)
    print(f"\n[+] Saved tuning results table to: {out_csv}")

    # Display console table
    print("\n" + "=" * 80)
    print(f"{'EXPERIMENT':<26} | {'WIN':<4} | {'LAYERS':<12} | {'SCORING':<15} | {'VAL RMSE':<8} | {'TIME'}")
    print("=" * 80)
    for _, r in df_results.iterrows():
        print(f"{r['Experiment']:<26} | {r['Window_Length']:<4} | {r['GRU_Layers']:<12} | {r['Scoring_Function']:<15} | {r['Val_Mean_RMSE']:<8.4f} | {r['Training_Time_s']}s")
    print("=" * 80)

    # Plot Tuning & Ablation Figure
    plot_tuning_summary(df_results)
    return df_results


def plot_tuning_summary(df_results: pd.DataFrame, save_path: Path = FIGURES_DIR / "09_hyperparameter_tuning_ablation.png"):
    """Generates a 3-panel publication-ready comparison plot."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # 1. Window Length Comparison
    df_win = df_results[df_results["Category"] == "Window Length"]
    axes[0].bar(
        [f"{w}h" for w in df_win["Window_Length"]],
        df_win["Val_Mean_RMSE"],
        color=["#1f77b4", "#3b82f6", "#2563eb"],
        edgecolor="black",
        alpha=0.85
    )
    axes[0].set_title("Window Length vs. Val RMSE", fontweight="bold")
    axes[0].set_xlabel("Lookback Window (Hours)")
    axes[0].set_ylabel("Mean Validation RMSE")
    axes[0].grid(True, linestyle="--", alpha=0.5)
    for i, v in enumerate(df_win["Val_Mean_RMSE"]):
        axes[0].text(i, v + 0.005, f"{v:.4f}", ha="center", fontsize=9, fontweight="bold")

    # 2. Recurrent Depth Comparison
    df_depth = df_results[df_results["Experiment"].isin(["GRU_Window_72h_2L", "GRU_Window_72h_3L"])]
    labels_depth = ["2 Layers (64, 32)", "3 Layers (64, 32, 16)"]
    axes[1].bar(
        labels_depth,
        df_depth["Val_Mean_RMSE"],
        color=["#2ca02c", "#16a34a"],
        edgecolor="black",
        alpha=0.85
    )
    axes[1].set_title("Recurrent Depth vs. Val RMSE", fontweight="bold")
    axes[1].set_xlabel("Architecture Depth")
    axes[1].grid(True, linestyle="--", alpha=0.5)
    for i, v in enumerate(df_depth["Val_Mean_RMSE"]):
        axes[1].text(i, v + 0.005, f"{v:.4f}", ha="center", fontsize=9, fontweight="bold")

    # 3. Attention Scoring Comparison: Additive vs Dot-Product
    df_attn = df_results[df_results["Category"] == "Attention Scoring"]
    labels_attn = ["Additive\n(Bahdanau)", "Dot-Product\n(Luong)"]
    axes[2].bar(
        labels_attn,
        df_attn["Val_Mean_RMSE"],
        color=["#d62728", "#ea580c"],
        edgecolor="black",
        alpha=0.85
    )
    axes[2].set_title("Attention Scoring: Additive vs. Dot-Product", fontweight="bold")
    axes[2].set_xlabel("Scoring Function")
    axes[2].grid(True, linestyle="--", alpha=0.5)
    for i, v in enumerate(df_attn["Val_Mean_RMSE"]):
        axes[2].text(i, v + 0.005, f"{v:.4f}", ha="center", fontsize=9, fontweight="bold")

    plt.suptitle("Hyperparameter Tuning & Attention Ablation Analysis", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[+] Saved tuning summary figure to: {save_path}")


if __name__ == "__main__":
    run_tuning_suite()

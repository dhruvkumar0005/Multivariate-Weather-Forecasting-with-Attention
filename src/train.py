"""Training Engine for Multivariate Weather Forecasting Models.

Trains:
1. Baseline LSTM
2. Stacked GRU
3. GRU with Temporal Attention
under identical optimization conditions, loss functions, and early stopping.
"""

import json
import time
import numpy as np
import tensorflow as tf
from pathlib import Path
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

from src.config import (
    MODELS_DIR,
    METRICS_DIR,
    EPOCHS,
    PATIENCE_EARLY_STOPPING,
    PATIENCE_REDUCE_LR,
    RANDOM_SEED
)
from src.features import build_preprocessed_splits
from src.sequences import create_sliding_window_dataset
from src.models import (
    build_baseline_lstm,
    build_stacked_gru,
    build_gru_with_attention,
    compile_model
)

# Set seeds for reproducibility
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


def train_single_model(model_name: str, model: tf.keras.Model, train_ds, val_ds, is_attention: bool = False):
    """Trains a given architecture with standard callbacks and logs timing."""
    checkpoint_path = MODELS_DIR / f"{model_name}.keras"
    history_file = METRICS_DIR / f"{model_name}_history.json"

    if checkpoint_path.exists() and history_file.exists():
        print(f"[+] Model {model_name} already trained and saved. Skipping to next model.")
        return model, None

    print(f"\n{'='*20} Training: {model_name} {'='*20}")
    monitor_metric = "val_loss"

    callbacks = [
        EarlyStopping(
            monitor=monitor_metric,
            mode="min",
            patience=PATIENCE_EARLY_STOPPING,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor=monitor_metric,
            mode="min",
            factor=0.5,
            patience=PATIENCE_REDUCE_LR,
            min_lr=1e-5,
            verbose=1
        ),
        ModelCheckpoint(
            filepath=checkpoint_path,
            monitor=monitor_metric,
            mode="min",
            save_best_only=True,
            verbose=1
        )
    ]

    start_time = time.time()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )
    elapsed_time = time.time() - start_time
    print(f"[+] {model_name} training completed in {elapsed_time:.1f} seconds.")

    # Save training history
    history_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    history_dict["training_time_seconds"] = elapsed_time
    history_file = METRICS_DIR / f"{model_name}_history.json"
    with open(history_file, "w") as f:
        json.dump(history_dict, f, indent=4)
    print(f"[+] Saved training history to: {history_file}")

    return model, history_dict


def run_all_training():
    """Builds preprocessed sequences and trains all three models sequentially."""
    print("[*] Preparing preprocessed datasets...")
    (X_tr, y_tr), (X_v, y_v), (X_te, y_te), preprocessor, _ = build_preprocessed_splits()

    print("[*] Creating tf.data.Dataset sliding window pipelines...")
    train_ds = create_sliding_window_dataset(X_tr, y_tr, shuffle=True)
    val_ds = create_sliding_window_dataset(X_v, y_v, shuffle=False)

    # 1. Baseline LSTM
    lstm_model = build_baseline_lstm()
    compile_model(lstm_model, is_attention=False)
    train_single_model("Baseline_LSTM", lstm_model, train_ds, val_ds, is_attention=False)

    # 2. Stacked GRU
    gru_model = build_stacked_gru()
    compile_model(gru_model, is_attention=False)
    train_single_model("Stacked_GRU", gru_model, train_ds, val_ds, is_attention=False)

    # 3. GRU with Temporal Attention
    # For attention training, dataset targets must be formatted to match dual outputs: (Y, dummy_attn)
    # Or train with output dictionary
    def map_attention_targets(x, y):
        # Dummy zero weights for attention loss since attention loss is None
        return x, {"multivariate_output": y, "temporal_attention": tf.zeros((tf.shape(y)[0], 72, 1))}

    train_ds_attn = train_ds.map(map_attention_targets)
    val_ds_attn = val_ds.map(map_attention_targets)

    attn_model = build_gru_with_attention()
    compile_model(attn_model, is_attention=True)
    train_single_model("GRU_with_Attention", attn_model, train_ds_attn, val_ds_attn, is_attention=True)

    print("\n[+] All three models trained and serialized successfully!")


if __name__ == "__main__":
    run_all_training()

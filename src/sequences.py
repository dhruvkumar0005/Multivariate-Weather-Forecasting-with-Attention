"""Sequence Generator Module for Sliding-Window Time-Series Forecasting.

Constructs:
Input X: past N_timesteps (e.g. 72 hours) x features (8)
Target Y: future timestep(s) (e.g. t+1) x target variables (4)
"""

import numpy as np
import tensorflow as tf
from src.config import WINDOW_LENGTH, HORIZON, BATCH_SIZE


def create_sliding_window_dataset(
    data: np.ndarray,
    targets: np.ndarray,
    window_length: int = WINDOW_LENGTH,
    horizon: int = HORIZON,
    batch_size: int = BATCH_SIZE,
    shuffle: bool = False,
    multi_step: bool = False,
    horizon_steps: int = 6
) -> tf.data.Dataset:
    """Creates a high-performance tf.data.Dataset of sliding windows.

    Args:
        data: Scaled feature array of shape (N, num_features).
        targets: Scaled target array of shape (N, num_targets).
        window_length: Past timesteps to look back (default: 72 hours).
        horizon: Step ahead to forecast for single-step mode (default: 1 step ahead).
        batch_size: Mini-batch size.
        shuffle: Whether to shuffle batches during training (preserves intra-window order).
        multi_step: If True, forecasts sequence of next horizon_steps timesteps (shape: batch, horizon_steps, targets).
        horizon_steps: Number of forecast steps ahead when multi_step is True.
    """
    if multi_step:
        total_samples = len(data) - window_length - horizon_steps + 1
        if total_samples <= 0:
            raise ValueError(f"Data length {len(data)} is too short for window {window_length} + multi-step horizon {horizon_steps}")

        X_arr, y_arr = extract_arrays_for_evaluation(
            data, targets, window_length=window_length, horizon=horizon, multi_step=True, horizon_steps=horizon_steps
        )
        dataset = tf.data.Dataset.from_tensor_slices((X_arr, y_arr))
        if shuffle:
            dataset = dataset.shuffle(buffer_size=10000, seed=42)
        dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        return dataset

    total_samples = len(data) - window_length - horizon + 1
    if total_samples <= 0:
        raise ValueError(f"Data length {len(data)} is too short for window {window_length} + horizon {horizon}")

    # Start and end positions for inputs and targets
    dataset = tf.keras.utils.timeseries_dataset_from_array(
        data=data,
        targets=targets[window_length + horizon - 1:],
        sequence_length=window_length,
        sequence_stride=1,
        sampling_rate=1,
        batch_size=batch_size,
        shuffle=shuffle
    )

    # Prefetch for performance
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


def extract_arrays_for_evaluation(
    data: np.ndarray,
    targets: np.ndarray,
    window_length: int = WINDOW_LENGTH,
    horizon: int = HORIZON,
    multi_step: bool = False,
    horizon_steps: int = 6
):
    """Extracts explicit numpy arrays for sliding windows for test set evaluation.

    Single-step:
        X shape: (N_windows, window_length, num_features)
        y shape: (N_windows, num_targets)
    Multi-step:
        X shape: (N_windows, window_length, num_features)
        y shape: (N_windows, horizon_steps, num_targets)
    """
    if multi_step:
        total_samples = len(data) - window_length - horizon_steps + 1
        X = np.empty((total_samples, window_length, data.shape[1]), dtype=np.float32)
        y = np.empty((total_samples, horizon_steps, targets.shape[1]), dtype=np.float32)
        for i in range(total_samples):
            X[i] = data[i : i + window_length]
            y[i] = targets[i + window_length : i + window_length + horizon_steps]
        return X, y

    total_samples = len(data) - window_length - horizon + 1
    X = np.empty((total_samples, window_length, data.shape[1]), dtype=np.float32)
    y = np.empty((total_samples, targets.shape[1]), dtype=np.float32)

    for i in range(total_samples):
        X[i] = data[i : i + window_length]
        y[i] = targets[i + window_length + horizon - 1]

    return X, y


if __name__ == "__main__":
    from src.features import build_preprocessed_splits

    (X_tr, y_tr), (X_v, y_v), (X_te, y_te), preprocessor, _ = build_preprocessed_splits()
    train_ds = create_sliding_window_dataset(X_tr, y_tr, shuffle=True)
    val_ds = create_sliding_window_dataset(X_v, y_v, shuffle=False)
    test_ds = create_sliding_window_dataset(X_te, y_te, shuffle=False)

    for x_batch, y_batch in train_ds.take(1):
        print(f"[+] Sample Batch X shape: {x_batch.shape} (Expected: ({BATCH_SIZE}, {WINDOW_LENGTH}, 8))")
        print(f"[+] Sample Batch Y shape: {y_batch.shape} (Expected: ({BATCH_SIZE}, 4))")

"""Model Architectures Module.

Implements:
1. Multi-Output Baseline LSTM
2. Multi-Output Stacked GRU (2 layers)
3. Multi-Output Stacked GRU + Temporal Additive Attention
"""

import tensorflow as tf
from tensorflow.keras import layers, models
from src.attention import TemporalAdditiveAttention
from src.config import WINDOW_LENGTH, TARGET_COLS, ALL_FEATURE_COLS, LEARNING_RATE


def build_baseline_lstm(
    input_shape=(WINDOW_LENGTH, len(ALL_FEATURE_COLS)),
    num_targets=len(TARGET_COLS),
    dropout_rate=0.2
) -> tf.keras.Model:
    """Builds a fair 2-layer multi-output Baseline LSTM model."""
    inputs = layers.Input(shape=input_shape, name="sequence_input")
    x = layers.LSTM(64, return_sequences=True, name="lstm_layer_1")(inputs)
    x = layers.Dropout(dropout_rate, name="dropout_1")(x)
    x = layers.LSTM(32, return_sequences=False, name="lstm_layer_2")(x)
    x = layers.Dropout(dropout_rate, name="dropout_2")(x)
    x = layers.Dense(32, activation="relu", name="dense_shared")(x)
    outputs = layers.Dense(num_targets, activation="linear", name="multivariate_output")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="Baseline_LSTM")
    return model


def build_stacked_gru(
    input_shape=(WINDOW_LENGTH, len(ALL_FEATURE_COLS)),
    num_targets=len(TARGET_COLS),
    dropout_rate=0.2,
    gru_layers=(64, 32)
) -> tf.keras.Model:
    """Builds a multi-output Stacked GRU model with configurable depth (2 or 3 layers)."""
    inputs = layers.Input(shape=input_shape, name="sequence_input")
    x = inputs
    for idx, units in enumerate(gru_layers):
        is_last = (idx == len(gru_layers) - 1)
        x = layers.GRU(units, return_sequences=not is_last, name=f"gru_layer_{idx+1}")(x)
        x = layers.Dropout(dropout_rate, name=f"dropout_{idx+1}")(x)

    x = layers.Dense(32, activation="relu", name="dense_shared")(x)
    outputs = layers.Dense(num_targets, activation="linear", name="multivariate_output")(x)

    name = f"Stacked_GRU_{len(gru_layers)}L" if len(gru_layers) != 2 else "Stacked_GRU"
    model = models.Model(inputs=inputs, outputs=outputs, name=name)
    return model


def build_gru_with_attention(
    input_shape=(WINDOW_LENGTH, len(ALL_FEATURE_COLS)),
    num_targets=len(TARGET_COLS),
    dropout_rate=0.2,
    attention_units=32,
    scoring="additive",
    gru_layers=(64, 32)
) -> tf.keras.Model:
    """Builds a Stacked GRU model augmented with a Temporal Attention mechanism (Additive or Dot-Product)."""
    inputs = layers.Input(shape=input_shape, name="sequence_input")
    x = inputs
    for idx, units in enumerate(gru_layers):
        # All recurrent layers return sequences so the final layer provides hidden states to attention
        x = layers.GRU(units, return_sequences=True, name=f"gru_layer_{idx+1}_seq")(x)
        x = layers.Dropout(dropout_rate, name=f"dropout_{idx+1}")(x)

    # Compute context vector and attention weights over time
    context_vector, attention_weights = TemporalAdditiveAttention(
        units=attention_units,
        scoring=scoring,
        return_attention_weights=True,
        name="temporal_attention"
    )(x)

    x = layers.Dense(32, activation="relu", name="dense_shared")(context_vector)
    outputs = layers.Dense(num_targets, activation="linear", name="multivariate_output")(x)

    model_name = "GRU_with_Attention"
    if scoring in ["dot_product", "dot"]:
        model_name = "GRU_with_DotProduct_Attention"

    model = models.Model(
        inputs=inputs,
        outputs=[outputs, attention_weights],
        name=model_name
    )
    return model


def compile_model(model: tf.keras.Model, is_attention: bool = False, lr: float = LEARNING_RATE) -> tf.keras.Model:
    """Compiles the model with Adam optimizer and Mean Squared Error loss."""
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    if is_attention:
        # We optimize the multi-target forecast output; attention weights have weight 0.0 in loss
        model.compile(
            optimizer=optimizer,
            loss={"multivariate_output": "mse", "temporal_attention": None},
            metrics={"multivariate_output": ["mae", "mse"]}
        )
    else:
        model.compile(
            optimizer=optimizer,
            loss="mse",
            metrics=["mae", "mse"]
        )
    return model


if __name__ == "__main__":
    lstm = build_baseline_lstm()
    compile_model(lstm, is_attention=False)
    lstm.summary()

    gru = build_stacked_gru()
    compile_model(gru, is_attention=False)
    gru.summary()

    attn_gru = build_gru_with_attention()
    compile_model(attn_gru, is_attention=True)
    attn_gru.summary()
    print("[+] All three architectures instantiated, configured, and verified.")

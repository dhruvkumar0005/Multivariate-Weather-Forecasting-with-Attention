"""Custom Temporal Attention Layer for Time-Series Recurrent Architectures.

Supports both:
1. Bahdanau-style Additive Attention: e_t = v^T * tanh(W * h_t + b)
2. Luong-style Scaled Dot-Product Attention: e_t = (h_t * q) / sqrt(d)
Softmax is strictly computed over the TIME dimension (axis=1).
"""

import tensorflow as tf
from tensorflow.keras import layers


@tf.keras.utils.register_keras_serializable(package="CustomLayers")
class TemporalAdditiveAttention(layers.Layer):
    """Temporal Attention Mechanism over recurrent hidden states along the TIME dimension (axis=1).

    Supports:
    - 'additive' (Bahdanau):
        u_t = tanh(W * h_t + b)
        e_t = v^T * u_t
    - 'dot_product' (Luong / Scaled Dot-Product):
        e_t = (h_t * q) / sqrt(hidden_dim)

    Softmax along the time dimension (axis=1):
        alpha_t = softmax(e_t, axis=1)

    Context vector:
        c = sum(alpha_t * h_t, axis=1)
    """

    def __init__(self, units: int = 32, scoring: str = "additive", return_attention_weights: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.scoring = scoring.lower()
        self.return_attention_weights = return_attention_weights

    def build(self, input_shape):
        hidden_dim = input_shape[-1]
        if self.scoring in ["dot_product", "dot", "luong"]:
            # Learnable query projection for dot-product scoring
            self.query = self.add_weight(
                name="attention_query_vector",
                shape=(hidden_dim, 1),
                initializer="glorot_uniform",
                trainable=True,
            )
            self.scale = tf.math.sqrt(tf.cast(hidden_dim, tf.float32))
        else:
            # Bahdanau additive scoring
            self.W = self.add_weight(
                name="attention_weight_matrix",
                shape=(hidden_dim, self.units),
                initializer="glorot_uniform",
                trainable=True,
            )
            self.b = self.add_weight(
                name="attention_bias",
                shape=(self.units,),
                initializer="zeros",
                trainable=True,
            )
            self.v = self.add_weight(
                name="attention_score_vector",
                shape=(self.units, 1),
                initializer="glorot_uniform",
                trainable=True,
            )
        super().build(input_shape)

    def call(self, inputs):
        # inputs shape: (batch_size, timesteps, hidden_dim)
        if self.scoring in ["dot_product", "dot", "luong"]:
            score = tf.tensordot(inputs, self.query, axes=1) / self.scale
        else:
            u = tf.nn.tanh(tf.tensordot(inputs, self.W, axes=1) + self.b)
            score = tf.tensordot(u, self.v, axes=1)

        # Softmax strictly along the TIME dimension (axis=1)
        attention_weights = tf.nn.softmax(score, axis=1)

        # Weighted context vector: (batch, hidden_dim)
        context_vector = tf.reduce_sum(inputs * attention_weights, axis=1)

        if self.return_attention_weights:
            return context_vector, attention_weights
        return context_vector

    def get_config(self):
        config = super().get_config()
        config.update({
            "units": self.units,
            "scoring": self.scoring,
            "return_attention_weights": self.return_attention_weights,
        })
        return config


# Alias
TemporalAttention = TemporalAdditiveAttention

from __future__ import annotations

import tensorflow as tf
from neurolens.data_storage.ModelConfig import ModelConfig


class CNNClassifier(tf.keras.Model):
    """
    Lightweight residual-style 2D CNN classifier for 4-class brain tumor MRI.
    Supports dynamic input channel counts (1-ch MRI, 2-ch +LBP, 3-ch +LoG)
    and outputs raw class logits.
    """

    def __init__(self, config: ModelConfig, dropoutRate: float = 0.2, **kwargs) -> None:
        super().__init__(**kwargs)
        self.config = config
        self.dropout_rate = float(dropoutRate)
        self.num_classes = len(config.classNames)
        filters = list(config.filterCounts) or [16, 32, 64]

        self._blocks: list[tuple[tf.keras.layers.Layer, ...]] = []
        for idx, f in enumerate(filters):
            conv1 = tf.keras.layers.Conv2D(
                f, kernel_size=3, padding="same", use_bias=False, name=f"conv_{idx}_a"
            )
            bn1 = tf.keras.layers.BatchNormalization(name=f"bn_{idx}_a")
            act1 = tf.keras.layers.Activation("gelu", name=f"gelu_{idx}_a")

            conv2 = tf.keras.layers.Conv2D(
                f, kernel_size=3, padding="same", use_bias=False, name=f"conv_{idx}_b"
            )
            bn2 = tf.keras.layers.BatchNormalization(name=f"bn_{idx}_b")
            act2 = tf.keras.layers.Activation("gelu", name=f"gelu_{idx}_b")

            pool = tf.keras.layers.MaxPooling2D(pool_size=2, name=f"pool_{idx}")
            self._blocks.append((conv1, bn1, act1, conv2, bn2, act2, pool))

        self._gap = tf.keras.layers.GlobalAveragePooling2D(name="global_avg_pool")
        self._dropout = tf.keras.layers.Dropout(self.dropout_rate, name="head_dropout")
        self._dense_hidden = tf.keras.layers.Dense(
            max(64, filters[-1]), activation="gelu", name="head_fc"
        )
        self._classifier = tf.keras.layers.Dense(
            self.num_classes, activation=None, name="logits"
        )

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        x = tf.cast(inputs, tf.float32)
        for conv1, bn1, act1, conv2, bn2, act2, pool in self._blocks:
            x = conv1(x)
            x = bn1(x, training=training)
            x = act1(x)
            x = conv2(x)
            x = bn2(x, training=training)
            x = act2(x)
            x = pool(x)

        x = self._gap(x)
        x = self._dropout(x, training=training)
        x = self._dense_hidden(x)
        return self._classifier(x)
from __future__ import annotations

import inspect
from typing import Callable

import tensorflow as tf
from neurolens.data_storage.ModelConfig import ModelConfig

ActivationSpecType = (
    str
    | Callable[..., tf.Tensor]
    | tf.keras.layers.Layer
    | tuple[
        str | Callable[..., tf.Tensor] | tf.keras.layers.Layer,
        str | Callable[..., tf.Tensor] | tf.keras.layers.Layer,
    ]
)


class CSineLU(tf.keras.layers.Layer):
    """
    Custom learnable activation:
      f(x) = x                     for x >= 0
      f(x) = p - (p / w) * sin(xw) for x < 0
    with per-channel trainable parameters p and w.
    """

    def __init__(
        self,
        p_init: float = 0.0,
        w_init: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.p_init = float(p_init)
        self.w_init = float(w_init)
        self.p = None
        self.w = None

    def build(self, input_shape) -> None:
        channels = int(input_shape[-1])
        self.p = self.add_weight(
            name="p",
            shape=(channels,),
            initializer=tf.keras.initializers.Constant(self.p_init),
            trainable=True,
        )
        self.w = self.add_weight(
            name="w",
            shape=(channels,),
            initializer=tf.keras.initializers.Constant(self.w_init),
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        x = tf.cast(inputs, tf.float32)
        p = tf.cast(self.p, tf.float32)
        w = tf.cast(self.w, tf.float32)

        xw = x * w
        safe_w = tf.where(tf.abs(w) < 1e-4, tf.ones_like(w), w)
        sin_over_w = tf.where(
            tf.abs(w) < 1e-4,
            x * (1.0 - (xw * xw) / 6.0),
            tf.math.sin(xw) / safe_w,
        )
        neg_branch = p - p * sin_over_w
        return tf.where(x >= 0.0, x, neg_branch)

    def get_config(self) -> dict:
        cfg = super().get_config()
        cfg.update({"p_init": self.p_init, "w_init": self.w_init})
        return cfg


class ParametricSwish(tf.keras.layers.Layer):
    """Channel-wise parameterized Swish activation: f(x) = x * sigmoid(beta * x)."""

    def __init__(
        self,
        beta_init: float = 1.0,
        trainable_beta: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.beta_init = float(beta_init)
        self.trainable_beta = bool(trainable_beta)
        self.beta = None

    def build(self, input_shape) -> None:
        channels = int(input_shape[-1])
        self.beta = self.add_weight(
            name="beta",
            shape=(channels,),
            initializer=tf.keras.initializers.Constant(self.beta_init),
            trainable=self.trainable_beta,
        )
        super().build(input_shape)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        x = tf.cast(inputs, tf.float32)
        b = tf.cast(self.beta, tf.float32)
        return x * tf.math.sigmoid(b * x)

    def get_config(self) -> dict:
        cfg = super().get_config()
        cfg.update(
            {
                "beta_init": self.beta_init,
                "trainable_beta": self.trainable_beta,
            }
        )
        return cfg


def _make_leaky_relu(slope: float, name: str) -> tf.keras.layers.Layer:
    params = inspect.signature(tf.keras.layers.LeakyReLU.__init__).parameters
    if "negative_slope" in params:
        return tf.keras.layers.LeakyReLU(negative_slope=float(slope), name=name)
    return tf.keras.layers.LeakyReLU(alpha=float(slope), name=name)


def _celu_fn(x: tf.Tensor) -> tf.Tensor:
    return tf.maximum(x, 0.0) + tf.minimum(0.0, tf.exp(tf.minimum(x, 0.0)) - 1.0)


def _mish_fn(x: tf.Tensor) -> tf.Tensor:
    return x * tf.math.tanh(tf.math.softplus(x))


class CNNClassifier(tf.keras.Model):
    """
    Lightweight 2D CNN classifier for 4-class brain tumor MRI.
    Supports dynamic input channel counts, parametric activation functions
    (single activation or (conv_activation, head_activation) pair, defaulting to 'relu'),
    and outputs raw class logits.
    """

    def __init__(
        self,
        config: ModelConfig,
        dropoutRate: float = 0.2,
        activation: ActivationSpecType = "relu",
        l2: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.config = config
        self.dropout_rate = float(dropoutRate)
        self.l2 = float(l2)
        self.activation_spec = activation

        if isinstance(activation, (tuple, list)):
            if len(activation) != 2:
                raise ValueError(
                    "activation tuple must contain exactly (conv_activation, head_activation)"
                )
            conv_act, head_act = activation[0], activation[1]
        else:
            conv_act, head_act = activation, activation

        self.conv_activation_name = self._describe_activation(conv_act)
        self.head_activation_name = self._describe_activation(head_act)
        self.activation_name = (
            self.conv_activation_name
            if self.conv_activation_name == self.head_activation_name
            else f"{self.conv_activation_name}+{self.head_activation_name}"
        )

        self.num_classes = len(config.classNames)
        filters = list(config.filterCounts) or [16, 32, 64]

        reg = (
            tf.keras.regularizers.l2(self.l2 / 2.0)
            if self.l2 > 0.0
            else None
        )

        self._blocks: list[tuple[tf.keras.layers.Layer, ...]] = []
        for idx, f in enumerate(filters):
            conv1 = tf.keras.layers.Conv2D(
                f,
                kernel_size=3,
                padding="same",
                use_bias=False,
                kernel_regularizer=reg,
                name=f"conv_{idx}_a",
            )
            bn1 = tf.keras.layers.BatchNormalization(name=f"bn_{idx}_a")
            act1 = self._build_activation_layer(
                conv_act, name=f"act_{idx}_a", spatial=True
            )

            conv2 = tf.keras.layers.Conv2D(
                f,
                kernel_size=3,
                padding="same",
                use_bias=False,
                kernel_regularizer=reg,
                name=f"conv_{idx}_b",
            )
            bn2 = tf.keras.layers.BatchNormalization(name=f"bn_{idx}_b")
            act2 = self._build_activation_layer(
                conv_act, name=f"act_{idx}_b", spatial=True
            )

            pool = tf.keras.layers.MaxPooling2D(pool_size=2, name=f"pool_{idx}")
            self._blocks.append((conv1, bn1, act1, conv2, bn2, act2, pool))

        self._gap = tf.keras.layers.GlobalAveragePooling2D(name="global_avg_pool")
        self._dropout = tf.keras.layers.Dropout(self.dropout_rate, name="head_dropout")
        self._dense_hidden = tf.keras.layers.Dense(
            max(64, filters[-1]),
            activation=None,
            kernel_regularizer=reg,
            name="head_fc",
        )
        self._dense_act = self._build_activation_layer(
            head_act, name="head_act", spatial=False
        )
        self._classifier = tf.keras.layers.Dense(
            self.num_classes,
            activation=None,
            kernel_regularizer=reg,
            name="logits",
        )

    @staticmethod
    def _describe_activation(
        activation: str | Callable[..., tf.Tensor] | tf.keras.layers.Layer | None,
    ) -> str:
        if activation is None:
            return "linear"
        if isinstance(activation, str):
            return activation.strip()
        if isinstance(activation, tf.keras.layers.Layer):
            return type(activation).__name__
        return getattr(activation, "__name__", type(activation).__name__)

    @staticmethod
    def _build_activation_layer(
        activation: str | Callable[..., tf.Tensor] | tf.keras.layers.Layer | None,
        *,
        name: str,
        spatial: bool,
    ) -> tf.keras.layers.Layer:
        if activation is None:
            return tf.keras.layers.Activation("linear", name=name)

        if isinstance(activation, str):
            raw = activation.strip()
            key = raw.lower().replace("-", "_")

            if key == "relu":
                return tf.keras.layers.ReLU(name=name)
            if key == "relu6":
                return tf.keras.layers.ReLU(max_value=6.0, name=name)
            if key in {"leaky_relu", "lrelu"}:
                return _make_leaky_relu(0.1, name=name)
            if key.startswith("leaky_relu_"):
                slope = float(key[len("leaky_relu_") :])
                return _make_leaky_relu(slope, name=name)
            if key == "prelu":
                if spatial:
                    return tf.keras.layers.PReLU(shared_axes=[1, 2], name=name)
                return tf.keras.layers.PReLU(name=name)
            if key == "elu":
                return tf.keras.layers.ELU(alpha=1.0, name=name)
            if key.startswith("elu_"):
                alpha = float(key[len("elu_") :])
                return tf.keras.layers.ELU(alpha=alpha, name=name)
            if key == "celu":
                return tf.keras.layers.Activation(_celu_fn, name=name)
            if key == "mish":
                return tf.keras.layers.Activation(_mish_fn, name=name)
            if key in {"csinelu", "c_sinelu"}:
                return CSineLU(p_init=0.0, w_init=1.0, name=name)
            if key in {"trainable_swish", "parametric_swish"}:
                return ParametricSwish(beta_init=1.0, trainable_beta=True, name=name)
            if key.startswith("swish_beta_"):
                beta = float(key[len("swish_beta_") :])
                return ParametricSwish(beta_init=beta, trainable_beta=False, name=name)
            if key == "silu":
                return tf.keras.layers.Activation("swish", name=name)

            return tf.keras.layers.Activation(raw, name=name)

        if isinstance(activation, type) and issubclass(activation, tf.keras.layers.Layer):
            return activation(name=name)

        if isinstance(activation, tf.keras.layers.Layer):
            cfg = activation.get_config()
            cfg["name"] = name
            return type(activation).from_config(cfg)

        if callable(activation):
            sig = inspect.signature(activation)
            required = [
                p
                for p in sig.parameters.values()
                if p.default is inspect.Parameter.empty
                and p.kind
                in (
                    inspect.Parameter.POSITIONAL_ONLY,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                )
            ]
            if not required:
                created = activation()
                if isinstance(created, tf.keras.layers.Layer):
                    created._name = name
                    return created
            return tf.keras.layers.Activation(activation, name=name)

        raise TypeError(f"Unsupported activation specification: {activation!r}")

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
        x = self._dense_act(x)
        return self._classifier(x)
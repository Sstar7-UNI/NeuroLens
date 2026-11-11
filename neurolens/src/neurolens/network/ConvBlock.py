import keras
from keras import layers


class ConvBlock(layers.Layer):
    def __init__(self, filters: int, kernelSize: int = 3, poolSize: int = 2, l2: float = 0.0):
        super().__init__()
        self.kernelSize = kernelSize
        self.poolSize = poolSize
        self._convolution = layers.Conv2D(
            filters,
            kernelSize,
            padding="same",
            kernel_initializer=keras.initializers.HeNormal(),
            kernel_regularizer=keras.regularizers.l2(l2 / 2.0) if l2 > 0 else None,
        )
        self._normalization = layers.BatchNormalization()
        self._activation = layers.ReLU()
        self._pooling = layers.MaxPool2D(poolSize) if poolSize else None
    def call(self, inputs, training=False):
        x = self._convolution(inputs)
        x = self._normalization(x, training=training)
        x = self._activation(x)
        if self._pooling is not None:
            x = self._pooling(x)
        return x
    def outputChannels(self) -> int:
        return self._convolution.filters

import keras
from keras import Model, layers

from neurolens.data_storage.ModelConfig import ModelConfig
from neurolens.network.ConvBlock import ConvBlock


class CNNClassifier(Model):
    def __init__(self, config: ModelConfig, dropoutRate: float = 0.0, l2: float = 0.0):
        super().__init__()
        self._blocks = []
        blockCount = len(config.filterCounts)

        for i in range(blockCount):
            poolSize = 2 if i < blockCount - 1 else 0
            self._blocks.append(
                ConvBlock(config.filterCounts[i], poolSize=poolSize, l2=l2)
            )

        self._globalPooling = layers.GlobalAveragePooling2D()
        self._dropout = layers.Dropout(dropoutRate) if dropoutRate > 0 else None
        self._classifier = layers.Dense(
            len(config.classNames),
            kernel_initializer=keras.initializers.LecunNormal(),
            kernel_regularizer=keras.regularizers.l2(l2 / 2.0) if l2 > 0 else None,
        )

    def extractFeatures(self, inputs, training=False):
        x = inputs
        for block in self._blocks:
            x = block(x, training=training)
        return self._globalPooling(x)

    def call(self, inputs, training=False):
        x = self.extractFeatures(inputs, training=training)
        if self._dropout is not None:
            x = self._dropout(x, training=training)
        return self._classifier(x)

    def receptiveField(self) -> int:
        r = 1
        jump = 1

        for block in self._blocks:
            r += (block.kernelSize - 1) * jump
            if block.poolSize:
                r += (block.poolSize - 1) * jump
                jump *= block.poolSize

        return r

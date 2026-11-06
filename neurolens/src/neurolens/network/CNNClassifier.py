import keras
from keras import Model
from keras.src.utils.torch_utils_test import Classifier
from tensorflow.keras import layers
from neurolens.data_storage import ModelConfig
from neurolens.network.ConvBlock import ConvBlock


class CNNClassifier(Model):
    def __init__(self, config: ModelConfig,
                 dropoutRate:float = 0.0,
                 l2:float = 0.0):
        super().__init__()
        self._blocks = []
        blockCnt = len(config.filterCounts)
        for i in range(blockCnt):
            if i < blockCnt - 1:
                poolSize = 2
            else:
                poolSize = 0 #ultima nu are pooling
            block = ConvBlock(config.filterCounts[i],poolSize=poolSize,l2=l2)
            self._blocks.append(block)
        self._globalPooling = layers.GlobalAveragePooling2D()

        if dropoutRate > 0:
            self._dropout = layers.Dropout(dropoutRate)
        else:
            self._dropout = None

        if l2 > 0:
            regularizer = keras.regularizers.l2(l2=l2)
        else:
            regularizer = None

        self._classifier = layers.Dense(
            len(config.classNames),
            kernel_initializer=keras.initializers.HeNormal(),
            kernel_regularizer=regularizer,
        ) # fara activare

    def extractFeatures(self,inputs,training=False):
        x = inputs
        for block in self._blocks:
            x = block(x, training=training)
        return self._globalPooling(x)
    def call(self, inputs, training=False):
        x = self.extractFeatures(inputs,training=training)
        if self._dropout is not None:
            x = self._dropout(x,training=training)
        return self.classifier(x)
    def receptiveField(self) -> int:
        r = 1
        jump = 1
        for block in self._blocks:
            r+= (block.kernelSize - 1) * jump
            if block.poolSize:
                r+= (block.poolSize - 1) * jump
                jump *= block.poolSize
        return r


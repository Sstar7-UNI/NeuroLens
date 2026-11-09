import math
import time
import tensorflow as tf
from tensorflow import keras

from neurolens.TumorClass import TumorClass
from neurolens.controller import TrainingConfig
from neurolens.data_storage import ModelConfig
from neurolens.data_storage.ModelRepository import ModelRepository
from neurolens.network.CNNClassifier import CNNClassifier
from neurolens.preprocessing.AugmentedBatchGenerator import AugmentedBatchGenerator
from neurolens.preprocessing.ImagePreprocessor import ImagePreprocessor
from neurolens.results.TrainingReport import TrainingReport

SHUFFLE_BUFFER = 1024
class EpochTimer(keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.durationMs = []
        self._start = 0.0
    def on_train_begin(self,logs = None):
        self.durationMs = []
    def onepoch_begin(self,logs = None):
        self._start = time.perf_counter()
    def on_epoch_end(self,epoch, logs = None):
        elapsed = time.perf_counter() - self._start
        self.durationMs.append(elapsed*1000)
class Learner:
    def __init__(self, config:TrainingConfig,modelConfig: ModelConfig,
                 preprocessor:ImagePreprocessor, repository:ModelRepository):
        self._config = config
        self._modelConfig = modelConfig
        self._preprocessor = preprocessor
        self._repository = repository
        self._generator = AugmentedBatchGenerator(config.augmentation, preprocessor)
        self._timer = EpochTimer()

    def train(self, split) -> TrainingReport:
        cfg = self._config
        keras.utils.set_random_seed(cfg.seed)

        trainSamples = split.getTrainingSet()
        validationSamples = split.getValidationSet()

        self._preprocessor.fitNormalization(trainSamples)

        trainDataset = self.buildTrainingDataset(trainSamples)
        validationDataset = self._preprocessor.buildDataset(validationSamples)
        validationDataset = validationDataset.batch(cfg.batchSize)
        validationDataset = validationDataset.prefetch(tf.data.AUTOTUNE)

        if cfg.decoupledDecay:
            l2 = 0.0
        else:
            l2 = cfg.weightDecay

        model = CNNClassifier(self._modelConfig, dropoutRate=cfg.dropoutRate, l2=l2)

        f1Metric = keras.metrics.F1Score(average="macro", name="macro_f1")
        model.compile(
            optimizer=self.buildOptimizer(),
            loss=self.buildLoss(),
            metrics=[f1Metric],
        )

        classWeights = self.computeClassWeights(trainSamples)
        if len(classWeights) == 0:
            classWeights = None

        callbacks = [self.buildEarlyStopping(), self.buildCheckpoint(), self._timer]
        history = model.fit(
            trainDataset,
            validation_data=validationDataset,
            epochs=cfg.epochs,
            callbacks=callbacks,
            class_weight=classWeights,
        )

        self._modelConfig.epochsTrained = len(history.history["loss"])
        self._modelConfig.parameterCount = int(model.count_params())
        self._modelConfig.normalizationMean = self._normalizationMean()
        self._modelConfig.normalizationStd = self._normalizationStd()
        self._repository.saveConfig(self._modelConfig)

        return self.toReport(history)

    def buildTrainingDataset(self, samples):
        if self._config.useAugmentation:
            self._generator.reset(samples)
            dataset = self._generator.asDataset()
        else:
            dataset = self._preprocessor.buildDataset(samples)

        dataset = dataset.shuffle(SHUFFLE_BUFFER, seed=self._config.seed)
        dataset = dataset.batch(self._config.batchSize)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset

    def buildOptimizer(self):
        if self._config.decoupledDecay:
            return keras.optimizers.Adam(learning_rate=self._config.learningRate,
                                         weight_decay=self._config.weightDecay)
        return keras.optimizers.Adam(learning_rate=self._config.learningRate)

    def buildLoss(self):
        return keras.losses.CategoricalCrossentropy(from_logits=True)

    def buildEarlyStopping(self):
        return keras.callbacks.EarlyStopping(
            monitor="val_macro_f1", mode="max",
            patience=self._config.patience, restore_best_weights=True)

    def buildCheckpoint(self):
        return keras.callbacks.ModelCheckpoint(
            filepath=self._repository.getCheckpointPath(),
            monitor="val_macro_f1", mode="max",
            save_best_only=True, save_weights_only=True)

    def computeClassWeights(self, samples):
        weights = {}
        if not self._config.useClassWeights:
            return weights

        counts = {}
        for k in TumorClass:
            counts[k.id] = 0
        for sample in samples:
            counts[sample.label.id] += 1

        n = len(samples)
        for k in TumorClass:
            count = counts[k.id]
            if count == 0:
                count = 1
            weights[k.id] = n / (len(TumorClass) * count)
        return weights

    def toReport(self, history) -> TrainingReport:
        h = history.history
        validationF1 = self._toFloatList(h["val_macro_f1"])

        bestEpoch = 0  # 0-based
        for i in range(len(validationF1)):
            if validationF1[i] > validationF1[bestEpoch]:
                bestEpoch = i

        return TrainingReport(
            epochsCompleted=len(validationF1),
            bestEpoch=bestEpoch,
            bestValidationMacroF1=validationF1[bestEpoch],
            trainLoss=self._toFloatList(h["loss"]),
            validationLoss=self._toFloatList(h["val_loss"]),
            validationMacroF1=validationF1,
            epochDurationsMs=self._toFloatList(self._timer.durationsMs),
        )

    def _toFloatList(self, values):
        result = []
        for value in values:
            result.append(float(value))
        return result

    def _normalizationMean(self) -> float:
        flat = tf.reshape(self._preprocessor.normalization.mean, [-1])
        return float(flat[0])

    def _normalizationStd(self) -> float:
        flat = tf.reshape(self._preprocessor.normalization.variance, [-1])
        return math.sqrt(float(flat[0]))
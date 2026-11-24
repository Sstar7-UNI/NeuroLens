from concurrent.futures import ThreadPoolExecutor
import math
import os
import time

import numpy as np
import tensorflow as tf
from tensorflow import keras
from neurolens.data_management.TumorClass import TumorClass
from neurolens.controller import TrainingConfig
from neurolens.data_storage import ModelConfig
from neurolens.data_storage.ModelRepository import ModelRepository
from neurolens.network.CNNClassifier import CNNClassifier
from neurolens.preprocessing.AugmentedBatchGenerator import AugmentedBatchGenerator
from neurolens.preprocessing.ImagePreprocessor import ImagePreprocessor
from neurolens.results.TrainingReport import TrainingReport

SHUFFLE_BUFFER = 1024
BALANCE_MACRO_WEIGHT = 0.70
BALANCE_MIN_WEIGHT = 0.30


class EpochTimer(keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.durationMs = []
        self._start = 0.0

    def on_train_begin(self, logs=None):
        self.durationMs = []
        self._start = 0.0

    def on_epoch_begin(self, epoch, logs=None):
        self._start = time.perf_counter()

    def on_epoch_end(self, epoch, logs=None):
        if self._start:
            self.durationMs.append(
                (time.perf_counter() - self._start) * 1000.0
            )


class EmaBalancedF1(keras.callbacks.Callback):
    def __init__(
        self,
        validationDataset,
        classCount: int,
        alpha: float = 0.4,
        macroWeight: float = BALANCE_MACRO_WEIGHT,
        minWeight: float = BALANCE_MIN_WEIGHT,
    ):
        super().__init__()
        self._dataset = validationDataset
        self._classCount = classCount
        self._alpha = alpha
        self._macroWeight = macroWeight
        self._minWeight = minWeight
        self._labels = None
        self._ema = None

    def on_train_begin(self, logs=None):
        self._ema = None
        self._labels = np.concatenate(
            [
                np.argmax(labels.numpy(), axis=1)
                for _, labels in self._dataset
            ]
        )

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            return

        predictions = np.argmax(
            self.model.predict(self._dataset, verbose=0),
            axis=1,
        )

        matrix = np.zeros(
            (self._classCount, self._classCount),
            dtype=np.int64,
        )
        np.add.at(matrix, (self._labels, predictions), 1)

        tp = np.diag(matrix).astype(np.float64)
        fp = matrix.sum(axis=0) - tp
        fn = matrix.sum(axis=1) - tp
        denominator = 2.0 * tp + fp + fn
        perClass = np.divide(
            2.0 * tp,
            denominator,
            out=np.zeros_like(tp),
            where=denominator > 0.0,
        )

        balanced = (
            self._macroWeight * float(np.mean(perClass))
            + self._minWeight * float(np.min(perClass))
        )

        if self._ema is None:
            self._ema = balanced
        else:
            self._ema = (
                self._alpha * balanced
                + (1.0 - self._alpha) * self._ema
            )

        logs["val_min_class_f1"] = float(np.min(perClass))
        logs["val_ema_balanced_f1"] = float(self._ema)


class Learner:
    def __init__(
        self,
        config: TrainingConfig,
        modelConfig: ModelConfig,
        preprocessor: ImagePreprocessor,
        repository: ModelRepository,
    ):
        self._config = config
        self._modelConfig = modelConfig
        self._preprocessor = preprocessor
        self._repository = repository
        self._generator = AugmentedBatchGenerator(
            config.augmentation,
            preprocessor,
        )
        self._timer = EpochTimer()
        self._model = None
        self._on_phase = None

    def _emit_phase(self, event: str, **info) -> None:
        cb = getattr(self, "_on_phase", None)
        if callable(cb):
            cb(event, info)

    def train(self, split) -> TrainingReport:
        cfg = self._config
        keras.utils.set_random_seed(cfg.seed)
        trainSamples = split.getTrainingSet()
        validationSamples = split.getValidationSet()
        nTrain = len(trainSamples)
        nVal = len(validationSamples)
        batchSize = cfg.batchSize
        useAugmentation = cfg.useAugmentation
        copiesPerImage = cfg.augmentation.copiesPerImage if useAugmentation else 0
        examplesPerEpoch = nTrain * (1 + copiesPerImage)
        stepsPerEpoch = math.ceil(examplesPerEpoch / batchSize)
        print(
            f"[TRAIN] N_train={nTrain} | N_val={nVal} | "
            f"B={batchSize} | augmentation={useAugmentation} | "
            f"K={copiesPerImage} | examples={examplesPerEpoch} | "
            f"steps={stepsPerEpoch}",
            flush=True
        )
        normalizationStart = time.perf_counter()
        self._preprocessor.fitNormalization(trainSamples)
        normalizationTime = time.perf_counter() - normalizationStart
        print(
            f"[PERF] Normalization: {normalizationTime:.2f}s",
            flush=True
        )
        datasetStart = time.perf_counter()
        trainDataset = self.buildOptimizedTrainingDataset(trainSamples)
        trainDataset = trainDataset.apply(tf.data.experimental.assert_cardinality(stepsPerEpoch))
        datasetTime = time.perf_counter() - datasetStart
        print(
            f"[PERF] Dataset construction: {datasetTime:.2f}s",
            flush=True
        )
        validationDataset = self._preprocessor.buildDataset(validationSamples)
        validationDataset = validationDataset.batch(cfg.batchSize)
        validationDataset = validationDataset.prefetch(tf.data.AUTOTUNE)
        l2 = 0.0 if cfg.decoupledDecay else cfg.weightDecay
        model = CNNClassifier(
            self._modelConfig,
            dropoutRate=cfg.dropoutRate,
            l2=l2
        )
        model.compile(
            optimizer=self.buildOptimizer(),
            loss=self.buildLoss(),
            steps_per_execution=16,
            metrics=[
                keras.metrics.F1Score(
                    average="macro",
                    name="macro_f1",
                )
            ],
        )
        classWeights = self.computeClassWeights(trainSamples)
        if not classWeights:
            classWeights = None

        callbacks = [
            EmaBalancedF1(
                validationDataset,
                len(TumorClass),
                alpha=cfg.emaAlpha,
            ),
            self.buildEarlyStopping(),
            self.buildCheckpoint(),
            self._timer,
        ]
        self._emit_phase(
            "compile_end",
            duration_s=time.perf_counter() - t_compile,
        )

        self._emit_phase("fit_start", epochs=cfg.epochs)
        t_fit = time.perf_counter()
        history = model.fit(
            trainDataset,
            validation_data=validationDataset,
            epochs=cfg.epochs,
            callbacks=callbacks,
            class_weight=classWeights,
            shuffle=False,
            verbose=0,
        )
        self._emit_phase(
            "fit_end",
            duration_s=time.perf_counter() - t_fit,
            epochs_completed=len(history.history["loss"]),
        )
        self._model = model
        self._modelConfig.epochsTrained = len(history.history["loss"])
        self._modelConfig.parameterCount = int(model.count_params())
        self._modelConfig.normalizationMean = self._normalizationMean()
        self._modelConfig.normalizationStd = self._normalizationStd()
        self._repository.saveConfig(self._modelConfig)
        return self.toReport(history)

    def buildTrainingDataset(self, samples):
        if self._config.useAugmentation:
            copies = self._config.augmentation.copiesPerImage
            base_imgs = [self._preprocessor.preprocess(r.imagePath) for r in samples]
            base_labels = [int(r.label) for r in samples]

            workers = 1
            if workers > 1 and len(base_imgs) >= 32:
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    aug_groups = list(
                        pool.map(
                            lambda img: self._preprocessor.augment(img, copies),
                            base_imgs,
                        )
                    )
            else:
                aug_groups = [
                    self._preprocessor.augment(img, copies) for img in base_imgs
                ]

            all_imgs = list(base_imgs)
            all_labels = list(base_labels)
            for lbl, aug_list in zip(base_labels, aug_groups):
                all_imgs.extend(aug_list)
                all_labels.extend([lbl] * len(aug_list))

            allImages = tf.stack(all_imgs, axis=0)
            allLabels = tf.one_hot(all_labels, len(TumorClass))
            dataset = tf.data.Dataset.from_tensor_slices(
                (allImages, allLabels)
            )
            shuffleBuffer = int(allImages.shape[0])
        else:
            dataset = self._preprocessor.buildDataset(samples).cache()
            shuffleBuffer = max(len(samples), SHUFFLE_BUFFER)

        dataset = dataset.shuffle(
            shuffleBuffer,
            seed=self._config.seed,
            reshuffle_each_iteration=True,
        )
        dataset = dataset.batch(self._config.batchSize)

        options = tf.data.Options()
        options.experimental_deterministic = False
        dataset = dataset.with_options(options)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        return dataset

    def buildOptimizer(self):
        if self._config.decoupledDecay:
            return keras.optimizers.Adam(
                learning_rate=self._config.learningRate,
                weight_decay=self._config.weightDecay,
            )

        return keras.optimizers.Adam(
            learning_rate=self._config.learningRate
        )

    def buildLoss(self):
        return keras.losses.CategoricalCrossentropy(
            from_logits=True
        )

    def buildEarlyStopping(self):
        return keras.callbacks.EarlyStopping(
            monitor="val_ema_balanced_f1",
            mode="max",
            min_delta=1e-5,
            patience=self._config.patience,
            restore_best_weights=True,
        )

    def _getCheckpointPath(self):
        getter = getattr(
            self._repository,
            "getCheckpointPath",
            None,
        )

        if callable(getter):
            checkpointPath = getter()
        else:
            checkpointPath = None

            for attribute in (
                "_checkpoint",
                "_checkpointPath",
                "_checkpoint_path",
                "checkpointPath",
                "checkpoint_path",
                "checkpoint",
                "_weightsPath",
                "_weights_path",
                "weightsPath",
                "weights_path",
            ):
                value = getattr(self._repository, attribute, None)

                if value is not None and not callable(value):
                    checkpointPath = value
                    break

        if checkpointPath is None:
            raise AttributeError(
                "ModelRepository must expose its checkpoint path"
            )

        checkpointPath = os.fspath(checkpointPath)
        parent = os.path.dirname(checkpointPath)

        if parent:
            os.makedirs(parent, exist_ok=True)

        return checkpointPath

    def buildCheckpoint(self):
        return keras.callbacks.ModelCheckpoint(
            filepath=self._getCheckpointPath(),
            monitor="val_ema_balanced_f1",
            mode="max",
            save_best_only=True,
            save_weights_only=True,
        )

    def computeClassWeights(self, samples):
        if not getattr(self._config, "useClassWeights", False):
            return {}

        counts = {
            int(classId): 0
            for classId in TumorClass
        }

        for sample in samples:
            counts[int(sample.label)] += 1

        total = len(samples)
        weights = {}

        for classId in TumorClass:
            count = counts[int(classId)]

            if count > 0:
                weights[int(classId)] = (
                    total / (len(TumorClass) * count)
                )

        return weights

    def toReport(self, history) -> TrainingReport:
        h = history.history

        validationF1 = self._toFloatList(
            h["val_macro_f1"]
        )
        validationMinClassF1 = self._toFloatList(
            h["val_min_class_f1"]
        )
        validationEmaBalancedF1 = self._toFloatList(
            h["val_ema_balanced_f1"]
        )

        bestEpoch = max(
            range(len(validationEmaBalancedF1)),
            key=validationEmaBalancedF1.__getitem__,
        )

        return TrainingReport(
            epochsCompleted=len(validationF1),
            bestEpoch=bestEpoch,
            bestValidationMacroF1=validationF1[bestEpoch],
            trainLoss=self._toFloatList(h["loss"]),
            validationLoss=self._toFloatList(h["val_loss"]),
            validationMacroF1=validationF1,
            epochDurationsMs=self._toFloatList(
                self._timer.durationMs
            ),
            bestValidationEmaBalancedF1=validationEmaBalancedF1[bestEpoch],
            bestValidationMinClassF1=validationMinClassF1[bestEpoch],
            validationEmaBalancedF1=validationEmaBalancedF1,
            validationMinClassF1=validationMinClassF1,
        )

    @staticmethod
    def _toFloatList(values):
        return [float(value) for value in values]

    def _normalizationMean(self) -> float:
        flat = tf.reshape(
            self._preprocessor.normalization.mean,
            [-1],
        )

        return float(flat[0].numpy())

    def _normalizationStd(self) -> float:
        flat = tf.reshape(
            self._preprocessor.normalization.variance,
            [-1],
        )
        variance = float(flat[0].numpy())
        return math.sqrt(max(variance, keras.backend.epsilon()))

    def buildTrainingSource(self, samples) -> tf.data.Dataset:
        paths = [record.imagePath for record in samples]
        labels = [int(record.label) for record in samples]
        dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
        dataset = dataset.shuffle(buffer_size=len(samples),seed=self._config.seed,reshuffle_each_iteration=True)
        return dataset

    def decodeTrainingSource(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        new_dataset = dataset.map(lambda path, label: (self._preprocessor.decode(path), label),
                                  num_parallel_calls=tf.data.AUTOTUNE)
        return new_dataset

    def applyTrainingOperators(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        preprocessed_dataset = dataset.map(lambda image, label: (self._preprocessor.applyOperators(image),label),
                                           num_parallel_calls=1,deterministic=True)
        return preprocessed_dataset

    def normalizeTrainingSource(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        normalized_dataset = dataset.map(lambda image, label: (self._preprocessor.normalize(image), label),
                                         num_parallel_calls=1,deterministic=True)
        return normalized_dataset

    def buildPreprocessedTrainingDataset(self, samples) -> tf.data.Dataset:
        dataset = self.buildTrainingSource(samples)
        dataset = self.decodeTrainingSource(dataset)
        dataset = self.applyTrainingOperators(dataset)
        dataset = self.normalizeTrainingSource(dataset)
        return dataset

    def expandTrainingBatch(self, images: tf.Tensor, labels: tf.Tensor, copyIndex: tf.Tensor):
        return tf.cond(tf.equal(copyIndex, 0), lambda: (images, labels),
                       lambda: (self._preprocessor.augmentBatch(images), labels))

    def buildAugmentedBlockDataset(self,images: tf.Tensor,labels: tf.Tensor,copiesPerImage: int) -> tf.data.Dataset:
        dataset = tf.data.Dataset.range(copiesPerImage+1)
        dataset = dataset.map(lambda index: self.expandTrainingBatch(images, labels, index),
                              num_parallel_calls=1,deterministic=True)
        return dataset

    def buildAugmentedTrainingBlocks(self, samples) -> tf.data.Dataset:
        dataset = self.buildPreprocessedTrainingDataset(samples)
        cfg = self._config
        copiesPerImage = cfg.augmentation.copiesPerImage if cfg.useAugmentation else 0
        blockSize = min(32, cfg.batchSize)
        dataset = dataset.batch(blockSize, drop_remainder=False)
        dataset = dataset.flat_map(lambda images, labels: self.buildAugmentedBlockDataset(images, labels, copiesPerImage))
        return dataset

    def unbatchTrainingBlocks(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        dataset = dataset.unbatch()
        return dataset

    def shuffleTrainingExamples(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        dataset = dataset.shuffle(buffer_size= 1024, seed=self._config.seed,
                                  reshuffle_each_iteration=True)
        return dataset

    def batchTrainingExamples(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        dataset = dataset.batch(self._config.batchSize, drop_remainder=False)
        return dataset

    def prefetchTrainingBatches(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset

    def buildOptimizedTrainingDataset(self, samples) -> tf.data.Dataset:
        dataset = self.buildAugmentedTrainingBlocks(samples)
        dataset = self.unbatchTrainingBlocks(dataset)
        dataset = self.shuffleTrainingExamples(dataset)
        dataset = dataset.map(
            lambda image, label: (image,tf.one_hot(label, 4)),
            num_parallel_calls=1,
            deterministic=True
        )
        dataset = self.batchTrainingExamples(dataset)
        dataset = self.prefetchTrainingBatches(dataset)
        return dataset

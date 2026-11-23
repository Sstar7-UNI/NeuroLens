import random
from typing import Dict, List, Optional

import tensorflow as tf

from ..data_management.TumorClass import TumorClass
from ..controller.AugmentationConfig import AugmentationConfig
from ..data_management.ImageRecord import ImageRecord
from .AugmentedBatch import AugmentedBatch
from .ImagePreprocessor import ImagePreprocessor


class AugmentedBatchGenerator:
    def __init__(self, config: AugmentationConfig, preprocessor: ImagePreprocessor):
        self._config: AugmentationConfig = config
        self._preprocessor: ImagePreprocessor = preprocessor
        self._samples: List[ImageRecord] = []
        self._pendingByClass: Dict[TumorClass, List[ImageRecord]] = {c: [] for c in TumorClass}
        self._current: Optional[AugmentedBatch] = None
        self._augmentFn = tf.function(self._augmentTensor, reduce_retracing=True)

    def _augmentTensor(self, x: tf.Tensor) -> tf.Tensor:
        p = self._preprocessor
        x = p._rotation(x, training=True)
        x = p._translation(x, training=True)
        return p._zoom(x, training=True)

    def reset(self, samples: List[ImageRecord]):
        self._samples = list(samples)
        rng = random.Random(self._config.seed)
        self._pendingByClass = {c: [] for c in TumorClass}
        for r in self._samples:
            self._pendingByClass[r.label].append(r)
        for items in self._pendingByClass.values():
            rng.shuffle(items)
        self._current = None

    def hasNext(self) -> bool:
        return any(len(v) > 0 for v in self._pendingByClass.values())

    def nextBatch(self) -> AugmentedBatch:
        if not self.hasNext():
            raise StopIteration("no more batches")
        if self._current is not None:
            self._current.release()
        n = self._config.imagesPerClass
        copies = self._config.copiesPerImage
        batch = AugmentedBatch()
        indices: List[int] = []
        images: List[tf.Tensor] = []
        for items in self._pendingByClass.values():
            take = items[:n]
            del items[:n]
            for r in take:
                img = self._preprocessor.preprocess(r.imagePath)
                indices.append(batch.add(r, img))
                images.append(img)
        if copies > 0 and images:
            stacked = tf.repeat(tf.stack(images, axis=0), copies, axis=0)
            augmented = tf.unstack(self._augmentFn(stacked), axis=0)
            for k, a in enumerate(augmented):
                batch.addAugmented(a, indices[k // copies])
        self._current = batch
        return batch

    def _gen(self):
        self.reset(self._samples)
        while self.hasNext():
            batch = self.nextBatch()
            for img, lab in batch.toDataset():
                yield img, lab

    def asDataset(self) -> tf.data.Dataset:
        h, w = self._preprocessor._height, self._preprocessor._width
        return tf.data.Dataset.from_generator(
            self._gen,
            output_signature=(
                tf.TensorSpec(shape=(h, w, 1), dtype=tf.float32),
                tf.TensorSpec(shape=(4,), dtype=tf.float32),
            ),
        )
from __future__ import annotations

import math
from typing import Dict, List, Tuple

import tensorflow as tf
from tensorflow.keras import layers

from ..data_management.ImageRecord import ImageRecord
from .ImageOperator import ImageOperator


class ImagePreprocessor:
    def __init__(
        self,
        width: int = 224,
        height: int = 224,
        pipeline: List[ImageOperator] | None = None,
        pipeline_key: str = "baseline",
        cache: Dict[Tuple[str, str], tf.Tensor] | None = None,
    ) -> None:
        self._height = height
        self._width = width
        self._pipeline: List[ImageOperator] = list(pipeline) if pipeline is not None else []
        self._pipeline_key = pipeline_key
        self._cache: Dict[Tuple[str, str], tf.Tensor] = cache if cache is not None else {}
        self._normalization = layers.Normalization(axis=-1)
        self._rotation = layers.RandomRotation(0.05, fill_mode="constant")
        self._translation = layers.RandomTranslation(
            0.05, 0.05, fill_mode="constant"
        )
        self._zoom = layers.RandomZoom(
            (-0.1, 0.1), fill_mode="constant"
        )

    @property
    def normalization(self):
        return self._normalization

    @property
    def pipeline(self) -> List[ImageOperator]:
        return self._pipeline

    def setPipeline(
        self,
        pipeline: List[ImageOperator],
        pipeline_key: str | None = None,
    ) -> None:
        self._pipeline = list(pipeline)
        if pipeline_key is not None:
            self._pipeline_key = pipeline_key
        else:
            self._cache.clear()

    def setNormalization(self, mean: float, std: float) -> None:
        mean = float(mean)
        std = float(std)

        if not math.isfinite(mean) or not math.isfinite(std) or std < 0:
            raise ValueError("Normalization parameters must be finite and std non-negative")

        variance = max(std * std, tf.keras.backend.epsilon())

        self._normalization = layers.Normalization(
            axis=-1,
            mean=[mean],
            variance=[variance],
        )

    def decode(self, path: str) -> tf.Tensor:
        contents = tf.io.read_file(path)
        image = tf.io.decode_image(
            contents,
            channels=1,
            expand_animations=False,
        )
        image = tf.image.convert_image_dtype(image, tf.float32)

        return tf.image.resize_with_pad(
            image,
            self._height,
            self._width,
        )

    def _run_pipeline_tensor(self, path: str) -> tf.Tensor:
        cache_id = (self._pipeline_key, path)
        if cache_id in self._cache:
            return self._cache[cache_id]

        image = self.decode(path)
        for op in self._pipeline:
            image = op.apply(image)

        self._cache[cache_id] = image
        return image

    def preprocess(self, path: str) -> tf.Tensor:
        image = self._run_pipeline_tensor(path)

        return tf.squeeze(
            self._normalization(tf.expand_dims(image, 0)),
            0,
        )

    def fitNormalization(self, samples: List[ImageRecord]) -> None:
        if not samples:
            return

        has_fittable = any(
            hasattr(op, "fit") and callable(op.fit) for op in self._pipeline
        )
        if has_fittable:
            current_tensors = [self.decode(r.imagePath) for r in samples]
            for op in self._pipeline:
                if hasattr(op, "fit") and callable(op.fit):
                    imgs_2d = [t.numpy()[:, :, 0] for t in current_tensors]
                    op.fit(imgs_2d)
                current_tensors = [op.apply(t) for t in current_tensors]
            for r, tensor in zip(samples, current_tensors):
                self._cache[(self._pipeline_key, r.imagePath)] = tensor
            stacked = tf.stack(current_tensors, axis=0)
        else:
            tensors = [self._run_pipeline_tensor(r.imagePath) for r in samples]
            stacked = tf.stack(tensors, axis=0)

        self._normalization.adapt(stacked)

    def augment(self, image: tf.Tensor, copies: int) -> List[tf.Tensor]:
        output = []

        for _ in range(copies):
            x = tf.expand_dims(image, 0)
            x = self._rotation(x, training=True)
            x = self._translation(x, training=True)
            x = self._zoom(x, training=True)

            output.append(tf.squeeze(x, 0))

        return output

    def buildDataset(self, samples: List[ImageRecord]) -> tf.data.Dataset:
        if not samples:
            empty_x = tf.zeros((0, self._height, self._width, 1), dtype=tf.float32)
            empty_y = tf.zeros((0, 4), dtype=tf.float32)
            return tf.data.Dataset.from_tensor_slices((empty_x, empty_y))

        xs = tf.stack([self.preprocess(r.imagePath) for r in samples], axis=0)
        ys = tf.one_hot([int(r.label) for r in samples], depth=4)
        return tf.data.Dataset.from_tensor_slices((xs, ys))
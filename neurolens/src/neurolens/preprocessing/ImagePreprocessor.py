
import math
from typing import List

import tensorflow as tf
from tensorflow.keras import layers

from .ImageOperator import ImageOperator
from ..data_management.ImageRecord import ImageRecord


class ImagePreprocessor:
    def __init__(self, width: int = 224, height: int = 224):
        self._height = height
        self._width = width
        self._pipeline: List[ImageOperator] = []
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

    def preprocess(self, path: str) -> tf.Tensor:
        image = self.decode(path)

        for op in self._pipeline:
            image = op.apply(image)

        return tf.squeeze(
            self._normalization(tf.expand_dims(image, 0)),
            0,
        )

    def fitNormalization(self, samples: List[ImageRecord]) -> None:
        paths = [record.imagePath for record in samples]

        dataset = tf.data.Dataset.from_tensor_slices(paths)
        dataset = dataset.map(self.decode).batch(32)

        self._normalization.adapt(dataset)

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
        paths = [record.imagePath for record in samples]
        labels = [int(record.label) for record in samples]

        dataset = tf.data.Dataset.from_tensor_slices(
            (paths, labels)
        )

        return dataset.map(
            lambda path, label: (
                self.preprocess(path),
                tf.one_hot(label, 4),
            )
        )

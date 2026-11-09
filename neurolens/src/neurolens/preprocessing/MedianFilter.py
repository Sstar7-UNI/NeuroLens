from neurolens.preprocessing.ImageOperator import ImageOperator
from neurolens.preprocessing.PaddingMode import PaddingMode
import tensorflow as tf

class MedianFilter(ImageOperator):

    def __init__(self, kernelSize: int, padding: PaddingMode) -> None:
        if kernelSize <= 0 or kernelSize % 2 == 0:
            raise ValueError("kernelSize must be an odd and positive number")
        if not isinstance(padding, PaddingMode):
            raise ValueError("padding must be a PaddingMode")
        self._kernelSize = kernelSize
        self._padding = padding

    def apply(self, input: tf.Tensor) -> tf.Tensor:
        radius = self._kernelSize // 2
        paddings = [[radius, radius],[radius, radius],[0, 0]]
        if self._padding == PaddingMode.ZERO:
            padded = tf.pad(input, paddings, constant_values=0.0)
        elif self._padding == PaddingMode.REFLECT:
            padded = tf.pad(input, paddings, mode="REFLECT")
        elif self._padding == PaddingMode.REPLICATE:
            top = tf.repeat(input[:1, :, :], repeats=radius, axis=0)
            bottom = tf.repeat(input[-1:, :, :], repeats=radius, axis=0)
            padded = tf.concat([top, input, bottom], axis=0)
            left = tf.repeat(padded[:, :1, :], repeats=radius, axis=1)
            right = tf.repeat(padded[:, -1:, :], repeats=radius, axis=1)
            padded = tf.concat([left, padded, right], axis=1)
        image_batch = tf.expand_dims(padded, axis=0)
        patches = tf.image.extract_patches(
            images=image_batch,
            sizes=[1, self._kernelSize, self._kernelSize, 1],
            strides=[1, 1, 1, 1],
            rates=[1, 1, 1, 1],
            padding="VALID"
        )
        sorted_patches = tf.sort(patches, axis=-1)
        middle = (self._kernelSize * self._kernelSize) // 2
        medians = sorted_patches[:, :, :, middle:middle + 1]
        return tf.squeeze(medians, axis=0)
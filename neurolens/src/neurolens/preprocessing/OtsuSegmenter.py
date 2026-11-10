import tensorflow as tf
from neurolens.preprocessing.ImageOperator import ImageOperator

class OtsuSegmenter(ImageOperator):

    def __init__(self) -> None:
        super().__init__()
        self._threshold = tf.Variable(-1, dtype=tf.int32, trainable=False)
        self._separability = tf.Variable(0.0, dtype=tf.float64, trainable=False)

    def betweenClassVariance(self, histogram: tf.Tensor, t: tf.Tensor) -> tf.Tensor:
        levels = tf.cast(tf.range(tf.size(histogram)), histogram.dtype)
        histogram0 = histogram[:t + 1]
        histogram1 = histogram[t + 1:]
        w0 = tf.reduce_sum(histogram0)
        w1 = tf.reduce_sum(histogram1)
        def validVariance():
            mean0 = tf.reduce_sum(levels[:t + 1] * histogram0) / w0
            mean1 = tf.reduce_sum(levels[t + 1:] * histogram1) / w1
            return w0 * w1 * (mean0 - mean1) ** 2
        def invalidVariance():
            return tf.constant(-1.0, dtype=histogram.dtype)
        return tf.cond((w0 > 0) & (w1 > 0),validVariance,invalidVariance)

    def computeThreshold(self, histogram: tf.Tensor) -> tf.Tensor:
        numberOfLevels = tf.size(histogram)
        candidates = tf.range(numberOfLevels - 1)
        def evaluateThreshold(t):
            return self.betweenClassVariance(histogram, t)
        variances = tf.map_fn(
            evaluateThreshold,
            candidates,
            fn_output_signature=tf.TensorSpec(shape=(),dtype=histogram.dtype)
        )
        bestVariance = tf.reduce_max(variances)
        def validThreshold():
            return tf.argmax(variances, output_type=tf.int32)
        def constantThreshold():
            return tf.argmax(histogram, output_type=tf.int32)
        threshold = tf.cond(bestVariance >= 0,validThreshold,constantThreshold)
        levels = tf.cast(tf.range(numberOfLevels), histogram.dtype)
        globalMean = tf.reduce_sum(levels * histogram)
        totalVariance = tf.reduce_sum(histogram * (levels - globalMean) ** 2)
        separability = tf.math.divide_no_nan(
            tf.maximum(bestVariance, tf.zeros_like(bestVariance)),
            totalVariance
        )
        separability = tf.clip_by_value(separability, 0.0, 1.0)
        self._threshold.assign(threshold)
        self._separability.assign(tf.cast(separability, self._separability.dtype))
        return threshold

    def apply(self, input: tf.Tensor) -> tf.Tensor:
        numberOfLevels = 256
        counts = tf.histogram_fixed_width(
            input,
            value_range=[0.0, 1.0],
            nbins=numberOfLevels
        )
        histogram = counts / tf.size(input)
        threshold = self.computeThreshold(histogram)
        intervals = tf.histogram_fixed_width_bins(
            input,
            value_range=[0.0, 1.0],
            nbins=numberOfLevels
        )
        mask = intervals > threshold
        return tf.cast(mask, input.dtype)
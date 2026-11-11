from neurolens.preprocessing.ImageOperator import ImageOperator
import tensorflow as tf

class HistogramEqualizer(ImageOperator):

    def __init__(self, levels: int) -> None:
        if levels < 2:
            raise ValueError("Levels must be greater than or equal to 2")
        self._levels = levels

    def computeHistogram(self, input: tf.Tensor)-> tf.Tensor:
        pixeliPerInterval=tf.histogram_fixed_width(input,value_range=[0.0,1.0],nbins=self._levels)
        pixeliTotali=tf.size(input)
        return pixeliPerInterval/pixeliTotali

    def buildMapping(self, histogram: tf.Tensor) -> tf.Tensor:
        mapping=tf.math.cumsum(histogram)
        mapping= mapping * (self._levels - 1)
        mapping=tf.round(mapping)
        mapping=tf.cast(mapping, tf.int32)
        return mapping

    def apply(self, input: tf.Tensor) -> tf.Tensor:
        histogram = self.computeHistogram(input)
        mapping = self.buildMapping(histogram)
        intervals = tf.histogram_fixed_width_bins(input,value_range=[0.0,1.0],nbins=self._levels)
        newLevels = tf.gather(mapping, intervals)
        result = tf.cast(newLevels, input.dtype) / (self._levels - 1)
        return result
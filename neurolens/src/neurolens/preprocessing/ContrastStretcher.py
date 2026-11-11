from neurolens.preprocessing.ImageOperator import ImageOperator
import tensorflow as tf

class ContrastStretcher(ImageOperator):

    def __init__(self, lower: float, upper: float):
        if not (0 <= lower < upper <= 1):
            raise ValueError("lower and upper must be between 0 and 1")
        self._lower = lower
        self._upper = upper

    def apply(self,input: tf.Tensor) -> tf.Tensor:
        aux = (input - self._lower) / (self._upper - self._lower)
        result = tf.clip_by_value(aux, 0.0, 1.0)
        return result
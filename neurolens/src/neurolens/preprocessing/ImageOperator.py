from abc import ABC, abstractmethod
import tensorflow as tf
class ImageOperator(ABC):
    @abstractmethod
    def apply(self,input:tf.Tensor) -> tf.Tensor:
        pass
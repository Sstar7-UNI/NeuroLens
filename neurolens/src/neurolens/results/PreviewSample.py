from dataclass import dataclass
from neurolens.src.neurolens.TumorClass import TumorClass
import tensorflow as tf


@dataclass
class PreviewSample:
   _imagePath: str
   _label:TumorClass
   _original: tf.Tensor
   _preprocessed: tf.Tensor
   _augumented: List[tf.Tensor]
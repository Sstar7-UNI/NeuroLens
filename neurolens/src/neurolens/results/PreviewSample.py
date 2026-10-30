from dataclasses import dataclass
from typing import List

import tensorflow as tf

from ..data_management.TumorClass import TumorClass


@dataclass
class PreviewSample:
    _imagePath: str
    _label: TumorClass
    _original: tf.Tensor
    _preprocessed: tf.Tensor
    _augmented: List[tf.Tensor]
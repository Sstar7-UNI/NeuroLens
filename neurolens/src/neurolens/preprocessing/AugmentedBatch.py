from typing import List,Dict
import tensorflow as tf

from neurolens.src.neurolens.TumorClass import TumorClass
from ..data_management.ImageRecord import ImageRecord

class AugmentedBatch:
    def __init__(self):
        self.records:List[ImageRecord] = []
        self._preprocessed:List[tf.Tensor] = []
        self._augmented:List[tf.Tensor] = []
        self._augmentedSource:List[int] = []
        self._released: Boolean = false
    def size(self):
        return len(self.records)
    def classCounts(self) -> Dict[TumorClass, int]:
        counts = {c:0 for c in TumorClass}
        for r in self._records:
            counts[r.label] += 1
        return counts
    def toDataset(self) -> tf.data.Dataset:
        images = self._preprocessed + self._augmented
        labels = [int(r.label) for r in self._records]
        labels += [int(self._records[i].label) for i in self._augmentedSource]
        ds = tf.data.Dataset.from_tensor_slices((tf.stack(images),tf.one_hot(labels,4)))
        return ds
    def release(self):
        self._records.clear()
        self._preprocessed.clear()
        self._augmented.clear()
        self._augmentedSource.clear()
        self._released = True
    def isReleased(self):
        return self._released
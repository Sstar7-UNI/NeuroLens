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

    def add(self, record: ImageRecord, preprocessed: tf.Tensor) -> int:
        self._records.append(record)
        self._preprocessed.append(preprocessed)
        return len(self._records) - 1

    def addAugmented(self, image: tf.Tensor, sourceIndex: int) -> None:
        self._augmented.append(image)
        self._augmentedSource.append(sourceIndex)

    def recordAt(self, i: int) -> ImageRecord:
        return self._records[i]

    def preprocessedAt(self, i: int) -> tf.Tensor:
        return self._preprocessed[i]

    def augmentedFor(self, i: int) -> List[tf.Tensor]:
        return [a for a, s in zip(self._augmented, self._augmentedSource) if s == i]
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
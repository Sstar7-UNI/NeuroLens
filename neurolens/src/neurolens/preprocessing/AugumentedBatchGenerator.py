from typing import Dict, List

from neurolens.src.neurolens.TumorClass import TumorClass
from neurolens.src.neurolens.data_management.ImageRecord import ImageRecord


class AugumentedBatchGenerator:
    def __init__(self):
        self._config: AugumentationConfig
        self._preprocessor:ImagePreprocessor
        self._pendingByClass: Dict[TumorClass,List[ImageRecord]]
        self._current: AugumentedBatch
    def reset(self, samples:List[ImageRecord]):
        pass
    def hasNext(self) -> Boolean:
        pass
    def nextBatch(self) -> AugumentedBatch:
        pass
    def asDataset(self) -> tf.data.Dataset:
        pass
from typing import List

from neurolens.src.neurolens.TumorClass import TumorClass
from neurolens.src.neurolens.results.PreviewSample import PreviewSample


class PreprocessingPreview:
    def __init__(self, batchIndex: int = 0, samples: List[PreviewSample] = None):
        self._batchIndex: int = batchIndex
        self._samples: List[PreviewSample] = samples if samples is not None else []

    def getSamples(self, label: TumorClass) -> List[PreviewSample]:
        return [s for s in self._samples if s._label == label]

    def totalCounts(self) -> int:
        return len(self._samples)

    def augumentedCounts(self) -> int:
        return sum(len(s._augmented) for s in self._samples)

    def release(self):
        self._samples.clear()
from neurolens.src.neurolens.TumorClass import TumorClass
from neurolens.src.neurolens.results.PreviewSample import PreviewSample


class PreprocessingPreview:
    def __init__(self):
        self._batchIndex: int = 0
        self._samples:List[PreviewSample]

    def getSamples(self, label:TumorClass) -> List[PreviewSample]:
       return [s for s in self._samples if s._label == label]
    def totalCounts(self) -> int:
        return len(self._samples)
    def augumentedCounts(self) -> int:
        return sum(len(s.augmented) for s in self._samples)
    def release(self):
        return self._samples.clear()
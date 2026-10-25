from neurolens.src.neurolens.TumorClass import TumorClass
from neurolens.src.neurolens.results.PreviewSample import PreviewSample


class PreprocessingPreview:
    def __init__(self):
        self._batchIndex: int = 0
        self._samples:List[PreviewSample]

    def getSamples(self, label:TumorClass) -> List[PreviewSample]:
        pass
    def totalCounts(self) -> int:
        pass
    def augumentedCounts(self) -> int:
        pass
    def release(self):
        pass
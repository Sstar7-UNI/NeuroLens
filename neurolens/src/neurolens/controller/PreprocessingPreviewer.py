from .AugmentationConfig import AugmentationConfig
from ..data_management.DatasetSplit import DatasetSplit
from ..preprocessing.AugmentedBatch import AugmentedBatch
from ..preprocessing.AugmentedBatchGenerator import AugmentedBatchGenerator
from ..preprocessing.ImagePreprocessor import ImagePreprocessor
from ..results.PreprocessingPreview import PreprocessingPreview
from ..results.PreviewSample import PreviewSample


class PreprocessingPreviewer:
    def __init__(self, preprocessor: ImagePreprocessor):
        self._preprocessor: ImagePreprocessor = preprocessor
        self._generator: AugmentedBatchGenerator = None
        self._batchIndex: int = 0
        self.current: PreprocessingPreview = None

    def start(self, split: DatasetSplit, config: AugmentationConfig) -> PreprocessingPreview:
        self._generator = AugmentedBatchGenerator(config, self._preprocessor)
        self._generator.reset(split.getTrainingSet())
        self._batchIndex = 0
        self.current = None
        return self.next()

    def next(self) -> PreprocessingPreview:
        if self._generator is None or not self._generator.hasNext():
            return None
        if self.current is not None:
            self.current.release()
        batch = self._generator.nextBatch()
        self.current = self.__toPreview(batch)
        self._batchIndex += 1
        return self.current

    def __toPreview(self, batch: AugmentedBatch) -> PreprocessingPreview:
        samples = []
        for i in range(batch.size()):
            r = batch.recordAt(i)
            samples.append(PreviewSample(
                r.imagePath,
                r.label,
                self._preprocessor.decode(r.imagePath),
                batch.preprocessedAt(i),
                batch.augmentedFor(i),
            ))
        return PreprocessingPreview(self._batchIndex, samples)
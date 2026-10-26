from typing import List

from .ImageRecord import ImageRecord


class DatasetSplit:
    def __init__(
            self,
            trainingSamples: list[ImageRecord] | None = None,
            validationSamples: list[ImageRecord] | None = None,
            testSamples: list[ImageRecord] | None = None
    ) -> None:
        self._trainingSamples = (list(trainingSamples)if trainingSamples is not None else [])
        self._validationSamples = (list(validationSamples)if validationSamples is not None else [])
        self._testSamples = (list(testSamples)if testSamples is not None else [])
    def getTrainingSet(self) -> List[ImageRecord]:
        return self._trainingSamples
    def getValidationSet(self) -> List[ImageRecord]:
        return self._validationSamples
    def getTestSet(self) -> List[ImageRecord]:
        return self._testSamples
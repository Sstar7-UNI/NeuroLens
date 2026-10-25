from typing import List

from neurolens.src.neurolens.data_management.ImageRecord import ImageRecord


class DatasetSplit:
    def __init__(self):
        self._trainingSamples: List[ImageRecord] = []
        self._validationSamples: List[ImageRecord] = []
        self._testSamples: List[ImageRecord] = []
    def getTrainingSet(self) -> List[ImageRecord]:
        return self._trainingSamples
    def getValidationSet(self) -> List[ImageRecord]:
        return self._validationSamples
    def getTestSet(self) -> List[ImageRecord]:
        return self._testSamples
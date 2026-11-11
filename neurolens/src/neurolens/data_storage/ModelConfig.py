
from typing import List


class ModelConfig:
    def __init__(
            self,
            inputWidth: int,
            inputHeight: int,
            classNames: List[str],
            filterCounts: List[int],
            normalizationMean: float,
            normalizationStd: float,
            randomSeed: int,
            epochsTrained: int,
            parameterCount: int
    ):
        self._inputWidth = inputWidth
        self._inputHeight = inputHeight
        self._classNames = classNames
        self._filterCounts = filterCounts
        self._normalizationMean = normalizationMean
        self._normalizationStd = normalizationStd
        self._randomSeed = randomSeed
        self._epochsTrained = epochsTrained
        self._parameterCount = parameterCount

    @property
    def inputWidth(self):
        return self._inputWidth

    @inputWidth.setter
    def inputWidth(self, value):
        self._inputWidth = value

    @property
    def inputHeight(self):
        return self._inputHeight

    @inputHeight.setter
    def inputHeight(self, value):
        self._inputHeight = value

    @property
    def classNames(self):
        return self._classNames

    @classNames.setter
    def classNames(self, value):
        self._classNames = value

    @property
    def filterCounts(self):
        return self._filterCounts

    @filterCounts.setter
    def filterCounts(self, value):
        self._filterCounts = value

    @property
    def normalizationMean(self):
        return self._normalizationMean

    @normalizationMean.setter
    def normalizationMean(self, value):
        self._normalizationMean = value

    @property
    def normalizationStd(self):
        return self._normalizationStd

    @normalizationStd.setter
    def normalizationStd(self, value):
        self._normalizationStd = value

    @property
    def randomSeed(self):
        return self._randomSeed

    @randomSeed.setter
    def randomSeed(self, value):
        self._randomSeed = value

    @property
    def epochsTrained(self):
        return self._epochsTrained

    @epochsTrained.setter
    def epochsTrained(self, value):
        self._epochsTrained = value

    @property
    def parameterCount(self):
        return self._parameterCount

    @parameterCount.setter
    def parameterCount(self, value):
        self._parameterCount = value

    def validate(self) -> bool:
        return (
            self._inputWidth > 0
            and self._inputHeight > 0
            and self._classNames is not None
            and len(self._classNames) > 0
            and self._filterCounts is not None
            and len(self._filterCounts) > 0
        )

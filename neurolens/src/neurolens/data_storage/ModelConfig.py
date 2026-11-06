from typing import List
class ModelConfig:
    def __init__(self, inputWidth:int, inputHeight:int,
                 classNames:List[str], filterCounts:List[int],
                 normalizationMean:float,normalizationStd:float,
                 randomSeed:int, epochsTrained:int,parameterCount:int
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

    def validate(self) -> bool:
        return self._inputWidth > 0 and self._inputHeight > 0 and self._classNames is not None and len(self._classNames) > 0 and len(self._filterCounts) > 0

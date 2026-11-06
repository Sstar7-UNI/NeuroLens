from typing import List
class TrainingReport:
    def __init__(self, epochsCompleted:int, bestEpoch:int, bestValidationMacroF1:float, trainLoss:List[float], validationMacroF1:List[float], epochDurationMs:List[float]):
        self._epochsCompleted = epochsCompleted
        self._bestEpoch = bestEpoch
        self._bestValidationMacroF1 = bestValidationMacroF1
        self._trainLoss = trainLoss
        self._validationMacroF1 = validationMacroF1
        self._epochDurationMs = epochDurationMs
    def getTrainingHistory(self) ->List[float]:
        return self._trainLoss

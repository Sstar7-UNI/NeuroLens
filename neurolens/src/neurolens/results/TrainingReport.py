
class TrainingReport:
    def __init__(
        self,
        epochsCompleted: int,
        bestEpoch: int,
        bestValidationMacroF1: float,
        trainLoss: list[float],
        validationMacroF1: list[float] | None = None,
        epochDurationMs: list[float] | None = None,
        validationLoss: list[float] | None = None,
        epochDurationsMs: list[float] | None = None,
    ):
        self.epochsCompleted = epochsCompleted
        self.bestEpoch = bestEpoch
        self.bestValidationMacroF1 = bestValidationMacroF1
        self.trainLoss = trainLoss
        self.validationLoss = (
            validationLoss if validationLoss is not None else []
        )
        self.validationMacroF1 = (
            validationMacroF1 if validationMacroF1 is not None else []
        )
        self.epochDurationsMs = (
            epochDurationsMs
            if epochDurationsMs is not None
            else epochDurationMs if epochDurationMs is not None else []
        )

        self.epochDurationMs = self.epochDurationsMs

        self._epochsCompleted = self.epochsCompleted
        self._bestEpoch = self.bestEpoch
        self._bestValidationMacroF1 = self.bestValidationMacroF1
        self._trainLoss = self.trainLoss
        self._validationLoss = self.validationLoss
        self._validationMacroF1 = self.validationMacroF1
        self._epochDurationMs = self.epochDurationsMs

    def getTrainingHistory(self) -> list[float]:
        return self.trainLoss

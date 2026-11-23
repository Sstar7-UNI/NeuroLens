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
        bestValidationEmaBalancedF1: float = 0.0,
        bestValidationMinClassF1: float = 0.0,
        validationEmaBalancedF1: list[float] | None = None,
        validationMinClassF1: list[float] | None = None,
    ):
        self.epochsCompleted = epochsCompleted
        self.bestEpoch = bestEpoch
        self.bestValidationMacroF1 = bestValidationMacroF1
        self.trainLoss = trainLoss
        self.validationLoss = (validationLoss if validationLoss is not None else [])
        self.validationMacroF1 = (validationMacroF1 if validationMacroF1 is not None else [])
        self.epochDurationsMs = (
            epochDurationsMs
            if epochDurationsMs is not None
            else epochDurationMs if epochDurationMs is not None else []
        )

        self.epochDurationMs = self.epochDurationsMs

        self.bestValidationEmaBalancedF1 = bestValidationEmaBalancedF1
        self.bestValidationMinClassF1 = bestValidationMinClassF1
        self.validationEmaBalancedF1 = (
            validationEmaBalancedF1
            if validationEmaBalancedF1 is not None
            else []
        )
        self.validationMinClassF1 = (
            validationMinClassF1
            if validationMinClassF1 is not None
            else []
        )

        self._epochsCompleted = self.epochsCompleted
        self._bestEpoch = self.bestEpoch
        self._bestValidationMacroF1 = self.bestValidationMacroF1
        self._trainLoss = self.trainLoss
        self._validationLoss = self.validationLoss
        self._validationMacroF1 = self.validationMacroF1
        self._epochDurationMs = self.epochDurationsMs
        self._bestValidationEmaBalancedF1 = self.bestValidationEmaBalancedF1
        self._bestValidationMinClassF1 = self.bestValidationMinClassF1
        self._validationEmaBalancedF1 = self.validationEmaBalancedF1
        self._validationMinClassF1 = self.validationMinClassF1

    def getTrainingHistory(self) -> list[float]:
        return self.trainLoss

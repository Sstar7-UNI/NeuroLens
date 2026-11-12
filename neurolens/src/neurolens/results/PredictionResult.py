from neurolens.data_management.TumorClass import TumorClass

class PredictionResult:
    def __init__(
        self,
        predictedClass: TumorClass,
        classProbabilites: list[float],
        inferenceTimeMs: float
    ):
        self.predictedClass = predictedClass
        self.classProbabilites = list(classProbabilites)
        self.inferenceTimeMs = inferenceTimeMs

    def getPredictedLabel(self) -> str:
        return self.predictedClass.name

    def getConfidence(self) -> float:
        return self.classProbabilites[int(self.predictedClass)]
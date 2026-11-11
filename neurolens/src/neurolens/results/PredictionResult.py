from neurolens.data_management.TumorClass import TumorClass
from typing import List

class PredictionResult:
    def __init__(self,predictedClass:TumorClass,
                 classProbabilites:List[float],
                 inferenceTimeMs:float):
        def getPredictedLabel(self) -> str:
            return self.predictedClass.name
        def getConfidence(self) -> float:
            return self.classProbabilites[self.predictedClass.id]
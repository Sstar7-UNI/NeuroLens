import math

from neurolens.data_management.TumorClass import TumorClass

class Misclassification():

    def __init__(self, imagePath: str, trueLabel: TumorClass, predictedLabel: TumorClass, scores: list[float]) -> None:
        if trueLabel == predictedLabel:
            raise ValueError("True and predicted labels have to be different")
        self._imagePath = imagePath
        self._trueLabel = trueLabel
        self._predictedLabel = predictedLabel
        if not len(scores) == len(TumorClass):
            raise ValueError("There has to be a score for every tumor class")
        for score in scores:
            if not math.isfinite(score):
                raise ValueError("Scores must be finite")
        self._scores = list(scores)


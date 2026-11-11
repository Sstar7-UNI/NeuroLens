from neurolens.evaluation.ConfusionMatrix import ConfusionMatrix
from neurolens.evaluation.Misclassification import Misclassification

class EvaluationReport:
    def __init__(self, confusionMatrix: ConfusionMatrix, misclassifications: list[Misclassification]) -> None:
        self._confusionMatrix = confusionMatrix
        self._misclassifications = list(misclassifications)
        self._accuracy = confusionMatrix.accuracy()
        self._empiricalError = confusionMatrix.empiricalError()
        self._macroPrecision = confusionMatrix.macroPrecision()
        self._macroSensitivity = confusionMatrix.macroSensitivity()
        self._macroF1 = confusionMatrix.macroF1()
        self._classMetrics = list(confusionMatrix.classMetrics())

    def generateReport(self) -> str:
        lines = [
            "Evaluation report\n",
            f"Accuracy: {self._accuracy:.4f}",
            f"Empirical error: {self._empiricalError:.4f}",
            f"Macro precision: {self._macroPrecision:.4f}",
            f"Macro sensitivity: {self._macroSensitivity:.4f}",
            f"Macro F1: {self._macroF1:.4f}",
            f"\nMisclassified images: {len(self._misclassifications)}"
        ]
        return "\n".join(lines)
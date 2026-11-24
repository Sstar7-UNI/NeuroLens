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
        ]

        labels = [metric._label.name for metric in self._classMetrics]
        counts = self._confusionMatrix._counts

        lines.append("\nConfusion matrix (rows: actual, columns: predicted):")

        lines.append(
            f'{"Actual / Predicted":<20}' +
            "".join(f"{label:>12}" for label in labels)
        )

        for i, label in enumerate(labels):
            lines.append(
                f"{label:<20}" +
                "".join(f"{value:>12}" for value in counts[i])
            )

        lines.append("\nPer-class metrics:")

        lines.append(
            f'{"Class":<16}'
            f'{"Precision":>10}'
            f'{"Recall":>10}'
            f'{"Specificity":>14}'
            f'{"F1":>10}'
        )

        for metric in self._classMetrics:
            lines.append(
                f"{metric._label.name:<16}"
                f"{metric._precision:>10.4f}"
                f"{metric._sensitivity:>10.4f}"
                f"{metric._specificy:>14.4f}"
                f"{metric._f1:>10.4f}"
            )

        lines.append(
            f"\nMisclassified images: {len(self._misclassifications)}"
        )

        return "\n".join(lines)


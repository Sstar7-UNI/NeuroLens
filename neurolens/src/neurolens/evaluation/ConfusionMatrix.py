# evaluation/confusion_matrix.py
import tensorflow as tf

from neurolens.data_management import TumorClass
from neurolens.evaluation.ClassMetrics import ClassMetrics


class ConfusionMatrix:
    def __init__(self, counts, zeroDivisionValue: float = 0.0):
        # 4x4 list of lists, rows = true labels, columns = predictions
        self._counts = counts
        self._zeroDivisionValue = zeroDivisionValue

    @staticmethod
    def fromPredictions(trueLabels, predicted, zeroDivisionValue: float = 0.0):
        counts = tf.math.confusion_matrix(
            tf.cast(trueLabels, tf.int32),
            tf.cast(predicted, tf.int32),
            num_classes=len(TumorClass),
        )
        return ConfusionMatrix(counts.numpy().tolist(), zeroDivisionValue)

    def _div(self, num, den) -> float:
        if den == 0:
            return self._zeroDivisionValue
        return num / den

    def _rowSum(self, row: int) -> int:
        total = 0
        for col in range(len(TumorClass)):
            total += self._counts[row][col]
        return total

    def _columnSum(self, col: int) -> int:
        total = 0
        for row in range(len(TumorClass)):
            total += self._counts[row][col]
        return total

    def _trace(self) -> int:
        total = 0
        for i in range(len(TumorClass)):
            total += self._counts[i][i]
        return total

    def _total(self) -> int:
        total = 0
        for row in range(len(TumorClass)):
            total += self._rowSum(row)
        return total

    def truePositives(self, k: TumorClass) -> int:
        return self._counts[k.id][k.id]

    def falsePositives(self, k: TumorClass) -> int:
        return self._columnSum(k.id) - self.truePositives(k)

    def falseNegatives(self, k: TumorClass) -> int:
        return self._rowSum(k.id) - self.truePositives(k)

    def trueNegatives(self, k: TumorClass) -> int:
        return (self._total() - self.truePositives(k)
                - self.falsePositives(k) - self.falseNegatives(k))

    def accuracy(self) -> float:
        return self._div(self._trace(), self._total())

    def empiricalError(self) -> float:
        return 1.0 - self.accuracy()

    def macroPrecision(self) -> float:
        metrics = self.classMetrics()
        total = 0.0
        for m in metrics:
            total += m.precision
        return total / len(metrics)

    def macroSensitivity(self) -> float:
        metrics = self.classMetrics()
        total = 0.0
        for m in metrics:
            total += m.sensitivity
        return total / len(metrics)

    def macroF1(self) -> float:
        # mean of per-class F1, not F1 of macro precision/recall
        metrics = self.classMetrics()
        total = 0.0
        for m in metrics:
            total += m.f1
        return total / len(metrics)

    def classMetrics(self):
        result = []
        for k in TumorClass:
            tp = self.truePositives(k)
            fp = self.falsePositives(k)
            fn = self.falseNegatives(k)
            tn = self.trueNegatives(k)
            precision = self._div(tp, tp + fp)
            sensitivity = self._div(tp, tp + fn)
            specificity = self._div(tn, tn + fp)
            f1 = self._div(2 * precision * sensitivity, precision + sensitivity)
            result.append(ClassMetrics(label=k, precision=precision,
                                       sensitivity=sensitivity,
                                       specificity=specificity, f1=f1))
        return result

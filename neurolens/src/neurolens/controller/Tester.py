import tensorflow as tf

from neurolens.evaluation.ConfusionMatrix import ConfusionMatrix
from neurolens.results.EvaluationReport import EvaluationReport

BATCH_SIZE = 32
class Tester:
    def __init__(self, preprocessor, repository, rule, batchSize: int = BATCH_SIZE):
        self._preprocessor = preprocessor
        self._repository = repository
        self._rule = rule
        self._batchSize = batchSize
        self._samples = []
        self._probabilities = []
        self._predicted = []

    def evaluate(self, split) -> EvaluationReport:
        samples = list(split.getTestSet())
        config = self._repository.loadConfig()
        self._preprocessor.setNormalization(config.normalizationMean, config.normalizationStd)
        model = self._repository.loadModel(config)
        probabilities = []

        for start in range(0, len(samples), self._batchSize):
            chunk = samples[start:start + self._batchSize]
            images = [self._preprocessor.preprocess(sample.imagePath) for sample in chunk]
            batch = tf.stack(images)
            logits = model.predict(batch, verbose=0)
            batchProbabilities = tf.nn.softmax(logits, axis=-1).numpy()
            probabilities.extend([[float(value) for value in row] for row in batchProbabilities])

        predicted = [self._rule.decide(row) for row in probabilities]
        trueIds = [int(sample.label) for sample in samples]
        predictedIds = [int(prediction) for prediction in predicted]

        self._samples = samples
        self._probabilities = probabilities
        self._predicted = predicted

        matrix = ConfusionMatrix.fromPredictions(tf.constant(trueIds), tf.constant(predictedIds))
        report = EvaluationReport(matrix, matrix.classMetrics(), self.collectMisclassifications())
        self._repository.saveEvaluation(report)
        return report

    def collectMisclassifications(self):
        result = []
        for sample, predicted, probabilities in zip(
            self._samples, self._predicted, self._probabilities
        ):
            if int(predicted) != int(sample.label):
                result.append({
                    "imagePath": sample.imagePath,
                    "trueClass": sample.label,
                    "predictedClass": predicted,
                    "probabilities": list(probabilities),
                })
        return result

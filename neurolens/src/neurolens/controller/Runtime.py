import time
import tensorflow as tf

from neurolens.network import DecisionRule
from neurolens.results.PredictionResult import PredictionResult


class Runtime:
    def __init__(self, preprocessor, repository, rule: DecisionRule):
        self._preprocessor = preprocessor
        self._repository = repository
        self._rule = rule
        self._model = None

    def _getModel(self):
        if self._model is None:
            self._model = self._repository.loadModel()
            cfg = self._repository.loadConfig()
            #pre incalzire
            dummy = tf.zeros((1, cfg.inputHeight, cfg.inputWidth, 1))
            self._model.predict(dummy, verbose=0)
        return self._model

    def inference(self, imagePath: str) -> PredictionResult:
        model = self._getModel()

        start = time.perf_counter()
        image = self._preprocessor.preprocess(imagePath)
        batch = tf.expand_dims(image, axis=0)
        logits = model.predict(batch, verbose=0)
        probabilities = tf.nn.softmax(logits, axis=-1)[0]
        predictedClass = self._rule.decide(probabilities)
        elapsedMs = (time.perf_counter() - start) * 1000.0

        classProbabilities = []
        for value in probabilities.numpy():
            classProbabilities.append(float(value))

        return PredictionResult(
            predictedClass=predictedClass,
            classProbabilities=classProbabilities,
            inferenceTimeMs=elapsedMs,
        )
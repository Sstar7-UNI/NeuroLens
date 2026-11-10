import tensorflow as tf

from neurolens.evaluation.ConfusionMatrix import ConfusionMatrix
from neurolens.network import DecisionRule
from neurolens.results import EvaluationReport

BATCH_SIZE = 32
class Tester:
    def __init__(self,preprocessor,repository,rule:DecisionRule):
        self._preprocessor = preprocessor
        self._repository = repository
        self._rule = rule
        self._samples=[]
        self._probabilities=[] # lista de 4 floaturi pe sample
        self._predicted = []
    def evaluate(self,split) -> EvaluationReport:
        samples = split.getTestSet()
        if len(samples) == 0:
            raise ValueError("testsplit e gol")
        model = self._repository.loadModel()

        probabilities = []
        for start in range(0,len(samples),BATCH_SIZE):
            chunk = samples[start:start+BATCH_SIZE]
            images = []
            for sample in chunk:
                images.append(self._preprocessor.preprocess(sample.imagePath))
            batch = tf.stack(images)

            logits = model.predict(batch, verbose=0)
            batchProbabilities = tf.nn.softmax(logits, axis=-1).numpy()
            for row in batchProbabilities:
                values = []
                for value in row:
                    values.append(float(value))
                probabilities.append(values)
        predicted = []
        trueIds = []
        predictedIds  = []
        for i in range(len(samples)):
            predictedClass = self._rule.decide(probabilities[i])
            predicted.append(predictedClass)
            trueIds.append(samples[i].label.id)
            predictedIds.append(predictedClass.id)
        self._samples = samples
        self._probabilities = probabilities
        self._predicted = predicted

        matrix = ConfusionMatrix.fromPredictions(tf.constant(trueIds), tf.constant(predictedIds))
        report = EvaluationReport(matrix,matrix.classMetrics(), self.collectMisclassifications())
        self._repository.saveEvaluation(report)
        return report

    '''
    De terminat dupa ce legam modulele
    def collectMisclassifications(self):
        result = []
        for i in range(len(self._samples)):
            sample = self._samples[i]
            predictedClass = self._predicted[i]
            if(predictedClass != sample.label):
    '''
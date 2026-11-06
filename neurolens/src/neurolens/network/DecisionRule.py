import tensorflow as tf

from neurolens.TumorClass import TumorClass


class DecisionRule:
    def __init__(self,costMatrix=None):
        n = len(TumorClass)
        if costMatrix is None:
            costMatrix = 1.0 - tf.eye(n)
        costMatrix = tf.convert_to_tensor(costMatrix,dtype=tf.float32)
        if tuple(costMatrix.shape) != (n,n):
            raise ValueError("Matricea trebuie sa fie patratica")
        self.costMatrix = costMatrix
    def expectedCost(self, action:TumorClass, probabilities) -> float:
        p = tf.convert_to_tensor(probabilities,dtype=tf.float32)
        return float(tf.reduce_sum(self._costMatrix[action.id]*p))
    def decide(self,probabilities) -> TumorClass:
        costs = tf.constant([self.expectedCost(a,probabilities) for a in TumorClass])
        return TumorClass(int(tf.math.argmax(-costs,axis=0)))
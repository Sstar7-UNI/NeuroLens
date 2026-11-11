import tensorflow as tf

from neurolens.data_management.TumorClass import TumorClass


class DecisionRule:
    def __init__(self, costMatrix=None):
        n = len(TumorClass)
        if costMatrix is None:
            costMatrix = 1.0 - tf.eye(n, dtype=tf.float32)

        matrix = tf.convert_to_tensor(costMatrix, dtype=tf.float32)
        if tuple(matrix.shape) != (n, n):
            raise ValueError(f"cost matrix must have shape ({n}, {n})")
        if not bool(tf.reduce_all(tf.math.is_finite(matrix)).numpy()):
            raise ValueError("cost matrix must contain only finite values")

        self._costMatrix = matrix

    def expectedCost(self, action: TumorClass, probabilities) -> float:
        p = tf.convert_to_tensor(probabilities, dtype=tf.float32)
        if p.shape.rank != 1 or p.shape[0] != len(TumorClass):
            raise ValueError("probabilities must be a vector with one value per class")

        return float(tf.reduce_sum(self._costMatrix[int(action)] * p).numpy())

    def decide(self, probabilities) -> TumorClass:
        costs = tf.stack([
            self.expectedCost(action, probabilities)
            for action in TumorClass
        ])
        return TumorClass(int(tf.argmin(costs, axis=0).numpy()))

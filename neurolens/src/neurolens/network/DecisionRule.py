from __future__ import annotations

import numpy as np


class DecisionRule:
    """Converts model logits or calibrated probabilities into class predictions."""

    @staticmethod
    def to_probabilities(outputs: np.ndarray) -> np.ndarray:
        arr = np.asarray(outputs, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[None, :]
        row_sums = np.sum(arr, axis=-1, keepdims=True)
        is_prob = (
            np.all(arr >= -1e-5)
            and np.all(arr <= 1.0 + 1e-5)
            and np.allclose(row_sums, 1.0, atol=1e-3)
        )
        if is_prob:
            return np.clip(arr, 0.0, 1.0)
        shifted = arr - np.max(arr, axis=-1, keepdims=True)
        exp_vals = np.exp(shifted)
        return exp_vals / np.maximum(np.sum(exp_vals, axis=-1, keepdims=True), 1e-12)

    @staticmethod
    def predict(
        outputs: np.ndarray, class_multipliers: np.ndarray | None = None
    ) -> np.ndarray:
        probs = DecisionRule.to_probabilities(outputs)
        if class_multipliers is not None:
            w = np.asarray(class_multipliers, dtype=np.float32)
            probs = probs * w[None, :]
        return np.argmax(probs, axis=-1).astype(np.int32)
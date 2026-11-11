import json
from pathlib import Path

from neurolens.data_storage.ModelConfig import ModelConfig


class ModelRepository:
    def __init__(self, checkpoint: str, confPath: str):
        self._checkpoint = Path(checkpoint)
        self._confPath = Path(confPath)

    def getCheckpointPath(self) -> str:
        return str(self._checkpoint)

    def saveConfig(self, config: ModelConfig) -> None:
        if not config.validate():
            raise ValueError("Invalid model configuration")

        data = {
            "inputWidth": config.inputWidth,
            "inputHeight": config.inputHeight,
            "classNames": config.classNames,
            "filterCounts": config.filterCounts,
            "normalizationMean": config.normalizationMean,
            "normalizationStd": config.normalizationStd,
            "randomSeed": config.randomSeed,
            "epochsTrained": config.epochsTrained,
            "parameterCount": config.parameterCount,
        }

        self._confPath.parent.mkdir(parents=True, exist_ok=True)
        with self._confPath.open("w", encoding="utf-8") as file:
            json.dump(data, file, indent=4, ensure_ascii=False)

    def loadConfig(self) -> ModelConfig:
        with self._confPath.open("r", encoding="utf-8") as file:
            data = json.load(file)

        config = ModelConfig(**data)
        if not config.validate():
            raise ValueError("Invalid model configuration")
        return config

    def loadModel(self, config: ModelConfig = None):
        import tensorflow as tf
        from neurolens.network.CNNClassifier import CNNClassifier

        if config is None:
            config = self.loadConfig()

        if not self._checkpoint.is_file():
            raise FileNotFoundError(
                f"Model weights not found: {self._checkpoint}"
            )

        model = CNNClassifier(config)
        dummy = tf.zeros(
            (1, config.inputHeight, config.inputWidth, 1),
            dtype=tf.float32,
        )
        model(dummy, training=False)
        model.load_weights(str(self._checkpoint))
        return model

    def savePrediction(self, result) -> None:
        data = {
            "predictedClass": result.getPredictedLabel(),
            "confidence": result.getConfidence(),
            "inferenceTimeMs": result.inferenceTimeMs,
        }

        path = self._confPath.with_suffix(".predictions.jsonl")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(data, ensure_ascii=False) + "\n")

    def saveEvaluation(self, report) -> None:
        path = self._confPath.with_suffix(".evaluation.txt")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as file:
            file.write(report.generateReport())


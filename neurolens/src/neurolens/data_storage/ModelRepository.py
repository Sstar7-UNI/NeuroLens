import json
from pathlib import Path
import h5py
from neurolens.data_storage.ModelConfig import ModelConfig

class ModelRepository:
    def __init__(self,checkpoint:str,confPath:str):
        self._checkpoint = checkpoint
        self._confPath = confPath

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

        configPath = Path(self._confPath)
        configPath.parent.mkdir(parents=True, exist_ok=True)

        with configPath.open("w", encoding="utf-8") as file:
            json.dump(data, file, indent=4, ensure_ascii=False)

    def loadConfig(self) -> ModelConfig:
        with Path(self._confPath).open("r", encoding="utf-8") as file:
            data = json.load(file)

        config = ModelConfig(**data)

        if not config.validate():
            raise ValueError("Invalid model configuration")

        return config

    def loadModel(self, config: ModelConfig = None):
        import tensorflow as tf
        from neurolens.network.CNNClassifier import CNNClassifier
        from neurolens.network.LegacyCNNClassifier import LegacyCNNClassifier

        if config is None:
            config = self.loadConfig()

        checkpointPath = Path(self._checkpoint)

        if not checkpointPath.is_file():
            raise FileNotFoundError(
                f"Model weights not found: {checkpointPath}"
            )

        with h5py.File(checkpointPath, "r") as file:
            legacy = (
                "layers" in file
                and "conv_block" in file["layers"]
            )

        if legacy:
            model = LegacyCNNClassifier(config)
        else:
            model = CNNClassifier(config)

        dummy = tf.zeros(
            (1, config.inputHeight, config.inputWidth, 1),
            dtype=tf.float32,
        )

        model(dummy, training=False)
        model.load_weights(str(checkpointPath))

        return model

    def savePrediction(self, result) -> None:
        data = {
            "predictedClass": result.getPredictedLabel(),
            "confidence": result.getConfidence(),
            "inferenceTimeMs": result.inferenceTimeMs,
        }

        path = Path(self._confPath).with_suffix(".predictions.jsonl")
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(data, ensure_ascii=False) + "\n")

    def saveEvaluation(self, report) -> None:
        path = Path(self._confPath).with_suffix(".evaluation.txt")
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("w", encoding="utf-8") as file:
            file.write(report.generateReport())
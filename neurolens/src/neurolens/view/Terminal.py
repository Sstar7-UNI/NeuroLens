
import glob
import os
import sys

from neurolens.controller.AugmentationConfig import AugmentationConfig
from neurolens.controller.Learner import Learner
from neurolens.controller.Runtime import Runtime
from neurolens.controller.Tester import Tester
from neurolens.controller.TrainingConfig import TrainingConfig
from neurolens.data_management.DatasetManager import DatasetManager
from neurolens.data_management.ImageValidator import ImageValidator
from neurolens.data_management.TumorClass import TumorClass
from neurolens.data_storage.ModelConfig import ModelConfig
from neurolens.data_storage.ModelRepository import ModelRepository
from neurolens.network.DecisionRule import DecisionRule
from neurolens.preprocessing.ImagePreprocessor import ImagePreprocessor


USAGE = """usage:
  neurolens init-model    filters=N,N,... [model=NAME]
  neurolens learn         dataset=PATH [augment=off|N] [preprocessing=none|learn|a,b,c] [batch_size=N|all] [model=NAME]
  neurolens evaluate      dataset=PATH [model=NAME]
  neurolens runtime-test  [model=NAME] src=PATH"""


class Terminal:
    def __init__(self, dependencies=None):
        self._dependencies = {
            "DatasetManager": DatasetManager,
            "ModelConfig": ModelConfig,
            "ModelRepository": ModelRepository,
            "ImagePreprocessor": ImagePreprocessor,
            "Learner": Learner,
            "Tester": Tester,
            "Runtime": Runtime,
            "DecisionRule": DecisionRule,
            "ImageValidator": ImageValidator,
            "TrainingConfig": TrainingConfig,
            "AugmentationConfig": AugmentationConfig,
        }

        if dependencies:
            self._dependencies.update(dependencies)

    @staticmethod
    def _parse(tokens):
        options = {}

        for token in tokens:
            parts = token.split("=", 1)

            if len(parts) != 2 or not parts[0] or not parts[1]:
                raise ValueError(f"expected key=value, got '{token}'")

            key = parts[0].lower()

            if key in options:
                raise ValueError(f"option '{key}' appears more than once")

            options[key] = parts[1]

        return options

    @staticmethod
    def _modelPaths(name):
        if not isinstance(name, str) or not name.strip():
            raise ValueError("model name must not be empty")

        name = name.strip()

        if not os.path.isabs(name) and "/" not in name:
            name = os.path.join("models", name)

        if name.endswith(".weights.h5"):
            return (
                name,
                name[:-len(".weights.h5")] + ".json",
            )

        if name.endswith(".json"):
            base = name[:-len(".json")]
            return (
                base + ".weights.h5",
                name,
            )

        return (
            name + ".weights.h5",
            name + ".json",
        )

    def _get(self, name):
        return self._dependencies[name]

    def _repository(self, options):
        weightsPath, configPath = self._modelPaths(
            options.get("model", "best")
        )

        return self._get("ModelRepository")(weightsPath, configPath)

    def _preprocessor(self, repository):
        config = None
        loadConfig = getattr(repository, "loadConfig", None)

        if callable(loadConfig):
            config = loadConfig()

        width = getattr(config, "inputWidth", 224)
        height = getattr(config, "inputHeight", 224)

        return self._get("ImagePreprocessor")(width, height)

    def _datasetSplit(self, options, seed=42):
        datasetPath = options.get("dataset")

        if not datasetPath:
            raise ValueError("dataset=PATH is required")

        manager = self._get("DatasetManager")(datasetPath)
        manifest = manager.loadDataset()

        return manager.createSplits(manifest, seed)

    def run(self, arguments):
        if not arguments:
            print(USAGE, file=sys.stderr)
            return 2

        if arguments[0] in ("--help", "-h", "help"):
            print(USAGE)
            return 0

        commands = {
            "init-model": self._initModel,
            "learn": self._learn,
            "evaluate": self._evaluate,
            "runtime-test": self._runtimeTest,
        }

        command = arguments[0]

        if command not in commands:
            print(f"neurolens: unknown command '{command}'", file=sys.stderr)
            print(USAGE, file=sys.stderr)
            return 2

        try:
            options = self._parse(arguments[1:])
            commands[command](options)
            return 0
        except (ValueError, OSError, RuntimeError, TypeError) as error:
            print(f"neurolens: {error}", file=sys.stderr)
            return 2

    def _check(self, options, allowed):
        for key in options:
            if key not in allowed:
                raise ValueError(f"unknown option '{key}'")

    def _initModel(self, options):
        self._check(options, ["filters", "model", "width", "height"])

        filterText = options.get("filters")

        if not filterText:
            raise ValueError(
                "init-model requires an explicit filters=N,N,... configuration"
            )

        try:
            filters = [int(value) for value in filterText.split(",")]
        except ValueError:
            raise ValueError("filters must contain comma-separated integers")

        if not filters or any(value <= 0 for value in filters):
            raise ValueError("all filter counts must be positive integers")

        width = int(options.get("width", "224"))
        height = int(options.get("height", "224"))

        if width <= 0 or height <= 0:
            raise ValueError("model dimensions must be positive")

        config = self._get("ModelConfig")(
            inputWidth=width,
            inputHeight=height,
            classNames=[tumorClass.name for tumorClass in TumorClass],
            filterCounts=filters,
            normalizationMean=0.0,
            normalizationStd=1.0,
            randomSeed=42,
            epochsTrained=0,
            parameterCount=0,
        )

        repository = self._repository(options)
        repository.saveConfig(config)

        print("created model configuration")

    def _learn(self, options):
        self._check(
            options,
            ["dataset", "augment", "preprocessing", "batch_size", "model"],
        )

        preprocessing = options.get("preprocessing", "none")

        if preprocessing == "learn":
            raise ValueError("preprocessing=learn is not implemented")

        if preprocessing != "none":
            raise ValueError("custom preprocessing pipelines are not implemented")

        augmentationText = options.get("augment", "off")

        try:
            copies = 0 if augmentationText == "off" else int(augmentationText)
        except ValueError:
            raise ValueError("augment must be 'off' or a non-negative integer")

        if copies < 0:
            raise ValueError("augment must be 'off' or a non-negative integer")

        split = self._datasetSplit(options)
        batchText = options.get("batch_size", "32")

        if batchText == "all":
            batchSize = len(split.getTrainingSet())
        else:
            try:
                batchSize = int(batchText)
            except ValueError:
                raise ValueError("batch_size must be a positive integer or 'all'")

        if batchSize <= 0:
            raise ValueError("batch_size must be a positive integer or 'all'")

        augmentationConfig = self._get("AugmentationConfig")(
            imagesPerClass=100,
            copiesPerImage=max(copies, 1),
            seed=42,
        )

        trainingConfig = self._get("TrainingConfig")(
            epochs=40,
            batchSize=batchSize,
            learningRate=0.001,
            weightDecay=0.0001,
            dropoutRate=0.0,
            decoupledDecay=False,
            patience=5,
            useAugmentation=copies > 0,
            augmentation=augmentationConfig,
            seed=42,
        )

        repository = self._repository(options)
        modelConfig = repository.loadConfig()
        preprocessor = self._preprocessor(repository)

        learner = self._get("Learner")(
            trainingConfig,
            modelConfig,
            preprocessor,
            repository,
        )

        report = learner.train(split)

        print(
            f"best epoch {report.bestEpoch + 1} of "
            f"{report.epochsCompleted}, "
            f"val macro F1 {report.bestValidationMacroF1:.4f}"
        )

    def _evaluate(self, options):
        self._check(options, ["dataset", "model"])

        split = self._datasetSplit(options)
        repository = self._repository(options)
        preprocessor = self._preprocessor(repository)
        decisionRule = self._get("DecisionRule")()

        tester = self._get("Tester")(
            preprocessor,
            repository,
            decisionRule,
        )

        report = tester.evaluate(split)
        print(report.generateReport())

    def _runtimeTest(self, options):
        self._check(options, ["model", "src"])

        source = options.get("src")

        if not source:
            raise ValueError("runtime-test requires src=PATH")

        if os.path.isdir(source):
            extensions = (".jpg", ".jpeg", ".png", ".bmp")
            images = sorted(
                path
                for path in glob.glob(
                    os.path.join(source, "**", "*"),
                    recursive=True,
                )
                if os.path.isfile(path)
                and os.path.splitext(path)[1].lower() in extensions
            )
        elif os.path.isfile(source):
            if os.path.splitext(source)[1].lower() not in (
                ".jpg", ".jpeg", ".png", ".bmp"
            ):
                raise ValueError("unsupported image format")

            images = [source]
        else:
            raise ValueError(f"source path does not exist: {source}")

        if not images:
            raise ValueError("no supported images found in source directory")

        repository = self._repository(options)
        preprocessor = self._preprocessor(repository)
        decisionRule = self._get("DecisionRule")()
        validator = self._get("ImageValidator")()

        runtime = self._get("Runtime")(
            preprocessor,
            repository,
            decisionRule,
        )

        for path in images:
            if not validator.validate(path):
                print(f"{path}: not a valid image")
                continue

            result = runtime.inference(path)

            if hasattr(result, "getPredictedLabel"):
                label = result.getPredictedLabel()
            else:
                label = getattr(result, "predictedClass", str(result))

            if hasattr(result, "getConfidence"):
                confidence = result.getConfidence()
            elif hasattr(result, "getProbability"):
                confidence = result.getProbability()
            else:
                confidence = None

            elapsed = getattr(result, "inferenceTimeMs", None)

            output = f"{path}: {label}"

            if confidence is not None:
                output += f"  {confidence:.1%}"

            if elapsed is not None:
                output += f"  {elapsed:.0f} ms"

            print(output)


def main():
    return Terminal().run(sys.argv[1:])


if __name__ == "__main__":
    sys.exit(main())

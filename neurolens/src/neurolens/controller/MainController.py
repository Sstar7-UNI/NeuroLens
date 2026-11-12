from pathlib import Path
from typing import TypedDict
from neurolens.data_management.DatasetManager import DatasetManager
from neurolens.data_management.DatasetManifest import DatasetManifest
from neurolens.data_management.ImageRecord import ImageRecord
from neurolens.data_management.SplitType import SplitType
from neurolens.data_management.TumorClass import TumorClass
from neurolens.controller.AugmentationConfig import AugmentationConfig
from neurolens.controller.PreprocessingPreviewer import PreprocessingPreviewer
from neurolens.preprocessing.ImagePreprocessor import ImagePreprocessor
from neurolens.data_management.ImageValidator import ImageValidator
from neurolens.controller.TrainingConfig import TrainingConfig
from neurolens.controller.Learner import Learner
from neurolens.data_storage.ModelRepository import ModelRepository
from neurolens.results.TrainingReport import TrainingReport
from neurolens.preprocessing.ContrastStretcher import ContrastStretcher
from neurolens.preprocessing.GaussianFilter import GaussianFilter
from neurolens.preprocessing.MedianFilter import MedianFilter
from neurolens.preprocessing.HistogramEqualizer import HistogramEqualizer
from neurolens.preprocessing.OtsuSegmenter import OtsuSegmenter
from neurolens.preprocessing.MorphologyOperator import MorphologyOperator
from neurolens.preprocessing.PaddingMode import PaddingMode
from neurolens.preprocessing.Connectivity import Connectivity
from neurolens.controller.Tester import Tester
from neurolens.controller.Runtime import Runtime
from neurolens.network.DecisionRule import DecisionRule



class PreviewImageData(TypedDict):
    imagePath: str
    filename: str
    tumorClass: TumorClass
    split: SplitType
    fileExtension: str | None
    isAugmented: bool | None
    originalImage: str | None
    transformations: list[str] | None

class MainController:
    def __init__(self) -> None:
        self._datasetManager: DatasetManager | None = None
        self._datasetManifest: DatasetManifest | None = None
        self._selectedSplit: SplitType = SplitType.TRAINING
        self._images: list[ImageRecord] = []
        self._currentIndex: int = 0
        self._previewer = None
        self._selectedRepository = None
        self._preprocessingSteps = []

    def loadDataset(self, datasetPath: str) -> None:
        if not isinstance(datasetPath, str) or not datasetPath.strip():
            raise ValueError("Dataset path must be a non-empty string.")
        manager = DatasetManager(datasetPath)
        manifest = manager.loadDataset()
        images = [
            record
            for record in manifest.getSamples()
            if record._split == SplitType.TRAINING
        ]

        self._datasetManager = manager
        self._datasetManifest = manifest
        self._selectedSplit = SplitType.TRAINING
        self._images = images
        self._currentIndex = 0
        self._previewer = None

    def selectSplit(self, split: SplitType) -> None:
        if not isinstance(split, SplitType) or split not in (SplitType.TRAINING,SplitType.TEST,):
            raise ValueError("Preview supports only TRAINING and TEST splits.")
        self._selectedSplit = split
        self._images = (
            [
                record
                for record in self._datasetManifest.getSamples()
                if record._split == split
            ]
            if self._datasetManifest is not None
            else []
        )

        self._currentIndex = 0

    def nextImage(self) -> bool:
        if self._currentIndex + 1 >= len(self._images):
            return False
        self._currentIndex += 1
        return True

    def previousImage(self) -> bool:
        if self._currentIndex <= 0 or not self._images:
            return False
        self._currentIndex -= 1
        return True

    def getCurrentImage(self) -> PreviewImageData | None:
        if not self._images:
            return None
        record = self._images[self._currentIndex]
        path = Path(record._imagePath)
        if not path.is_file():
            raise FileNotFoundError(f"Preview image not found: {path}")
        return {
            "imagePath": str(path),
            "filename": path.name,
            "tumorClass": record._label,
            "split": record._split,
            "fileExtension": path.suffix.lstrip(".").upper() or None,
            "isAugmented": None,
            "originalImage": None,
            "transformations": None,
        }

    def getSelectedSplit(self) -> SplitType:
        return self._selectedSplit

    def getImageCount(self) -> int:
        return len(self._images)

    def getCurrentIndex(self) -> int:
        return self._currentIndex

    def canGoPrevious(self) -> bool:
        return self._currentIndex > 0 and bool(self._images)

    def canGoNext(self) -> bool:
        return self._currentIndex + 1 < len(self._images)

    def previewPreprocessing(self, config: AugmentationConfig):
        if self._datasetManager is None or self._datasetManifest is None:
            raise ValueError("Dataset must be loaded before augmentation preview.")
        if not config.validate():
            raise ValueError("Invalid augmentation configuration.")
        split = self._datasetManager.createSplits(self._datasetManifest,config.seed)
        preprocesor = ImagePreprocessor(operators=self._buildPreprocessingOperators(self._preprocessingSteps))
        preprocesor.fitNormalization(split.getTrainingSet())
        previewer = PreprocessingPreviewer(preprocesor)
        preview = previewer.start(split, config)
        self._previewer = previewer
        return preview

    def nextPreviewBatch(self):
        if self._previewer is None:
            raise ValueError("Previewer must be initialized before augmentation preview.")
        return self._previewer.next()

    def validateInput(self, path: str) -> bool:
        validator = ImageValidator()
        return validator.validate(path)

    def selectModel(self, weightsPath: str, configPath: str) -> None:
        repository = ModelRepository(weightsPath, configPath)
        repository.loadConfig()
        self._selectedRepository = repository

    def train(self, config: TrainingConfig) -> TrainingReport:
        if self._datasetManager is None or self._datasetManifest is None:
            raise ValueError("Load a dataset before training.")

        if self._selectedRepository is None:
            raise ValueError("Select a model before training.")

        split = self._datasetManager.createSplits(
            self._datasetManifest, config.seed
        )

        trainingSamples = split.getTrainingSet()

        if not trainingSamples:
            raise ValueError("Training set is empty.")

        if not split.getValidationSet():
            raise ValueError("Validation set is empty.")

        if config.batchSize == 0:
            config.batchSize = len(trainingSamples)

        if config.batchSize <= 0:
            raise ValueError("Batch size must be positive.")

        modelConfig = self._selectedRepository.loadConfig()

        preprocessor = ImagePreprocessor(
            modelConfig.inputWidth,
            modelConfig.inputHeight,
            operators=self._buildPreprocessingOperators(
                self._preprocessingSteps
            )
        )

        learner = Learner(
            config,
            modelConfig,
            preprocessor,
            self._selectedRepository
        )

        return learner.train(split)

    def selectPreprocessing(self, steps: list[dict]) -> None:
        if not isinstance(steps, list):
            raise ValueError("Preprocessing steps must be a list.")

        selected = [dict(step) for step in steps]

        self._buildPreprocessingOperators(selected)
        self._preprocessingSteps = selected

    def _buildPreprocessingOperators(self, steps: list[dict]):
        operators = []
        binary_mask = False

        for step in steps:
            name = step["type"]

            if name == "ContrastStretcher":
                operator = ContrastStretcher(
                    step["lower"], step["upper"]
                )

            elif name == "GaussianFilter":
                operator = GaussianFilter(
                    step["sigma"],
                    step["kernelSize"],
                    PaddingMode[step["padding"]]
                )

            elif name == "MedianFilter":
                operator = MedianFilter(
                    step["kernelSize"],
                    PaddingMode[step["padding"]]
                )

            elif name == "HistogramEqualizer":
                operator = HistogramEqualizer(step["levels"])

            elif name == "OtsuSegmenter":
                operator = OtsuSegmenter()

            elif name == "MorphologyOperator":
                if not binary_mask:
                    raise ValueError(
                        "MorphologyOperator requires a binary mask "
                        "from OtsuSegmenter immediately before it "
                        "or before other morphology operations."
                    )
                operator = MorphologyOperator(
                    Connectivity[step["connectivity"]]
                )

            else:
                raise ValueError(f"Unknown preprocessing operator: {name}")

            operators.append(operator)

            if name == "OtsuSegmenter":
                binary_mask = True
            elif name != "MorphologyOperator":
                binary_mask = False

        return operators

    def evaluate(self):
        if self._datasetManager is None or self._datasetManifest is None:
            raise ValueError("Load a dataset before evaluation.")

        if self._selectedRepository is None:
            raise ValueError("Select a model before evaluation.")

        split = self._datasetManager.createSplits(
            self._datasetManifest, 42
        )

        if not split.getTestSet():
            raise ValueError("Test set is empty.")

        modelConfig = self._selectedRepository.loadConfig()

        preprocessor = ImagePreprocessor(
            modelConfig.inputWidth,
            modelConfig.inputHeight,
            operators=self._buildPreprocessingOperators(
                self._preprocessingSteps
            )
        )

        tester = Tester(
            preprocessor,
            self._selectedRepository,
            DecisionRule()
        )

        return tester.evaluate(split)

    def classify(self, imagePath: str):
        if self._selectedRepository is None:
            raise ValueError("Select a model before classification.")

        if not self.validateInput(imagePath):
            raise ValueError("Invalid image.")

        modelConfig = self._selectedRepository.loadConfig()

        preprocessor = ImagePreprocessor(
            modelConfig.inputWidth,
            modelConfig.inputHeight,
            operators=self._buildPreprocessingOperators(
                self._preprocessingSteps
            )
        )

        runtime = Runtime(
            preprocessor,
            self._selectedRepository,
            DecisionRule()
        )

        return runtime.inference(imagePath)



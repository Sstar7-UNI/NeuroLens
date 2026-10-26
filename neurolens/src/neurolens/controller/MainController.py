from pathlib import Path
from typing import TypedDict
from neurolens.data_management.DatasetManager import DatasetManager
from neurolens.data_management.DatasetManifest import DatasetManifest
from neurolens.data_management.ImageRecord import ImageRecord
from neurolens.data_management.SplitType import SplitType
from neurolens.data_management.TumorClass import TumorClass

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

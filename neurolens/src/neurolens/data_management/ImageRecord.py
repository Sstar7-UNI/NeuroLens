from neurolens.src.neurolens.SplitType import SplitType
from neurolens.src.neurolens.TumorClass import TumorClass
import os


class ImageRecord:
    def __init__(self, imagePath: str, label: TumorClass, split: SplitType):
        self._imagePath: str = imagePath
        self._label: TumorClass = label
        self._split: SplitType = split

    @property
    def imagePath(self) -> str:
        return self._imagePath

    @property
    def label(self) -> TumorClass:
        return self._label

    @property
    def split(self) -> SplitType:
        return self._split

    def validate(self) -> bool:
        return (
            isinstance(self._imagePath, str) and
            os.path.isfile(self._imagePath) and
            isinstance(self._label, TumorClass) and
            isinstance(self._split, SplitType)
        )
from neurolens.src.neurolens.SplitType import SplitType
from neurolens.src.neurolens.TumorClass import TumorClass
import os

class ImageRecord:
    _imagePath:String
    _label:TumorClass
    _split:SplitType
    def validate(self) -> bool:
        return (
            isinstance(self._imagePath, str) and
            os.path.isfile(self._imagePath) and
            isinstance(self._label, TumorClass) and
            isinstance(self._split, SplitType)
        )
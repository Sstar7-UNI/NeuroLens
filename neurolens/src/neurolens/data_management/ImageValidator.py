
from pathlib import Path

import tensorflow as tf

from .TumorClass import TumorClass


class ImageValidator:

    def __init__(self) -> None:
        self._supportedFormats: list[str] = [".jpg", ".jpeg", ".png", ".bmp"]
        self._expectedChannels: int = 3
    def validate(self, path: str) -> bool:
        if not isinstance(path, str) or not path.strip():
            return False
        image_path = Path(path)
        extension = image_path.suffix.lower()
        if extension not in self._supportedFormats:
            return False
        if not image_path.is_file():
            return False
        try:
            contents = tf.io.read_file(str(image_path))
            signatures = (
                b"\xff\xd8\xff",
                b"\x89PNG\r\n\x1a\n",
                b"BM"
            )

            if not contents.numpy().startswith(signatures):
                return False
            image = tf.io.decode_image(
                contents,
                channels=self._expectedChannels,
                expand_animations=False
            )
            shape = tf.shape(image).numpy()
            return (
                len(shape) == 3
                and shape[0] > 0
                and shape[1] > 0
                and shape[2] == self._expectedChannels
            )
        except (tf.errors.OpError, ValueError, TypeError, OSError):
            return False

    def hasValidLabel(self, path: str) -> bool:
        if not isinstance(path, str) or not path.strip():
            return False
        directory_label = Path(path).parent.name

        try:
            TumorClass.fromName(directory_label)
        except ValueError:
            return False
        return True

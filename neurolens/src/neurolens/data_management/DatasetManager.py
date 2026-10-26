import hashlib
import shutil
import stat
import tempfile
import zipfile

from pathlib import Path, PurePosixPath
from .DatasetManifest import DatasetManifest
from .ImageRecord import ImageRecord
from .ImageValidator import ImageValidator
from .SplitType import SplitType
from .TumorClass import TumorClass
from .DatasetSplit import DatasetSplit


class DatasetManager:

    def __init__(self, datasetPath: str) -> None:
        self._datasetPath: str = datasetPath

    def loadDataset(self) -> DatasetManifest:
        source = Path(self._datasetPath).expanduser().resolve()
        if source.is_file() and source.suffix.lower() == ".zip":
            root = self._extractZip(source)
        elif source.is_dir():
            root = source
        else:
            raise FileNotFoundError(f"Dataset directory or ZIP not found: {source}")
        validator = ImageValidator()
        records: list[ImageRecord] = []
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            if path.suffix.lower() not in validator._supportedFormats:
                continue
            if not validator.hasValidLabel(str(path)):
                raise ValueError(f"Unknown tumor class for image: {path}")
            split = self._detectSplit(path, root)
            if split is None:
                raise ValueError(f"Missing dataset split for: {path}")
            if not validator.validate(str(path)):
                raise ValueError(f"Invalid image: {path}")
            records.append(
                ImageRecord(
                    imagePath=str(path),
                    label=TumorClass.fromName(path.parent.name),
                    split=split
                )
            )
        if not records:
            raise ValueError("No supported images found in dataset.")
        return DatasetManifest(records)

    def validateDataset(self) -> bool:
        try:
            self.loadDataset()
            return True
        except (OSError,ValueError,RuntimeError,zipfile.BadZipFile):
            return False

    def createSplits(self,manifest: DatasetManifest,seed: int) -> "DatasetSplit":

        if not isinstance(seed, int):
            raise TypeError("seed must be an integer")
        records = manifest.getSamples()
        training = [
            r for r in records
            if r._split == SplitType.TRAINING
        ]

        validation = [
            r for r in records
            if r._split == SplitType.VALIDATION
        ]

        testing = [
            r for r in records
            if r._split == SplitType.TEST
        ]

        if len(training) + len(validation) + len(testing) != len(records):
            raise ValueError("Manifest contains an invalid split")
        return DatasetSplit(training, validation, testing)

    def _detectSplit(self,imagePath: Path,root: Path) -> SplitType | None:

        aliases = {
            "training": SplitType.TRAINING,
            "train": SplitType.TRAINING,
            "validation": SplitType.VALIDATION,
            "val": SplitType.VALIDATION,
            "testing": SplitType.TEST,
            "test": SplitType.TEST
        }
        ancestors = imagePath.relative_to(root).parts[:-2]
        matches = {
            aliases[part.lower()]
            for part in ancestors
            if part.lower() in aliases
        }

        if len(matches) > 1:
            raise ValueError( f"Conflicting splits for: {imagePath}")
        return next(iter(matches), None)

    def _extractZip(self, archivePath: Path) -> Path:
        digest = hashlib.sha256()
        with archivePath.open("rb") as file:
            for block in iter(lambda: file.read(1024 * 1024),b""):
                digest.update(block)

        destination = archivePath.with_name(
            f"{archivePath.stem}_extracted_{digest.hexdigest()[:12]}"
        )
        supported = set(ImageValidator()._supportedFormats)
        with zipfile.ZipFile(archivePath) as archive:
            entries: list[tuple[zipfile.ZipInfo, PurePosixPath]] = []
            names: set[str] = set()
            total_size = 0
            for info in archive.infolist():
                raw = info.filename.replace("\\", "/")
                if (raw.startswith("/")or ".." in raw.split("/")or ":" in raw.split("/")[0]):
                    raise ValueError(f"Unsafe archive path: {info.filename}")
                if stat.S_ISLNK(info.external_attr >> 16):
                    raise ValueError(f"Symbolic link not allowed: {info.filename}")
                if info.is_dir():
                    continue
                relative = PurePosixPath(raw)
                if relative.suffix.lower() not in supported:
                    continue
                key = str(relative).casefold()
                if key in names:
                    raise ValueError(f"Duplicate image entry: {info.filename}")
                names.add(key)
                if info.file_size > 256 * 1024 * 1024:
                    raise ValueError(f"Image too large: {info.filename}")
                total_size += info.file_size
                if total_size > 4 * 1024 * 1024 * 1024:
                    raise ValueError("ZIP exceeds the allowed image size limit")
                if info.file_size > max(info.compress_size, 1) * 2000:
                    raise ValueError(f"Suspicious compression ratio: {info.filename}")
                entries.append((info, relative))
            if not entries:
                raise ValueError("No supported images found in ZIP")
            if destination.exists():
                if not destination.is_dir() or not all(
                    (destination.joinpath(*rel.parts)).is_file()
                    and (destination.joinpath(*rel.parts)).stat().st_size== info.file_size
                    for info, rel in entries):
                    raise ValueError(f"Existing extraction directory is incomplete: {destination}")
                return destination
            temporary = Path(tempfile.mkdtemp(prefix="neurolens_extract_", dir=archivePath.parent))
            try:
                for info, relative in entries:
                    target = temporary.joinpath(*relative.parts)
                    target.parent.mkdir(parents=True, exist_ok=True)

                    with (archive.open(info) as src,target.open("xb") as dst):
                        shutil.copyfileobj(src, dst)
                temporary.rename(destination)
                return destination
            finally:
                if temporary.exists():
                    shutil.rmtree(temporary)

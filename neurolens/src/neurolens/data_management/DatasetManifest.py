import json
from pathlib import Path
from .ImageRecord import ImageRecord
from .SplitType import SplitType
from .TumorClass import TumorClass


class DatasetManifest:
    def __init__(self,records: list[ImageRecord] | None = None) -> None:
        self._records: list[ImageRecord] = (list(records) if records is not None else [])
    def getSamples(self) -> list[ImageRecord]:
        return self._records.copy()

    def getClassDistribution(self) -> dict[TumorClass, int]:
        distribution = { tumor_class: 0 for tumor_class in TumorClass }
        for record in self._records:
            distribution[record._label] += 1
        return distribution

    def save(self, path: str) -> None:
        serialized_records = []
        for record in self._records:
            if not isinstance(record, ImageRecord) or not record.validate():
                raise ValueError("Cannot save an invalid ImageRecord.")

            serialized_records.append({
                "imagePath": record._imagePath,
                "label": record._label.name,
                "split": record._split.value,
            })

        with Path(path).open("w", encoding="utf-8") as file:
            json.dump({"records": serialized_records}, file, indent=4)

    def load(self, path: str) -> None:
        with Path(path).open("r", encoding="utf-8") as file:
            content = json.load(file)
        if not isinstance(content, dict) or not isinstance( content.get("records"), list):
            raise ValueError("Manifest must contain a 'records' list.")
        loaded_records = []
        for index, item in enumerate(content["records"]):
            try:
                if not isinstance(item, dict):
                    raise ValueError("Record must be a JSON object.")
                record = ImageRecord(
                    imagePath=item["imagePath"],
                    label=TumorClass.fromName(item["label"]),
                    split=SplitType(item["split"]),
                )
                if not record.validate():
                    raise ValueError("Invalid ImageRecord fields.")
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Invalid record at index {index}: {exc}") from exc
            loaded_records.append(record)
        self._records = loaded_records

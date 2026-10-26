from enum import IntEnum

class TumorClass(IntEnum):
    GLIOMA = 0
    MENINGIOMA = 1
    NO_TUMOR = 2
    PITUITARY = 3
    @staticmethod
    def fromName(name: str) -> "TumorClass":
        key = name.strip().lower().replace("_", " ").replace(" ", "").replace("-", "")
        aliases = {
            "glioma": TumorClass.GLIOMA,
            "meningioma": TumorClass.MENINGIOMA,
            "notumor": TumorClass.NO_TUMOR,
            "pituitary": TumorClass.PITUITARY,
        }
        if key not in aliases:
            raise ValueError(f"Unknown tumor class: {name}")
        return aliases[key]
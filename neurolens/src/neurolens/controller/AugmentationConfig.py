class AugmentationConfig:
    def __init__(self, imagesPerClass: int = 8, copiesPerImage: int = 2, seed: int = 42):
        self.imagesPerClass: int = imagesPerClass
        self.copiesPerImage: int = copiesPerImage
        self.seed: int = seed

    def validate(self) -> bool:
        return (
            isinstance(self.imagesPerClass, int) and self.imagesPerClass > 0 and
            isinstance(self.copiesPerImage, int) and self.copiesPerImage >= 0 and
            isinstance(self.seed, int)
        )
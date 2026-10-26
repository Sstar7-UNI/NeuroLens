class AugmentationConfig:
    def __init__(self, imagesPerClass: int = 8, copiesPerImage: int = 2, seed: int = 42):
        self.imagesPerClass: int = imagesPerClass
        self.copiesPerImage: int = copiesPerImage
        self.seed: int = se
    def validate(self):
        return (
            isinstance(self.imagesPerClass,Integer) and
            isinstance(self.copiesPerImage,Integer) and
            isinstance(self.seed,Integer)
        )
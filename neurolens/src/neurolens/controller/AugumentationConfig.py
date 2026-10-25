class AugumentationConfig:
    imagesPerClass:Integer
    copiesPerImage:Integer
    seed:Integer
    def validate(self):
        return (
            isinstance(self.imagesPerClass,Integer) and
            isinstance(self.copiesPerImage,Integer) and
            isinstance(self.seed,Integer)
        )
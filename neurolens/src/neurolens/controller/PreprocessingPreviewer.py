from neurolens.src.neurolens.controller.AugumentationConfig import AugumentationConfig


class PreprocessingPreviewer:
    def __init__(self):
        self._preprocessor: ImagePreprocessor
        self._generator:AugmentedBatchGenerator
        self.current: PreprocessingPreview
    def start(self, split:DatasetSplit, config:AugumentationConfig):
        pass
    def next(self) -> PreprocessingPreview:
        pass
    def __toPreview(self):
        pass
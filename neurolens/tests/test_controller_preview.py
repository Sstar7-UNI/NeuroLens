from neurolens.controller.MainController import MainController
from neurolens.controller.AugmentationConfig import AugmentationConfig


def testControllerPreview():
    controller = MainController()
    dataset_path = r"G:\Downloads\archive"
    controller.loadDataset(dataset_path)
    config = AugmentationConfig(imagesPerClass=2,copiesPerImage=1,seed=42)
    preview = controller.previewPreprocessing(config)
    if preview is None:
        print("No preview generated.")
        return
    print("First batch:")
    print("Original images:", preview.totalCounts())
    print("Augmented images:", preview.augumentedCounts())
    next_preview = controller.nextPreviewBatch()
    if next_preview is None:
        print("No more batches available.")
    else:
        print("\nSecond batch:")
        print("Original images:", next_preview.totalCounts())
        print("Augmented images:", next_preview.augumentedCounts())
if __name__ == "__main__":
    testControllerPreview()
import sys
from pathlib import Path
import tensorflow as tf
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QLabel, QHBoxLayout,
                               QGroupBox, QSizePolicy, QPushButton, QFileDialog, QComboBox, QSpinBox, QMessageBox)
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from controller.MainController import MainController
from neurolens.data_management.SplitType import SplitType
from neurolens.view.ImageAdapter import ImageAdapter
from neurolens.controller.AugmentationConfig import AugmentationConfig
from neurolens.data_management.TumorClass import TumorClass


class ImageLabel(QLabel):

    def __init__(self):
        super().__init__()
        self.original_pixmap = QPixmap()

    def setImage(self, pixmap):
        self.original_pixmap = pixmap
        self.updateImage()

    def updateImage(self):
        if self.original_pixmap.isNull():
            return
        size = self.contentsRect().size()
        scaled_pixmap = self.original_pixmap.scaled(size, Qt.AspectRatioMode.KeepAspectRatio,Qt.TransformationMode.SmoothTransformation)
        self.setPixmap(scaled_pixmap)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.updateImage()

    def clearImage(self, message="No image available"):
        self.original_pixmap = QPixmap()
        self.clear()
        self.setText(message)


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()

        self.image_adapter = ImageAdapter()
        self.controller = MainController()

        self._preview_samples = []
        self._preview_sample_index = 0
        self._current_preview_sample = None
        self._preview_image_index = 0

        self.setWindowTitle("NeuroLens")
        self.resize(900, 600)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        left_panel = QGroupBox("Image Preview")
        right_panel = QGroupBox("Control Panel")

        main_layout.addWidget(left_panel, 3)
        main_layout.addWidget(right_panel, 2)

        left_layout = QVBoxLayout(left_panel)
        right_layout = QVBoxLayout(right_panel)

        self.setupImagePreview(left_layout)
        self.setupControlsPanel(right_layout)
        self.updateNavigationControls(0, 0)

    def setupImagePreview(self, layout):
        self.image_label = ImageLabel()
        self.image_label.setMinimumSize(160, 120)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setStyleSheet("background-color: #202020; color: white;")
        self.image_label.setText("No image available")
        self.image_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        layout.addWidget(self.image_label, 1)  # aici 1 e factorul stretch

        navigation_layout = QHBoxLayout()
        layout.addLayout(navigation_layout)

        self.previous_button = QPushButton("Previous")
        self.next_button = QPushButton("Next")
        self.position_label = QLabel()
        self.position_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.previous_button.clicked.connect(self.onPreviousClicked)
        self.next_button.clicked.connect(self.onNextClicked)

        navigation_layout.addWidget(self.previous_button)
        navigation_layout.addWidget(self.position_label)
        navigation_layout.addWidget(self.next_button)

        self.metadata_label = QLabel("No Metadata Available")
        self.metadata_label.setWordWrap(True)
        layout.addWidget(self.metadata_label)

        preprocessing_preview_widget = QGroupBox("Preprocessing Preview")
        preprocessing_preview_layout = QVBoxLayout(preprocessing_preview_widget)

        self.preprocessing_image_label = ImageLabel()
        self.preprocessing_image_label.setText("No preprocessing preview available")
        self.preprocessing_image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preprocessing_image_label.setMinimumSize(160, 120)
        self.preprocessing_image_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        preprocessing_preview_layout.addWidget(self.preprocessing_image_label)

        preview_navigation_layout = QHBoxLayout()

        self.preview_previous_button = QPushButton("Previous")
        self.preview_next_button = QPushButton("Next")
        self.preview_position_label = QLabel("0 / 0")
        self.preview_position_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.preview_previous_button.setEnabled(False)
        self.preview_next_button.setEnabled(False)
        self.preview_previous_button.clicked.connect(self.onPreviewPreviousClicked)
        self.preview_next_button.clicked.connect(self.onPreviewNextClicked)

        preview_navigation_layout.addWidget(self.preview_previous_button)
        preview_navigation_layout.addWidget(self.preview_position_label)
        preview_navigation_layout.addWidget(self.preview_next_button)
        preprocessing_preview_layout.addLayout(preview_navigation_layout)

        sample_navigation_layout = QHBoxLayout()

        self.sample_previous_button = QPushButton("Previous Sample")
        self.sample_next_button = QPushButton("Next Sample")
        self.sample_position_label = QLabel("0 / 0")
        self.sample_position_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.sample_previous_button.setEnabled(False)
        self.sample_next_button.setEnabled(False)
        self.sample_previous_button.clicked.connect(self.onPreviewSamplePreviousClicked)
        self.sample_next_button.clicked.connect(self.onPreviewSampleNextClicked)

        sample_navigation_layout.addWidget(self.sample_previous_button)
        sample_navigation_layout.addWidget(self.sample_position_label)
        sample_navigation_layout.addWidget(self.sample_next_button)
        preprocessing_preview_layout.addLayout(sample_navigation_layout)

        self.preprocessing_metadata_label = QLabel("No metadata available")
        self.preprocessing_metadata_label.setWordWrap(True)
        preprocessing_preview_layout.addWidget(self.preprocessing_metadata_label)

        layout.addWidget(preprocessing_preview_widget, 1)

    def setupControlsPanel(self, layout):
        self.load_button = QPushButton("Load Dataset")
        self.load_button.clicked.connect(self.onLoadDataset)
        layout.addWidget(self.load_button)

        split_widget = QWidget()
        split_widget.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        split_layout = QVBoxLayout(split_widget)

        split_label = QLabel("Dataset Split")
        self.choose_split = QComboBox()
        self.choose_split.setEnabled(False)
        self.choose_split.addItems(["Training", "Testing"])
        self.choose_split.currentIndexChanged.connect(self.onSplitChanged)

        split_layout.addWidget(split_label)
        split_layout.addWidget(self.choose_split)
        layout.addWidget(split_widget)

        augmentation_widget = QGroupBox("Augmentation Settings")
        augmentation_widget.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        augmentation_layout = QVBoxLayout(augmentation_widget)

        images_per_class_label = QLabel("Images per Class")
        augmentation_layout.addWidget(images_per_class_label)
        self.images_per_class_spinbox = QSpinBox()
        self.images_per_class_spinbox.setRange(1, 100)
        self.images_per_class_spinbox.setValue(8)
        augmentation_layout.addWidget(self.images_per_class_spinbox)

        copies_per_image_label = QLabel("Copies per image")
        augmentation_layout.addWidget(copies_per_image_label)
        self.copies_per_image_spinbox = QSpinBox()
        self.copies_per_image_spinbox.setRange(0, 100)
        self.copies_per_image_spinbox.setValue(2)
        augmentation_layout.addWidget(self.copies_per_image_spinbox)

        seed_label = QLabel("Seed")
        augmentation_layout.addWidget(seed_label)
        self.seed_spinbox = QSpinBox()
        self.seed_spinbox.setRange(0, 99999)
        self.seed_spinbox.setValue(42)
        augmentation_layout.addWidget(self.seed_spinbox)

        self.preview_button = QPushButton("Generate Preview")
        self.preview_button.setEnabled(False)
        self.preview_button.clicked.connect(self.previewPreprocessing)
        augmentation_layout.addWidget(self.preview_button)

        self.next_preview_batch_button = QPushButton("Next Preview Batch")
        self.next_preview_batch_button.setEnabled(False)
        self.next_preview_batch_button.clicked.connect(self.showNextPreviewBatch)
        augmentation_layout.addWidget(self.next_preview_batch_button)

        layout.addWidget(augmentation_widget)
        layout.addStretch()

    def onLoadDataset(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Dataset")
        if not folder:
            return
        try:
            self.controller.loadDataset(folder)
            self._clearPreprocessingPreview()
            self.choose_split.setEnabled(True)
            self.preview_button.setEnabled(True)
            self.onSplitChanged(self.choose_split.currentIndex())
        except Exception as error:
            self.displayError(f"Failed to load dataset:\n{error}")

    def onSplitChanged(self, index):
        if index == 0:
            selected_split = SplitType.TRAINING
        else:
            selected_split = SplitType.TEST
        self.controller.selectSplit(selected_split)
        self.refreshPreview()

    def onPreviousClicked(self):
        if self.controller.canGoPrevious():
            self.controller.previousImage()
        self.refreshPreview()

    def onNextClicked(self):
        if self.controller.canGoNext():
            self.controller.nextImage()
        self.refreshPreview()

    def refreshPreview(self):
        self.updateCurrentImage()
        self.updateNavigationControls(self.controller.getCurrentIndex(),self.controller.getImageCount())

    def updateNavigationControls(self, current_index, image_count):
        if image_count == 0:
            self.position_label.setText("0 / 0")
            self.previous_button.setEnabled(False)
            self.next_button.setEnabled(False)
            return
        else:
            self.position_label.setText(f"{current_index + 1}/{image_count}")
        self.previous_button.setEnabled(current_index > 0)
        self.next_button.setEnabled(current_index < image_count - 1)

    def updateCurrentImage(self):
        image_data = self.controller.getCurrentImage()
        self.updateImageMetadata(image_data)

        if image_data is None:
            self.image_label.clearImage()
            return

        image_path = image_data["imagePath"]
        pixmap = QPixmap(str(image_path))

        if not pixmap.isNull():
            self.image_label.setImage(pixmap)
        else:
            self.image_label.clearImage()

    def updateImageMetadata(self, image_data):
        if image_data is None:
            self.metadata_label.setText("No Metadata Available")
            return

        image_path = image_data.get("imagePath", "Unknown")
        filename = image_data.get("filename", "Unknown")
        tumorClass = image_data.get("tumorClass")
        split = image_data.get("split")
        fileExtension = image_data.get("fileExtension", "Unknown")

        metadata_text = (
            f"Filename: {filename}\n"
            f"Image Path: {image_path}\n"
            f"File Extension: {fileExtension}\n"
            f"Split: {split.name}\n"
            f"Tumor Class: {tumorClass.name}\n"
        )
        self.metadata_label.setText(metadata_text)

    def previewPreprocessing(self):
        config = AugmentationConfig(
            self.images_per_class_spinbox.value(),
            self.copies_per_image_spinbox.value(),
            self.seed_spinbox.value()
        )

        if not config.validate():
            self.displayError("Invalid augmentation configuration.")
            return

        self.preview_button.setEnabled(False)
        self.next_preview_batch_button.setEnabled(False)

        try:
            preview = self.controller.previewPreprocessing(config)
            self.displayPreview(preview)
            self.next_preview_batch_button.setEnabled(len(self._preview_samples) > 0)
        except Exception as error:
            self._clearPreprocessingPreview()
            self.displayError(f"Failed to generate preprocessing preview:\n{error}")
        finally:
            self.preview_button.setEnabled(True)

    def showNextPreviewBatch(self):
        self.next_preview_batch_button.setEnabled(False)

        try:
            preview = self.controller.nextPreviewBatch()

            if preview is None:
                self._clearPreprocessingPreview("No more preview batches available")
                return

            self.displayPreview(preview)
            self.next_preview_batch_button.setEnabled(len(self._preview_samples) > 0)
        except Exception as error:
            self._clearPreprocessingPreview()
            self.displayError(f"Failed to load next preview batch:\n{error}")

    def displayPreview(self, preview):
        if preview is None:
            self._clearPreprocessingPreview()
            return

        samples = []
        for tumor_class in TumorClass:
            samples.extend(preview.getSamples(tumor_class))

        if not samples:
            self._clearPreprocessingPreview("No images in this batch")
            return

        self._preview_samples = samples
        self._preview_sample_index = 0
        self._displaySelectedPreviewSample()

    def _displaySelectedPreviewSample(self):
        if not self._preview_samples:
            return

        self._current_preview_sample = self._preview_samples[self._preview_sample_index]
        self._preview_image_index = 0
        total_samples = len(self._preview_samples)

        self.sample_position_label.setText(f"{self._preview_sample_index + 1} / {total_samples}")
        self.sample_previous_button.setEnabled(self._preview_sample_index > 0)
        self.sample_next_button.setEnabled(self._preview_sample_index < total_samples - 1)

        self._updatePreviewImage()

    def onPreviewSamplePreviousClicked(self):
        if not self._preview_samples:
            return

        if self._preview_sample_index > 0:
            self._preview_sample_index -= 1
            self._displaySelectedPreviewSample()

    def onPreviewSampleNextClicked(self):
        if not self._preview_samples:
            return

        if self._preview_sample_index < len(self._preview_samples) - 1:
            self._preview_sample_index += 1
            self._displaySelectedPreviewSample()

    def onPreviewPreviousClicked(self):
        if self._current_preview_sample is None:
            return

        if self._preview_image_index > 0:
            self._preview_image_index -= 1
            self._updatePreviewImage()

    def onPreviewNextClicked(self):
        if self._current_preview_sample is None:
            return

        total_versions = 2 + len(self._current_preview_sample._augmented)
        if self._preview_image_index < total_versions - 1:
            self._preview_image_index += 1
            self._updatePreviewImage()

    def _updatePreviewImage(self):
        sample = self._current_preview_sample
        if sample is None:
            return

        index = self._preview_image_index
        total_versions = 2 + len(sample._augmented)

        if index == 0:
            tensor = sample._original
            image_type = "Original"
        elif index == 1:
            tensor = sample._preprocessed
            image_type = "Preprocessed"
        else:
            tensor = sample._augmented[index - 2]
            image_type = f"Augmented {index - 1}"

        if index > 0:
            minimum = tf.reduce_min(tensor)
            maximum = tf.reduce_max(tensor)
            tensor = tf.math.divide_no_nan(tensor - minimum, maximum - minimum)

        pixmap = self.image_adapter.tensorToPixmap(tensor)
        self.preprocessing_image_label.setImage(pixmap)

        filename = Path(sample._imagePath).name
        metadata_text = (
            f"Filename: {filename}\n"
            f"Tumor Class: {sample._label.name}\n"
            f"Augmented Copies: {len(sample._augmented)}\n"
            f"Displayed Image: {image_type}"
        )

        self.preprocessing_metadata_label.setText(metadata_text)
        self.preview_position_label.setText(f"{index + 1} / {total_versions}")
        self.preview_previous_button.setEnabled(index > 0)
        self.preview_next_button.setEnabled(index < total_versions - 1)

    def _clearPreprocessingPreview(self, message="No preprocessing preview available"):
        self._preview_samples = []
        self._preview_sample_index = 0
        self._current_preview_sample = None
        self._preview_image_index = 0

        self.preprocessing_image_label.clearImage(message)
        self.preprocessing_metadata_label.setText("No metadata available")

        self.preview_position_label.setText("0 / 0")
        self.preview_previous_button.setEnabled(False)
        self.preview_next_button.setEnabled(False)

        self.sample_position_label.setText("0 / 0")
        self.sample_previous_button.setEnabled(False)
        self.sample_next_button.setEnabled(False)

        self.next_preview_batch_button.setEnabled(False)

    def displayTensorPreview(self, tensor):
        pixmap = self.image_adapter.tensorToPixmap(tensor)
        self.image_label.setImage(pixmap)

    def displayError(self, message):
        QMessageBox.critical(self, "NeuroLens - Error", str(message))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())  #aparent exec() e blocant, deci exit nu se executa pana cand nu se opreste exec

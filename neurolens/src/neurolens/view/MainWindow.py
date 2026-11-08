import sys
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QLabel, QHBoxLayout,
                               QGroupBox, QSizePolicy, QPushButton, QFileDialog, QComboBox)
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from controller.MainController import MainController
from neurolens.data_management.SplitType import SplitType
from neurolens.view.ImageAdapter import ImageAdapter


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
        scaled_pixmap=self.original_pixmap.scaled(size, Qt.AspectRatioMode.KeepAspectRatio,
                                                  Qt.TransformationMode.SmoothTransformation)
        self.setPixmap(scaled_pixmap)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.updateImage()

    def clearImage(self, message="No image available"):
        self.original_pixmap = QPixmap()
        self.clear()
        self.setText(message)

class MainWindow(QMainWindow):

    def onNextClicked(self):
        if self.controller.canGoNext():
                self.controller.nextImage()
        self.refreshPreview()

    def onPreviousClicked(self):
        if self.controller.canGoPrevious():
            self.controller.previousImage()
        self.refreshPreview()

    def onLoadDataset(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Dataset")
        if not folder:
            return
        self.controller.loadDataset(folder)
        self.choose_split.setEnabled(True)
        self.onSplitChanged(self.choose_split.currentIndex())

    def onSplitChanged(self, index):
        if index == 0:
            selected_split=SplitType.TRAINING
        else:
            selected_split=SplitType.TEST
        self.controller.selectSplit(selected_split)
        self.refreshPreview()

    def updateNavigationControls(self,current_index,image_count):
        if image_count == 0:
            self.position_label.setText("0 / 0")
            self.previous_button.setEnabled(False)
            self.next_button.setEnabled(False)
            return
        else:
            self.position_label.setText(f"{current_index+1}/{image_count}")
        self.previous_button.setEnabled(current_index > 0)
        self.next_button.setEnabled(current_index < image_count-1)

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
        image_path = image_data.get("imagePath","Unknown")
        filename = image_data.get("filename","Unknown")
        tumorClass = image_data.get("tumorClass")
        split = image_data.get("split")
        fileExtension = image_data.get("fileExtension","Unknown")
        metadata_text=(f"Filename: {filename}\n"
                       f"Image Path: {image_path}\n"
                       f"File Extension: {fileExtension}\n"
                       f"Split: {split.name}\n"
                       f"Tumor Class: {tumorClass.name}\n")
        self.metadata_label.setText(metadata_text)

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
        layout.addStretch()

    def setupImagePreview(self,layout):
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

    def refreshPreview(self):
        self.updateCurrentImage()
        self.updateNavigationControls(self.controller.getCurrentIndex(),self.controller.getImageCount())

    def displayTensorPreview(self, tensor):
        pixmap = self.image_adapter.tensorToPixmap(tensor)
        self.image_label.setPixmap(pixmap)

    def __init__(self):
        super().__init__()
        self.image_adapter = ImageAdapter()
        self.controller = MainController()
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


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())#aparent exec() e blocant, deci exit nu se executa pana cand nu se opreste exec

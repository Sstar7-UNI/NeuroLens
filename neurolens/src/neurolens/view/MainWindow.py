import sys

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QApplication, QComboBox, QFileDialog, QGroupBox,
    QHBoxLayout, QLabel, QMainWindow, QMessageBox,
    QPushButton, QVBoxLayout, QWidget
)
from neurolens.controller.MainController import MainController
from neurolens.data_management.SplitType import SplitType


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.controller = MainController()
        self.setWindowTitle("NeuroLens")
        self.resize(950, 650)
        central = QWidget()
        self.setCentralWidget(central)
        mainLayout = QHBoxLayout(central)
        leftLayout = QVBoxLayout()
        mainLayout.addLayout(leftLayout, 3)
        self.splitSelector = QComboBox()
        self.splitSelector.addItem("Training", SplitType.TRAINING)
        self.splitSelector.addItem("Testing", SplitType.TEST)
        self.splitSelector.currentIndexChanged.connect(self.changeSplit)
        leftLayout.addWidget(self.splitSelector)
        self.imageLabel = QLabel("Load a dataset to preview images")
        self.imageLabel.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.imageLabel.setFixedSize(500, 330)
        self.imageLabel.setStyleSheet("background: #202020; color: white;")
        leftLayout.addWidget(self.imageLabel)
        navigation = QHBoxLayout()
        self.previousButton = QPushButton("Previous")
        self.nextButton = QPushButton("Next")
        self.counterLabel = QLabel("0 / 0")
        navigation.addWidget(self.previousButton)
        navigation.addWidget(self.counterLabel)
        navigation.addWidget(self.nextButton)
        leftLayout.addLayout(navigation)
        self.previousButton.clicked.connect(self.previousImage)
        self.nextButton.clicked.connect(self.nextImage)
        metadataBox = QGroupBox("Image Metadata")
        metadataLayout = QVBoxLayout(metadataBox)
        self.metadataLabel = QLabel("No image selected")
        self.metadataLabel.setWordWrap(True)
        metadataLayout.addWidget(self.metadataLabel)
        leftLayout.addWidget(metadataBox)
        leftLayout.addStretch()
        controls = QGroupBox("Control Panel")
        controlLayout = QVBoxLayout(controls)
        mainLayout.addWidget(controls, 2)
        loadButton = QPushButton("Load Dataset")
        loadButton.clicked.connect(self.loadDataset)
        controlLayout.addWidget(loadButton)
        for name in (
            "Start Training",
            "Stop Training",
            "Start Testing",
            "Export Model"
        ):
            button = QPushButton(name)
            button.setEnabled(False)
            controlLayout.addWidget(button)

        controlLayout.addStretch()

        self.showPreview()

    def loadDataset(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Dataset Folder")
        if not folder:
            return
        try:
            self.controller.loadDataset(folder)
        except Exception as error:
            QMessageBox.warning(self, "Dataset Error", str(error))
            return
        self.splitSelector.setCurrentIndex(0)
        self.showPreview()

    def changeSplit(self, index):
        split = self.splitSelector.itemData(index)
        self.controller.selectSplit(split)
        self.showPreview()

    def previousImage(self):
        self.controller.previousImage()
        self.showPreview()

    def nextImage(self):
        self.controller.nextImage()
        self.showPreview()

    def showPreview(self):
        count = self.controller.getImageCount()
        index = self.controller.getCurrentIndex()

        self.counterLabel.setText(f"{index + 1} / {count}" if count else "0 / 0")
        self.previousButton.setEnabled(self.controller.canGoPrevious())
        self.nextButton.setEnabled(self.controller.canGoNext())
        try:
            data = self.controller.getCurrentImage()
        except OSError as error:
            self.imageLabel.setText("Image unavailable")
            self.metadataLabel.setText(str(error))
            return

        if data is None:
            self.imageLabel.setText("No images available")
            self.metadataLabel.setText("No image selected")
            return

        pixmap = QPixmap(data["imagePath"])

        if pixmap.isNull():
            self.imageLabel.setText("Cannot display image")
            self.metadataLabel.setText("Cannot decode image")
            return

        self.imageLabel.setPixmap(
            pixmap.scaled(
                self.imageLabel.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
        )

        self.metadataLabel.setText(
            f"File: {data['filename']}\n"
            f"Class: {data['tumorClass'].name}\n"
            f"Split: {data['split'].name}\n"
            f"Size: {pixmap.width()} x {pixmap.height()}\n"
            f"Augmented: "
            f"{data['isAugmented'] if data['isAugmented'] is not None else 'Not available'}\n"
            f"Source: {data['originalImage'] or 'Not available'}\n"
            f"Transformations: "
            f"{', '.join(data['transformations']) if data['transformations'] else 'Not available'}"
        )


if __name__ == "__main__":
    app = QApplication(sys.argv)

    window = MainWindow()
    window.show()

    sys.exit(app.exec())

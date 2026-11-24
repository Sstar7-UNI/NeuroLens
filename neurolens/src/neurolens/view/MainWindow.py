import sys
from pathlib import Path
import tensorflow as tf
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QLabel, QHBoxLayout,
                               QGroupBox, QSizePolicy, QPushButton, QFileDialog, QComboBox, QSpinBox, QMessageBox)
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from neurolens.controller.MainController import MainController
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

    def setupLearnTab(self, learn_tab):
        layout = QVBoxLayout(learn_tab)

        layout.addWidget(QLabel("Training"))
        layout.addWidget(QLabel(
            "Configure a run, review its settings, then start training."
        ))

        settings_group = QGroupBox()
        settings_layout = QVBoxLayout(settings_group)

        # Training settings
        training_row = QHBoxLayout()
        training_info = QVBoxLayout()

        training_info.addWidget(QLabel("Training settings"))

        self.training_summary = QLabel()
        training_info.addWidget(self.training_summary)

        training_row.addLayout(training_info, 1)

        self.training_configure_button = QPushButton("Configure...")
        self.training_configure_button.clicked.connect(
            self.openTrainingSettings
        )

        training_row.addWidget(self.training_configure_button)
        settings_layout.addLayout(training_row)

        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        settings_layout.addWidget(separator)

        # Preprocessing
        preprocessing_row = QHBoxLayout()
        preprocessing_info = QVBoxLayout()

        preprocessing_info.addWidget(QLabel("Preprocessing"))

        self.preprocessing_summary = QLabel("No additional operators")
        preprocessing_info.addWidget(self.preprocessing_summary)

        preprocessing_row.addLayout(preprocessing_info, 1)

        self.preprocessing_configure_button = QPushButton("Configure...")
        self.preprocessing_configure_button.clicked.connect(
            self.openLearnPreprocessing
        )

        preprocessing_row.addWidget(self.preprocessing_configure_button)
        settings_layout.addLayout(preprocessing_row)

        layout.addWidget(settings_group)

        # Start training
        start_row = QHBoxLayout()

        self.training_requirement_label = QLabel(
            "Dataset and model required"
        )
        start_row.addWidget(self.training_requirement_label)
        start_row.addStretch()

        self.start_training_button = QPushButton("Start training")
        self.start_training_button.setEnabled(False)
        self.start_training_button.clicked.connect(self.startTraining)

        start_row.addWidget(self.start_training_button)
        layout.addLayout(start_row)


        # Training results
        results_group = QGroupBox("Training results")
        results_layout = QHBoxLayout(results_group)

        epochs_layout = QVBoxLayout()
        epochs_layout.addWidget(QLabel("Epochs completed"))
        self.training_epochs_label = QLabel("—")
        epochs_layout.addWidget(self.training_epochs_label)

        best_epoch_layout = QVBoxLayout()
        best_epoch_layout.addWidget(QLabel("Best epoch (EMA)"))
        self.training_best_epoch_label = QLabel("—")
        best_epoch_layout.addWidget(self.training_best_epoch_label)

        f1_layout = QVBoxLayout()
        f1_layout.addWidget(QLabel("Validation macro F1"))
        self.training_f1_label = QLabel("—")
        f1_layout.addWidget(self.training_f1_label)

        f1_layout.addWidget(QLabel("EMA balanced F1"))
        self.training_ema_f1_label = QLabel("—")
        f1_layout.addWidget(self.training_ema_f1_label)

        results_layout.addLayout(epochs_layout, 1)
        results_layout.addLayout(best_epoch_layout, 1)
        results_layout.addLayout(f1_layout, 1)

        layout.addWidget(results_group)
        layout.addStretch()

        self.updateTrainingSummary()

    def updateTrainingSummary(self):
        settings = self._training_settings

        batch = "all" if settings["all"] else str(settings["batch"])

        if settings["augment"]:
            augmentation = f'{settings["copies"]} copies'
        else:
            augmentation = "off"

        self.training_summary.setText(
            f'Batch size {batch} · '
            f'augmentation {augmentation} · '
            f'{settings["epochs"]} epochs'
        )

    def openTrainingSettings(self):
        settings = self._training_settings

        dialog = QDialog(self)
        dialog.setWindowTitle("Training settings")
        layout = QVBoxLayout(dialog)

        # Batch size
        layout.addWidget(QLabel("Batch size"))

        batch_input = QSpinBox()
        batch_input.setRange(1, 1000000)
        batch_input.setValue(settings["batch"])
        layout.addWidget(batch_input)

        all_check = QCheckBox("Use all training images in one batch")
        all_check.setChecked(settings["all"])

        all_check.toggled.connect(
            lambda checked: batch_input.setEnabled(not checked)
        )
        batch_input.setEnabled(not settings["all"])
        layout.addWidget(all_check)

        # Augmentation
        augment_check = QCheckBox("Enable augmentation")
        augment_check.setChecked(settings["augment"])
        layout.addWidget(augment_check)

        copies_widget = QWidget()
        copies_layout = QFormLayout(copies_widget)

        copies_input = QSpinBox()
        copies_input.setRange(1, 100)
        copies_input.setValue(settings["copies"])

        copies_layout.addRow("Copies per image", copies_input)

        copies_widget.setVisible(settings["augment"])
        augment_check.toggled.connect(copies_widget.setVisible)

        layout.addWidget(copies_widget)

        # Additional settings
        additional_toggle = QToolButton()
        additional_toggle.setText("Additional settings")
        additional_toggle.setCheckable(True)
        additional_toggle.setArrowType(Qt.ArrowType.RightArrow)
        additional_toggle.setToolButtonStyle(
            Qt.ToolButtonStyle.ToolButtonTextBesideIcon
        )

        layout.addWidget(additional_toggle)

        additional_widget = QWidget()
        additional_layout = QFormLayout(additional_widget)
        additional_widget.setVisible(False)

        additional_toggle.toggled.connect(additional_widget.setVisible)

        additional_toggle.toggled.connect(
            lambda checked: additional_toggle.setArrowType(
                Qt.ArrowType.DownArrow
                if checked else Qt.ArrowType.RightArrow
            )
        )

        epochs_input = QSpinBox()
        epochs_input.setRange(1, 1000)
        epochs_input.setValue(settings["epochs"])
        additional_layout.addRow("Epochs", epochs_input)

        learning_rate_input = QDoubleSpinBox()
        learning_rate_input.setDecimals(6)
        learning_rate_input.setRange(0.000001, 1.0)
        learning_rate_input.setSingleStep(0.0001)
        learning_rate_input.setValue(settings["learning_rate"])
        additional_layout.addRow("Learning rate", learning_rate_input)

        weight_decay_input = QDoubleSpinBox()
        weight_decay_input.setDecimals(6)
        weight_decay_input.setRange(0.0, 1.0)
        weight_decay_input.setSingleStep(0.0001)
        weight_decay_input.setValue(settings["weight_decay"])
        additional_layout.addRow("Weight decay", weight_decay_input)

        dropout_input = QDoubleSpinBox()
        dropout_input.setDecimals(2)
        dropout_input.setRange(0.0, 0.99)
        dropout_input.setSingleStep(0.05)
        dropout_input.setValue(settings["dropout"])
        additional_layout.addRow("Dropout rate", dropout_input)

        decoupled_check = QCheckBox("Decoupled weight decay")
        decoupled_check.setChecked(settings["decoupled"])
        additional_layout.addRow(decoupled_check)

        patience_input = QSpinBox()
        patience_input.setRange(1, 1000)
        patience_input.setValue(settings["patience"])
        additional_layout.addRow("Patience", patience_input)

        seed_input = QSpinBox()
        seed_input.setRange(0, 99999)
        seed_input.setValue(settings["seed"])
        additional_layout.addRow("Seed", seed_input)

        layout.addWidget(additional_widget)

        # Cancel / Apply
        buttons_layout = QHBoxLayout()
        buttons_layout.addStretch()

        cancel_button = QPushButton("Cancel")
        apply_button = QPushButton("Apply")

        cancel_button.clicked.connect(dialog.reject)
        apply_button.clicked.connect(dialog.accept)

        buttons_layout.addWidget(cancel_button)
        buttons_layout.addWidget(apply_button)
        layout.addLayout(buttons_layout)

        if dialog.exec() == QDialog.DialogCode.Accepted:
            self._training_settings = {
                "batch": batch_input.value(),
                "all": all_check.isChecked(),
                "augment": augment_check.isChecked(),
                "copies": copies_input.value(),
                "epochs": epochs_input.value(),
                "learning_rate": learning_rate_input.value(),
                "weight_decay": weight_decay_input.value(),
                "dropout": dropout_input.value(),
                "decoupled": decoupled_check.isChecked(),
                "patience": patience_input.value(),
                "seed": seed_input.value(),
            }

            self.updateTrainingSummary()

    def configurePreprocessingOperator(self, operator_type: str):
        if operator_type == "OtsuSegmenter":
            return {"type": "OtsuSegmenter"}

        dialog = QDialog(self)
        dialog.setWindowTitle(f"Configure {operator_type}")

        layout = QVBoxLayout(dialog)
        form = QFormLayout()
        layout.addLayout(form)

        fields = {}

        def doubleInput(minimum, maximum, value):
            widget = QDoubleSpinBox()
            widget.setDecimals(3)
            widget.setRange(minimum, maximum)
            widget.setSingleStep(0.05)
            widget.setValue(value)
            return widget

        def kernelInput():
            widget = QComboBox()
            widget.addItems(["1", "3", "5", "7", "9"])
            widget.setCurrentText("3")
            return widget

        def paddingInput():
            widget = QComboBox()
            widget.addItems(["ZERO", "REPLICATE", "REFLECT"])
            widget.setCurrentText("REFLECT")
            return widget

        if operator_type == "ContrastStretcher":
            fields["lower"] = doubleInput(0.0, 1.0, 0.05)
            fields["upper"] = doubleInput(0.0, 1.0, 0.95)

            form.addRow("Lower", fields["lower"])
            form.addRow("Upper", fields["upper"])

        elif operator_type == "GaussianFilter":
            fields["sigma"] = doubleInput(0.01, 20.0, 1.0)
            fields["kernelSize"] = kernelInput()
            fields["padding"] = paddingInput()

            form.addRow("Sigma", fields["sigma"])
            form.addRow("Kernel size", fields["kernelSize"])
            form.addRow("Padding", fields["padding"])

        elif operator_type == "MedianFilter":
            fields["kernelSize"] = kernelInput()
            fields["padding"] = paddingInput()

            form.addRow("Kernel size", fields["kernelSize"])
            form.addRow("Padding", fields["padding"])

        elif operator_type == "HistogramEqualizer":
            fields["levels"] = QSpinBox()
            fields["levels"].setRange(2, 256)
            fields["levels"].setValue(256)

            form.addRow("Levels", fields["levels"])

        elif operator_type == "MorphologyOperator":
            fields["connectivity"] = QComboBox()
            fields["connectivity"].addItems(["FOUR", "EIGHT"])
            fields["connectivity"].setCurrentText("EIGHT")

            form.addRow("Connectivity", fields["connectivity"])

        else:
            raise ValueError(f"Unknown operator: {operator_type}")

        buttons = QHBoxLayout()
        buttons.addStretch()

        cancel = QPushButton("Cancel")
        apply = QPushButton("Add operator")

        cancel.clicked.connect(dialog.reject)
        apply.clicked.connect(dialog.accept)

        buttons.addWidget(cancel)
        buttons.addWidget(apply)
        layout.addLayout(buttons)

        if dialog.exec() != QDialog.DialogCode.Accepted:
            return None

        step = {"type": operator_type}

        for name, widget in fields.items():
            if isinstance(widget, QComboBox):
                if name == "kernelSize":
                    step[name] = int(widget.currentText())
                else:
                    step[name] = widget.currentText()
            else:
                step[name] = widget.value()

        return step

    def openLearnPreprocessing(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Preprocessing")
        dialog.resize(500, 400)

        layout = QVBoxLayout(dialog)

        # Mode
        layout.addWidget(QLabel("Mode"))

        mode = QComboBox()
        mode.addItems([
            "None — no additional operators",
            "Manual — choose operators"
        ])

        if self._preprocessing_mode == "manual":
            mode.setCurrentIndex(1)

        layout.addWidget(mode)

        # Local copy of the selected pipeline
        steps = [
            step.copy()
            for step in self._manual_preprocessing_steps
        ]

        manual_group = QGroupBox("Manual preprocessing")
        manual_layout = QVBoxLayout(manual_group)

        manual_layout.addWidget(QLabel("Selected operators (in order)"))

        operators_list = QListWidget()
        manual_layout.addWidget(operators_list)

        # Operator selector
        selector_row = QHBoxLayout()

        operator_selector = QComboBox()
        operator_selector.addItems([
            "ContrastStretcher",
            "GaussianFilter",
            "MedianFilter",
            "HistogramEqualizer",
            "OtsuSegmenter",
            "MorphologyOperator"
        ])

        add_button = QPushButton("Add")

        selector_row.addWidget(operator_selector, 1)
        selector_row.addWidget(add_button)

        manual_layout.addLayout(selector_row)

        # Pipeline controls
        controls = QHBoxLayout()

        remove_button = QPushButton("Remove")
        up_button = QPushButton("Move up")
        down_button = QPushButton("Move down")

        controls.addWidget(remove_button)
        controls.addWidget(up_button)
        controls.addWidget(down_button)

        manual_layout.addLayout(controls)

        layout.addWidget(manual_group)

        manual_group.setVisible(mode.currentIndex() == 1)

        mode.currentIndexChanged.connect(
            lambda index: manual_group.setVisible(index == 1)
        )

        def refreshList():
            current = operators_list.currentRow()
            operators_list.clear()

            for index, step in enumerate(steps):
                parameters = ", ".join(
                    f"{key}={value}"
                    for key, value in step.items()
                    if key != "type"
                )

                description = f"{index + 1}. {step['type']}"

                if parameters:
                    description += f" ({parameters})"

                operators_list.addItem(description)

            if steps:
                operators_list.setCurrentRow(
                    max(0, min(current, len(steps) - 1))
                )

        def addOperator():
            operator_type = operator_selector.currentText()

            step = self.configurePreprocessingOperator(
                operator_type
            )

            if step is not None:
                steps.append(step)
                refreshList()
                operators_list.setCurrentRow(len(steps) - 1)

        def removeOperator():
            index = operators_list.currentRow()

            if index < 0:
                return

            steps.pop(index)
            refreshList()

        def moveUp():
            index = operators_list.currentRow()

            if index <= 0:
                return

            steps[index - 1], steps[index] = (
                steps[index],
                steps[index - 1]
            )

            refreshList()
            operators_list.setCurrentRow(index - 1)

        def moveDown():
            index = operators_list.currentRow()

            if index < 0 or index >= len(steps) - 1:
                return

            steps[index], steps[index + 1] = (
                steps[index + 1],
                steps[index]
            )

            refreshList()
            operators_list.setCurrentRow(index + 1)

        add_button.clicked.connect(addOperator)
        remove_button.clicked.connect(removeOperator)
        up_button.clicked.connect(moveUp)
        down_button.clicked.connect(moveDown)

        refreshList()

        buttons = QHBoxLayout()
        buttons.addStretch()

        cancel = QPushButton("Cancel")
        apply = QPushButton("Apply")

        buttons.addWidget(cancel)
        buttons.addWidget(apply)

        layout.addLayout(buttons)

        cancel.clicked.connect(dialog.reject)

        def applySettings():
            manual = mode.currentIndex() == 1

            if manual and not steps:
                QMessageBox.warning(
                    dialog,
                    "Preprocessing",
                    "Add at least one operator for Manual mode."
                )
                return

            selected_steps = steps if manual else []

            try:
                self.controller.selectPreprocessing(
                    selected_steps
                )
            except Exception as error:
                QMessageBox.warning(
                    dialog,
                    "Invalid preprocessing",
                    str(error)
                )
                return

            dialog.accept()

        apply.clicked.connect(applySettings)

        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        self._manual_preprocessing_steps = steps

        if mode.currentIndex() == 0:
            self._preprocessing_mode = "none"
            self.preprocessing_summary.setText(
                "No additional operators"
            )
        else:
            self._preprocessing_mode = "manual"
            self.preprocessing_summary.setText(
                f"Manual: {len(steps)} operators"
            )

        self._clearPreprocessingPreview()
        self.statusBar().showMessage(
            "Preprocessing configuration updated"
        )

    def openModelDialog(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Choose model")

        layout = QVBoxLayout(dialog)

        # Model weights
        layout.addWidget(QLabel("Model weights (.weights.h5)"))

        weights_row = QHBoxLayout()

        weights_path = QLineEdit(self._model_weights_path or "")
        weights_button = QPushButton("Choose file...")

        weights_row.addWidget(weights_path)
        weights_row.addWidget(weights_button)
        layout.addLayout(weights_row)

        def chooseWeights():
            path, _ = QFileDialog.getSaveFileName(
                dialog,
                "Choose model weights destination",
                weights_path.text(),
                "Keras weights (*.weights.h5)"
            )

            if path:
                weights_path.setText(path)

        weights_button.clicked.connect(chooseWeights)

        # Model configuration
        layout.addWidget(QLabel("Model configuration (.json)"))

        config_row = QHBoxLayout()

        config_path = QLineEdit(self._model_json_path or "")
        config_button = QPushButton("Choose file...")

        config_row.addWidget(config_path)
        config_row.addWidget(config_button)
        layout.addLayout(config_row)

        def chooseConfig():
            path, _ = QFileDialog.getOpenFileName(
                dialog,
                "Choose model configuration",
                config_path.text(),
                "JSON files (*.json)"
            )

            if path:
                config_path.setText(path)

        config_button.clicked.connect(chooseConfig)

        # Cancel / Select
        buttons = QHBoxLayout()
        buttons.addStretch()

        cancel = QPushButton("Cancel")
        select = QPushButton("Select")

        cancel.clicked.connect(dialog.reject)
        select.clicked.connect(dialog.accept)

        buttons.addWidget(cancel)
        buttons.addWidget(select)
        layout.addLayout(buttons)

        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        weights = weights_path.text().strip()
        config = config_path.text().strip()

        if not weights.endswith(".weights.h5"):
            self.displayError("Weights path must end with .weights.h5")
            return

        if not config.endswith(".json"):
            self.displayError("Model configuration must be a JSON file.")
            return

        try:
            self.controller.selectModel(weights, config)
        except Exception as error:
            self.displayError(f"Unable to select model:\n{error}")
            return

        self._model_weights_path = weights
        self._model_json_path = config

        self.model_label.setText(f"Model: {Path(config).stem}")
        self.statusBar().showMessage("Model selected")

        self.refreshLearnAvailability()

    def refreshLearnAvailability(self):
        ready = self._dataset_loaded and self._model_json_path is not None

        self.start_training_button.setEnabled(
            ready and not self._training_running
        )

        if self._training_running:
            message = "Training in progress..."
        elif not self._dataset_loaded and self._model_json_path is None:
            message = "Dataset and model required"
        elif not self._dataset_loaded:
            message = "Dataset required"
        elif self._model_json_path is None:
            message = "Model required"
        else:
            message = "Ready to train"

        self.training_requirement_label.setText(message)

    def startTraining(self):
        if self._training_running:
            return

        if not self._dataset_loaded or self._model_json_path is None:
            self.refreshLearnAvailability()
            return

        # Prevent accidental overwriting of existing weights
        if Path(self._model_weights_path).exists():
            answer = QMessageBox.question(
                self,
                "Overwrite model weights?",
                "The selected weights file may be overwritten. Continue?",
                QMessageBox.StandardButton.Yes |
                QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )

            if answer != QMessageBox.StandardButton.Yes:
                return

        settings = self._training_settings

        config = TrainingConfig(
            epochs=settings["epochs"],
            batchSize=0 if settings["all"] else settings["batch"],
            learningRate=settings["learning_rate"],
            weightDecay=settings["weight_decay"],
            dropoutRate=settings["dropout"],
            decoupledDecay=settings["decoupled"],
            patience=settings["patience"],
            useAugmentation=settings["augment"],
            augmentation=AugmentationConfig(
                imagesPerClass=100,
                copiesPerImage=settings["copies"] if settings["augment"] else 1,
                seed=settings["seed"]
            ),
            seed=settings["seed"]
        )

        self._training_running = True
        self.refreshLearnAvailability()

        self.training_epochs_label.setText("—")
        self.training_best_epoch_label.setText("—")
        self.training_f1_label.setText("—")
        self.training_ema_f1_label.setText("—")

        self.statusBar().showMessage("Training in progress...")

        QApplication.processEvents()

        try:
            report = self.controller.train(config)

            self.training_epochs_label.setText(
                str(report.epochsCompleted)
            )

            self.training_best_epoch_label.setText(
                str(report.bestEpoch + 1)
            )

            self.training_f1_label.setText(
                f"{report.bestValidationMacroF1:.4f}"
            )

            self.training_ema_f1_label.setText(
                f"{report.bestValidationEmaBalancedF1:.4f}"
            )

            self.statusBar().showMessage("Training completed")

        except Exception as error:
            self.displayError(f"Training failed:\n{error}")

        finally:
            self._training_running = False
            self.refreshLearnAvailability()

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

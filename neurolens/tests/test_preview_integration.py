
from pathlib import Path

from PySide6.QtGui import QColor, QImage
from PySide6.QtWidgets import QApplication

from neurolens.view.MainWindow import MainWindow


def create_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    image = QImage(16, 16, QImage.Format.Format_RGB888)
    image.fill(QColor("white"))

    assert image.save(str(path), "PNG")


def test_dataset_preview_integration(tmp_path: Path) -> None:
    app = QApplication.instance() or QApplication([])

    # Create a temporary dataset
    create_image(tmp_path / "Training" / "glioma" / "01.png")
    create_image(tmp_path / "Training" / "notumor" / "02.jpg")
    create_image(tmp_path / "Testing" / "pituitary" / "03.png")

    window = MainWindow()

    try:
        window.controller.loadDataset(str(tmp_path))
        window.showPreview()
        assert window.counterLabel.text() == "1 / 2"
        assert "01.png" in window.metadataLabel.text()
        assert not window.previousButton.isEnabled()
        assert window.nextButton.isEnabled()
        assert not window.imageLabel.pixmap().isNull()
        window.nextButton.click()
        assert window.counterLabel.text() == "2 / 2"
        assert "02.jpg" in window.metadataLabel.text()
        assert not window.nextButton.isEnabled()
        window.splitSelector.setCurrentIndex(1)
        assert window.counterLabel.text() == "1 / 1"
        assert "03.png" in window.metadataLabel.text()
        assert "PITUITARY" in window.metadataLabel.text()
        assert not window.previousButton.isEnabled()
        assert not window.nextButton.isEnabled()

    finally:
        window.close()

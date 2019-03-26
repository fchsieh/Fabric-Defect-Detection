import ntpath
import sys

import numpy
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

import cv2


class Demo(QWidget):
    def __init__(self):
        super().__init__()
        self.image = None
        self.imageForDisplay = None
        self.currentImage = None
        self.currentPosition = None

        # init UI
        self.label = QLabel()
        self.initUI()

    def initUI(self):
        # Set window title
        self.setWindowTitle("Read image demo")
        self.label.setText("Display image here")
        self.label.setAlignment(Qt.AlignCenter)

        # Add buttons
        button_ReadImage = QPushButton("Read image")
        button_ReadImage.clicked.connect(self.readImage)

        button_CropImage = QPushButton("Cut image")
        button_CropImage.clicked.connect(self.cropImage)

        # Set layout
        top_bar = QHBoxLayout()
        top_bar.addWidget(button_ReadImage)
        top_bar.addWidget(button_CropImage)

        root = QVBoxLayout(self)
        root.addLayout(top_bar)
        root.addWidget(self.label)

        self.resize(500, 500)

    def readImage(self):
        # Read image from current dir
        filename, _ = QFileDialog.getOpenFileName(
            self, "Open file", ".", "Image Files (*.png *.jpg *.jpeg)")
        if filename:
            self.image = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
            height, width = self.image.shape[:2]
            self.imageForDisplay = cv2.resize(
                self.image, (int(0.5 * width), int(0.5 * height)),
                interpolation=cv2.INTER_CUBIC)
            self.currentImage = filename
            self.displayImage()

    def cropImage(self):
        if self.currentImage is not None:
            img = cv2.imread(self.currentImage, cv2.IMREAD_UNCHANGED)
            basename = ntpath.basename(self.currentImage)
            dirname = ntpath.dirname(self.currentImage)
            # Return to original coordinate
            if self.currentPosition is not None:
                x = self.currentPosition[0] * 2
                y = self.currentPosition[1] * 2
                cropImg = img[y - 100:y + 100, x - 100:x + 100]
                newFilename = dirname + "/cropped_" + basename  # add prefix
                cv2.imwrite(newFilename, cropImg)

    def displayImage(self):
        size = self.imageForDisplay.shape
        step = self.imageForDisplay.size / size[0]
        qformat = QImage.Format_Indexed8

        if len(size) == 3:
            if size[2] == 4:
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888

        img = QImage(self.imageForDisplay, size[1], size[0], step, qformat)
        img = img.rgbSwapped()

        self.label.setPixmap(QPixmap.fromImage(img))
        self.resize(self.imageForDisplay.shape[1],
                    self.imageForDisplay.shape[0])

    def mouseReleaseEvent(self, QMouseEvent):
        if self.imageForDisplay is not None:
            self.currentPosition = [QMouseEvent.x(), QMouseEvent.y()]


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = Demo()
    window.setMinimumSize(500, 500)
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

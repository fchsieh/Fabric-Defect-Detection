# -*- coding: utf-8 -*-
import io
import math
import sys
import os

from PIL import Image, ImageDraw
from PIL.ImageQt import ImageQt
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *


class project(QWidget):
    def __init__(self):
        super().__init__()
        self.currentImage = None
        self.currentPosition = None
        self.imgResizeWidth = 1024
        self.imgResizeHeight = 642
        self.AreaHeight = 100
        self.AreaWidth = 100
        self.OriginX = 500
        self.OriginY = 511
        self.saveFileName = None
        self.fileCount = 1
        self.saveDir = None

        # init UI
        self.imgBlock = QLabel()
        self.imgtext = QLabel()
        self.imgSizetext = QLabel(" ")
        self.imgSizelabel = QLabel()
        self.AreaSizetext = QLabel()
        self.mouseloctext = QLabel()
        self.Ox_txtEdit = QLineEdit(str(self.OriginX))
        self.Oy_txtEdit = QLineEdit(str(self.OriginY))
        self.initUI()

    def initUI(self):
        # Set window title
        self.setWindowTitle("Project_2019_03")
        self.imgBlock.move(10, 20)

        # Set text
        self.imgBlock.setAlignment(Qt.AlignCenter)
        self.imgtext.setFixedHeight(15)
        self.imgSizetext.setFixedHeight(15)
        self.mouseloctext.setFixedHeight(15)
        self.mouseloctext.setText("擷取座標：")
        self.imgSizelabel.setFixedWidth(self.imgResizeWidth)
        self.imgSizelabel.setFixedHeight(self.imgResizeHeight)
        self.imgSizetext.setText(
            "載入轉換:" + str(self.imgResizeWidth) + "×" + str(self.imgResizeHeight) + "px"
        )
        self.AreaSizetext.setText(
            "區域大小:" + str(self.AreaWidth * 2) + "×" + str(self.AreaHeight * 2) + "px"
        )

        # Set LineEdit
        self.Ox_txtEdit.setMaxLength(4)
        self.Oy_txtEdit.setMaxLength(4)
        self.Ox_txtEdit.setAlignment(Qt.AlignCenter)
        self.Oy_txtEdit.setAlignment(Qt.AlignCenter)
        self.Ox_txtEdit.setInputMask("0000")
        self.Oy_txtEdit.setInputMask("0000")
        self.Ox_txtEdit.textChanged.connect(self.NewOriginX)
        self.Oy_txtEdit.textChanged.connect(self.NewOriginY)

        # Add buttons
        imgSavebtn = QPushButton("儲存位置")
        imgSavebtn.clicked.connect(self.saveFile)
        imgLoadbtn = QPushButton("載入圖片")
        imgLoadbtn.clicked.connect(self.getFile)
        imgCropbtn = QPushButton("擷取圖片")
        imgCropbtn.clicked.connect(self.cropImage)
        txtOriginlb = QLabel("原點(x,y):")

        # Set layout
        btn_Layout = QHBoxLayout()
        btn_Layout.addWidget(imgSavebtn)
        btn_Layout.addWidget(imgLoadbtn)
        btn_Layout.addWidget(imgCropbtn)

        Origintxt_Layout = QHBoxLayout()
        Origintxt_Layout.addWidget(txtOriginlb)
        Origintxt_Layout.addWidget(self.Ox_txtEdit)
        Origintxt_Layout.addWidget(self.Oy_txtEdit)

        Pointer_Layout = QHBoxLayout()
        Pointer_Layout.addWidget(self.mouseloctext)
        Pointer_Layout.addStretch(1)
        Pointer_Layout.addLayout(Origintxt_Layout)

        File_Layout = QHBoxLayout()
        File_Layout.addWidget(self.imgtext)
        File_Layout.addStretch(5)
        File_Layout.addWidget(self.AreaSizetext)
        File_Layout.addStretch(1)
        File_Layout.addWidget(self.imgSizetext)

        txt_Layout = QVBoxLayout()
        txt_Layout.addLayout(File_Layout)
        txt_Layout.addLayout(Pointer_Layout)

        root = QVBoxLayout(self)
        root.addLayout(btn_Layout)
        root.addLayout(txt_Layout)
        root.addWidget(self.imgBlock)
        self.resize(1024, 768)

    def saveFile(self):
        path = str(QFileDialog.getExistingDirectory(None, "Select directory"))
        self.saveDir = path

    def getFile(self):
        fname, _ = QFileDialog.getOpenFileName(
            self, "Open file", ".", "Image files (*.jpg *.gif *.png *.jpeg *.bmp)"
        )
        base = os.path.basename(fname)
        self.saveFileName = os.path.splitext(base)[0]
        img = QImage(fname)
        self.imgtext.setText(str(fname))
        result = img.scaled(
            self.imgSizelabel.width(),
            self.imgSizelabel.height(),
            Qt.IgnoreAspectRatio,
            Qt.SmoothTransformation,
        )
        self.imgBlock.setPixmap(QPixmap.fromImage(result))
        self.currentImage = fname
        QPixmapCache.clear()
        self.fileCount = 1

    def cropImage(self):
        if self.currentImage is not None:
            img = Image.open(self.currentImage)
            CenterPoint = self.OriginX * 2, self.OriginY * 2
            ClickPoint = (2 * self.currentPosition[0], 2 * self.currentPosition[1])
            Radians = math.atan2(
                ClickPoint[1] - CenterPoint[1], ClickPoint[0] - CenterPoint[0]
            )
            Degrees = math.degrees(Radians) + 90
            rotated_img = img.rotate(Degrees, expand=False, center=(ClickPoint))
            Area = (
                ClickPoint[0] - self.AreaWidth,
                ClickPoint[1] - self.AreaHeight,
                ClickPoint[0] + self.AreaWidth,
                ClickPoint[1] + self.AreaHeight,
            )
            cropped_img = rotated_img.crop(Area)
            if self.saveDir is None:
                self.saveDir = "."
            saveName = "%s/%s_(%sx%s)_[%s].bmp" % (
                self.saveDir,
                self.saveFileName,
                str(self.AreaHeight * 2),
                str(self.AreaWidth * 2),
                str(self.fileCount),
            )
            try:
                cropped_img.save(saveName)
            except:
                print("Save file error!")
            self.fileCount += 1

    def NewOriginX(self, text):
        if text is not None:
            self.OriginX = int(text)

    def NewOriginY(self, text):
        if text is not None:
            self.OriginY = int(text)

    def mousePressEvent(self, event):
        if self.currentImage is not None:
            if event.buttons() == Qt.LeftButton:
                mouselocstr = (
                    event.pos().x() - self.imgBlock.x(),
                    event.pos().y() - self.imgBlock.y(),
                )
                self.currentPosition = mouselocstr
                if (
                    mouselocstr[0] >= 0
                    and mouselocstr[1] >= 0
                    and mouselocstr[0] <= self.imgResizeWidth
                    and mouselocstr[1] <= self.imgResizeHeight
                ):
                    self.mouseloctext.setText("擷取座標：" + str(mouselocstr))
                    img = Image.open(self.currentImage)
                    CenterPoint = self.OriginX * 2, self.OriginY * 2
                    ClickPoint = (
                        2 * self.currentPosition[0],
                        2 * self.currentPosition[1],
                    )
                    AreaHeight = 200
                    AreaWidth = 150
                    AreaPoint = [
                        (
                            ClickPoint[0] - self.AreaWidth,
                            ClickPoint[1] - self.AreaHeight,
                        ),
                        (
                            ClickPoint[0] - self.AreaWidth,
                            ClickPoint[1] + self.AreaHeight,
                        ),
                        (
                            ClickPoint[0] + self.AreaWidth,
                            ClickPoint[1] + self.AreaHeight,
                        ),
                        (
                            ClickPoint[0] + self.AreaWidth,
                            ClickPoint[1] - self.AreaHeight,
                        ),
                        (
                            ClickPoint[0] - self.AreaWidth,
                            ClickPoint[1] - self.AreaHeight,
                        ),
                    ]
                    Radians = math.atan2(
                        ClickPoint[1] - CenterPoint[1], ClickPoint[0] - CenterPoint[0]
                    )
                    Degrees = math.degrees(Radians) + 90
                    theta = math.pi / 180 * (Degrees)
                    RotAreaPoint = [
                        (
                            ClickPoint[0]
                            + (ClickPoint[0] - self.AreaWidth - ClickPoint[0])
                            * math.cos(theta)
                            - (ClickPoint[1] - self.AreaHeight - ClickPoint[1])
                            * math.sin(theta),
                            ClickPoint[1]
                            + (ClickPoint[0] - self.AreaWidth - ClickPoint[0])
                            * math.sin(theta)
                            + (ClickPoint[1] - self.AreaHeight - ClickPoint[1])
                            * math.cos(theta),
                        ),
                        (
                            ClickPoint[0]
                            + (ClickPoint[0] - self.AreaWidth - ClickPoint[0])
                            * math.cos(theta)
                            - (ClickPoint[1] + self.AreaHeight - ClickPoint[1])
                            * math.sin(theta),
                            ClickPoint[1]
                            + (ClickPoint[0] - self.AreaWidth - ClickPoint[0])
                            * math.sin(theta)
                            + (ClickPoint[1] + self.AreaHeight - ClickPoint[1])
                            * math.cos(theta),
                        ),
                        (
                            ClickPoint[0]
                            + (ClickPoint[0] + self.AreaWidth - ClickPoint[0])
                            * math.cos(theta)
                            - (ClickPoint[1] + self.AreaHeight - ClickPoint[1])
                            * math.sin(theta),
                            ClickPoint[1]
                            + (ClickPoint[0] + self.AreaWidth - ClickPoint[0])
                            * math.sin(theta)
                            + (ClickPoint[1] + self.AreaHeight - ClickPoint[1])
                            * math.cos(theta),
                        ),
                        (
                            ClickPoint[0]
                            + (ClickPoint[0] + self.AreaWidth - ClickPoint[0])
                            * math.cos(theta)
                            - (ClickPoint[1] - self.AreaHeight - ClickPoint[1])
                            * math.sin(theta),
                            ClickPoint[1]
                            + (ClickPoint[0] + self.AreaWidth - ClickPoint[0])
                            * math.sin(theta)
                            + (ClickPoint[1] - self.AreaHeight - ClickPoint[1])
                            * math.cos(theta),
                        ),
                        (
                            ClickPoint[0]
                            + (ClickPoint[0] - self.AreaWidth - ClickPoint[0])
                            * math.cos(theta)
                            - (ClickPoint[1] - self.AreaHeight - ClickPoint[1])
                            * math.sin(theta),
                            ClickPoint[1]
                            + (ClickPoint[0] - self.AreaWidth - ClickPoint[0])
                            * math.sin(theta)
                            + (ClickPoint[1] - self.AreaHeight - ClickPoint[1])
                            * math.cos(theta),
                        ),
                    ]
                    draw = ImageDraw.Draw(img)
                    draw.ellipse(
                        (
                            CenterPoint[0] - 10,
                            CenterPoint[1] - 10,
                            CenterPoint[0] + 10,
                            CenterPoint[1] + 10,
                        ),
                        fill=(255),
                    )
                    draw.line(RotAreaPoint, fill=(255), width=3)
                    qim = ImageQt(img)
                    result = qim.scaled(
                        self.imgSizelabel.width(),
                        self.imgSizelabel.height(),
                        Qt.IgnoreAspectRatio,
                        Qt.SmoothTransformation,
                    )
                    self.imgBlock.setPixmap(QPixmap.fromImage(result))
                    QPixmapCache.clear()
                else:
                    self.mouseloctext.setText("擷取座標：")


if __name__ == "__main__":

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = project()
    window.show()
    sys.exit(app.exec_())

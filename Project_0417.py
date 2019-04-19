# -*- coding: utf-8 -*-

'''
Created on Apr 11, 2019

@author: Fu-Cheng Hsieh
         Yin-Tao Ling

'''

import io
import math
import sys
import os

from PIL import Image, ImageDraw
from PIL.ImageQt import ImageQt
from functools import partial
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *


class project(QWidget):
    def __init__(self):
        super().__init__()
        self.currentImage = None
        self.currentPosition = (500,511)
        self.imgResizeWidth = 1376
        self.imgResizeHeight = 864
        self.AreaHeight = 100
        self.AreaWidth = 100
        self.OriginX = 500
        self.OriginY = 511
        self.saveFileName = None
        self.fileCountN = 1
        self.fileCountE = [1,1,1,1]
        self.saveDir = None
        self.OriginbtnFlag = 0

        # init UI
        self.imgBlock = QLabel()

        self.imgINtxt = QLabel()
        self.imgINtxt.setWordWrap(True)
        self.imgINtxt.setAlignment(Qt.AlignLeft)
        self.imgOUTtxt = QLabel()
        self.imgOUTtxt.setWordWrap(True)
        self.imgOUTtxt.setFixedHeight(150)
        self.imgOUTtxt.setAlignment(Qt.AlignLeft)
        self.imgSizetext = QLabel()
        self.imgSizelabel = QLabel()
        self.AreaSizetext = QLabel()
        self.mouseloctext = QLabel()
        self.Ox_txtEdit = QLineEdit(str(self.OriginX))
        self.Oy_txtEdit = QLineEdit(str(self.OriginY))
        self.XCropSizeEdit = QLineEdit(str(self.AreaWidth*2))
        self.YCropSizeEdit = QLineEdit(str(self.AreaHeight*2))
        self.initUI()

    def initUI(self):

        # Set StyleSheet
        self.setStyleSheet("QGroupBox {border: 1px solid #CCCCCC ;border-radius: 2px;margin-top: 1ex;}")

        # Set window title
        self.setWindowTitle("絲餅樣本建立工具")

        # Set text
        self.imgBlock.setAlignment(Qt.AlignCenter)
        self.imgBlock.setFixedWidth(self.imgResizeWidth)
        self.mouseloctext.setText("擷取座標：")
        self.imgSizelabel.setFixedWidth(self.imgResizeWidth)
        self.imgSizelabel.setFixedHeight(self.imgResizeHeight)
        self.imgSizetext.setText(
            "載入轉換:" + str(self.imgResizeWidth) + "×" + str(self.imgResizeHeight) + "px"
        )
        self.AreaSizetext.setText(
            "擷取大小:" + str(self.AreaWidth * 2) + "×" + str(self.AreaHeight * 2) + "px"
        )

        # Set Origin LineEdit
        self.Ox_txtEdit.setMaxLength(4)
        self.Oy_txtEdit.setMaxLength(4)
        self.Ox_txtEdit.setAlignment(Qt.AlignCenter)
        self.Oy_txtEdit.setAlignment(Qt.AlignCenter)
        self.Ox_txtEdit.setInputMask("0000")
        self.Oy_txtEdit.setInputMask("0000")
        self.Ox_txtEdit.textChanged.connect(self.NewOriginX)
        self.Oy_txtEdit.textChanged.connect(self.NewOriginY)

        # Set CropSize LineEdit
        self.XCropSizeEdit.setMaxLength(3)
        self.YCropSizeEdit.setMaxLength(3)
        self.XCropSizeEdit.setAlignment(Qt.AlignCenter)
        self.YCropSizeEdit.setAlignment(Qt.AlignCenter)
        self.XCropSizeEdit.setInputMask("000")
        self.YCropSizeEdit.setInputMask("000")
        self.XCropSizeEdit.textChanged.connect(self.NewCropWidth)
        self.YCropSizeEdit.textChanged.connect(self.NewCropHeight)

        # Add buttons
        imgSavebtn = QPushButton("儲存位置")
        imgSavebtn.clicked.connect(self.saveFile)
        imgLoadbtn = QPushButton("載入圖片")
        imgLoadbtn.clicked.connect(self.getFile)
        self.imgCropNormalbtn = QPushButton("正常")
        self.imgCropNormalbtn.setFixedHeight(75)
        self.imgCropNormalbtn.setStyleSheet("QPushButton {background-color: #7CFC00 }")
        self.imgCropNormalbtn.clicked.connect(partial(self.cropImage,0))
        self.imgCropNormalbtn.setEnabled(False)
        self.imgCropECbtn = QPushButton("碰污")
        self.imgCropECbtn.setFixedHeight(75)
        self.imgCropECbtn.setStyleSheet("QPushButton {background-color: #FF4500 }")
        self.imgCropECbtn.clicked.connect(partial(self.cropImage,1))
        self.imgCropECbtn.setEnabled(False)
        self.imgCropEObtn = QPushButton("外溢")
        self.imgCropEObtn.setFixedHeight(75)
        self.imgCropEObtn.setStyleSheet("QPushButton {background-color: #FF4500 }")
        self.imgCropEObtn.clicked.connect(partial(self.cropImage,2))
        self.imgCropEObtn.setEnabled(False)
        self.imgCropEFbtn = QPushButton("夾絲")
        self.imgCropEFbtn.setFixedHeight(75)
        self.imgCropEFbtn.setStyleSheet("QPushButton {background-color: #FF4500 }")
        self.imgCropEFbtn.clicked.connect(partial(self.cropImage,3))
        self.imgCropEFbtn.setEnabled(False)
        self.imgCropESbtn = QPushButton("蛛網")
        self.imgCropESbtn.setFixedHeight(75)
        self.imgCropESbtn.setStyleSheet("QPushButton {background-color: #FF4500 }")
        self.imgCropESbtn.clicked.connect(partial(self.cropImage,4))
        self.imgCropESbtn.setEnabled(False)
        self.setOrigbtn = QPushButton("定位")
        self.setOrigbtn.setEnabled(False)
        self.setOrigbtn.clicked.connect(self.setOrigin)

        # Add Text
        txtOriginlb = QLabel("原點(x,y):")
        txtCropSizelb = QLabel("擷取大小(px):")
        txtCropbtnlb = QLabel("擷取圖片")
        txtCropbtnlb.setAlignment(Qt.AlignCenter)

        ### Set layout ###
        # Box 1 _ Img LOAD
        grpLOADtxt = QGroupBox("")
        grpLOADtxt.setFlat(True)
        FileIN_Layout = QVBoxLayout()
        FileIN_Layout.setSizeConstraint(0)
        FileIN_Layout.addWidget(imgLoadbtn)
        FileIN_Layout.addWidget(self.imgINtxt)
        grpLOADtxt.setLayout(FileIN_Layout)

        # Box 2 _ Origin Setting
        grpOrigtxt = QGroupBox()
        grpOrigtxt.setFlat(True)
        BOX2_Layout = QVBoxLayout()
        Sizetxt_Layout = QHBoxLayout()
        Sizetxt_Layout.addWidget(txtCropSizelb)
        Sizetxt_Layout.addWidget(self.XCropSizeEdit)
        Sizetxt_Layout.addWidget(self.YCropSizeEdit)
        Origintxt_Layout = QHBoxLayout()
        Origintxt_Layout.addWidget(txtOriginlb)
        Origintxt_Layout.addWidget(self.Ox_txtEdit)
        Origintxt_Layout.addWidget(self.Oy_txtEdit)
        Origintxt_Layout.addWidget(self.setOrigbtn)
        BOX2_Layout.addLayout(Origintxt_Layout)
        BOX2_Layout.addLayout(Sizetxt_Layout)
        grpOrigtxt.setLayout(BOX2_Layout)

        # Box 3 _ Img SAVE
        grpSAVEtxt = QGroupBox("")
        grpSAVEtxt.setFlat(True)
        FileOUT_Layout = QVBoxLayout()
        FileOUT_Layout.setSizeConstraint(0)
        FileOUT_Layout.addWidget(imgSavebtn)
        FileOUT_Layout.addWidget(self.imgOUTtxt)
        grpSAVEtxt.setLayout(FileOUT_Layout)

        # Box 4 _ Img Crop
        grpCroptxt = QGroupBox()
        grpCroptxt.setFlat(True)
        CropBox_Layout = QGridLayout()
        CropBox_Layout.addWidget(txtCropbtnlb, 0, 0, 1, 5)
        CropBox_Layout.addWidget(self.imgCropNormalbtn, 1, 0, 2, 1)
        CropBox_Layout.addWidget(self.imgCropECbtn, 1, 1, 2, 1)
        CropBox_Layout.addWidget(self.imgCropEObtn, 1, 2, 2, 1)
        CropBox_Layout.addWidget(self.imgCropEFbtn, 1, 3, 2, 1)
        CropBox_Layout.addWidget(self.imgCropESbtn, 1, 4, 2, 1)
        grpCroptxt.setLayout(CropBox_Layout)

        # Layout 3 _ Show Status 
        Status_Layout = QVBoxLayout()
        Status_Layout.addWidget(self.mouseloctext)
        Status_Layout.addWidget(self.AreaSizetext)
        Status_Layout.addWidget(self.imgSizetext)

        BOX = QVBoxLayout()
        BOX.addWidget(grpLOADtxt)
        BOX.addWidget(grpOrigtxt)
        BOX.addWidget(grpSAVEtxt)
        BOX.addWidget(grpCroptxt)
        BOX.addLayout(Status_Layout)

        root = QHBoxLayout(self)
        root.addLayout(BOX)
        root.addWidget(self.imgBlock)
        self.resize(1800, 768)
        self.move(30,10)

        # Set QTooltip
        QToolTip.setFont(QFont('SansSerif', 10))
        self.setOrigbtn.setToolTip('點擊後自訂原點位置，方框會以原點旋轉')
        imgSavebtn.setToolTip('選擇儲存圖片的位置，自動設立Normal與Error資料夾')
        imgLoadbtn.setToolTip('載入原始圖片')
        self.XCropSizeEdit.setToolTip('修改選取區塊大小')
        self.YCropSizeEdit.setToolTip('修改選取區塊大小')
        self.imgCropNormalbtn.setToolTip('確定選取區塊為<b>無瑕疵</b>，進行儲存')
        self.imgCropECbtn.setToolTip('確定選取區塊為<b>碰污</b>，進行儲存')
        self.imgCropEObtn.setToolTip('確定選取區塊為<b>外溢</b>，進行儲存')
        self.imgCropEFbtn.setToolTip('確定選取區塊為<b>夾絲</b>，進行儲存')
        self.imgCropESbtn.setToolTip('確定選取區塊為<b>蛛網</b>，進行儲存')

    def setOrigin(self):
        if self.currentImage is not None:
            self.OriginbtnFlag = 1
            self.setCursor(QCursor(Qt.CrossCursor))
            self.imgBlock.setToolTip('請選擇原點位置')
            self.imgBlock.setToolTipDuration(2000)

    def saveFile(self):
        path = str(QFileDialog.getExistingDirectory(None, "Select directory"))
        self.saveDir = path
        self.imgOUTtxt.setText(str(path))

    def getFile(self):
        fname, _ = QFileDialog.getOpenFileName(
            self, "Open file", ".", "Image files (*.jpg *.gif *.png *.jpeg *.bmp)"
        )
        if fname:
            self.imgBlock.setToolTip('請在想擷取的區塊進行點擊')
            self.imgBlock.setToolTipDuration(1800)
            base = os.path.basename(fname)
            self.saveFileName = os.path.splitext(base)[0]
            img = QImage(fname)
            self.imgINtxt.setText(str(fname))
            result = img.scaled(
                self.imgSizelabel.width(),
                self.imgSizelabel.height(),
                Qt.IgnoreAspectRatio,
                Qt.SmoothTransformation,
            )
            self.imgBlock.setPixmap(QPixmap.fromImage(result))
            self.currentImage = fname
            QPixmapCache.clear()
            self.fileCountN = 1
            self.fileCountE = [1,1,1,1]
            self.imgCropNormalbtn.setEnabled(True)
            self.imgCropECbtn.setEnabled(True)
            self.imgCropEObtn.setEnabled(True)
            self.imgCropEFbtn.setEnabled(True)
            self.imgCropESbtn.setEnabled(True)
            self.setOrigbtn.setEnabled(True)
            self.setOrigbtn.setCheckable(True)

    def cropImage(self,values):
        if self.currentImage is not None:
            img = Image.open(self.currentImage)
            CenterPoint = self.OriginX * 2.5, self.OriginY * 2.5
            ClickPoint = (2.5 * self.currentPosition[0], 2.5 * self.currentPosition[1])
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
            if values == 0:
                newpath = self.saveDir + "/Normal"
                if not os.path.exists(newpath):
                    os.makedirs(newpath)
                saveName = "%s/%s_(%sx%s)_[%s].bmp" % (
                    newpath,
                    self.saveFileName,
                    str(self.AreaHeight * 2),
                    str(self.AreaWidth * 2),
                    str(self.fileCountN),
                )
                cropped_img.save(saveName)
                self.imgOUTtxt.setText(str(self.saveDir)+"\n\n已儲存圖片\n"+str(saveName))
                self.fileCountN += 1
            elif values == 1:
                newpath = self.saveDir + "/Stain"
                if not os.path.exists(newpath):
                    os.makedirs(newpath)
                saveName = "%s/%s_(%sx%s)_[%s].bmp" % (
                    newpath,
                    self.saveFileName,
                    str(self.AreaHeight * 2),
                    str(self.AreaWidth * 2),
                    str(self.fileCountE[0]),
                )
                cropped_img.save(saveName)
                self.imgOUTtxt.setText(str(self.saveDir)+"\n\n已儲存圖片\n"+str(saveName))
                self.fileCountE[0] += 1
            elif values == 2:
                newpath = self.saveDir + "/Spill"
                if not os.path.exists(newpath):
                    os.makedirs(newpath)
                saveName = "%s/%s_(%sx%s)_[%s].bmp" % (
                    newpath,
                    self.saveFileName,
                    str(self.AreaHeight * 2),
                    str(self.AreaWidth * 2),
                    str(self.fileCountE[1]),
                )
                cropped_img.save(saveName)
                self.imgOUTtxt.setText(str(self.saveDir)+"\n\n已儲存圖片\n"+str(saveName))
                self.fileCountE[1] += 1
            elif values == 3:
                newpath = self.saveDir + "/Pinch"
                if not os.path.exists(newpath):
                    os.makedirs(newpath)
                saveName = "%s/%s_(%sx%s)_[%s].bmp" % (
                    newpath,
                    self.saveFileName,
                    str(self.AreaHeight * 2),
                    str(self.AreaWidth * 2),
                    str(self.fileCountE[2]),
                )
                cropped_img.save(saveName)
                self.imgOUTtxt.setText(str(self.saveDir)+"\n\n已儲存圖片\n"+str(saveName))
                self.fileCountE[2] += 1
            elif values == 4:
                newpath = self.saveDir + "/Spider"
                if not os.path.exists(newpath):
                    os.makedirs(newpath)
                saveName = "%s/%s_(%sx%s)_[%s].bmp" % (
                    newpath,
                    self.saveFileName,
                    str(self.AreaHeight * 2),
                    str(self.AreaWidth * 2),
                    str(self.fileCountE[3]),
                )
                cropped_img.save(saveName)
                self.imgOUTtxt.setText(str(self.saveDir)+"\n\n已儲存圖片\n"+str(saveName))
                self.fileCountE[3] += 1

    def NewCropWidth(self, text):
        if text is not None:
            self.AreaWidth = round(int(text)/2)
            self.AreaSizetext.setText(
            "擷取大小:" + str(self.AreaWidth * 2) + "×" + str(self.AreaHeight * 2) + "px"
            )

    def NewCropHeight(self, text):
        if text is not None:
            self.AreaHeight = round(int(text)/2)
            self.AreaSizetext.setText(
            "擷取大小:" + str(self.AreaWidth * 2) + "×" + str(self.AreaHeight * 2) + "px"
            )

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
                    event.pos().y() - self.imgBlock.y()
                )
                self.currentPosition = mouselocstr
                if (
                    mouselocstr[0] >= 0
                    and mouselocstr[1] >= 0
                    and mouselocstr[0] <= self.imgResizeWidth
                    and mouselocstr[1] <= self.imgResizeHeight
                ):
                    if self.OriginbtnFlag == 0:
                        self.mouseloctext.setText("擷取座標：" + str(mouselocstr))
                        img = Image.open(self.currentImage)
                        CenterPoint = self.OriginX * 2.5, self.OriginY * 2.5
                        ClickPoint = (
                            2.5 * self.currentPosition[0],
                            2.5 * self.currentPosition[1],
                        )
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
                        self.OriginX = self.currentPosition[0]
                        self.OriginY = self.currentPosition[1]
                        self.Ox_txtEdit.setText(str(self.OriginX))
                        self.Oy_txtEdit.setText(str(self.OriginY))
                        img = Image.open(self.currentImage)
                        CenterPoint = self.OriginX * 2.5, self.OriginY * 2.5
                        draw = ImageDraw.Draw(img)
                        draw.ellipse((  CenterPoint[0] - 10,
                                        CenterPoint[1] - 10,
                                        CenterPoint[0] + 10,
                                        CenterPoint[1] + 10,
                                    ),  fill=(255))
                        qim = ImageQt(img)
                        result = qim.scaled(
                            self.imgSizelabel.width(),
                            self.imgSizelabel.height(),
                            Qt.IgnoreAspectRatio,
                            Qt.SmoothTransformation,
                        )
                        self.imgBlock.setPixmap(QPixmap.fromImage(result))
                        QPixmapCache.clear
                        self.setOrigbtn.toggle()
                else:
                    self.mouseloctext.setText("擷取座標：")
                self.OriginbtnFlag = 0
                self.setCursor(QCursor(Qt.ArrowCursor))
                self.imgBlock.setToolTip('請在想擷取的區塊進行點擊')
                self.imgBlock.setToolTipDuration(1800)


if __name__ == "__main__":

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = project()
    window.show()
    sys.exit(app.exec_())

import sys

import numpy as np
import os
import math
import shutil
import cv2
from PyQt5 import QtGui, QtCore, QtWidgets
import random
from PyQt5.QtWidgets import QWidget, QApplication, QMainWindow, qApp, QAction


class MainWindow:
    def __init__(self):
        self.app = QtWidgets.QApplication(sys.argv)

        self.window = QtWidgets.QMainWindow()
        self.face_cascade = cv2.CascadeClassifier(
            '/home/ily19/PycharmProjects/haarcascadexmlneeded/haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier('/home/ily19/PycharmProjects/haarcascadexmlneeded/haarcascade_eye.xml')
        self.pathImg = "/home/ily19/PycharmProjects/imgProcessing/img"

        # Apply CSS
        self.stylesheet = """

                QMenuBar {
                    background-color:#2e2e2e;
                    color: #e0e0e0;

                }

                QMenuBar::item {
                    background: transparent;
                }

                QMenuBar::item:selected {
                    background-color:#ffffff;
                    color:#2e2e2e
                }

                QMenu{
                    background-color:#2e2e2e;
                    color:#ffffff;
                }

                QMenu::item:selected{
                    background-color:#ffffff;
                    color:#2e2e2e
                }

                QCheckBox {
                    spacing: 0px;
                }

                 QCheckBox::indicator {
                     width: 30px;
                     height: 30px;
                 }

                 QCheckBox::indicator:unchecked {
                     background-color:#ffffff;
                 }

                 QCheckBox::indicator:checked {
                     image: url(images/checkIcon.png);
                 }


                """

        self.initUi()
        self.app.setStyleSheet(self.stylesheet)
        self.window.showMaximized()
        self.window.setWindowTitle("Face Project")
        self.window.show()
        sys.exit(self.app.exec_())

    # define The GUI
    def initUi(self):

        # create the menu bar
        self.menubar = self.window.menuBar()

        # create the root menus
        self.FileMenu = self.menubar.addMenu('File')
        self.FaceDetectionMenu = self.menubar.addMenu('Face Detection')
        self.EyeDetectionMenu = self.menubar.addMenu('Eye Detection and Tracking')
        self.FaceAveraging = self.menubar.addMenu('Face Averaging')
        self.FaceSwap = self.menubar.addMenu('Face Swap')
        self.FaceMorph = self.menubar.addMenu('Face Morphing')
        self.others = self.menubar.addMenu('Others')

        # create the subMenu
        self.SaveImage = QAction('Save', self.window)
        self.SaveImage.setShortcut('CTRL+S')

        self.QuitApp = QAction("Quit", self.window)
        self.QuitApp.setShortcut("CTRL+Q")
        self.QuitApp.triggered.connect(self.quitApp)

        self.FaceDetectionInImageAction = QAction('Face Detection In Image', self.window)
        self.FaceDetectionInImageAction.triggered.connect(self.mainWidgetFaceDetectionInImage_Clicked)

        self.FaceDetectionRealTimeAction = QtWidgets.QAction('Real Time Face Detection', self.window)
        self.FaceDetectionRealTimeAction.triggered.connect(self.mainWidgetFaceDetectionInRealTimeBtn_clicked)

        self.EyeDetectionInImageAction = QtWidgets.QAction('Eye Detection In Image', self.window)
        self.EyeDetectionInImageAction.triggered.connect(self.mainWidgetEyeDetectionInImage_clicked)

        self.EyeDetectionRealTimeAction = QtWidgets.QAction('Real Time Eye Detection', self.window)
        self.EyeDetectionRealTimeAction.triggered.connect(self.mainWidgetEyeDetectionInRealTimeBtn_clicked)

        self.eyeTrackingRealTimeAction = QtWidgets.QAction('Real Time Eye Tracking', self.window)
        self.eyeTrackingRealTimeAction.triggered.connect(self.mainWidgetEyeTrackingImageBtn_clicked)

        self.faceAveragingImages = QtWidgets.QAction('Face Average In Images', self.window)
        self.faceAveragingImages.triggered.connect(self.mainWidgetFaceAveragingImagesBtn_clicked)

        self.faceAveragingRealTime = QtWidgets.QAction('Real Time Face Averaging', self.window)
        # self.faceAveragingRealTime.triggered.connect()

        self.FaceSwapInImages = QtWidgets.QAction('Face Swap In Images', self.window)
        self.FaceSwapInImages.triggered.connect(self.mainWidgetFaceSwapImagesBtn_clicked)

        self.FaceMorphingInImages = QtWidgets.QAction('Face Morphing', self.window)
        self.FaceMorphingInImages.triggered.connect(self.mainWidgetFaceMorphImagesBtn_clicked)

        self.Delaunary = QtWidgets.QAction('Delaunary', self.window)
        self.Delaunary.triggered.connect(self.mainWidgetDelaunaryImagesBtn_clicked)

        self.captureMotion = QtWidgets.QAction('Motion capture', self.window)
        self.captureMotion.triggered.connect(self.mainWidgetMotionImagesBtn_clicked)

        self.eyeTracker = QtWidgets.QAction('Eye Tracking', self.window)
        self.eyeTracker.triggered.connect(self.mainWidgetEyeTrackingImageBtn_clicked)

        # add the action to the menu
        self.FileMenu.addAction(self.SaveImage)
        self.FileMenu.addAction(self.QuitApp)

        self.FaceDetectionMenu.addAction(self.FaceDetectionInImageAction)
        self.FaceDetectionMenu.addAction(self.FaceDetectionRealTimeAction)

        self.EyeDetectionMenu.addAction(self.EyeDetectionInImageAction)
        self.EyeDetectionMenu.addAction(self.EyeDetectionRealTimeAction)
        self.EyeDetectionMenu.addAction(self.eyeTrackingRealTimeAction)

        self.FaceAveraging.addAction(self.faceAveragingImages)
        self.FaceAveraging.addAction(self.faceAveragingRealTime)

        self.FaceSwap.addAction(self.FaceSwapInImages)

        self.FaceMorph.addAction(self.FaceMorphingInImages)

        self.others.addAction(self.Delaunary)
        self.others.addAction(self.captureMotion)
        self.others.addAction(self.eyeTracker)
        # ----------------------------------------------------------------------------------------------------------------------------------
        # create the label to display the image
        self.label = QtWidgets.QLabel(self.window)
        self.label.setGeometry(0, 67, 1070, 550)
        # self.label.setStyleSheet("background-color: #ffffff")
        self.label.setStyleSheet("border: 1px solid rgb(77,77,77); background-color: #1e1e1e")

        # create a label to display swapped images
        self.swapImageLabel = QtWidgets.QLabel(self.window)
        self.swapImageLabel.setGeometry(235, 67, 600, 550)
        self.swapImageLabel.setStyleSheet("background-color:#ff9900")
        self.swapImageLabel.setVisible(False)

        # create label to display delaunary images
        self.originalDelaunaryImageLabel = QtWidgets.QLabel(self.window)
        self.originalDelaunaryImageLabel.setGeometry(3, 67, 528, 550)
        self.originalDelaunaryImageLabel.setStyleSheet("background-color:#ff9900")
        self.originalDelaunaryImageLabel.setVisible(False)

        self.resulltDelaunaryImageLabel = QtWidgets.QLabel(self.window)
        self.resulltDelaunaryImageLabel.setGeometry(538, 67, 528, 550)
        self.resulltDelaunaryImageLabel.setStyleSheet("background-color:#ff9900")
        self.resulltDelaunaryImageLabel.setVisible(False)

        self.displayVideoLabel = QtWidgets.QLabel(self.window)
        self.displayVideoLabel.setGeometry(235, 67, 600, 550)
        self.displayVideoLabel.setStyleSheet("background-color:#f44336; color:#e2e2e2;")
        self.displayVideoLabel.setVisible(False)

        self.buttonToDestroyCamera = QtWidgets.QPushButton("Stop Camera", self.displayVideoLabel)
        self.buttonToDestroyCamera.setGeometry(500, 500, 90, 30)
        self.buttonToDestroyCamera.setStyleSheet("border:1px solid #f44336")
        self.buttonToDestroyCamera.clicked.connect(self.hideDisplayedVideoLabel)

        # create the right panel to display the widgets
        self.rightPanel = QtWidgets.QLabel(self.window)
        self.rightPanel.setFrameStyle(QtWidgets.QFrame.Panel)
        self.rightPanel.setGeometry(1076, 67, 288, 550)
        self.rightPanel.setStyleSheet("background-color: #ffffff;")

        # create the toolBar panel
        self.toolBarPanel = QtWidgets.QLabel(self.window)
        self.toolBarPanel.setFrameStyle(QtWidgets.QFrame.Panel)
        self.toolBarPanel.setGeometry(0, 28, 1366, 38)
        self.toolBarPanel.setStyleSheet("background-color: #2e2e2e;border:1px solid #2e2e2e")

        # what will be on toolBar
        self.searchField = QtWidgets.QTextEdit(self.toolBarPanel)
        self.searchField.setGeometry(550, 3, 400, 32)
        self.searchField.setStyleSheet(
            "border: 1px solid #f44336; background-color:#e2e2e2; color:#1e1e1e; font-size: 15px")

        self.searchBtn = QtWidgets.QPushButton('Search', self.toolBarPanel)
        self.searchBtn.setGeometry(440, 3, 100, 32)
        self.searchBtn.setStyleSheet("background-color:#f44336; color:#e2e2e2")
        self.searchBtn.setIconSize(QtCore.QSize(30, 30))
        self.searchBtn.setIcon(QtGui.QIcon("images/search.png"))

        self.logoText = QtWidgets.QPushButton("FaceTool", self.toolBarPanel)
        self.logoText.setGeometry(2, 1, 125, 36)
        self.logoText.setStyleSheet(
            "background-color: #2e2e2e; border: 1px solid #2e2e2e; color: #e2e2e2; font-size: 17px")
        self.logoText.setIconSize(QtCore.QSize(30, 30))
        self.logoText.setIcon(QtGui.QIcon("images/logo2.png"))

        self.registerBtn = QtWidgets.QPushButton('Create an Account', self.toolBarPanel)
        self.registerBtn.setGeometry(1100, 3, 150, 32)
        self.registerBtn.setStyleSheet("background-color:#e2e2e2; color:#1e1e1e")
        self.registerBtn.setIconSize(QtCore.QSize(30, 30))
        self.registerBtn.setIcon(QtGui.QIcon("images/web_hi_res_5127.png"))

        self.loginBtn = QtWidgets.QPushButton('Log In', self.toolBarPanel)
        self.loginBtn.setGeometry(1265, 3, 100, 32)
        self.loginBtn.setStyleSheet("background-color:#e2e2e2; color:#1e1e1e")
        self.loginBtn.setIconSize(QtCore.QSize(30, 30))
        self.loginBtn.setIcon(QtGui.QIcon("images/web_hi_res_5127.png"))

        # create widgets for the right panel
        self.mainWidget = QtWidgets.QWidget(self.rightPanel)
        self.mainWidget.resize(288, 550)
        self.mainWidget.setStyleSheet("background-color: #2e2e2e; color:#ffffff")
        self.mainWidget.setVisible(True)
        # create the widget panel associated to 'mainWidgetFaceDetectionInImage' button
        self.FaceDetectionInImage_Widget = QtWidgets.QWidget(self.rightPanel)
        self.FaceDetectionInImage_Widget.resize(288, 550)
        self.FaceDetectionInImage_Widget.setStyleSheet("background-color: #2e2e2e; color:#ffffff")
        self.FaceDetectionInImage_Widget.setVisible(False)

        # create the widget panel associated to 'mainWidgetEyeDetectionInImageBtn' button
        self.EyeDetectionInImage_Widget = QtWidgets.QWidget(self.rightPanel)
        self.EyeDetectionInImage_Widget.resize(288, 550)
        self.EyeDetectionInImage_Widget.setStyleSheet("background-color: #2e2e2e; color:#ffffff")
        self.EyeDetectionInImage_Widget.setVisible(False)

        # create the widget panel associated to 'mainWidgetFaceAveragingImagesBtn' button
        self.FaceAveragingImages_Widget = QtWidgets.QWidget(self.rightPanel)
        self.FaceAveragingImages_Widget.resize(288, 550)
        self.FaceAveragingImages_Widget.setStyleSheet("background-color: #2e2e2e; color:#ffffff")
        self.FaceAveragingImages_Widget.setVisible(False)

        # create a label that contains the widget with all the images used in this project
        self.imageSectionLabel = QtWidgets.QLabel(self.window)
        self.imageSectionLabel.setGeometry(0, 618, 1366, 98)
        self.imageSectionLabel.setStyleSheet("background-color:#2e2e2e")

        # create the widget that contains all the images
        self.imageSectionLabelWidget = QtWidgets.QWidget(self.imageSectionLabel)
        self.imageSectionLabelWidget.resize(1366, 98)
        self.imageSectionLabelWidget.setStyleSheet("background-color:#2e2e2e")

        # create the widget panel associated to 'mainWidgetFaceSwapImagesBtn' button
        self.FaceSwapImages_Widget = QtWidgets.QWidget(self.rightPanel)
        self.FaceSwapImages_Widget.resize(288, 550)
        self.FaceSwapImages_Widget.setStyleSheet("background-color: #2e2e2e; color:#ffffff")
        self.FaceSwapImages_Widget.setVisible(False)

        # create the widget 2 for mainWidgetImagesBtn for swap function
        self.FaceSwapImages_Widget2 = QtWidgets.QWidget(self.rightPanel)
        self.FaceSwapImages_Widget2.resize(288, 550)
        self.FaceSwapImages_Widget2.setStyleSheet("background-color: #2e2e2e; color:#ffffff")
        self.FaceSwapImages_Widget2.setVisible(False)

        # create the widget panel associated to 'mainWidgetFaceMorphImagesBtn' button
        self.FaceMorphImages_Widget = QtWidgets.QWidget(self.rightPanel)
        self.FaceMorphImages_Widget.resize(288, 550)
        self.FaceMorphImages_Widget.setStyleSheet("background-color: #2e2e2e; color:#ffffff")
        self.FaceMorphImages_Widget.setVisible(False)

        # create the widget 2 for mainWidgetImagesBtn for face morphing
        self.FaceMorphImages_Widget2 = QtWidgets.QWidget(self.rightPanel)
        self.FaceMorphImages_Widget2.resize(288, 550)
        self.FaceMorphImages_Widget2.setStyleSheet("background-color: #2e2e2e; color:#ffffff")
        self.FaceMorphImages_Widget2.setVisible(False)

        # create the widget for Delaunary images
        self.DelaunaryImages_Widget = QtWidgets.QWidget(self.rightPanel)
        self.DelaunaryImages_Widget.resize(288, 550)
        self.DelaunaryImages_Widget.setStyleSheet("background-color: #2e2e2e; color:#ffffff")
        self.DelaunaryImages_Widget.setVisible(False)

        # associate buttons to the right mainWidget
        self.CreateMainWidgetButtons()
        # associate buttons to the FaceDetectionInImage_Widget
        self.createFaceDetectionInImageWidgetButtons()
        # associate buttons to the EyeDetectionInImage_Widget
        self.createEyeDetectionInImageWidgetButtons()
        # associate buttons to the FaceAveragingImages_widget
        self.createFaceAveragingImagesWidgetButtons()
        # associate buttons to the FaceSwapImages_Widget
        self.createFaceSwapImagesWidgetButtons()
        # associate buttons to the FaceMorphImages_widget
        self.createFaceMorphImagesWidgetButtons()
        # associate buttons to the Delaunary widget
        self.createDelaunaryImagesWidgetButtons()
        # associate the images section
        self.displayAllTheImagesUsed()

    # create a slot for 'loadImageBtn_FaceDetectionInImage_Widget' button

    def loadImageBtn_FaceDetectionInImage_Widget_clicked(self):
        self.filename, _ = QtWidgets.QFileDialog().getOpenFileName(None, "Load an image ", "/home/ily19/Desktop/",
                                                                   "Image Files(*.jpg *.png *.JPG *.jpeg)")

        self.image = cv2.imread(self.filename)
        self.displayImage()

    # create a slot for 'mainWidgetEyeCaptureimagesBtn' clicked
    def mainWidgetEyeTrackingImageBtn_clicked(self):
        self.eyeTracking()

    # create a slot for 'mainWidgetMotionImagesBtn' clicked
    def mainWidgetMotionImagesBtn_clicked(self):
        self.motionCapture()

    # create a slot for 'mainWidget_DelaunaryImagesbtn'
    def mainWidgetDelaunaryImagesBtn_clicked(self):
        self.mainWidget.setVisible(False)
        self.DelaunaryImages_Widget.setVisible(True)

    # create a slot for 'cancelBtn_DelaunaryImages_widget' button
    def cancelBtn_DelaunaryImages_Widget_clicked(self):
        self.mainWidget.setVisible(True)
        self.DelaunaryImages_Widget.setVisible(False)
        self.originalDelaunaryImageLabel.setVisible(False)
        self.resulltDelaunaryImageLabel.setVisible(False)

    # create a slot for 'DelaunaryBtn_DelaunaryImages_widget_widget_clicked' button
    def DelaunaryBtn_DelaunaryImages_widget_widget_clicked(self):
        self.displayDelaunary()

    # create a slot for 'mainWidgetFaceMorphImagesBtn_clicked' button
    def mainWidgetFaceMorphImagesBtn_clicked(self):
        self.mainWidget.setVisible(False)
        self.FaceMorphImages_Widget.setVisible(True)
        self.FaceMorphImages_Widget2.setVisible(True)

    # create a slot for 'FaceMorphBtn_FaceMorphImages_Widget' button
    def FaceMorphBtn_FaceMorphImages_widget_clicked(self):
        self.displayMorphImages()

    # create a slot for 'cancelBtn_FaceMorphImages_WidgetBtn' button
    def cancelBtn_FaceMorphImages_Widget_clicked(self):
        self.mainWidget.setVisible(True)
        self.FaceMorphImages_Widget.setVisible(False)
        self.FaceMorphImages_Widget2.setVisible(False)
        self.swapImageLabel.setVisible(False)

    # create a slot for 'mainWidgetFaceSwapImagesBtn_clicked' button
    def mainWidgetFaceSwapImagesBtn_clicked(self):
        self.mainWidget.setVisible(False)
        self.FaceSwapImages_Widget.setVisible(True)
        self.FaceSwapImages_Widget2.setVisible(True)

    # create a slot for 'FaceSwapBtn_FaceSwapImages_Widget' button
    def FaceSwapBtn_FaceSwapImages_widget_clicked(self):
        self.displaySwappedFaces()

    # create a slot for 'cancelBtn_FaceSwapImages_WidgetBtn' button
    def cancelBtn_FaceSwapImages_Widget_clicked(self):
        self.mainWidget.setVisible(True)
        self.FaceSwapImages_Widget.setVisible(False)
        self.FaceSwapImages_Widget2.setVisible(False)
        self.swapImageLabel.setVisible(False)

    # creata a slot for 'FaceAverageBtn_FaceAveragingImages_widget' button
    def FaceAverageBtn_FaceAveragingImages_widget_clicked(self):
        self.swapImageLabel.setVisible(False)
        self.displayAveragedFaces()

    # create a slot for 'cancelBtn_FaceAveragingImages_WidgetBtn' button
    def cancelBtn_FaceAveragingImages_Widget_clicked(self):
        self.mainWidget.setVisible(True)
        self.FaceAveragingImages_Widget.setVisible(False)

    # create a slot for the Quit action
    def quitApp(self):
        qApp.quit()

    # create a slot for 'mainWidgetFaceAveragingImagesBtn_clicked' button
    def mainWidgetFaceAveragingImagesBtn_clicked(self):
        self.mainWidget.setVisible(False)
        self.FaceAveragingImages_Widget.setVisible(True)

    # create a slot for the 'mainWidgetEyeDetectionInRealTimeBtn' button
    def mainWidgetEyeDetectionInRealTimeBtn_clicked(self):
        self.realTimeEyeDetection()

    # create a slot for the 'findEyesBtn_EyeDetectionInImage_Widget' button
    def findEyesBtn_EyeDetectionInImage_Widget_clicked(self):
        self.findEeyes()

    # create a slot for 'mainWidgetEyeDetectionInImageBtn' button when it's clicked
    def mainWidgetEyeDetectionInImage_clicked(self):
        self.mainWidget.setVisible(False)
        self.EyeDetectionInImage_Widget.setVisible(True)

    # create a slot for 'mainWidgetFaceDetectionInRealTimeBtn' button
    def mainWidgetFaceDetectionInRealTimeBtn_clicked(self):
        self.realTimeFaceDetection()

    # create a slot for 'findFacesBtn_FaceDetectionInImage_Widget' button
    def findFacesBtn_FaceDetectionInImage_Widget_clicked(self):
        self.findFaces()

    # create a slot for 'cancelBtn_FaceDetectionInImage_Widget' button
    def cancelBtn_FaceDetectionInImage_Widget_clicked(self):
        self.mainWidget.setVisible(True)
        self.FaceDetectionInImage_Widget.setVisible(False)

    # create a slot for 'cancelBtn_EyeDetectionInImage_Widget' button
    def cancelBtn_EyeDetectionInImage_Widget_clicked(self):
        self.mainWidget.setVisible(True)
        self.EyeDetectionInImage_Widget.setVisible(False)

    # create a slot for 'mainWidgetFaceDetectionInImageBtn' when it is clicked
    def mainWidgetFaceDetectionInImage_Clicked(self):
        self.mainWidget.setVisible(False)
        self.FaceDetectionInImage_Widget.setVisible(True)

    # ----------------------------------------------------------------------Motion Capture function ---------------------------------------------------------
    def motionCapture(self):

        self.capture = cv2.VideoCapture(0)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 400)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 300)

        self.timer = QtCore.QTimer(self.window)
        self.timer.timeout.connect(self.updateMFrame)
        self.timer.start(5)

    def diffImg(self, t0, t1, t2):
        d1 = cv2.absdiff(t2, t1)
        d2 = cv2.absdiff(t1, t0)
        return cv2.bitwise_and(d1, d2)

    def updateMFrame(self):
        # Read three images first:
        t_minus = cv2.cvtColor(self.capture.read()[1], cv2.COLOR_RGB2GRAY)
        t = cv2.cvtColor(self.capture.read()[1], cv2.COLOR_RGB2GRAY)
        t_plus = cv2.cvtColor(self.capture.read()[1], cv2.COLOR_RGB2GRAY)

        self.displayRealTimeInQlabel(self.diffImg(t_minus, t, t_plus), 1)

        # Read next image
        t_minus = t
        t = t_plus
        t_plus = cv2.cvtColor(self.capture.read()[1], cv2.COLOR_RGB2GRAY)

    # -------------------------------------------------------eye tracking--------------------------------------------------------------------------------
    def eyeTracking(self):
        self.capture = cv2.VideoCapture(0)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 400)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 300)

        self.timer = QtCore.QTimer(self.window)
        self.timer.timeout.connect(self.updateEyeTrackingFrame)
        self.timer.start(5)

    def updateEyeTrackingFrame(self):
        MIN_FACE_SIZE = 100
        MAX_FACE_SIZE = 300

        ret, frameBig = self.capture.read()

        # If frame opened successfully
        if ret == True:

            # Fixing the scaling factor
            scale = 640.0 / frameBig.shape[1]

            # Resizing the image
            frame = cv2.resize(frameBig, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

            # Converting to grayscale
            frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect faces
            faces = self.face_cascade.detectMultiScale(frameGray, scaleFactor=1.1, minNeighbors=5, flags=0,
                                                       minSize=(MIN_FACE_SIZE, MIN_FACE_SIZE),
                                                       maxSize=(MAX_FACE_SIZE, MAX_FACE_SIZE))

            # Loop over each detected face
            for i in xrange(0, len(faces)):
                # Dimension parameters for bounding rectangle for face
                x, y, width, height = faces[i];

                # Calculating the dimension parameters for eyes from the dimensions parameters of the face
                ex, ey, ewidth, eheight = int(x + 0.125 * width), int(y + 0.25 * height), int(0.75 * width), int(
                    0.25 * height)

                # Drawing the bounding rectangle around the face
                cv2.rectangle(frame, (ex, ey), (ex + ewidth, ey + eheight), (0, 0, 255), 2)

                # Display the resulting frame
            self.displayRealTimeInQlabel(frame, 1)

    # -----------------------------------------------------display all the images used in this project-------------------------------------------------------------
    def displayAllTheImagesUsed(self):
        self.image1PathS = "/home/ily19/Desktop/images/5.jpg"
        self.image2PathS = "/home/ily19/Desktop/images/8.jpg"
        self.image3PathS = "/home/ily19/Desktop/images/15.jpg"
        self.image4PathS = "/home/ily19/Desktop/images/12.jpg"
        self.image5PathS = "/home/ily19/Desktop/images/16.jpg"
        self.image6PathS = "/home/ily19/Desktop/images/2.jpg"
        self.image7PathS = "/home/ily19/Desktop/images/9.jpg"
        self.image8PathS = "/home/ily19/Desktop/images/7.jpg"
        self.image9PathS = "/home/ily19/Desktop/images/1.jpg"
        self.image10PathS = "/home/ily19/Desktop/images/3.jpg"
        self.image11PathS = "/home/ily19/Desktop/images/4.jpg"
        self.image12PathS = "/home/ily19/Desktop/images/10.jpg"
        self.image13PathS = "/home/ily19/Desktop/images/11.jpg"
        self.image14PathS = "/home/ily19/Desktop/images/17.jpg"
        self.image15PathS = "/home/ily19/Desktop/images/14.jpg"
        self.image16PathS = "/home/ily19/Desktop/images/13.jpg"
        self.image17PathS = "/home/ily19/Desktop/images/18.jpg"

        # create the widget that contains the scrollArea
        self.widgetContainigScrollAreaS = QtWidgets.QWidget(self.imageSectionLabelWidget)
        self.widgetContainigScrollAreaS.resize(1736, 82)
        self.widgetContainigScrollAreaS.setStyleSheet("background-color:#2e2e2e")

        # create the scollArea
        self.scrollAreaS = QtWidgets.QScrollArea(self.imageSectionLabelWidget)
        self.scrollAreaS.setWidget(self.widgetContainigScrollAreaS)
        self.scrollAreaS.setFixedWidth(1366)
        self.scrollAreaS.setFixedHeight(98)
        # self.scrollAreaS.setStyleSheet("border-top: 1px solid #4e4e4e; border-left:1px solid #4e4e4e;border-right:1px solid #4e4e4e")

        # create the labels
        self.labelimage1 = QtWidgets.QLabel(self.widgetContainigScrollAreaS)
        self.labelimage1.setGeometry(2, 0, 100, 82)
        self.labelimage1.setStyleSheet("background-color: #f44336")
        self.pixmap1 = QtGui.QPixmap(self.image1PathS)
        self.labelimage1.setScaledContents(True)
        self.labelimage1.setPixmap(self.pixmap1)

        self.labelimage2 = QtWidgets.QLabel(self.widgetContainigScrollAreaS)
        self.labelimage2.setGeometry(104, 0, 100, 82)
        self.labelimage2.setStyleSheet("background-color: #f44336")
        self.pixmap2 = QtGui.QPixmap(self.image2PathS)
        self.labelimage2.setScaledContents(True)
        self.labelimage2.setPixmap(self.pixmap2)

        self.labelimage3 = QtWidgets.QLabel(self.widgetContainigScrollAreaS)
        self.labelimage3.setGeometry(206, 0, 100, 82)
        self.labelimage3.setStyleSheet("background-color: #f44336")
        self.pixmap3 = QtGui.QPixmap(self.image3PathS)
        self.labelimage3.setScaledContents(True)
        self.labelimage3.setPixmap(self.pixmap3)

        self.labelimage4 = QtWidgets.QLabel(self.widgetContainigScrollAreaS)
        self.labelimage4.setGeometry(308, 0, 100, 82)
        self.labelimage4.setStyleSheet("background-color: #f44336")
        self.pixmap4 = QtGui.QPixmap(self.image4PathS)
        self.labelimage4.setScaledContents(True)
        self.labelimage4.setPixmap(self.pixmap4)

        self.labelimage5 = QtWidgets.QLabel(self.widgetContainigScrollAreaS)
        self.labelimage5.setGeometry(410, 0, 100, 82)
        self.labelimage5.setStyleSheet("background-color: #f44336")
        self.pixmap5 = QtGui.QPixmap(self.image5PathS)
        self.labelimage5.setScaledContents(True)
        self.labelimage5.setPixmap(self.pixmap5)

        self.labelimage6 = QtWidgets.QLabel(self.widgetContainigScrollAreaS)
        self.labelimage6.setGeometry(512, 0, 100, 82)
        self.labelimage6.setStyleSheet("background-color: #f44336")
        self.pixmap6 = QtGui.QPixmap(self.image6PathS)
        self.labelimage6.setScaledContents(True)
        self.labelimage6.setPixmap(self.pixmap6)

        self.labelimage7 = QtWidgets.QLabel(self.widgetContainigScrollAreaS)
        self.labelimage7.setGeometry(614, 0, 100, 82)
        self.labelimage7.setStyleSheet("background-color: #f44336")
        self.pixmap7 = QtGui.QPixmap(self.image7PathS)
        self.labelimage7.setScaledContents(True)
        self.labelimage7.setPixmap(self.pixmap7)

        self.labelimage8 = QtWidgets.QLabel(self.widgetContainigScrollAreaS)
        self.labelimage8.setGeometry(716, 0, 100, 82)
        self.labelimage8.setStyleSheet("background-color: #f44336")
        self.pixmap8 = QtGui.QPixmap(self.image8PathS)
        self.labelimage8.setScaledContents(True)
        self.labelimage8.setPixmap(self.pixmap8)

        self.labelimage9 = QtWidgets.QLabel(self.widgetContainigScrollAreaS)
        self.labelimage9.setGeometry(818, 0, 100, 82)
        self.labelimage9.setStyleSheet("background-color: #f44336")
        self.pixmap9 = QtGui.QPixmap(self.image9PathS)
        self.labelimage9.setScaledContents(True)
        self.labelimage9.setPixmap(self.pixmap9)

        self.labelimage10 = QtWidgets.QLabel(self.widgetContainigScrollAreaS)
        self.labelimage10.setGeometry(920, 0, 100, 82)
        self.labelimage10.setStyleSheet("background-color: #f44336")
        self.pixmap10 = QtGui.QPixmap(self.image10PathS)
        self.labelimage10.setScaledContents(True)
        self.labelimage10.setPixmap(self.pixmap10)

        self.labelimage11 = QtWidgets.QLabel(self.widgetContainigScrollAreaS)
        self.labelimage11.setGeometry(1022, 0, 100, 82)
        self.labelimage11.setStyleSheet("background-color: #f44336")
        self.pixmap11 = QtGui.QPixmap(self.image11PathS)
        self.labelimage11.setScaledContents(True)
        self.labelimage11.setPixmap(self.pixmap11)

        self.labelimage12 = QtWidgets.QLabel(self.widgetContainigScrollAreaS)
        self.labelimage12.setGeometry(1124, 0, 100, 82)
        self.labelimage12.setStyleSheet("background-color: #f44336")
        self.pixmap12 = QtGui.QPixmap(self.image12PathS)
        self.labelimage12.setScaledContents(True)
        self.labelimage12.setPixmap(self.pixmap12)

        self.labelimage13 = QtWidgets.QLabel(self.widgetContainigScrollAreaS)
        self.labelimage13.setGeometry(1226, 0, 100, 82)
        self.labelimage13.setStyleSheet("background-color: #f44336")
        self.pixmap13 = QtGui.QPixmap(self.image13PathS)
        self.labelimage13.setScaledContents(True)
        self.labelimage13.setPixmap(self.pixmap13)

        self.labelimage14 = QtWidgets.QLabel(self.widgetContainigScrollAreaS)
        self.labelimage14.setGeometry(1328, 0, 100, 82)
        self.labelimage14.setStyleSheet("background-color: #f44336")
        self.pixmap14 = QtGui.QPixmap(self.image14PathS)
        self.labelimage14.setScaledContents(True)
        self.labelimage14.setPixmap(self.pixmap14)

        self.labelimage15 = QtWidgets.QLabel(self.widgetContainigScrollAreaS)
        self.labelimage15.setGeometry(1430, 0, 100, 82)
        self.labelimage15.setStyleSheet("background-color: #f44336")
        self.pixmap15 = QtGui.QPixmap(self.image15PathS)
        self.labelimage15.setScaledContents(True)
        self.labelimage15.setPixmap(self.pixmap15)

        self.labelimage16 = QtWidgets.QLabel(self.widgetContainigScrollAreaS)
        self.labelimage16.setGeometry(1532, 0, 100, 82)
        self.labelimage16.setStyleSheet("background-color: #f44336")
        self.pixmap16 = QtGui.QPixmap(self.image16PathS)
        self.labelimage16.setScaledContents(True)
        self.labelimage16.setPixmap(self.pixmap16)

        self.labelimage17 = QtWidgets.QLabel(self.widgetContainigScrollAreaS)
        self.labelimage17.setGeometry(1634, 0, 100, 82)
        self.labelimage17.setStyleSheet("background-color: #f44336")
        self.pixmap17 = QtGui.QPixmap(self.image17PathS)
        self.labelimage17.setScaledContents(True)
        self.labelimage17.setPixmap(self.pixmap17)

    # ---------------------------------------------------Delaunary image function-------------------------------------------------------------------------
    # create the definition for the delaunary images
    def createDelaunaryImagesWidgetButtons(self):
        # images to be placed in the labels below
        self.image1PathD = "/home/ily19/Desktop/images/5.jpg"
        self.image2PathD = "/home/ily19/Desktop/images/8.jpg"
        self.image3PathD = "/home/ily19/Desktop/images/15.jpg"
        self.image4PathD = "/home/ily19/Desktop/images/12.jpg"
        self.image5PathD = "/home/ily19/Desktop/images/16.jpg"
        self.image6PathD = "/home/ily19/Desktop/images/2.jpg"
        self.image7PathD = "/home/ily19/Desktop/images/9.jpg"
        self.image8PathD = "/home/ily19/Desktop/images/7.jpg"

        self.cancelBtn_DelaunaryImages_widget = QtWidgets.QPushButton('Cancel', self.DelaunaryImages_Widget)
        self.cancelBtn_DelaunaryImages_widget.setGeometry(10, 500, 125, 30)
        self.cancelBtn_DelaunaryImages_widget.clicked.connect(self.cancelBtn_DelaunaryImages_Widget_clicked)

        self.DelaunaryBtn_DelaunaryImages_widget = QtWidgets.QPushButton('Average Faces', self.DelaunaryImages_Widget)
        self.DelaunaryBtn_DelaunaryImages_widget.setGeometry(155, 500, 125, 30)
        self.DelaunaryBtn_DelaunaryImages_widget.clicked.connect(
            self.DelaunaryBtn_DelaunaryImages_widget_widget_clicked)

        # create the widget that contains the scrollArea
        self.widgetContainigScrollArea_DelaunaryImages = QtWidgets.QWidget(self.DelaunaryImages_Widget)
        self.widgetContainigScrollArea_DelaunaryImages.resize(272, 550)
        self.widgetContainigScrollArea_DelaunaryImages.setStyleSheet("background-color:#ffffff")

        # create the scollArea
        self.scrollArea = QtWidgets.QScrollArea(self.DelaunaryImages_Widget)
        self.scrollArea.setWidget(self.widgetContainigScrollArea_DelaunaryImages)
        self.scrollArea.setFixedWidth(288)
        self.scrollArea.setFixedHeight(460)
        self.scrollArea.setStyleSheet(
            "border-top: 1px solid #2e2e2e; border-left:1px solid #2e2e2e;border-right:1px solid #2e2e2e")

        # create the labels that contains all the images to average at the end
        self.label1_image1D = QtWidgets.QLabel(self.widgetContainigScrollArea_DelaunaryImages)
        self.label1_image1D.setGeometry(0, 0, 134, 134)
        self.label1_image1D.setStyleSheet("background-color: #f44336")
        self.pixmap1D = QtGui.QPixmap(self.image1PathD)
        self.label1_image1D.setScaledContents(True)
        self.label1_image1D.setPixmap(self.pixmap1D)

        self.label2_image2D = QtWidgets.QLabel(self.widgetContainigScrollArea_DelaunaryImages)
        self.label2_image2D.setGeometry(137, 0, 134, 134)
        self.label2_image2D.setStyleSheet("background-color: #f44336")
        self.pixmap2D = QtGui.QPixmap(self.image2PathD)
        self.label2_image2D.setScaledContents(True)
        self.label2_image2D.setPixmap(self.pixmap2D)

        self.label3_image3D = QtWidgets.QLabel(self.widgetContainigScrollArea_DelaunaryImages)
        self.label3_image3D.setGeometry(0, 137, 134, 134)
        self.label3_image3D.setStyleSheet("background-color: #f44336")
        self.pixmap3D = QtGui.QPixmap(self.image3PathD)
        self.label3_image3D.setScaledContents(True)
        self.label3_image3D.setPixmap(self.pixmap3D)

        self.label4_image4D = QtWidgets.QLabel(self.widgetContainigScrollArea_DelaunaryImages)
        self.label4_image4D.setGeometry(137, 137, 134, 134)
        self.label4_image4D.setStyleSheet("background-color: #f44336")
        self.pixmap4D = QtGui.QPixmap(self.image4PathD)
        self.label4_image4D.setScaledContents(True)
        self.label4_image4D.setPixmap(self.pixmap4D)

        self.label5_image5D = QtWidgets.QLabel(self.widgetContainigScrollArea_DelaunaryImages)
        self.label5_image5D.setGeometry(0, 274, 134, 134)
        self.label5_image5D.setStyleSheet("background-color: #f44336")
        self.pixmap5D = QtGui.QPixmap(self.image5PathD)
        self.label5_image5D.setScaledContents(True)
        self.label5_image5D.setPixmap(self.pixmap5D)

        self.label6_image6D = QtWidgets.QLabel(self.widgetContainigScrollArea_DelaunaryImages)
        self.label6_image6D.setGeometry(137, 274, 134, 134)
        self.label6_image6D.setStyleSheet("background-color: #f44336")
        self.pixmap6D = QtGui.QPixmap(self.image6PathD)
        self.label6_image6D.setScaledContents(True)
        self.label6_image6D.setPixmap(self.pixmap6D)

        self.label7_image7D = QtWidgets.QLabel(self.widgetContainigScrollArea_DelaunaryImages)
        self.label7_image7D.setGeometry(0, 411, 134, 134)
        self.label7_image7D.setStyleSheet("background-color: #f44336")
        self.pixmap7D = QtGui.QPixmap(self.image7PathD)
        self.label7_image7D.setScaledContents(True)
        self.label7_image7D.setPixmap(self.pixmap7D)

        self.label8_image8D = QtWidgets.QLabel(self.widgetContainigScrollArea_DelaunaryImages)
        self.label8_image8D.setGeometry(137, 411, 134, 134)
        self.label8_image8D.setStyleSheet("background-color: #f44336")
        self.pixmap8D = QtGui.QPixmap(self.image8PathD)
        self.label8_image8D.setScaledContents(True)
        self.label8_image8D.setPixmap(self.pixmap8D)

        # create a checkbox on the images
        self.checkbox_image1D = QtWidgets.QCheckBox(self.label1_image1D)
        self.checkbox_image1D.move(102, 102)
        self.checkbox_image1D.stateChanged.connect(self.checkIfButtonImage1IsCheckedD)

        self.checkbox_image2D = QtWidgets.QCheckBox(self.label2_image2D)
        self.checkbox_image2D.move(102, 102)
        self.checkbox_image2D.stateChanged.connect(self.checkIfButtonImage2IsCheckedD)

        self.checkbox_image3D = QtWidgets.QCheckBox(self.label3_image3D)
        self.checkbox_image3D.move(102, 102)
        self.checkbox_image3D.stateChanged.connect(self.checkIfButtonImage3IsCheckedD)

        self.checkbox_image4D = QtWidgets.QCheckBox(self.label4_image4D)
        self.checkbox_image4D.move(102, 102)
        self.checkbox_image4D.stateChanged.connect(self.checkIfButtonImage4IsCheckedD)

        self.checkbox_image5D = QtWidgets.QCheckBox(self.label5_image5D)
        self.checkbox_image5D.move(102, 102)
        self.checkbox_image5D.stateChanged.connect(self.checkIfButtonImage5IsCheckedD)

        self.checkbox_image6D = QtWidgets.QCheckBox(self.label6_image6D)
        self.checkbox_image6D.move(102, 102)
        self.checkbox_image6D.stateChanged.connect(self.checkIfButtonImage6IsCheckedD)

        self.checkbox_image7D = QtWidgets.QCheckBox(self.label7_image7D)
        self.checkbox_image7D.move(102, 102)
        self.checkbox_image7D.stateChanged.connect(self.checkIfButtonImage7IsCheckedD)

        self.checkbox_image8D = QtWidgets.QCheckBox(self.label8_image8D)
        self.checkbox_image8D.move(102, 102)
        self.checkbox_image8D.stateChanged.connect(self.checkIfButtonImage8IsCheckedD)

    # ----------------------------------------------------------face morphing function --------------------------------------------------------------------
    # create a definition for 'mainWidgetFaceSwapImagesBtn' button
    def createFaceMorphImagesWidgetButtons(self):
        # images to be placed in the labels below
        self.imageMorph1Path = "/home/ily19/PycharmProjects/imgProcessing/originalFaceMorphImages/8.jpg"
        self.imageMorph2Path = "/home/ily19/PycharmProjects/imgProcessing/originalFaceMorphImages/16.jpg"
        self.imageMorph3Path = "/home/ily19/PycharmProjects/imgProcessing/originalFaceMorphImages/2.jpg"

        self.cancelBtn_FaceMorphImages_widget = QtWidgets.QPushButton('Cancel', self.FaceMorphImages_Widget2)
        self.cancelBtn_FaceMorphImages_widget.setGeometry(10, 500, 125, 30)
        self.cancelBtn_FaceMorphImages_widget.clicked.connect(self.cancelBtn_FaceMorphImages_Widget_clicked)

        # create the widget that contains the scrollArea for the first images
        self.widgetContainigScrollArea_FaceMorphImages = QtWidgets.QWidget(self.FaceMorphImages_Widget)
        self.widgetContainigScrollArea_FaceMorphImages.resize(272, 550)
        self.widgetContainigScrollArea_FaceMorphImages.setStyleSheet("background-color:#ffffff")

        # create the widget that contains the scrollArea for the second images
        self.widgetContainigScrollArea_FaceMorphImages2 = QtWidgets.QWidget(self.FaceMorphImages_Widget2)
        self.widgetContainigScrollArea_FaceMorphImages2.resize(272, 550)
        self.widgetContainigScrollArea_FaceMorphImages2.setStyleSheet("background-color:#ffffff")

        # create the scollArea for the first images
        self.scrollArea = QtWidgets.QScrollArea(self.FaceMorphImages_Widget2)
        self.scrollArea.setWidget(self.widgetContainigScrollArea_FaceMorphImages)
        self.scrollArea.setFixedWidth(288)
        self.scrollArea.setFixedHeight(230)
        self.scrollArea.setStyleSheet(
            "border-top: 1px solid #2e2e2e; border-left:1px solid #2e2e2e;border-right:1px solid #2e2e2e")

        # create the labels that contains all the images to swap at the end
        self.label1_imageMorph1 = QtWidgets.QLabel(self.widgetContainigScrollArea_FaceMorphImages)
        self.label1_imageMorph1.setGeometry(0, 0, 134, 134)
        self.label1_imageMorph1.setStyleSheet("background-color: #f44336;")
        self.pixmapMorph1 = QtGui.QPixmap(self.imageMorph1Path)
        self.label1_imageMorph1.setScaledContents(True)
        self.label1_imageMorph1.setPixmap(self.pixmapMorph1)

        self.label2_imageMorph2 = QtWidgets.QLabel(self.widgetContainigScrollArea_FaceMorphImages)
        self.label2_imageMorph2.setGeometry(137, 0, 134, 134)
        self.label2_imageMorph2.setStyleSheet("background-color: #f44336")
        self.pixmapMorph2 = QtGui.QPixmap(self.imageMorph2Path)
        self.label2_imageMorph2.setScaledContents(True)
        self.label2_imageMorph2.setPixmap(self.pixmapMorph2)

        self.label3_imageMorph3 = QtWidgets.QLabel(self.widgetContainigScrollArea_FaceMorphImages)
        self.label3_imageMorph3.setGeometry(0, 137, 134, 134)
        self.label3_imageMorph3.setStyleSheet("background-color: #f44336")
        self.pixmapMorph3 = QtGui.QPixmap(self.imageMorph3Path)
        self.label3_imageMorph3.setScaledContents(True)
        self.label3_imageMorph3.setPixmap(self.pixmapMorph3)

        self.label4_imageMorph4 = QtWidgets.QLabel(self.widgetContainigScrollArea_FaceMorphImages)
        self.label4_imageMorph4.setGeometry(137, 137, 134, 134)
        self.label4_imageMorph4.setStyleSheet("background-color: #4e4e4e")
        self.label4_imageMorph4.setScaledContents(True)

        self.label5_imageMorph5 = QtWidgets.QLabel(self.widgetContainigScrollArea_FaceMorphImages)
        self.label5_imageMorph5.setGeometry(0, 274, 134, 134)
        self.label5_imageMorph5.setStyleSheet("background-color: #4e4e4e")
        # self.pixmapMorph5 = QtGui.QPixmap(self.imageMorph5Path)
        self.label5_imageMorph5.setScaledContents(True)
        # self.label5_imageMorph5.setPixmap(self.pixmapMorph5)

        self.label6_imageMorph6 = QtWidgets.QLabel(self.widgetContainigScrollArea_FaceMorphImages)
        self.label6_imageMorph6.setGeometry(137, 274, 134, 134)
        self.label6_imageMorph6.setStyleSheet("background-color: #4e4e4e")
        # self.pixmapMorph6 = QtGui.QPixmap(self.imageMorph6Path)
        self.label6_imageMorph6.setScaledContents(True)
        # self.label6_imageMorph6.setPixmap(self.pixmapMorph6)

        self.label7_imageMorph7 = QtWidgets.QLabel(self.widgetContainigScrollArea_FaceMorphImages)
        self.label7_imageMorph7.setGeometry(0, 411, 134, 134)
        self.label7_imageMorph7.setStyleSheet("background-color: #4e4e4e")
        # self.pixmapMorph7 = QtGui.QPixmap(self.imageMorph7Path)
        self.label7_imageMorph7.setScaledContents(True)
        # self.label7_imageMorph7.setPixmap(self.pixmapMorph7)

        self.label8_imageMorph8 = QtWidgets.QLabel(self.widgetContainigScrollArea_FaceMorphImages)
        self.label8_imageMorph8.setGeometry(137, 411, 134, 134)
        self.label8_imageMorph8.setStyleSheet("background-color: #4e4e4e")
        # self.pixmapMorph8 = QtGui.QPixmap(self.imageMorph8Path)
        self.label8_imageMorph8.setScaledContents(True)
        # self.label8_imageMorph8.setPixmap(self.pixmapMorph8)

        # create a checkbox on the images
        self.checkbox_imageMorph1 = QtWidgets.QCheckBox(self.label1_imageMorph1)
        self.checkbox_imageMorph1.move(102, 102)
        self.checkbox_imageMorph1.clicked.connect(self.checkIfFaceMorphImageIsClicked_1)

        self.checkbox_imageMorph2 = QtWidgets.QCheckBox(self.label2_imageMorph2)
        self.checkbox_imageMorph2.move(102, 102)
        self.checkbox_imageMorph2.clicked.connect(self.checkIfFaceMorphImageIsClicked_2)

        self.checkbox_imageMorph3 = QtWidgets.QCheckBox(self.label3_imageMorph3)
        self.checkbox_imageMorph3.move(102, 102)
        self.checkbox_imageMorph3.clicked.connect(self.checkIfFaceMorphImageIsClicked_3)

        self.checkbox_imageMorph4 = QtWidgets.QCheckBox(self.label4_imageMorph4)
        self.checkbox_imageMorph4.move(102, 102)
        self.checkbox_imageMorph4.clicked.connect(self.checkIfFaceMorphImageIsClicked_4)

        self.checkbox_imageMorph5 = QtWidgets.QCheckBox(self.label5_imageMorph5)
        self.checkbox_imageMorph5.move(102, 102)
        self.checkbox_imageMorph5.clicked.connect(self.checkIfFaceMorphImageIsClicked_5)

        self.checkbox_imageMorph6 = QtWidgets.QCheckBox(self.label6_imageMorph6)
        self.checkbox_imageMorph6.move(102, 102)
        self.checkbox_imageMorph6.clicked.connect(self.checkIfFaceMorphImageIsClicked_6)

        self.checkbox_imageMorph7 = QtWidgets.QCheckBox(self.label7_imageMorph7)
        self.checkbox_imageMorph7.move(102, 102)
        self.checkbox_imageMorph7.clicked.connect(self.checkIfFaceMorphImageIsClicked_7)

        self.checkbox_imageMorph8 = QtWidgets.QCheckBox(self.label8_imageMorph8)
        self.checkbox_imageMorph8.move(102, 102)
        self.checkbox_imageMorph8.clicked.connect(self.checkIfFaceMorphImageIsClicked_8)

        # create all the definition of all the scroll area 2

        self.FaceSwapBtn_FaceMorphImages_widget = QtWidgets.QPushButton('Swap Faces', self.FaceMorphImages_Widget2)
        self.FaceSwapBtn_FaceMorphImages_widget.setGeometry(155, 500, 125, 30)
        self.FaceSwapBtn_FaceMorphImages_widget.clicked.connect(self.FaceMorphBtn_FaceMorphImages_widget_clicked)

        # create the scollArea for the second images
        self.scrollArea2 = QtWidgets.QScrollArea(self.FaceMorphImages_Widget2)
        self.scrollArea2.setWidget(self.widgetContainigScrollArea_FaceMorphImages2)
        self.scrollArea2.setFixedWidth(288)
        self.scrollArea2.setFixedHeight(230)
        self.scrollArea2.move(0, 240)
        self.scrollArea2.setStyleSheet(
            "border-top: 1px solid #2e2e2e; border-left:1px solid #2e2e2e; border-right:1px solid #2e2e2e")

        self.label1_imageMorph1 = QtWidgets.QLabel(self.widgetContainigScrollArea_FaceMorphImages2)
        self.label1_imageMorph1.setGeometry(0, 0, 134, 134)
        self.label1_imageMorph1.setStyleSheet("background-color: #f44336;")
        self.pixmapMorph1 = QtGui.QPixmap(self.imageMorph1Path)
        self.label1_imageMorph1.setScaledContents(True)
        self.label1_imageMorph1.setPixmap(self.pixmapMorph1)

        self.label2_imageMorph2 = QtWidgets.QLabel(self.widgetContainigScrollArea_FaceMorphImages2)
        self.label2_imageMorph2.setGeometry(137, 0, 134, 134)
        self.label2_imageMorph2.setStyleSheet("background-color: #f44336")
        self.pixmapMorph2 = QtGui.QPixmap(self.imageMorph2Path)
        self.label2_imageMorph2.setScaledContents(True)
        self.label2_imageMorph2.setPixmap(self.pixmapMorph2)

        self.label3_imageMorph3 = QtWidgets.QLabel(self.widgetContainigScrollArea_FaceMorphImages2)
        self.label3_imageMorph3.setGeometry(0, 137, 134, 134)
        self.label3_imageMorph3.setStyleSheet("background-color: #f44336")
        self.pixmapMorph3 = QtGui.QPixmap(self.imageMorph3Path)
        self.label3_imageMorph3.setScaledContents(True)
        self.label3_imageMorph3.setPixmap(self.pixmapMorph3)

        self.label4_imageMorph4 = QtWidgets.QLabel(self.widgetContainigScrollArea_FaceMorphImages2)
        self.label4_imageMorph4.setGeometry(137, 137, 134, 134)
        self.label4_imageMorph4.setStyleSheet("background-color: #4e4e4e")
        self.label4_imageMorph4.setScaledContents(True)

        self.label5_imageMorph5 = QtWidgets.QLabel(self.widgetContainigScrollArea_FaceMorphImages2)
        self.label5_imageMorph5.setGeometry(0, 274, 134, 134)
        self.label5_imageMorph5.setStyleSheet("background-color: #4e4e4e")
        self.label5_imageMorph5.setScaledContents(True)

        self.label6_imageMorph6 = QtWidgets.QLabel(self.widgetContainigScrollArea_FaceMorphImages2)
        self.label6_imageMorph6.setGeometry(137, 274, 134, 134)
        self.label6_imageMorph6.setStyleSheet("background-color: #4e4e4e")
        self.label6_imageMorph6.setScaledContents(True)

        self.label7_imageMorph7 = QtWidgets.QLabel(self.widgetContainigScrollArea_FaceMorphImages2)
        self.label7_imageMorph7.setGeometry(0, 411, 134, 134)
        self.label7_imageMorph7.setStyleSheet("background-color: #4e4e4e")
        self.label7_imageMorph7.setScaledContents(True)

        self.label8_imageMorph8 = QtWidgets.QLabel(self.widgetContainigScrollArea_FaceMorphImages2)
        self.label8_imageMorph8.setGeometry(137, 411, 134, 134)
        self.label8_imageMorph8.setStyleSheet("background-color: #4e4e4e")
        self.label8_imageMorph8.setScaledContents(True)

        # create a checkbox on the images
        self.checkbox_imageMorph1_2 = QtWidgets.QCheckBox(self.label1_imageMorph1)
        self.checkbox_imageMorph1_2.move(102, 102)
        self.checkbox_imageMorph1_2.clicked.connect(self.checkIfFaceMorphImagesClicked_1_2)

        self.checkbox_imageMorph2_2 = QtWidgets.QCheckBox(self.label2_imageMorph2)
        self.checkbox_imageMorph2_2.move(102, 102)
        self.checkbox_imageMorph2_2.clicked.connect(self.checkIfFaceMorphImagesClicked_2_2)

        self.checkbox_imageMorph3_2 = QtWidgets.QCheckBox(self.label3_imageMorph3)
        self.checkbox_imageMorph3_2.move(102, 102)
        self.checkbox_imageMorph3_2.clicked.connect(self.checkIfFaceMorphImagesClicked_3_2)

        self.checkbox_imageMorph4_2 = QtWidgets.QCheckBox(self.label4_imageMorph4)
        self.checkbox_imageMorph4_2.move(102, 102)
        self.checkbox_imageMorph4_2.clicked.connect(self.checkIfFaceMorphImagesClicked_4_2)

        self.checkbox_imageMorph5_2 = QtWidgets.QCheckBox(self.label5_imageMorph5)
        self.checkbox_imageMorph5_2.move(102, 102)
        self.checkbox_imageMorph5_2.clicked.connect(self.checkIfFaceMorphImagesClicked_5_2)

        self.checkbox_imageMorph6_2 = QtWidgets.QCheckBox(self.label6_imageMorph6)
        self.checkbox_imageMorph6_2.move(102, 102)
        self.checkbox_imageMorph6_2.clicked.connect(self.checkIfFaceMorphImagesClicked_6_2)

        self.checkbox_imageMorph7_2 = QtWidgets.QCheckBox(self.label7_imageMorph7)
        self.checkbox_imageMorph7_2.move(102, 102)
        self.checkbox_imageMorph7_2.clicked.connect(self.checkIfFaceMorphImagesClicked_7_2)

        self.checkbox_imageMorph8_2 = QtWidgets.QCheckBox(self.label8_imageMorph8)
        self.checkbox_imageMorph8_2.move(102, 102)
        self.checkbox_imageMorph8_2.clicked.connect(self.checkIfFaceMorphImagesClicked_8_2)

    # ----------------------------------------------------------end face morphing--------------------------------------------------------------------------

    # -------------------------------------------------------------------------------face swap Function--------------------------------------------------
    # create a definition for 'mainWidgetFaceSwapImagesBtn' button
    def createFaceSwapImagesWidgetButtons(self):
        # images to be placed in the labels below
        self.imageSwap1Path = "/home/ily19/PycharmProjects/imgProcessing/originalFaceSwapImages/8.jpg"
        self.imageSwap2Path = "/home/ily19/PycharmProjects/imgProcessing/originalFaceSwapImages/16.jpg"
        self.imageSwap3Path = "/home/ily19/PycharmProjects/imgProcessing/originalFaceSwapImages/2.jpg"
        self.imageSwap4Path = "/home/ily19/PycharmProjects/imgProcessing/originalFaceSwapImages/5.jpg"
        self.imageSwap5Path = "/home/ily19/PycharmProjects/imgProcessing/originalFaceSwapImages/7.jpg"
        self.imageSwap6Path = "/home/ily19/PycharmProjects/imgProcessing/originalFaceSwapImages/15.jpg"
        self.imageSwap7Path = "/home/ily19/PycharmProjects/imgProcessing/originalFaceSwapImages/12.jpg"
        self.imageSwap8Path = "/home/ily19/PycharmProjects/imgProcessing/originalFaceSwapImages/9.jpg"

        self.cancelBtn_FaceSwapImages_widget = QtWidgets.QPushButton('Cancel', self.FaceSwapImages_Widget2)
        self.cancelBtn_FaceSwapImages_widget.setGeometry(10, 500, 125, 30)
        self.cancelBtn_FaceSwapImages_widget.clicked.connect(self.cancelBtn_FaceSwapImages_Widget_clicked)

        # create the widget that contains the scrollArea for the first images
        self.widgetContainigScrollArea_FaceSwapImages = QtWidgets.QWidget(self.FaceSwapImages_Widget)
        self.widgetContainigScrollArea_FaceSwapImages.resize(272, 550)
        self.widgetContainigScrollArea_FaceSwapImages.setStyleSheet("background-color:#ffffff")

        # create the widget that contains the scrollArea for the second images
        self.widgetContainigScrollArea_FaceSwapImages2 = QtWidgets.QWidget(self.FaceSwapImages_Widget2)
        self.widgetContainigScrollArea_FaceSwapImages2.resize(272, 550)
        self.widgetContainigScrollArea_FaceSwapImages2.setStyleSheet("background-color:#ffffff")

        # create the scollArea for the first images
        self.scrollArea = QtWidgets.QScrollArea(self.FaceSwapImages_Widget2)
        self.scrollArea.setWidget(self.widgetContainigScrollArea_FaceSwapImages)
        self.scrollArea.setFixedWidth(288)
        self.scrollArea.setFixedHeight(230)
        self.scrollArea.setStyleSheet(
            "border-top: 1px solid #2e2e2e; border-left:1px solid #2e2e2e; border-right:1px solid #2e2e2e")

        # create the labels that contains all the images to swap at the end
        self.label1_imageSwap1 = QtWidgets.QLabel(self.widgetContainigScrollArea_FaceSwapImages)
        self.label1_imageSwap1.setGeometry(0, 0, 134, 134)
        self.label1_imageSwap1.setStyleSheet("background-color: #f44336;")
        self.pixmapSwap1 = QtGui.QPixmap(self.imageSwap1Path)
        self.label1_imageSwap1.setScaledContents(True)
        self.label1_imageSwap1.setPixmap(self.pixmapSwap1)

        self.label2_imageSwap2 = QtWidgets.QLabel(self.widgetContainigScrollArea_FaceSwapImages)
        self.label2_imageSwap2.setGeometry(137, 0, 134, 134)
        self.label2_imageSwap2.setStyleSheet("background-color: #f44336")
        self.pixmapSwap2 = QtGui.QPixmap(self.imageSwap2Path)
        self.label2_imageSwap2.setScaledContents(True)
        self.label2_imageSwap2.setPixmap(self.pixmapSwap2)

        self.label3_imageSwap3 = QtWidgets.QLabel(self.widgetContainigScrollArea_FaceSwapImages)
        self.label3_imageSwap3.setGeometry(0, 137, 134, 134)
        self.label3_imageSwap3.setStyleSheet("background-color: #f44336")
        self.pixmapSwap3 = QtGui.QPixmap(self.imageSwap3Path)
        self.label3_imageSwap3.setScaledContents(True)
        self.label3_imageSwap3.setPixmap(self.pixmapSwap3)

        self.label4_imageSwap4 = QtWidgets.QLabel(self.widgetContainigScrollArea_FaceSwapImages)
        self.label4_imageSwap4.setGeometry(137, 137, 134, 134)
        self.label4_imageSwap4.setStyleSheet("background-color: #f44336")
        self.pixmapSwap4 = QtGui.QPixmap(self.imageSwap4Path)
        self.label4_imageSwap4.setScaledContents(True)
        self.label4_imageSwap4.setPixmap(self.pixmapSwap4)

        self.label5_imageSwap5 = QtWidgets.QLabel(self.widgetContainigScrollArea_FaceSwapImages)
        self.label5_imageSwap5.setGeometry(0, 274, 134, 134)
        self.label5_imageSwap5.setStyleSheet("background-color: #f44336")
        self.pixmapSwap5 = QtGui.QPixmap(self.imageSwap5Path)
        self.label5_imageSwap5.setScaledContents(True)
        self.label5_imageSwap5.setPixmap(self.pixmapSwap5)

        self.label6_imageSwap6 = QtWidgets.QLabel(self.widgetContainigScrollArea_FaceSwapImages)
        self.label6_imageSwap6.setGeometry(137, 274, 134, 134)
        self.label6_imageSwap6.setStyleSheet("background-color: #f44336")
        self.pixmapSwap6 = QtGui.QPixmap(self.imageSwap6Path)
        self.label6_imageSwap6.setScaledContents(True)
        self.label6_imageSwap6.setPixmap(self.pixmapSwap6)

        self.label7_imageSwap7 = QtWidgets.QLabel(self.widgetContainigScrollArea_FaceSwapImages)
        self.label7_imageSwap7.setGeometry(0, 411, 134, 134)
        self.label7_imageSwap7.setStyleSheet("background-color: #f44336")
        self.pixmapSwap7 = QtGui.QPixmap(self.imageSwap7Path)
        self.label7_imageSwap7.setScaledContents(True)
        self.label7_imageSwap7.setPixmap(self.pixmapSwap7)

        self.label8_imageSwap8 = QtWidgets.QLabel(self.widgetContainigScrollArea_FaceSwapImages)
        self.label8_imageSwap8.setGeometry(137, 411, 134, 134)
        self.label8_imageSwap8.setStyleSheet("background-color: #f44336")
        self.pixmapSwap8 = QtGui.QPixmap(self.imageSwap8Path)
        self.label8_imageSwap8.setScaledContents(True)
        self.label8_imageSwap8.setPixmap(self.pixmapSwap8)

        # create a checkbox on the images
        self.checkbox_imageSwap1 = QtWidgets.QCheckBox(self.label1_imageSwap1)
        self.checkbox_imageSwap1.move(102, 102)
        self.checkbox_imageSwap1.clicked.connect(self.checkIfFaceSwapImageIsClicked_1)

        self.checkbox_imageSwap2 = QtWidgets.QCheckBox(self.label2_imageSwap2)
        self.checkbox_imageSwap2.move(102, 102)
        self.checkbox_imageSwap2.clicked.connect(self.checkIfFaceSwapImageIsClicked_2)

        self.checkbox_imageSwap3 = QtWidgets.QCheckBox(self.label3_imageSwap3)
        self.checkbox_imageSwap3.move(102, 102)
        self.checkbox_imageSwap3.clicked.connect(self.checkIfFaceSwapImageIsClicked_3)

        self.checkbox_imageSwap4 = QtWidgets.QCheckBox(self.label4_imageSwap4)
        self.checkbox_imageSwap4.move(102, 102)
        self.checkbox_imageSwap4.clicked.connect(self.checkIfFaceSwapImageIsClicked_4)

        self.checkbox_imageSwap5 = QtWidgets.QCheckBox(self.label5_imageSwap5)
        self.checkbox_imageSwap5.move(102, 102)
        self.checkbox_imageSwap5.clicked.connect(self.checkIfFaceSwapImageIsClicked_5)

        self.checkbox_imageSwap6 = QtWidgets.QCheckBox(self.label6_imageSwap6)
        self.checkbox_imageSwap6.move(102, 102)
        self.checkbox_imageSwap6.clicked.connect(self.checkIfFaceSwapImageIsClicked_6)

        self.checkbox_imageSwap7 = QtWidgets.QCheckBox(self.label7_imageSwap7)
        self.checkbox_imageSwap7.move(102, 102)
        self.checkbox_imageSwap7.clicked.connect(self.checkIfFaceSwapImageIsClicked_7)

        self.checkbox_imageSwap8 = QtWidgets.QCheckBox(self.label8_imageSwap8)
        self.checkbox_imageSwap8.move(102, 102)
        self.checkbox_imageSwap8.clicked.connect(self.checkIfFaceSwapImageIsClicked_8)

        # create all the definition of all the scroll area 2

        self.FaceSwapBtn_FaceSwapImages_widget = QtWidgets.QPushButton('Swap Faces', self.FaceSwapImages_Widget2)
        self.FaceSwapBtn_FaceSwapImages_widget.setGeometry(155, 500, 125, 30)
        self.FaceSwapBtn_FaceSwapImages_widget.clicked.connect(self.FaceSwapBtn_FaceSwapImages_widget_clicked)

        # create the scollArea for the second images
        self.scrollArea2 = QtWidgets.QScrollArea(self.FaceSwapImages_Widget2)
        self.scrollArea2.setWidget(self.widgetContainigScrollArea_FaceSwapImages2)
        self.scrollArea2.setFixedWidth(288)
        self.scrollArea2.setFixedHeight(230)
        self.scrollArea2.move(0, 240)
        self.scrollArea2.setStyleSheet(
            "border-top: 1px solid #2e2e2e; border-left:1px solid #2e2e2e; border-right:1px solid #2e2e2e")

        self.label1_imageSwap1 = QtWidgets.QLabel(self.widgetContainigScrollArea_FaceSwapImages2)
        self.label1_imageSwap1.setGeometry(0, 0, 134, 134)
        self.label1_imageSwap1.setStyleSheet("background-color: #f44336;")
        self.pixmapSwap1 = QtGui.QPixmap(self.imageSwap1Path)
        self.label1_imageSwap1.setScaledContents(True)
        self.label1_imageSwap1.setPixmap(self.pixmapSwap1)

        self.label2_imageSwap2 = QtWidgets.QLabel(self.widgetContainigScrollArea_FaceSwapImages2)
        self.label2_imageSwap2.setGeometry(137, 0, 134, 134)
        self.label2_imageSwap2.setStyleSheet("background-color: #f44336")
        self.pixmapSwap2 = QtGui.QPixmap(self.imageSwap2Path)
        self.label2_imageSwap2.setScaledContents(True)
        self.label2_imageSwap2.setPixmap(self.pixmapSwap2)

        self.label3_imageSwap3 = QtWidgets.QLabel(self.widgetContainigScrollArea_FaceSwapImages2)
        self.label3_imageSwap3.setGeometry(0, 137, 134, 134)
        self.label3_imageSwap3.setStyleSheet("background-color: #f44336")
        self.pixmapSwap3 = QtGui.QPixmap(self.imageSwap3Path)
        self.label3_imageSwap3.setScaledContents(True)
        self.label3_imageSwap3.setPixmap(self.pixmapSwap3)

        self.label4_imageSwap4 = QtWidgets.QLabel(self.widgetContainigScrollArea_FaceSwapImages2)
        self.label4_imageSwap4.setGeometry(137, 137, 134, 134)
        self.label4_imageSwap4.setStyleSheet("background-color: #f44336")
        self.pixmapSwap4 = QtGui.QPixmap(self.imageSwap4Path)
        self.label4_imageSwap4.setScaledContents(True)
        self.label4_imageSwap4.setPixmap(self.pixmapSwap4)

        self.label5_imageSwap5 = QtWidgets.QLabel(self.widgetContainigScrollArea_FaceSwapImages2)
        self.label5_imageSwap5.setGeometry(0, 274, 134, 134)
        self.label5_imageSwap5.setStyleSheet("background-color: #f44336")
        self.pixmapSwap5 = QtGui.QPixmap(self.imageSwap5Path)
        self.label5_imageSwap5.setScaledContents(True)
        self.label5_imageSwap5.setPixmap(self.pixmapSwap5)

        self.label6_imageSwap6 = QtWidgets.QLabel(self.widgetContainigScrollArea_FaceSwapImages2)
        self.label6_imageSwap6.setGeometry(137, 274, 134, 134)
        self.label6_imageSwap6.setStyleSheet("background-color: #f44336")
        self.pixmapSwap6 = QtGui.QPixmap(self.imageSwap6Path)
        self.label6_imageSwap6.setScaledContents(True)
        self.label6_imageSwap6.setPixmap(self.pixmapSwap6)

        self.label7_imageSwap7 = QtWidgets.QLabel(self.widgetContainigScrollArea_FaceSwapImages2)
        self.label7_imageSwap7.setGeometry(0, 411, 134, 134)
        self.label7_imageSwap7.setStyleSheet("background-color: #f44336")
        self.pixmapSwap7 = QtGui.QPixmap(self.imageSwap7Path)
        self.label7_imageSwap7.setScaledContents(True)
        self.label7_imageSwap7.setPixmap(self.pixmapSwap7)

        self.label8_imageSwap8 = QtWidgets.QLabel(self.widgetContainigScrollArea_FaceSwapImages2)
        self.label8_imageSwap8.setGeometry(137, 411, 134, 134)
        self.label8_imageSwap8.setStyleSheet("background-color: #f44336")
        self.pixmapSwap8 = QtGui.QPixmap(self.imageSwap8Path)
        self.label8_imageSwap8.setScaledContents(True)
        self.label8_imageSwap8.setPixmap(self.pixmapSwap8)

        # create a checkbox on the images
        self.checkbox_imageSwap1_2 = QtWidgets.QCheckBox(self.label1_imageSwap1)
        self.checkbox_imageSwap1_2.move(102, 102)
        self.checkbox_imageSwap1_2.clicked.connect(self.checkIfFaceSwapImagesClicked_1_2)

        self.checkbox_imageSwap2_2 = QtWidgets.QCheckBox(self.label2_imageSwap2)
        self.checkbox_imageSwap2_2.move(102, 102)
        self.checkbox_imageSwap2_2.clicked.connect(self.checkIfFaceSwapImagesClicked_2_2)

        self.checkbox_imageSwap3_2 = QtWidgets.QCheckBox(self.label3_imageSwap3)
        self.checkbox_imageSwap3_2.move(102, 102)
        self.checkbox_imageSwap3_2.clicked.connect(self.checkIfFaceSwapImagesClicked_3_2)

        self.checkbox_imageSwap4_2 = QtWidgets.QCheckBox(self.label4_imageSwap4)
        self.checkbox_imageSwap4_2.move(102, 102)
        self.checkbox_imageSwap4_2.clicked.connect(self.checkIfFaceSwapImagesClicked_4_2)

        self.checkbox_imageSwap5_2 = QtWidgets.QCheckBox(self.label5_imageSwap5)
        self.checkbox_imageSwap5_2.move(102, 102)
        self.checkbox_imageSwap5_2.clicked.connect(self.checkIfFaceSwapImagesClicked_5_2)

        self.checkbox_imageSwap6_2 = QtWidgets.QCheckBox(self.label6_imageSwap6)
        self.checkbox_imageSwap6_2.move(102, 102)
        self.checkbox_imageSwap6_2.clicked.connect(self.checkIfFaceSwapImagesClicked_6_2)

        self.checkbox_imageSwap7_2 = QtWidgets.QCheckBox(self.label7_imageSwap7)
        self.checkbox_imageSwap7_2.move(102, 102)
        self.checkbox_imageSwap7_2.clicked.connect(self.checkIfFaceSwapImagesClicked_7_2)

        self.checkbox_imageSwap8_2 = QtWidgets.QCheckBox(self.label8_imageSwap8)
        self.checkbox_imageSwap8_2.move(102, 102)
        self.checkbox_imageSwap8_2.clicked.connect(self.checkIfFaceSwapImagesClicked_8_2)

    # *----------------------------------------------------------------------end face swap function----------------------------------------------------

    # -------------------------------------------------------------------face averaging function-------------------------------------------------------
    # create a definition for  'mainWidgetFaceAveragingImagesBtn'
    def createFaceAveragingImagesWidgetButtons(self):
        # images to be placed in the labels below
        self.image1Path = "/home/ily19/Desktop/images/5.jpg"
        self.image2Path = "/home/ily19/Desktop/images/8.jpg"
        self.image3Path = "/home/ily19/Desktop/images/15.jpg"
        self.image4Path = "/home/ily19/Desktop/images/12.jpg"
        self.image5Path = "/home/ily19/Desktop/images/16.jpg"
        self.image6Path = "/home/ily19/Desktop/images/2.jpg"
        self.image7Path = "/home/ily19/Desktop/images/9.jpg"
        self.image8Path = "/home/ily19/Desktop/images/7.jpg"

        self.cancelBtn_FaceAveragingImages_widget = QtWidgets.QPushButton('Cancel', self.FaceAveragingImages_Widget)
        self.cancelBtn_FaceAveragingImages_widget.setGeometry(10, 470, 125, 30)
        self.cancelBtn_FaceAveragingImages_widget.clicked.connect(self.cancelBtn_FaceAveragingImages_Widget_clicked)

        self.FaceAverageBtn_FaceAveragingImages_widget = QtWidgets.QPushButton('Average Faces',
                                                                               self.FaceAveragingImages_Widget)
        self.FaceAverageBtn_FaceAveragingImages_widget.setGeometry(155, 470, 125, 30)
        self.FaceAverageBtn_FaceAveragingImages_widget.clicked.connect(
            self.FaceAverageBtn_FaceAveragingImages_widget_clicked)

        # create the widget that contains the scrollArea
        self.widgetContainigScrollArea_FaceAveragingImages = QtWidgets.QWidget(self.FaceAveragingImages_Widget)
        self.widgetContainigScrollArea_FaceAveragingImages.resize(272, 550)
        self.widgetContainigScrollArea_FaceAveragingImages.setStyleSheet("background-color:#ffffff")

        # create the scollArea
        self.scrollArea = QtWidgets.QScrollArea(self.FaceAveragingImages_Widget)
        self.scrollArea.setWidget(self.widgetContainigScrollArea_FaceAveragingImages)
        self.scrollArea.setFixedWidth(288)
        self.scrollArea.setFixedHeight(460)
        self.scrollArea.setStyleSheet(
            "border-top: 1px solid #2e2e2e; border-left:1px solid #2e2e2e;border-right:1px solid #2e2e2e")

        # create the labels that contains all the images to average at the end
        self.label1_image1 = QtWidgets.QLabel(self.widgetContainigScrollArea_FaceAveragingImages)
        self.label1_image1.setGeometry(0, 0, 134, 134)
        self.label1_image1.setStyleSheet("background-color: #f44336")
        self.pixmap1 = QtGui.QPixmap(self.image1Path)
        self.label1_image1.setScaledContents(True)
        self.label1_image1.setPixmap(self.pixmap1)

        self.label2_image2 = QtWidgets.QLabel(self.widgetContainigScrollArea_FaceAveragingImages)
        self.label2_image2.setGeometry(137, 0, 134, 134)
        self.label2_image2.setStyleSheet("background-color: #f44336")
        self.pixmap2 = QtGui.QPixmap(self.image2Path)
        self.label2_image2.setScaledContents(True)
        self.label2_image2.setPixmap(self.pixmap2)

        self.label3_image3 = QtWidgets.QLabel(self.widgetContainigScrollArea_FaceAveragingImages)
        self.label3_image3.setGeometry(0, 137, 134, 134)
        self.label3_image3.setStyleSheet("background-color: #f44336")
        self.pixmap3 = QtGui.QPixmap(self.image3Path)
        self.label3_image3.setScaledContents(True)
        self.label3_image3.setPixmap(self.pixmap3)

        self.label4_image4 = QtWidgets.QLabel(self.widgetContainigScrollArea_FaceAveragingImages)
        self.label4_image4.setGeometry(137, 137, 134, 134)
        self.label4_image4.setStyleSheet("background-color: #f44336")
        self.pixmap4 = QtGui.QPixmap(self.image4Path)
        self.label4_image4.setScaledContents(True)
        self.label4_image4.setPixmap(self.pixmap4)

        self.label5_image5 = QtWidgets.QLabel(self.widgetContainigScrollArea_FaceAveragingImages)
        self.label5_image5.setGeometry(0, 274, 134, 134)
        self.label5_image5.setStyleSheet("background-color: #f44336")
        self.pixmap5 = QtGui.QPixmap(self.image5Path)
        self.label5_image5.setScaledContents(True)
        self.label5_image5.setPixmap(self.pixmap5)

        self.label6_image6 = QtWidgets.QLabel(self.widgetContainigScrollArea_FaceAveragingImages)
        self.label6_image6.setGeometry(137, 274, 134, 134)
        self.label6_image6.setStyleSheet("background-color: #f44336")
        self.pixmap6 = QtGui.QPixmap(self.image6Path)
        self.label6_image6.setScaledContents(True)
        self.label6_image6.setPixmap(self.pixmap6)

        self.label7_image7 = QtWidgets.QLabel(self.widgetContainigScrollArea_FaceAveragingImages)
        self.label7_image7.setGeometry(0, 411, 134, 134)
        self.label7_image7.setStyleSheet("background-color: #f44336")
        self.pixmap7 = QtGui.QPixmap(self.image7Path)
        self.label7_image7.setScaledContents(True)
        self.label7_image7.setPixmap(self.pixmap7)

        self.label8_image8 = QtWidgets.QLabel(self.widgetContainigScrollArea_FaceAveragingImages)
        self.label8_image8.setGeometry(137, 411, 134, 134)
        self.label8_image8.setStyleSheet("background-color: #f44336")
        self.pixmap8 = QtGui.QPixmap(self.image8Path)
        self.label8_image8.setScaledContents(True)
        self.label8_image8.setPixmap(self.pixmap8)

        # create a checkbox on the images
        self.checkbox_image1 = QtWidgets.QCheckBox(self.label1_image1)
        self.checkbox_image1.move(102, 102)
        self.checkbox_image1.stateChanged.connect(self.checkIfButtonImage1IsChecked)

        self.checkbox_image2 = QtWidgets.QCheckBox(self.label2_image2)
        self.checkbox_image2.move(102, 102)
        self.checkbox_image2.stateChanged.connect(self.checkIfButtonImage2IsChecked)

        self.checkbox_image3 = QtWidgets.QCheckBox(self.label3_image3)
        self.checkbox_image3.move(102, 102)
        self.checkbox_image3.stateChanged.connect(self.checkIfButtonImage3IsChecked)

        self.checkbox_image4 = QtWidgets.QCheckBox(self.label4_image4)
        self.checkbox_image4.move(102, 102)
        self.checkbox_image4.stateChanged.connect(self.checkIfButtonImage4IsChecked)

        self.checkbox_image5 = QtWidgets.QCheckBox(self.label5_image5)
        self.checkbox_image5.move(102, 102)
        self.checkbox_image5.stateChanged.connect(self.checkIfButtonImage5IsChecked)

        self.checkbox_image6 = QtWidgets.QCheckBox(self.label6_image6)
        self.checkbox_image6.move(102, 102)
        self.checkbox_image6.stateChanged.connect(self.checkIfButtonImage6IsChecked)

        self.checkbox_image7 = QtWidgets.QCheckBox(self.label7_image7)
        self.checkbox_image7.move(102, 102)
        self.checkbox_image7.stateChanged.connect(self.checkIfButtonImage7IsChecked)

        self.checkbox_image8 = QtWidgets.QCheckBox(self.label8_image8)
        self.checkbox_image8.move(102, 102)
        self.checkbox_image8.stateChanged.connect(self.checkIfButtonImage8IsChecked)

    # create a function to check the clicked checkboxes------------------------------------------------------------------------------------------------------------------
    def checkBoxesCliclked(self):
        if (self.checkbox_image1.isChecked() == False & self.checkbox_image2.isChecked() == False
                & self.checkbox_image3.isChecked() == False & self.checkbox_image4.isChecked() == False & self.checkbox_image5.isChecked() == False
                & self.checkbox_image6.isChecked() == False & self.checkbox_image7.isChecked() == False & self.checkbox_image8.isChecked() == False):
            self.msgBox = QtWidgets.QMessageBox()
            self.msgBox.setText('What to do?')
            self.msgBox.addButton(QtWidgets.QPushButton('Accept'), QtWidgets.QMessageBox.YesRole)
            self.msgBox.exec_()

    def checkIfButtonImage1IsChecked(self):
        if (self.checkbox_image1.isChecked()):
            shutil.move("/home/ily19/PycharmProjects/imgProcessing/neededImages/5.jpg",
                        "/home/ily19/PycharmProjects/imgProcessing/img/5.jpg")
            shutil.move("/home/ily19/PycharmProjects/imgProcessing/neededImages/5.jpg.txt",
                        "/home/ily19/PycharmProjects/imgProcessing/img/5.jpg.txt")
        else:
            shutil.move("/home/ily19/PycharmProjects/imgProcessing/img/5.jpg",
                        "/home/ily19/PycharmProjects/imgProcessing/neededImages/5.jpg")
            shutil.move("/home/ily19/PycharmProjects/imgProcessing/img/5.jpg.txt",
                        "/home/ily19/PycharmProjects/imgProcessing/neededImages/5.jpg.txt")

    def checkIfButtonImage2IsChecked(self):
        if (self.checkbox_image2.isChecked()):
            shutil.move("/home/ily19/PycharmProjects/imgProcessing/neededImages/8.jpg",
                        "/home/ily19/PycharmProjects/imgProcessing/img/8.jpg")
            shutil.move("/home/ily19/PycharmProjects/imgProcessing/neededImages/8.jpg.txt",
                        "/home/ily19/PycharmProjects/imgProcessing/img/8.jpg.txt")
        else:
            shutil.move("/home/ily19/PycharmProjects/imgProcessing/img/8.jpg",
                        "/home/ily19/PycharmProjects/imgProcessing/neededImages/8.jpg")
            shutil.move("/home/ily19/PycharmProjects/imgProcessing/img/8.jpg.txt",
                        "/home/ily19/PycharmProjects/imgProcessing/neededImages/8.jpg.txt")

    def checkIfButtonImage3IsChecked(self):
        if (self.checkbox_image3.isChecked()):
            shutil.move("/home/ily19/PycharmProjects/imgProcessing/neededImages/15.jpg",
                        "/home/ily19/PycharmProjects/imgProcessing/img/15.jpg")
            shutil.move("/home/ily19/PycharmProjects/imgProcessing/neededImages/15.jpg.txt",
                        "/home/ily19/PycharmProjects/imgProcessing/img/15.jpg.txt")
        else:
            shutil.move("/home/ily19/PycharmProjects/imgProcessing/img/15.jpg",
                        "/home/ily19/PycharmProjects/imgProcessing/neededImages/15.jpg")
            shutil.move("/home/ily19/PycharmProjects/imgProcessing/img/15.jpg.txt",
                        "/home/ily19/PycharmProjects/imgProcessing/neededImages/15.jpg.txt")

    def checkIfButtonImage4IsChecked(self):
        if (self.checkbox_image4.isChecked()):
            shutil.move("/home/ily19/PycharmProjects/imgProcessing/neededImages/12.jpg",
                        "/home/ily19/PycharmProjects/imgProcessing/img/12.jpg")
            shutil.move("/home/ily19/PycharmProjects/imgProcessing/neededImages/12.jpg.txt",
                        "/home/ily19/PycharmProjects/imgProcessing/img/12.jpg.txt")
        else:
            shutil.move("/home/ily19/PycharmProjects/imgProcessing/img/12.jpg",
                        "/home/ily19/PycharmProjects/imgProcessing/neededImages/12.jpg")
            shutil.move("/home/ily19/PycharmProjects/imgProcessing/img/12.jpg.txt",
                        "/home/ily19/PycharmProjects/imgProcessing/neededImages/12.jpg.txt")

    def checkIfButtonImage5IsChecked(self):
        if (self.checkbox_image5.isChecked()):
            shutil.move("/home/ily19/PycharmProjects/imgProcessing/neededImages/16.jpg",
                        "/home/ily19/PycharmProjects/imgProcessing/img/16.jpg")
            shutil.move("/home/ily19/PycharmProjects/imgProcessing/neededImages/16.jpg.txt",
                        "/home/ily19/PycharmProjects/imgProcessing/img/16.jpg.txt")
        else:
            shutil.move("/home/ily19/PycharmProjects/imgProcessing/img/16.jpg",
                        "/home/ily19/PycharmProjects/imgProcessing/neededImages/16.jpg")
            shutil.move("/home/ily19/PycharmProjects/imgProcessing/img/16.jpg.txt",
                        "/home/ily19/PycharmProjects/imgProcessing/neededImages/16.jpg.txt")

    def checkIfButtonImage6IsChecked(self):
        if (self.checkbox_image6.isChecked()):
            shutil.move("/home/ily19/PycharmProjects/imgProcessing/neededImages/2.jpg",
                        "/home/ily19/PycharmProjects/imgProcessing/img/2.jpg")
            shutil.move("/home/ily19/PycharmProjects/imgProcessing/neededImages/2.jpg.txt",
                        "/home/ily19/PycharmProjects/imgProcessing/img/2.jpg.txt")
        else:
            shutil.move("/home/ily19/PycharmProjects/imgProcessing/img/2.jpg",
                        "/home/ily19/PycharmProjects/imgProcessing/neededImages/2.jpg")
            shutil.move("/home/ily19/PycharmProjects/imgProcessing/img/2.jpg.txt",
                        "/home/ily19/PycharmProjects/imgProcessing/neededImages/2.jpg.txt")

    def checkIfButtonImage7IsChecked(self):
        if (self.checkbox_image7.isChecked()):
            shutil.move("/home/ily19/PycharmProjects/imgProcessing/neededImages/9.jpg",
                        "/home/ily19/PycharmProjects/imgProcessing/img/9.jpg")
            shutil.move("/home/ily19/PycharmProjects/imgProcessing/neededImages/9.jpg.txt",
                        "/home/ily19/PycharmProjects/imgProcessing/img/9.jpg.txt")
        else:
            shutil.move("/home/ily19/PycharmProjects/imgProcessing/img/9.jpg",
                        "/home/ily19/PycharmProjects/imgProcessing/neededImages/9.jpg")
            shutil.move("/home/ily19/PycharmProjects/imgProcessing/img/9.jpg.txt",
                        "/home/ily19/PycharmProjects/imgProcessing/neededImages/9.jpg.txt")

    def checkIfButtonImage8IsChecked(self):
        if (self.checkbox_image8.isChecked()):
            shutil.move("/home/ily19/PycharmProjects/imgProcessing/neededImages/7.jpg",
                        "/home/ily19/PycharmProjects/imgProcessing/img/7.jpg")
            shutil.move("/home/ily19/PycharmProjects/imgProcessing/neededImages/7.jpg.txt",
                        "/home/ily19/PycharmProjects/imgProcessing/img/7.jpg.txt")
        else:
            shutil.move("/home/ily19/PycharmProjects/imgProcessing/img/7.jpg",
                        "/home/ily19/PycharmProjects/imgProcessing/neededImages/7.jpg")
            shutil.move("/home/ily19/PycharmProjects/imgProcessing/img/7.jpg.txt",
                        "/home/ily19/PycharmProjects/imgProcessing/neededImages/7.jpg.txt")

    # ----------------------------------------------------------------------------------------------------------------------------------------------------------------------

    # ---------------------------------------------------------------------functoion for ckeckboxes in face morph-------------------------------------------------------------
    # image1 8.jpg
    def checkIfFaceMorphImageIsClicked_1(self):
        if (self.checkbox_imageMorph1.isChecked()):
            shutil.move("/home/ily19/PycharmProjects/imgProcessing/originalFaceMorphImages/8.jpg",
                        "/home/ily19/PycharmProjects/imgProcessing/morphImagesNeeded/filename1.jpg")
            shutil.move("/home/ily19/PycharmProjects/imgProcessing/originalFaceMorphImages/8.jpg.txt",
                        "/home/ily19/PycharmProjects/imgProcessing/morphImagesNeeded/filename1.jpg.txt")
        else:
            shutil.move("/home/ily19/PycharmProjects/imgProcessing/morphImagesNeeded/filename1.jpg",
                        "/home/ily19/PycharmProjects/imgProcessing/originalFaceMorphImages/8.jpg")
            shutil.move("/home/ily19/PycharmProjects/imgProcessing/morphImagesNeeded/filename1.jpg.txt",
                        "/home/ily19/PycharmProjects/imgProcessing/originalFaceMorphImages/8.jpg.txt")

    def checkIfFaceMorphImagesClicked_1_2(self):
        if (self.checkbox_imageMorph1_2.isChecked()):
            shutil.move("/home/ily19/PycharmProjects/imgProcessing/originalFaceMorphImages/8_2.jpg",
                        "/home/ily19/PycharmProjects/imgProcessing/morphImagesNeeded/filename2.jpg")
            shutil.move("/home/ily19/PycharmProjects/imgProcessing/originalFaceMorphImages/8_2.jpg.txt",
                        "/home/ily19/PycharmProjects/imgProcessing/morphImagesNeeded/filename2.jpg.txt")
        else:
            shutil.move("/home/ily19/PycharmProjects/imgProcessing/morphImagesNeeded/filename2.jpg",
                        "/home/ily19/PycharmProjects/imgProcessing/originalFaceMorphImages/8_2.jpg")
            shutil.move("/home/ily19/PycharmProjects/imgProcessing/morphImagesNeeded/filename2.jpg.txt",
                        "/home/ily19/PycharmProjects/imgProcessing/originalFaceMorphImages/8_2.jpg.txt")

    # image2 16.jpg
    def checkIfFaceMorphImageIsClicked_2(self):
        if (self.checkbox_imageMorph2.isChecked()):
            shutil.move("/home/ily19/PycharmProjects/imgProcessing/originalFaceMorphImages/16.jpg",
                        "/home/ily19/PycharmProjects/imgProcessing/morphImagesNeeded/filename1.jpg")
            shutil.move("/home/ily19/PycharmProjects/imgProcessing/originalFaceMorphImages/16.jpg.txt",
                        "/home/ily19/PycharmProjects/imgProcessing/morphImagesNeeded/filename1.jpg.txt")
        else:
            shutil.move("/home/ily19/PycharmProjects/imgProcessing/morphImagesNeeded/filename1.jpg",
                        "/home/ily19/PycharmProjects/imgProcessing/originalFaceMorphImages/16.jpg")
            shutil.move("/home/ily19/PycharmProjects/imgProcessing/morphImagesNeeded/filename1.jpg.txt",
                        "/home/ily19/PycharmProjects/imgProcessing/originalFaceMorphImages/16.jpg.txt")

    def checkIfFaceMorphImagesClicked_2_2(self):
        if (self.checkbox_imageMorph2_2.isChecked()):
            shutil.move("/home/ily19/PycharmProjects/imgProcessing/originalFaceMorphImages/16_2.jpg",
                        "/home/ily19/PycharmProjects/imgProcessing/morphImagesNeeded/filename2.jpg")
            shutil.move("/home/ily19/PycharmProjects/imgProcessing/originalFaceMorphImages/16_2.jpg.txt",
                        "/home/ily19/PycharmProjects/imgProcessing/morphImagesNeeded/filename2.jpg.txt")
        else:
            shutil.move("/home/ily19/PycharmProjects/imgProcessing/morphImagesNeeded/filename2.jpg",
                        "/home/ily19/PycharmProjects/imgProcessing/originalFaceMorphImages/16_2.jpg")
            shutil.move("/home/ily19/PycharmProjects/imgProcessing/morphImagesNeeded/filename2.jpg.txt",
                        "/home/ily19/PycharmProjects/imgProcessing/originalFaceMorphImages/16_2.jpg.txt")

    # image3 2.jpg
    def checkIfFaceMorphImageIsClicked_3(self):
        if (self.checkbox_imageMorph3.isChecked()):
            shutil.move("/home/ily19/PycharmProjects/imgProcessing/originalFaceMorphImages/2.jpg",
                        "/home/ily19/PycharmProjects/imgProcessing/morphImagesNeeded/filename1.jpg")
            shutil.move("/home/ily19/PycharmProjects/imgProcessing/originalFaceMorphImages/2.jpg.txt",
                        "/home/ily19/PycharmProjects/imgProcessing/morphImagesNeeded/filename1.jpg.txt")
        else:
            shutil.move("/home/ily19/PycharmProjects/imgProcessing/morphImagesNeeded/filename1.jpg",
                        "/home/ily19/PycharmProjects/imgProcessing/originalFaceMorphImages/2.jpg")
            shutil.move("/home/ily19/PycharmProjects/imgProcessing/morphImagesNeeded/filename1.jpg.txt",
                        "/home/ily19/PycharmProjects/imgProcessing/originalFaceMorphImages/2.jpg.txt")

    def checkIfFaceMorphImagesClicked_3_2(self):
        if (self.checkbox_imageMorph3_2.isChecked()):
            shutil.move("/home/ily19/PycharmProjects/imgProcessing/originalFaceMorphImages/2_2.jpg",
                        "/home/ily19/PycharmProjects/imgProcessing/morphImagesNeeded/filename2.jpg")
            shutil.move("/home/ily19/PycharmProjects/imgProcessing/originalFaceMorphImages/2_2.jpg.txt",
                        "/home/ily19/PycharmProjects/imgProcessing/morphImagesNeeded/filename2.jpg.txt")
        else:
            shutil.move("/home/ily19/PycharmProjects/imgProcessing/morphImagesNeeded/filename2.jpg",
                        "/home/ily19/PycharmProjects/imgProcessing/originalFaceMorphImages/2_2.jpg")
            shutil.move("/home/ily19/PycharmProjects/imgProcessing/morphImagesNeeded/filename2.jpg.txt",
                        "/home/ily19/PycharmProjects/imgProcessing/originalFaceMorphImages/2_2.jpg.txt")

    # image4 5.jpg
    def checkIfFaceMorphImageIsClicked_4(self):
        if (self.checkbox_imageMorph4.isChecked()):
            shutil.move("/home/ily19/PycharmProjects/imgProcessing/originalFaceMorphImages/5.jpg",
                        "/home/ily19/PycharmProjects/imgProcessing/morphImagesNeeded/filename1.jpg")
            shutil.move("/home/ily19/PycharmProjects/imgProcessing/originalFaceMorphImages/5.jpg.txt",
                        "/home/ily19/PycharmProjects/imgProcessing/morphImagesNeeded/filename1.jpg.txt")
        else:
            shutil.move("/home/ily19/PycharmProjects/imgProcessing/morphImagesNeeded/filename1.jpg",
                        "/home/ily19/PycharmProjects/imgProcessing/originalFaceMorphImages/5.jpg")
            shutil.move("/home/ily19/PycharmProjects/imgProcessing/morphImagesNeeded/filename1.jpg.txt",
                        "/home/ily19/PycharmProjects/imgProcessing/originalFaceMorphImages/5.jpg.txt")

    def checkIfFaceMorphImagesClicked_4_2(self):
        if (self.checkbox_imageMorph4_2.isChecked()):
            shutil.move("/home/ily19/PycharmProjects/imgProcessing/originalFaceMorphImages/5_2.jpg",
                        "/home/ily19/PycharmProjects/imgProcessing/morphImagesNeeded/filename2.jpg")
            shutil.move("/home/ily19/PycharmProjects/imgProcessing/originalFaceMorphImages/5_2.jpg.txt",
                        "/home/ily19/PycharmProjects/imgProcessing/morphImagesNeeded/filename2.jpg.txt")
        else:
            shutil.move("/home/ily19/PycharmProjects/imgProcessing/morphImagesNeeded/filename2.jpg",
                        "/home/ily19/PycharmProjects/imgProcessing/originalFaceMorphImages/5_2.jpg")
            shutil.move("/home/ily19/PycharmProjects/imgProcessing/morphImagesNeeded/filename2.jpg.txt",
                        "/home/ily19/PycharmProjects/imgProcessing/originalFaceMorphImages/5_2.jpg.txt")

    # image5 7.jpg
    def checkIfFaceMorphImageIsClicked_5(self):
        if (self.checkbox_imageMorph5.isChecked()):
            shutil.move("/home/ily19/PycharmProjects/imgProcessing/originalFaceMorphImages/7.jpg",
                        "/home/ily19/PycharmProjects/imgProcessing/morphImagesNeeded/filename1.jpg")
            shutil.move("/home/ily19/PycharmProjects/imgProcessing/originalFaceMorphImages/7.jpg.txt",
                        "/home/ily19/PycharmProjects/imgProcessing/morphImagesNeeded/filename1.jpg.txt")
        else:
            shutil.move("/home/ily19/PycharmProjects/imgProcessing/morphImagesNeeded/filename1.jpg",
                        "/home/ily19/PycharmProjects/imgProcessing/originalFaceMorphImages/7.jpg")
            shutil.move("/home/ily19/PycharmProjects/imgProcessing/morphImagesNeeded/filename1.jpg.txt",
                        "/home/ily19/PycharmProjects/imgProcessing/originalFaceMorphImages/7.jpg.txt")

    def checkIfFaceMorphImagesClicked_5_2(self):
        if (self.checkbox_imageMorph5_2.isChecked()):
            shutil.move("/home/ily19/PycharmProjects/imgProcessing/originalFaceMorphImages/7_2.jpg",
                        "/home/ily19/PycharmProjects/imgProcessing/morphImagesNeeded/filename2.jpg")
            shutil.move("/home/ily19/PycharmProjects/imgProcessing/originalFaceMorphImages/7_2.jpg.txt",
                        "/home/ily19/PycharmProjects/imgProcessing/morphImagesNeeded/filename2.jpg.txt")
        else:
            shutil.move("/home/ily19/PycharmProjects/imgProcessing/morphImagesNeeded/filename2.jpg",
                        "/home/ily19/PycharmProjects/imgProcessing/originalFaceMorphImages/7_2.jpg")
            shutil.move("/home/ily19/PycharmProjects/imgProcessing/morphImagesNeeded/filename2.jpg.txt",
                        "/home/ily19/PycharmProjects/imgProcessing/originalFaceMorphImages/7_2.jpg.txt")

    # image6 15.jpg
    def checkIfFaceMorphImageIsClicked_6(self):
        if (self.checkbox_imageMorph6.isChecked()):
            shutil.move("/home/ily19/PycharmProjects/imgProcessing/originalFaceMorphImages/15.jpg",
                        "/home/ily19/PycharmProjects/imgProcessing/morphImagesNeeded/filename1.jpg")
            shutil.move("/home/ily19/PycharmProjects/imgProcessing/originalFaceMorphImages/15.jpg.txt",
                        "/home/ily19/PycharmProjects/imgProcessing/morphImagesNeeded/filename1.jpg.txt")
        else:
            shutil.move("/home/ily19/PycharmProjects/imgProcessing/morphImagesNeeded/filename1.jpg",
                        "/home/ily19/PycharmProjects/imgProcessing/originalFaceMorphImages/15.jpg")
            shutil.move("/home/ily19/PycharmProjects/imgProcessing/morphImagesNeeded/filename1.jpg.txt",
                        "/home/ily19/PycharmProjects/imgProcessing/originalFaceMorphImages/15.jpg.txt")

    def checkIfFaceMorphImagesClicked_6_2(self):
        if (self.checkbox_imageMorph6_2.isChecked()):
            shutil.move("/home/ily19/PycharmProjects/imgProcessing/originalFaceMorphImages/15_2.jpg",
                        "/home/ily19/PycharmProjects/imgProcessing/morphImagesNeeded/filename2.jpg")
            shutil.move("/home/ily19/PycharmProjects/imgProcessing/originalFaceMorphImages/15_2.jpg.txt",
                        "/home/ily19/PycharmProjects/imgProcessing/morphImagesNeeded/filename2.jpg.txt")
        else:
            shutil.move("/home/ily19/PycharmProjects/imgProcessing/morphImagesNeeded/filename2.jpg",
                        "/home/ily19/PycharmProjects/imgProcessing/originalFaceMorphImages/15_2.jpg")
            shutil.move("/home/ily19/PycharmProjects/imgProcessing/morphImagesNeeded/filename2.jpg.txt",
                        "/home/ily19/PycharmProjects/imgProcessing/originalFaceMorphImages/15_2.jpg.txt")

    # image7 12.jpg
    def checkIfFaceMorphImageIsClicked_7(self):
        if (self.checkbox_imageMorph7.isChecked()):
            shutil.move("/home/ily19/PycharmProjects/imgProcessing/originalFaceMorphImages/12.jpg",
                        "/home/ily19/PycharmProjects/imgProcessing/morphImagesNeeded/filename1.jpg")
            shutil.move("/home/ily19/PycharmProjects/imgProcessing/originalFaceMorphImages/12.jpg.txt",
                        "/home/ily19/PycharmProjects/imgProcessing/morphImagesNeeded/filename1.jpg.txt")
        else:
            shutil.move("/home/ily19/PycharmProjects/imgProcessing/morphImagesNeeded/filename1.jpg",
                        "/home/ily19/PycharmProjects/imgProcessing/originalFaceMorphImages/12.jpg")
            shutil.move("/home/ily19/PycharmProjects/imgProcessing/morphImagesNeeded/filename1.jpg.txt",
                        "/home/ily19/PycharmProjects/imgProcessing/originalFaceMorphImages/12.jpg.txt")

    def checkIfFaceMorphImagesClicked_7_2(self):
        if (self.checkbox_imageMorph7_2.isChecked()):
            shutil.move("/home/ily19/PycharmProjects/imgProcessing/originalFaceMorphImages/12_2.jpg",
                        "/home/ily19/PycharmProjects/imgProcessing/morphImagesNeeded/filename2.jpg")
            shutil.move("/home/ily19/PycharmProjects/imgProcessing/originalFaceMorphImages/12_2.jpg.txt",
                        "/home/ily19/PycharmProjects/imgProcessing/morphImagesNeeded/filename2.jpg.txt")
        else:
            shutil.move("/home/ily19/PycharmProjects/imgProcessing/morphImagesNeeded/filename2.jpg",
                        "/home/ily19/PycharmProjects/imgProcessing/originalFaceMorphImages/12_2.jpg")
            shutil.move("/home/ily19/PycharmProjects/imgProcessing/morphImagesNeeded/filename2.jpg.txt",
                        "/home/ily19/PycharmProjects/imgProcessing/originalFaceMorphImages/12_2.jpg.txt")

    # image8 9.jpg
    def checkIfFaceMorphImageIsClicked_8(self):
        if (self.checkbox_imageMorph8.isChecked()):
            shutil.move("/home/ily19/PycharmProjects/imgProcessing/originalFaceMorphImages/9.jpg",
                        "/home/ily19/PycharmProjects/imgProcessing/morphImagesNeeded/filename1.jpg")
            shutil.move("/home/ily19/PycharmProjects/imgProcessing/originalFaceMorphImages/9.jpg.txt",
                        "/home/ily19/PycharmProjects/imgProcessing/morphImagesNeeded/filename1.jpg.txt")
        else:
            shutil.move("/home/ily19/PycharmProjects/imgProcessing/morphImagesNeeded/filename1.jpg",
                        "/home/ily19/PycharmProjects/imgProcessing/originalFaceMorphImages/9.jpg")
            shutil.move("/home/ily19/PycharmProjects/imgProcessing/morphImagesNeeded/filename1.jpg.txt",
                        "/home/ily19/PycharmProjects/imgProcessing/originalFaceMorphImages/9.jpg.txt")

    def checkIfFaceMorphImagesClicked_8_2(self):
        if (self.checkbox_imageMorph8_2.isChecked()):
            shutil.move("/home/ily19/PycharmProjects/imgProcessing/originalFaceMorphImages/9_2.jpg",
                        "/home/ily19/PycharmProjects/imgProcessing/morphImagesNeeded/filename2.jpg")
            shutil.move("/home/ily19/PycharmProjects/imgProcessing/originalFaceMorphImages/9_2.jpg.txt",
                        "/home/ily19/PycharmProjects/imgProcessing/morphImagesNeeded/filename2.jpg.txt")
        else:
            shutil.move("/home/ily19/PycharmProjects/imgProcessing/morphImagesNeeded/filename2.jpg",
                        "/home/ily19/PycharmProjects/imgProcessing/originalFaceMorphImages/9_2.jpg")
            shutil.move("/home/ily19/PycharmProjects/imgProcessing/morphImagesNeeded/filename2.jpg.txt",
                        "/home/ily19/PycharmProjects/imgProcessing/originalFaceMorphImages/9_2.jpg.txt")

    # ------------------------------------------------------------------------end checkboxes for in face morth---------------------------------------------------------------
    # create a definition functions for all the checkboxes of face swap images
    # image1 8.jpg
    def checkIfFaceSwapImageIsClicked_1(self):
        if (self.checkbox_imageSwap1.isChecked()):
            shutil.move("/home/ily19/PycharmProjects/imgProcessing/originalFaceSwapImages/8.jpg",
                        "/home/ily19/PycharmProjects/imgProcessing/swapImagesNeeded/filename1.jpg")
            shutil.move("/home/ily19/PycharmProjects/imgProcessing/originalFaceSwapImages/8.jpg.txt",
                        "/home/ily19/PycharmProjects/imgProcessing/swapImagesNeeded/filename1.jpg.txt")
        else:
            shutil.move("/home/ily19/PycharmProjects/imgProcessing/swapImagesNeeded/filename1.jpg",
                        "/home/ily19/PycharmProjects/imgProcessing/originalFaceSwapImages/8.jpg")
            shutil.move("/home/ily19/PycharmProjects/imgProcessing/swapImagesNeeded/filename1.jpg.txt",
                        "/home/ily19/PycharmProjects/imgProcessing/originalFaceSwapImages/8.jpg.txt")

    def checkIfFaceSwapImagesClicked_1_2(self):
        if (self.checkbox_imageSwap1_2.isChecked()):
            shutil.move("/home/ily19/PycharmProjects/imgProcessing/originalFaceSwapImages/8_2.jpg",
                        "/home/ily19/PycharmProjects/imgProcessing/swapImagesNeeded/filename2.jpg")
            shutil.move("/home/ily19/PycharmProjects/imgProcessing/originalFaceSwapImages/8_2.jpg.txt",
                        "/home/ily19/PycharmProjects/imgProcessing/swapImagesNeeded/filename2.jpg.txt")
        else:
            shutil.move("/home/ily19/PycharmProjects/imgProcessing/swapImagesNeeded/filename2.jpg",
                        "/home/ily19/PycharmProjects/imgProcessing/originalFaceSwapImages/8_2.jpg")
            shutil.move("/home/ily19/PycharmProjects/imgProcessing/swapImagesNeeded/filename2.jpg.txt",
                        "/home/ily19/PycharmProjects/imgProcessing/originalFaceSwapImages/8_2.jpg.txt")

    # image2 16.jpg
    def checkIfFaceSwapImageIsClicked_2(self):
        if (self.checkbox_imageSwap2.isChecked()):
            shutil.move("/home/ily19/PycharmProjects/imgProcessing/originalFaceSwapImages/16.jpg",
                        "/home/ily19/PycharmProjects/imgProcessing/swapImagesNeeded/filename1.jpg")
            shutil.move("/home/ily19/PycharmProjects/imgProcessing/originalFaceSwapImages/16.jpg.txt",
                        "/home/ily19/PycharmProjects/imgProcessing/swapImagesNeeded/filename1.jpg.txt")
        else:
            shutil.move("/home/ily19/PycharmProjects/imgProcessing/swapImagesNeeded/filename1.jpg",
                        "/home/ily19/PycharmProjects/imgProcessing/originalFaceSwapImages/16.jpg")
            shutil.move("/home/ily19/PycharmProjects/imgProcessing/swapImagesNeeded/filename1.jpg.txt",
                        "/home/ily19/PycharmProjects/imgProcessing/originalFaceSwapImages/16.jpg.txt")

    def checkIfFaceSwapImagesClicked_2_2(self):
        if (self.checkbox_imageSwap2_2.isChecked()):
            shutil.move("/home/ily19/PycharmProjects/imgProcessing/originalFaceSwapImages/16_2.jpg",
                        "/home/ily19/PycharmProjects/imgProcessing/swapImagesNeeded/filename2.jpg")
            shutil.move("/home/ily19/PycharmProjects/imgProcessing/originalFaceSwapImages/16_2.jpg.txt",
                        "/home/ily19/PycharmProjects/imgProcessing/swapImagesNeeded/filename2.jpg.txt")
        else:
            shutil.move("/home/ily19/PycharmProjects/imgProcessing/swapImagesNeeded/filename2.jpg",
                        "/home/ily19/PycharmProjects/imgProcessing/originalFaceSwapImages/16_2.jpg")
            shutil.move("/home/ily19/PycharmProjects/imgProcessing/swapImagesNeeded/filename2.jpg.txt",
                        "/home/ily19/PycharmProjects/imgProcessing/originalFaceSwapImages/16_2.jpg.txt")

    # image3 2.jpg
    def checkIfFaceSwapImageIsClicked_3(self):
        if (self.checkbox_imageSwap3.isChecked()):
            shutil.move("/home/ily19/PycharmProjects/imgProcessing/originalFaceSwapImages/2.jpg",
                        "/home/ily19/PycharmProjects/imgProcessing/swapImagesNeeded/filename1.jpg")
            shutil.move("/home/ily19/PycharmProjects/imgProcessing/originalFaceSwapImages/2.jpg.txt",
                        "/home/ily19/PycharmProjects/imgProcessing/swapImagesNeeded/filename1.jpg.txt")
        else:
            shutil.move("/home/ily19/PycharmProjects/imgProcessing/swapImagesNeeded/filename1.jpg",
                        "/home/ily19/PycharmProjects/imgProcessing/originalFaceSwapImages/2.jpg")
            shutil.move("/home/ily19/PycharmProjects/imgProcessing/swapImagesNeeded/filename1.jpg.txt",
                        "/home/ily19/PycharmProjects/imgProcessing/originalFaceSwapImages/2.jpg.txt")

    def checkIfFaceSwapImagesClicked_3_2(self):
        if (self.checkbox_imageSwap3_2.isChecked()):
            shutil.move("/home/ily19/PycharmProjects/imgProcessing/originalFaceSwapImages/2_2.jpg",
                        "/home/ily19/PycharmProjects/imgProcessing/swapImagesNeeded/filename2.jpg")
            shutil.move("/home/ily19/PycharmProjects/imgProcessing/originalFaceSwapImages/2_2.jpg.txt",
                        "/home/ily19/PycharmProjects/imgProcessing/swapImagesNeeded/filename2.jpg.txt")
        else:
            shutil.move("/home/ily19/PycharmProjects/imgProcessing/swapImagesNeeded/filename2.jpg",
                        "/home/ily19/PycharmProjects/imgProcessing/originalFaceSwapImages/2_2.jpg")
            shutil.move("/home/ily19/PycharmProjects/imgProcessing/swapImagesNeeded/filename2.jpg.txt",
                        "/home/ily19/PycharmProjects/imgProcessing/originalFaceSwapImages/2_2.jpg.txt")

    # image4 5.jpg
    def checkIfFaceSwapImageIsClicked_4(self):
        if (self.checkbox_imageSwap4.isChecked()):
            shutil.move("/home/ily19/PycharmProjects/imgProcessing/originalFaceSwapImages/5.jpg",
                        "/home/ily19/PycharmProjects/imgProcessing/swapImagesNeeded/filename1.jpg")
            shutil.move("/home/ily19/PycharmProjects/imgProcessing/originalFaceSwapImages/5.jpg.txt",
                        "/home/ily19/PycharmProjects/imgProcessing/swapImagesNeeded/filename1.jpg.txt")
        else:
            shutil.move("/home/ily19/PycharmProjects/imgProcessing/swapImagesNeeded/filename1.jpg",
                        "/home/ily19/PycharmProjects/imgProcessing/originalFaceSwapImages/5.jpg")
            shutil.move("/home/ily19/PycharmProjects/imgProcessing/swapImagesNeeded/filename1.jpg.txt",
                        "/home/ily19/PycharmProjects/imgProcessing/originalFaceSwapImages/5.jpg.txt")

    def checkIfFaceSwapImagesClicked_4_2(self):
        if (self.checkbox_imageSwap4_2.isChecked()):
            shutil.move("/home/ily19/PycharmProjects/imgProcessing/originalFaceSwapImages/5_2.jpg",
                        "/home/ily19/PycharmProjects/imgProcessing/swapImagesNeeded/filename2.jpg")
            shutil.move("/home/ily19/PycharmProjects/imgProcessing/originalFaceSwapImages/5_2.jpg.txt",
                        "/home/ily19/PycharmProjects/imgProcessing/swapImagesNeeded/filename2.jpg.txt")
        else:
            shutil.move("/home/ily19/PycharmProjects/imgProcessing/swapImagesNeeded/filename2.jpg",
                        "/home/ily19/PycharmProjects/imgProcessing/originalFaceSwapImages/5_2.jpg")
            shutil.move("/home/ily19/PycharmProjects/imgProcessing/swapImagesNeeded/filename2.jpg.txt",
                        "/home/ily19/PycharmProjects/imgProcessing/originalFaceSwapImages/5_2.jpg.txt")

    # image5 7.jpg
    def checkIfFaceSwapImageIsClicked_5(self):
        if (self.checkbox_imageSwap5.isChecked()):
            shutil.move("/home/ily19/PycharmProjects/imgProcessing/originalFaceSwapImages/7.jpg",
                        "/home/ily19/PycharmProjects/imgProcessing/swapImagesNeeded/filename1.jpg")
            shutil.move("/home/ily19/PycharmProjects/imgProcessing/originalFaceSwapImages/7.jpg.txt",
                        "/home/ily19/PycharmProjects/imgProcessing/swapImagesNeeded/filename1.jpg.txt")
        else:
            shutil.move("/home/ily19/PycharmProjects/imgProcessing/swapImagesNeeded/filename1.jpg",
                        "/home/ily19/PycharmProjects/imgProcessing/originalFaceSwapImages/7.jpg")
            shutil.move("/home/ily19/PycharmProjects/imgProcessing/swapImagesNeeded/filename1.jpg.txt",
                        "/home/ily19/PycharmProjects/imgProcessing/originalFaceSwapImages/7.jpg.txt")

    def checkIfFaceSwapImagesClicked_5_2(self):
        if (self.checkbox_imageSwap5_2.isChecked()):
            shutil.move("/home/ily19/PycharmProjects/imgProcessing/originalFaceSwapImages/7_2.jpg",
                        "/home/ily19/PycharmProjects/imgProcessing/swapImagesNeeded/filename2.jpg")
            shutil.move("/home/ily19/PycharmProjects/imgProcessing/originalFaceSwapImages/7_2.jpg.txt",
                        "/home/ily19/PycharmProjects/imgProcessing/swapImagesNeeded/filename2.jpg.txt")
        else:
            shutil.move("/home/ily19/PycharmProjects/imgProcessing/swapImagesNeeded/filename2.jpg",
                        "/home/ily19/PycharmProjects/imgProcessing/originalFaceSwapImages/7_2.jpg")
            shutil.move("/home/ily19/PycharmProjects/imgProcessing/swapImagesNeeded/filename2.jpg.txt",
                        "/home/ily19/PycharmProjects/imgProcessing/originalFaceSwapImages/7_2.jpg.txt")

    # image6 15.jpg
    def checkIfFaceSwapImageIsClicked_6(self):
        if (self.checkbox_imageSwap6.isChecked()):
            shutil.move("/home/ily19/PycharmProjects/imgProcessing/originalFaceSwapImages/15.jpg",
                        "/home/ily19/PycharmProjects/imgProcessing/swapImagesNeeded/filename1.jpg")
            shutil.move("/home/ily19/PycharmProjects/imgProcessing/originalFaceSwapImages/15.jpg.txt",
                        "/home/ily19/PycharmProjects/imgProcessing/swapImagesNeeded/filename1.jpg.txt")
        else:
            shutil.move("/home/ily19/PycharmProjects/imgProcessing/swapImagesNeeded/filename1.jpg",
                        "/home/ily19/PycharmProjects/imgProcessing/originalFaceSwapImages/15.jpg")
            shutil.move("/home/ily19/PycharmProjects/imgProcessing/swapImagesNeeded/filename1.jpg.txt",
                        "/home/ily19/PycharmProjects/imgProcessing/originalFaceSwapImages/15.jpg.txt")

    def checkIfFaceSwapImagesClicked_6_2(self):
        if (self.checkbox_imageSwap6_2.isChecked()):
            shutil.move("/home/ily19/PycharmProjects/imgProcessing/originalFaceSwapImages/15_2.jpg",
                        "/home/ily19/PycharmProjects/imgProcessing/swapImagesNeeded/filename2.jpg")
            shutil.move("/home/ily19/PycharmProjects/imgProcessing/originalFaceSwapImages/15_2.jpg.txt",
                        "/home/ily19/PycharmProjects/imgProcessing/swapImagesNeeded/filename2.jpg.txt")
        else:
            shutil.move("/home/ily19/PycharmProjects/imgProcessing/swapImagesNeeded/filename2.jpg",
                        "/home/ily19/PycharmProjects/imgProcessing/originalFaceSwapImages/15_2.jpg")
            shutil.move("/home/ily19/PycharmProjects/imgProcessing/swapImagesNeeded/filename2.jpg.txt",
                        "/home/ily19/PycharmProjects/imgProcessing/originalFaceSwapImages/15_2.jpg.txt")

    # image7 12.jpg
    def checkIfFaceSwapImageIsClicked_7(self):
        if (self.checkbox_imageSwap7.isChecked()):
            shutil.move("/home/ily19/PycharmProjects/imgProcessing/originalFaceSwapImages/12.jpg",
                        "/home/ily19/PycharmProjects/imgProcessing/swapImagesNeeded/filename1.jpg")
            shutil.move("/home/ily19/PycharmProjects/imgProcessing/originalFaceSwapImages/12.jpg.txt",
                        "/home/ily19/PycharmProjects/imgProcessing/swapImagesNeeded/filename1.jpg.txt")
        else:
            shutil.move("/home/ily19/PycharmProjects/imgProcessing/swapImagesNeeded/filename1.jpg",
                        "/home/ily19/PycharmProjects/imgProcessing/originalFaceSwapImages/12.jpg")
            shutil.move("/home/ily19/PycharmProjects/imgProcessing/swapImagesNeeded/filename1.jpg.txt",
                        "/home/ily19/PycharmProjects/imgProcessing/originalFaceSwapImages/12.jpg.txt")

    def checkIfFaceSwapImagesClicked_7_2(self):
        if (self.checkbox_imageSwap7_2.isChecked()):
            shutil.move("/home/ily19/PycharmProjects/imgProcessing/originalFaceSwapImages/12_2.jpg",
                        "/home/ily19/PycharmProjects/imgProcessing/swapImagesNeeded/filename2.jpg")
            shutil.move("/home/ily19/PycharmProjects/imgProcessing/originalFaceSwapImages/12_2.jpg.txt",
                        "/home/ily19/PycharmProjects/imgProcessing/swapImagesNeeded/filename2.jpg.txt")
        else:
            shutil.move("/home/ily19/PycharmProjects/imgProcessing/swapImagesNeeded/filename2.jpg",
                        "/home/ily19/PycharmProjects/imgProcessing/originalFaceSwapImages/12_2.jpg")
            shutil.move("/home/ily19/PycharmProjects/imgProcessing/swapImagesNeeded/filename2.jpg.txt",
                        "/home/ily19/PycharmProjects/imgProcessing/originalFaceSwapImages/12_2.jpg.txt")

    # image8 9.jpg
    def checkIfFaceSwapImageIsClicked_8(self):
        if (self.checkbox_imageSwap8.isChecked()):
            shutil.move("/home/ily19/PycharmProjects/imgProcessing/originalFaceSwapImages/9.jpg",
                        "/home/ily19/PycharmProjects/imgProcessing/swapImagesNeeded/filename1.jpg")
            shutil.move("/home/ily19/PycharmProjects/imgProcessing/originalFaceSwapImages/9.jpg.txt",
                        "/home/ily19/PycharmProjects/imgProcessing/swapImagesNeeded/filename1.jpg.txt")
        else:
            shutil.move("/home/ily19/PycharmProjects/imgProcessing/swapImagesNeeded/filename1.jpg",
                        "/home/ily19/PycharmProjects/imgProcessing/originalFaceSwapImages/9.jpg")
            shutil.move("/home/ily19/PycharmProjects/imgProcessing/swapImagesNeeded/filename1.jpg.txt",
                        "/home/ily19/PycharmProjects/imgProcessing/originalFaceSwapImages/9.jpg.txt")

    def checkIfFaceSwapImagesClicked_8_2(self):
        if (self.checkbox_imageSwap8_2.isChecked()):
            shutil.move("/home/ily19/PycharmProjects/imgProcessing/originalFaceSwapImages/9_2.jpg",
                        "/home/ily19/PycharmProjects/imgProcessing/swapImagesNeeded/filename2.jpg")
            shutil.move("/home/ily19/PycharmProjects/imgProcessing/originalFaceSwapImages/9_2.jpg.txt",
                        "/home/ily19/PycharmProjects/imgProcessing/swapImagesNeeded/filename2.jpg.txt")
        else:
            shutil.move("/home/ily19/PycharmProjects/imgProcessing/swapImagesNeeded/filename2.jpg",
                        "/home/ily19/PycharmProjects/imgProcessing/originalFaceSwapImages/9_2.jpg")
            shutil.move("/home/ily19/PycharmProjects/imgProcessing/swapImagesNeeded/filename2.jpg.txt",
                        "/home/ily19/PycharmProjects/imgProcessing/originalFaceSwapImages/9_2.jpg.txt")

    # ------------------------------------checkboxes functions for Delaunary-------------------------------------------------------------------------
    def checkIfButtonImage1IsCheckedD(self):
        if (self.checkbox_image1D.isChecked()):
            shutil.move("/home/ily19/PycharmProjects/imgProcessing/neededImagesD/5.jpg",
                        "/home/ily19/PycharmProjects/imgProcessing/imgD/filename1.jpg")
            shutil.move("/home/ily19/PycharmProjects/imgProcessing/neededImagesD/5.jpg.txt",
                        "/home/ily19/PycharmProjects/imgProcessing/imgD/filename1.txt")
        else:
            shutil.move("/home/ily19/PycharmProjects/imgProcessing/imgD/filename1.jpg",
                        "/home/ily19/PycharmProjects/imgProcessing/neededImagesD/5.jpg")
            shutil.move("/home/ily19/PycharmProjects/imgProcessing/imgD/filename1.txt",
                        "/home/ily19/PycharmProjects/imgProcessing/neededImagesD/5.jpg.txt")

    def checkIfButtonImage2IsCheckedD(self):
        if (self.checkbox_image2D.isChecked()):
            shutil.move("/home/ily19/PycharmProjects/imgProcessing/neededImagesD/8.jpg",
                        "/home/ily19/PycharmProjects/imgProcessing/imgD/filename1.jpg")
            shutil.move("/home/ily19/PycharmProjects/imgProcessing/neededImagesD/8.jpg.txt",
                        "/home/ily19/PycharmProjects/imgProcessing/imgD/filename1.txt")
        else:
            shutil.move("/home/ily19/PycharmProjects/imgProcessing/imgD/filename1.jpg",
                        "/home/ily19/PycharmProjects/imgProcessing/neededImagesD/8.jpg")
            shutil.move("/home/ily19/PycharmProjects/imgProcessing/imgD/filename1.txt",
                        "/home/ily19/PycharmProjects/imgProcessing/neededImagesD/8.jpg.txt")

    def checkIfButtonImage3IsCheckedD(self):
        if (self.checkbox_image3D.isChecked()):
            shutil.move("/home/ily19/PycharmProjects/imgProcessing/neededImagesD/15.jpg",
                        "/home/ily19/PycharmProjects/imgProcessing/imgD/filename1.jpg")
            shutil.move("/home/ily19/PycharmProjects/imgProcessing/neededImagesD/15.jpg.txt",
                        "/home/ily19/PycharmProjects/imgProcessing/imgD/filename1.txt")
        else:
            shutil.move("/home/ily19/PycharmProjects/imgProcessing/imgD/filename1.jpg",
                        "/home/ily19/PycharmProjects/imgProcessing/neededImagesD/15.jpg")
            shutil.move("/home/ily19/PycharmProjects/imgProcessing/imgD/filename1.txt",
                        "/home/ily19/PycharmProjects/imgProcessing/neededImagesD/15.jpg.txt")

    def checkIfButtonImage4IsCheckedD(self):
        if (self.checkbox_image4D.isChecked()):
            shutil.move("/home/ily19/PycharmProjects/imgProcessing/neededImagesD/12.jpg",
                        "/home/ily19/PycharmProjects/imgProcessing/imgD/filename1.jpg")
            shutil.move("/home/ily19/PycharmProjects/imgProcessing/neededImagesD/12.jpg.txt",
                        "/home/ily19/PycharmProjects/imgProcessing/imgD/filename1.txt")
        else:
            shutil.move("/home/ily19/PycharmProjects/imgProcessing/imgD/filename1.jpg",
                        "/home/ily19/PycharmProjects/imgProcessing/neededImagesD/12.jpg")
            shutil.move("/home/ily19/PycharmProjects/imgProcessing/imgD/filename1.txt",
                        "/home/ily19/PycharmProjects/imgProcessing/neededImagesD/12.jpg.txt")

    def checkIfButtonImage5IsCheckedD(self):
        if (self.checkbox_image5D.isChecked()):
            shutil.move("/home/ily19/PycharmProjects/imgProcessing/neededImagesD/16.jpg",
                        "/home/ily19/PycharmProjects/imgProcessing/imgD/filename1.jpg")
            shutil.move("/home/ily19/PycharmProjects/imgProcessing/neededImagesD/16.jpg.txt",
                        "/home/ily19/PycharmProjects/imgProcessing/imgD/filename1.txt")
        else:
            shutil.move("/home/ily19/PycharmProjects/imgProcessing/imgD/filename1.jpg",
                        "/home/ily19/PycharmProjects/imgProcessing/neededImagesD/16.jpg")
            shutil.move("/home/ily19/PycharmProjects/imgProcessing/imgD/filename1.txt",
                        "/home/ily19/PycharmProjects/imgProcessing/neededImagesD/16.jpg.txt")

    def checkIfButtonImage6IsCheckedD(self):
        if (self.checkbox_image6D.isChecked()):
            shutil.move("/home/ily19/PycharmProjects/imgProcessing/neededImagesD/2.jpg",
                        "/home/ily19/PycharmProjects/imgProcessing/imgD/filename1.jpg")
            shutil.move("/home/ily19/PycharmProjects/imgProcessing/neededImagesD/2.jpg.txt",
                        "/home/ily19/PycharmProjects/imgProcessing/imgD/filename1.txt")
        else:
            shutil.move("/home/ily19/PycharmProjects/imgProcessing/imgD/filename1.jpg",
                        "/home/ily19/PycharmProjects/imgProcessing/neededImagesD/2.jpg")
            shutil.move("/home/ily19/PycharmProjects/imgProcessing/imgD/filename1.txt",
                        "/home/ily19/PycharmProjects/imgProcessing/neededImagesD/2.jpg.txt")

    def checkIfButtonImage7IsCheckedD(self):
        if (self.checkbox_image7D.isChecked()):
            shutil.move("/home/ily19/PycharmProjects/imgProcessing/neededImagesD/9.jpg",
                        "/home/ily19/PycharmProjects/imgProcessing/imgD/filename1.jpg")
            shutil.move("/home/ily19/PycharmProjects/imgProcessing/neededImagesD/9.jpg.txt",
                        "/home/ily19/PycharmProjects/imgProcessing/imgD/filename1.txt")
        else:
            shutil.move("/home/ily19/PycharmProjects/imgProcessing/imgD/filename1.jpg",
                        "/home/ily19/PycharmProjects/imgProcessing/neededImagesD/9.jpg")
            shutil.move("/home/ily19/PycharmProjects/imgProcessing/imgD/filename1.txt",
                        "/home/ily19/PycharmProjects/imgProcessing/neededImagesD/9.jpg.txt")

    def checkIfButtonImage8IsCheckedD(self):
        if (self.checkbox_image8D.isChecked()):
            shutil.move("/home/ily19/PycharmProjects/imgProcessing/neededImagesD/7.jpg",
                        "/home/ily19/PycharmProjects/imgProcessing/imgD/filename1.jpg")
            shutil.move("/home/ily19/PycharmProjects/imgProcessing/neededImagesD/7.jpg.txt",
                        "/home/ily19/PycharmProjects/imgProcessing/imgD/filename1.txt")
        else:
            shutil.move("/home/ily19/PycharmProjects/imgProcessing/imgD/filename1.jpg",
                        "/home/ily19/PycharmProjects/imgProcessing/neededImagesD/7.jpg")
            shutil.move("/home/ily19/PycharmProjects/imgProcessing/imgD/filename1.txt",
                        "/home/ily19/PycharmProjects/imgProcessing/neededImagesD/7.jpg.txt")

    # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # check if an image exists in the folder image:
    def checkIfImageExistsThenMoveIt(self):
        pass

    # create the definition (buttons...) for 'EyeDetectionInImage_widget'
    def createEyeDetectionInImageWidgetButtons(self):
        self.exampleImage_EyeDetectionInImage_Widget = "images/Screenshot from 2018-04-12 12-27-44.png"

        self.cancelBtn_EyeDetectionInImage_Widget = QtWidgets.QPushButton('Cancel', self.EyeDetectionInImage_Widget)
        self.cancelBtn_EyeDetectionInImage_Widget.setGeometry(10, 470, 125, 30)
        self.cancelBtn_EyeDetectionInImage_Widget.clicked.connect(self.cancelBtn_EyeDetectionInImage_Widget_clicked)

        self.loadImageBtn_EyeDetectionInImage_widget = QtWidgets.QPushButton('Load an Image',
                                                                             self.EyeDetectionInImage_Widget)
        self.loadImageBtn_EyeDetectionInImage_widget.setGeometry(155, 470, 125, 30)
        self.loadImageBtn_EyeDetectionInImage_widget.clicked.connect(
            self.loadImageBtn_FaceDetectionInImage_Widget_clicked)

        self.findEyesBtn_EyeDetectionInImage_Widget = QtWidgets.QPushButton('Find Eyes',
                                                                            self.EyeDetectionInImage_Widget)
        self.findEyesBtn_EyeDetectionInImage_Widget.setGeometry(10, 430, 270, 30)
        self.findEyesBtn_EyeDetectionInImage_Widget.clicked.connect(self.findEyesBtn_EyeDetectionInImage_Widget_clicked)

        self.labelToDisplayExampleImage_EyeDetectionInImage_Widget = QtWidgets.QLabel(self.EyeDetectionInImage_Widget)
        self.labelToDisplayExampleImage_EyeDetectionInImage_Widget.setGeometry(10, 10, 270, 400)
        self.labelToDisplayExampleImage_EyeDetectionInImage_Widget.setStyleSheet("background-color:#ffffff")
        self.labelToDisplayExampleImage_EyeDetectionInImage_Widget.setPixmap(
            QtGui.QPixmap(self.exampleImage_EyeDetectionInImage_Widget))
        self.labelToDisplayExampleImage_EyeDetectionInImage_Widget.setAlignment(
            QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)

    # create the definition (buttons..) for 'FaceDetectionInImage_widget'
    def createFaceDetectionInImageWidgetButtons(self):
        self.exampleImage_FaceDetectionInImage_Widget = "images/Screenshot from 2018-04-12 11-05-11.png"

        self.cancelBtn_FaceDetectionInImage_Widget = QtWidgets.QPushButton("Cancel", self.FaceDetectionInImage_Widget)
        self.cancelBtn_FaceDetectionInImage_Widget.setGeometry(10, 470, 125, 30)
        self.cancelBtn_FaceDetectionInImage_Widget.clicked.connect(self.cancelBtn_FaceDetectionInImage_Widget_clicked)

        self.loadImageBtn_FaceDetectionInImage_Widget = QtWidgets.QPushButton("Load an Image",
                                                                              self.FaceDetectionInImage_Widget)
        self.loadImageBtn_FaceDetectionInImage_Widget.setGeometry(155, 470, 125, 30)
        self.loadImageBtn_FaceDetectionInImage_Widget.clicked.connect(
            self.loadImageBtn_FaceDetectionInImage_Widget_clicked)

        self.findFacesBtn_FaceDetectionInImage_Widget = QtWidgets.QPushButton("Find Faces",
                                                                              self.FaceDetectionInImage_Widget)
        self.findFacesBtn_FaceDetectionInImage_Widget.setGeometry(10, 430, 270, 30)
        self.findFacesBtn_FaceDetectionInImage_Widget.clicked.connect(
            self.findFacesBtn_FaceDetectionInImage_Widget_clicked)

        self.labelToDisplayExampleImage_FaceDetectionInImage_Widget = QtWidgets.QLabel(self.FaceDetectionInImage_Widget)
        self.labelToDisplayExampleImage_FaceDetectionInImage_Widget.setGeometry(10, 10, 270, 400)
        self.labelToDisplayExampleImage_FaceDetectionInImage_Widget.setStyleSheet("background-color:#ffffff")
        self.labelToDisplayExampleImage_FaceDetectionInImage_Widget.setPixmap(
            QtGui.QPixmap(self.exampleImage_FaceDetectionInImage_Widget))
        self.labelToDisplayExampleImage_FaceDetectionInImage_Widget.setAlignment(
            QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)

    # create the definition for 'MainWidget'
    def CreateMainWidgetButtons(self):
        self.mainWidgetFaceDetectionInImageBtn = QtWidgets.QPushButton("Face Detection In Image", self.mainWidget)
        self.mainWidgetFaceDetectionInImageBtn.setGeometry(0, 0, 288, 40)
        self.mainWidgetFaceDetectionInImageBtn.setStyleSheet("Text-align:left;padding-left: 10px;font-size:13px;")
        self.mainWidgetFaceDetectionInImageBtn.setIcon(QtGui.QIcon("images/web_hi_res_5123.png"))
        self.mainWidgetFaceDetectionInImageBtn.setIconSize(QtCore.QSize(35, 35))
        self.mainWidgetFaceDetectionInImageBtn.clicked.connect(self.mainWidgetFaceDetectionInImage_Clicked)

        self.mainWidgetFaceDetectionInRealTimeBtn = QtWidgets.QPushButton("Face Detection In Real Time",
                                                                          self.mainWidget)
        self.mainWidgetFaceDetectionInRealTimeBtn.setGeometry(0, 40, 288, 40)
        self.mainWidgetFaceDetectionInRealTimeBtn.setStyleSheet(
            "Text-align:left;padding-left: 10px;font-size:13px; border-top:1px solid #9e9e9e")
        self.mainWidgetFaceDetectionInRealTimeBtn.setIcon(QtGui.QIcon("images/web_hi_res_5125.png"))
        self.mainWidgetFaceDetectionInRealTimeBtn.setIconSize(QtCore.QSize(35, 35))
        self.mainWidgetFaceDetectionInRealTimeBtn.clicked.connect(self.mainWidgetFaceDetectionInRealTimeBtn_clicked)

        self.mainWidgetEyeDetectionInImageBtn = QtWidgets.QPushButton("Eye Detection In Image", self.mainWidget)
        self.mainWidgetEyeDetectionInImageBtn.setGeometry(0, 80, 288, 40)
        self.mainWidgetEyeDetectionInImageBtn.setStyleSheet(
            "Text-align:left;padding-left: 10px;font-size:13px; border-top:1px solid #9e9e9e")
        self.mainWidgetEyeDetectionInImageBtn.setIcon(QtGui.QIcon("images/web_hi_res_512.png"))
        self.mainWidgetEyeDetectionInImageBtn.setIconSize(QtCore.QSize(35, 35))
        self.mainWidgetEyeDetectionInImageBtn.clicked.connect(self.mainWidgetEyeDetectionInImage_clicked)

        self.mainWidgetEyeDetectionInRealTimeBtn = QtWidgets.QPushButton('Eye Detection In Real Time', self.mainWidget)
        self.mainWidgetEyeDetectionInRealTimeBtn.setGeometry(0, 120, 288, 40)
        self.mainWidgetEyeDetectionInRealTimeBtn.setStyleSheet(
            "Text-align:left;padding-left: 10px;font-size:13px; border-top:1px solid #9e9e9e")
        self.mainWidgetEyeDetectionInRealTimeBtn.setIcon(QtGui.QIcon("images/web_hi_res_5122.png"))
        self.mainWidgetEyeDetectionInRealTimeBtn.setIconSize(QtCore.QSize(35, 35))
        self.mainWidgetEyeDetectionInRealTimeBtn.clicked.connect(self.mainWidgetEyeDetectionInRealTimeBtn_clicked)

        self.mainWidgetFaceAveragingImagesBtn = QtWidgets.QPushButton('Face Averaging Images', self.mainWidget)
        self.mainWidgetFaceAveragingImagesBtn.setGeometry(0, 160, 288, 40)
        self.mainWidgetFaceAveragingImagesBtn.setStyleSheet(
            "Text-align:left;padding-left: 10px;font-size:13px; border-top:1px solid #9e9e9e")
        self.mainWidgetFaceAveragingImagesBtn.setIcon(QtGui.QIcon("images/web_hi_res_5124.png"))
        self.mainWidgetFaceAveragingImagesBtn.setIconSize(QtCore.QSize(35, 35))
        self.mainWidgetFaceAveragingImagesBtn.clicked.connect(self.mainWidgetFaceAveragingImagesBtn_clicked)

        self.mainWidgetFaceAveragingInRealTimeBtn = QtWidgets.QPushButton('Real Time Face Averaging', self.mainWidget)
        self.mainWidgetFaceAveragingInRealTimeBtn.setGeometry(0, 200, 288, 40)
        self.mainWidgetFaceAveragingInRealTimeBtn.setStyleSheet(
            "Text-align:left;padding-left: 10px;font-size:13px; border-top:1px solid #9e9e9e")
        self.mainWidgetFaceAveragingInRealTimeBtn.setIcon(QtGui.QIcon("images/web_hi_res_5126.png"))
        self.mainWidgetFaceAveragingInRealTimeBtn.setIconSize(QtCore.QSize(35, 35))
        # self.mainWidgetFaceAveragingInRealTimeBtn.clicked.connect()

        self.mainWidgetFaceSwapImagesBtn = QtWidgets.QPushButton('Face Swap Images', self.mainWidget)
        self.mainWidgetFaceSwapImagesBtn.setGeometry(0, 240, 288, 40)
        self.mainWidgetFaceSwapImagesBtn.setStyleSheet(
            "Text-align:left;padding-left: 10px;font-size:13px; border-top:1px solid #9e9e9e")
        self.mainWidgetFaceSwapImagesBtn.setIcon(QtGui.QIcon("images/web_hi_res_5127.png"))
        self.mainWidgetFaceSwapImagesBtn.setIconSize(QtCore.QSize(35, 35))
        self.mainWidgetFaceSwapImagesBtn.clicked.connect(self.mainWidgetFaceSwapImagesBtn_clicked)

        self.mainWidgetFaceMorphImagesBtn = QtWidgets.QPushButton('Face Morphing Images', self.mainWidget)
        self.mainWidgetFaceMorphImagesBtn.setGeometry(0, 280, 288, 40)
        self.mainWidgetFaceMorphImagesBtn.setStyleSheet(
            "Text-align:left;padding-left: 10px;font-size:13px; border-top:1px solid #9e9e9e")
        self.mainWidgetFaceMorphImagesBtn.setIcon(QtGui.QIcon("images/web_hi_res_5128.png"))
        self.mainWidgetFaceMorphImagesBtn.setIconSize(QtCore.QSize(35, 35))
        self.mainWidgetFaceMorphImagesBtn.clicked.connect(self.mainWidgetFaceMorphImagesBtn_clicked)

        self.mainWidgetDelaunaryImagesBtn = QtWidgets.QPushButton('Get Delaunary Image', self.mainWidget)
        self.mainWidgetDelaunaryImagesBtn.setGeometry(0, 320, 288, 40)
        self.mainWidgetDelaunaryImagesBtn.setStyleSheet(
            "Text-align:left;padding-left: 10px;font-size:13px; border-top:1px solid #9e9e9e")
        self.mainWidgetDelaunaryImagesBtn.setIcon(QtGui.QIcon("images/web_hi_res_51210.png"))
        self.mainWidgetDelaunaryImagesBtn.setIconSize(QtCore.QSize(35, 35))
        self.mainWidgetDelaunaryImagesBtn.clicked.connect(self.mainWidgetDelaunaryImagesBtn_clicked)

        self.mainWidgetMotionImageBtn = QtWidgets.QPushButton('Real Time Motion Capture', self.mainWidget)
        self.mainWidgetMotionImageBtn.setGeometry(0, 360, 288, 40)
        self.mainWidgetMotionImageBtn.setStyleSheet(
            "Text-align:left;padding-left: 10px;font-size:13px; border-top:1px solid #9e9e9e")
        self.mainWidgetMotionImageBtn.setIcon(QtGui.QIcon("images/web_hi_res_5129.png"))
        self.mainWidgetMotionImageBtn.setIconSize(QtCore.QSize(35, 35))
        self.mainWidgetMotionImageBtn.clicked.connect(self.mainWidgetMotionImagesBtn_clicked)

        self.mainWidgetEyeTrackingImageBtn = QtWidgets.QPushButton('Real Time Eye Tracking', self.mainWidget)
        self.mainWidgetEyeTrackingImageBtn.setGeometry(0, 400, 288, 40)
        self.mainWidgetEyeTrackingImageBtn.setStyleSheet(
            "Text-align:left;padding-left: 10px;font-size:13px; border-top:1px solid #9e9e9e; border-bottom:1px solid #9e9e9e")
        self.mainWidgetEyeTrackingImageBtn.setIcon(QtGui.QIcon("images/web_hi_res_51211.png"))
        self.mainWidgetEyeTrackingImageBtn.setIconSize(QtCore.QSize(35, 35))
        self.mainWidgetEyeTrackingImageBtn.clicked.connect(self.mainWidgetEyeTrackingImageBtn_clicked)

        # display an image in a qlabel

    def displayImage(self):

        qFormat = QtGui.QImage.Format_Indexed8
        if len(self.image.shape) == 3:
            if (self.image.shape[2]) == 4:
                qFormat = QtGui.QImage.Format_RGBA8888
            else:
                qFormat = QtGui.QImage.Format_RGB888

        img = QtGui.QImage(self.image, self.image.shape[1], self.image.shape[0], self.image.strides[0], qFormat)
        # BGR > RGB
        img = img.rgbSwapped()
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setPixmap(QtGui.QPixmap.fromImage(img))
        # self.label.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)

    # find faces if they exixt on a selected image
    def findFaces(self):
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.faces = self.face_cascade.detectMultiScale(self.gray, 1.3, 5)
        for (x, y, w, h) in self.faces:
            cv2.rectangle(self.image, (x, y), (x + w, y + h), (0, 0, 255), 1)

        self.displayImage()

    # hide the display video label
    def hideDisplayedVideoLabel(self):
        self.timer.stop()
        self.displayVideoLabel.setVisible(False)
        self.capture.release()

    # function to perform real time face detection
    def realTimeFaceDetection(self):
        self.capture = cv2.VideoCapture(0)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 550)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 600)

        self.timer = QtCore.QTimer(self.window)
        self.timer.timeout.connect(self.updateFrame)
        self.timer.start(5)

    def updateFrame(self):
        ret, self.image = self.capture.read()
        self.image = cv2.flip(self.image, 1)

        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.realfaces = self.face_cascade.detectMultiScale(self.gray, 1.3, 5)
        for (x, y, w, h) in self.realfaces:
            cv2.rectangle(self.image, (x, y), (x + w, y + h), (0, 0, 255), 1)

        self.displayRealTimeInQlabel(self.image, 1)

    # function to display real time video on qlabel
    def displayRealTimeInQlabel(self, image, window=1):
        qFormat = QtGui.QImage.Format_Indexed8
        if len(image.shape) == 3:
            if (image.shape[2]) == 4:
                qFormat = QtGui.QImage.Format_RGBA8888
            else:
                qFormat = QtGui.QImage.Format_RGB888

        outimg = QtGui.QImage(image, image.shape[1], image.shape[0], image.strides[0], qFormat)
        # BGR > RGB
        outimg = outimg.rgbSwapped()

        if window == 1:
            self.displayVideoLabel.setVisible(True)
            self.displayVideoLabel.setPixmap(QtGui.QPixmap.fromImage(outimg))
            self.displayVideoLabel.setScaledContents(True)

    # find eyes in a selected image if they exist
    def findEeyes(self):
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.faces = self.face_cascade.detectMultiScale(self.gray, 1.3, 5)
        for (x, y, w, h) in self.faces:
            # cv2.rectangle(self.image, (x, y), (x + w, y + h), (0, 0, 255), 1)
            self.roi_gray = self.gray[y:y + h, x:x + w]
            self.roi_color = self.image[y:y + h, x:x + w]
            self.eyes = self.eye_cascade.detectMultiScale(self.roi_gray)
            for (ex, ey, ew, eh) in self.eyes:
                cv2.rectangle(self.roi_color, (ex, ey), (ex + ew, ey + eh), (255, 0, 0), 1)

        self.displayImage()

    # function to perform real time eye detection
    def realTimeEyeDetection(self):
        self.capture = cv2.VideoCapture(0)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 550)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 600)

        self.timer = QtCore.QTimer(self.window)
        self.timer.timeout.connect(self.updateEyeDetectionFrame)
        self.timer.start(5)

    def updateEyeDetectionFrame(self):
        ret, self.image = self.capture.read()
        self.image = cv2.flip(self.image, 1)

        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.realfaces = self.face_cascade.detectMultiScale(self.gray, 1.3, 5)
        for (x, y, w, h) in self.realfaces:
            # cv2.rectangle(self.image, (x, y), (x + w, y + h), (0, 0, 255), 1)
            self.roi_gray = self.gray[y:y + h, x:x + w]
            self.roi_color = self.image[y:y + h, x:x + w]
            self.eyes = self.eye_cascade.detectMultiScale(self.roi_gray)
            for (ex, ey, ew, eh) in self.eyes:
                cv2.rectangle(self.roi_color, (ex, ey), (ex + ew, ey + eh), (255, 0, 0), 1)

        self.displayRealTimeInQlabel(self.image, 1)

    # ---------------------------------------------------Face averaging outside code--------------------------------------------------------------------
    # Read points from text files in directory
    def readPoints(self, path):
        # Create an array of array of points.
        pointsArray = [];

        # List all files in the directory and read points from text files one by one
        for filePath in sorted(os.listdir(path)):

            if filePath.endswith(".txt"):

                # Create an array of points.
                points = [];

                # Read points from filePath
                with open(os.path.join(path, filePath)) as file:
                    for line in file:
                        x, y = line.split(',')
                        points.append((int(x), int(y)))

                # Store array of points
                pointsArray.append(points)

        return pointsArray;

    # Read all jpg images in folder.
    def readImages(self, path):
        # Create array of array of images.
        imagesArray = [];

        # List all files in the directory and read points from text files one by one
        for filePath in sorted(os.listdir(path)):

            if filePath.endswith(".jpg"):
                # Read image found.
                img = cv2.imread(os.path.join(path, filePath));

                # Convert to floating point
                img = np.float32(img) / 255.0;

                # Add to array of images
                imagesArray.append(img);

        return imagesArray;

    # Compute similarity transform given two sets of two points.
    # OpenCV requires 3 pairs of corresponding points.
    # We are faking the third one.

    def similarityTransform(self, inPoints, outPoints):
        s60 = math.sin(60 * math.pi / 180);
        c60 = math.cos(60 * math.pi / 180);

        inPts = np.copy(inPoints).tolist();
        outPts = np.copy(outPoints).tolist();

        xin = c60 * (inPts[0][0] - inPts[1][0]) - s60 * (inPts[0][1] - inPts[1][1]) + inPts[1][0];
        yin = s60 * (inPts[0][0] - inPts[1][0]) + c60 * (inPts[0][1] - inPts[1][1]) + inPts[1][1];

        inPts.append([np.int(xin), np.int(yin)]);

        xout = c60 * (outPts[0][0] - outPts[1][0]) - s60 * (outPts[0][1] - outPts[1][1]) + outPts[1][0];
        yout = s60 * (outPts[0][0] - outPts[1][0]) + c60 * (outPts[0][1] - outPts[1][1]) + outPts[1][1];

        outPts.append([np.int(xout), np.int(yout)]);

        tform = cv2.estimateRigidTransform(np.array([inPts]), np.array([outPts]), False);

        return tform;

    # Check if a point is inside a rectangle
    def rectContains(self, rect, point):
        if point[0] < rect[0]:
            return False
        elif point[1] < rect[1]:
            return False
        elif point[0] > rect[2]:
            return False
        elif point[1] > rect[3]:
            return False
        return True

    # Calculate delanauy triangle
    def calculateDelaunayTriangles(self, rect, points):
        # Create subdiv
        subdiv = cv2.Subdiv2D(rect);

        # Insert points into subdiv
        for p in points:
            subdiv.insert((p[0], p[1]));

        # List of triangles. Each triangle is a list of 3 points ( 6 numbers )
        triangleList = subdiv.getTriangleList();

        # Find the indices of triangles in the points array

        delaunayTri = []

        for t in triangleList:
            pt = []
            pt.append((t[0], t[1]))
            pt.append((t[2], t[3]))
            pt.append((t[4], t[5]))

            pt1 = (t[0], t[1])
            pt2 = (t[2], t[3])
            pt3 = (t[4], t[5])

            if self.rectContains(rect, pt1) and self.rectContains(rect, pt2) and self.rectContains(rect, pt3):
                ind = []
                for j in xrange(0, 3):
                    for k in xrange(0, len(points)):
                        if (abs(pt[j][0] - points[k][0]) < 1.0 and abs(pt[j][1] - points[k][1]) < 1.0):
                            ind.append(k)
                if len(ind) == 3:
                    delaunayTri.append((ind[0], ind[1], ind[2]))

        return delaunayTri

    def constrainPoint(self, p, w, h):
        p = (min(max(p[0], 0), w - 1), min(max(p[1], 0), h - 1))
        return p;

    # Apply affine transform calculated using srcTri and dstTri to src and
    # output an image of size.
    def applyAffineTransform(self, src, srcTri, dstTri, size):
        # Given a pair of triangles, find the affine transform.
        warpMat = cv2.getAffineTransform(np.float32(srcTri), np.float32(dstTri))

        # Apply the Affine Transform just found to the src image
        dst = cv2.warpAffine(src, warpMat, (size[0], size[1]), None, flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REFLECT_101)

        return dst

    # Warps and alpha blends triangular regions from img1 and img2 to img
    def warpTriangle(self, img1, img2, t1, t2):
        # Find bounding rectangle for each triangle
        r1 = cv2.boundingRect(np.float32([t1]))
        r2 = cv2.boundingRect(np.float32([t2]))

        # Offset points by left top corner of the respective rectangles
        t1Rect = []
        t2Rect = []
        t2RectInt = []

        for i in xrange(0, 3):
            t1Rect.append(((t1[i][0] - r1[0]), (t1[i][1] - r1[1])))
            t2Rect.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))
            t2RectInt.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))

        # Get mask by filling triangle
        mask = np.zeros((r2[3], r2[2], 3), dtype=np.float32)
        cv2.fillConvexPoly(mask, np.int32(t2RectInt), (1.0, 1.0, 1.0), 16, 0);

        # Apply warpImage to small rectangular patches
        img1Rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]

        size = (r2[2], r2[3])

        img2Rect = self.applyAffineTransform(img1Rect, t1Rect, t2Rect, size)

        img2Rect = img2Rect * mask

        # Copy triangular region of the rectangular patch to the output image
        img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] = img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] * (
                (1.0, 1.0, 1.0) - mask)

        img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] = img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] + img2Rect

        # the function to display the average face image

    def displayAveragedFaces(self):
        path = 'img/'

        # Dimensions of output image
        w = 600;
        h = 600;

        # Read points for all images
        allPoints = self.readPoints(path);
        # print np.shape(allPoints)

        # Read all images
        images = self.readImages(path);
        # print images

        # Eye corners
        eyecornerDst = [(np.int(0.3 * w), np.int(h / 3)), (np.int(0.7 * w), np.int(h / 3))];

        imagesNorm = [];
        pointsNorm = [];

        # Add boundary points for delaunay triangulation
        boundaryPts = np.array(
            [(0, 0), (w / 2, 0), (w - 1, 0), (w - 1, h / 2), (w - 1, h - 1), (w / 2, h - 1), (0, h - 1), (0, h / 2)]);

        # Initialize location of average points to 0s
        pointsAvg = np.array([(0, 0)] * (len(allPoints[0]) + len(boundaryPts)), np.float32());
        # print np.shape(pointsAvg)
        n = len(allPoints[0]);
        # print n

        numImages = len(images)

        # Warp images and trasnform landmarks to output coordinate system,
        # and find average of transformed landmarks.

        for i in xrange(0, numImages):
            points1 = allPoints[i];

            # Corners of the eye in input image
            eyecornerSrc = [allPoints[i][36], allPoints[i][45]];

            # Compute similarity transform
            tform = self.similarityTransform(eyecornerSrc, eyecornerDst);

            # Apply similarity transformation
            img = cv2.warpAffine(images[i], tform, (w, h));

            # Apply similarity transform on points
            points2 = np.reshape(np.array(points1), (68, 1, 2));

            points = cv2.transform(points2, tform);

            points = np.float32(np.reshape(points, (68, 2)));

            # Append boundary points. Will be used in Delaunay Triangulation
            points = np.append(points, boundaryPts, axis=0)

            # Calculate location of average landmark points.
            pointsAvg = pointsAvg + points / numImages;

            pointsNorm.append(points);
            imagesNorm.append(img);

        # Delaunay triangulation
        rect = (0, 0, w, h);
        dt = self.calculateDelaunayTriangles(rect, np.array(pointsAvg));

        # Output image
        output = np.zeros((h, w, 3), np.float32());

        # Warp input images to average image landmarks
        for i in xrange(0, len(imagesNorm)):
            img = np.zeros((h, w, 3), np.float32());
            # Transform triangles one by one
            for j in xrange(0, len(dt)):
                tin = [];
                tout = [];

                for k in xrange(0, 3):
                    pIn = pointsNorm[i][dt[j][k]];
                    pIn = self.constrainPoint(pIn, w, h);

                    pOut = pointsAvg[dt[j][k]];
                    pOut = self.constrainPoint(pOut, w, h);

                    tin.append(pIn);
                    tout.append(pOut);

                self.warpTriangle(imagesNorm[i], img, tin, tout);

            # Add image intensities for averaging
            output = output + img;

        # Divide by numImages to get average
        output = output / numImages;
        # Display result
        # the image output is a float32 image so we have to convert it first to uint8
        self.image = (output * 255).round().astype(np.uint8)
        self.displayImage()

    # ----------------------------------------------------------------------------face swap functions---------------------------------------------
    # Read points from text file
    def readSwapPoints(self, path):
        # Create an array of points.
        points = [];

        # Read points
        with open(path) as file:
            for line in file:
                x, y = line.split()
                points.append((int(x), int(y)))

        return points

    # Apply affine transform calculated using srcTri and dstTri to src and
    # output an image of size.
    def applyAffineSwapTransform(self, src, srcTri, dstTri, size):

        # Given a pair of triangles, find the affine transform.
        warpMat = cv2.getAffineTransform(np.float32(srcTri), np.float32(dstTri))

        # Apply the Affine Transform just found to the src image
        dst = cv2.warpAffine(src, warpMat, (size[0], size[1]), None, flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REFLECT_101)

        return dst

    # Check if a point is inside a rectangle
    def rectSwapContains(self, rect, point):
        if point[0] < rect[0]:
            return False
        elif point[1] < rect[1]:
            return False
        elif point[0] > rect[0] + rect[2]:
            return False
        elif point[1] > rect[1] + rect[3]:
            return False
        return True

    # calculate delanauy triangle
    def calculateSwapDelaunayTriangles(self, rect, points):
        # create subdiv
        subdiv = cv2.Subdiv2D(rect);

        # Insert points into subdiv
        for p in points:
            subdiv.insert(p)

        triangleList = subdiv.getTriangleList();

        delaunayTri = []

        pt = []

        for t in triangleList:
            pt.append((t[0], t[1]))
            pt.append((t[2], t[3]))
            pt.append((t[4], t[5]))

            pt1 = (t[0], t[1])
            pt2 = (t[2], t[3])
            pt3 = (t[4], t[5])

            if self.rectSwapContains(rect, pt1) and self.rectSwapContains(rect, pt2) and self.rectSwapContains(rect,
                                                                                                               pt3):
                ind = []
                # Get face-points (from 68 face detector) by coordinates
                for j in xrange(0, 3):
                    for k in xrange(0, len(points)):
                        if (abs(pt[j][0] - points[k][0]) < 1.0 and abs(pt[j][1] - points[k][1]) < 1.0):
                            ind.append(k)
                            # Three points form a triangle. Triangle array corresponds to the file tri.txt in FaceMorph
                if len(ind) == 3:
                    delaunayTri.append((ind[0], ind[1], ind[2]))

            pt = []

        return delaunayTri

    # Warps and alpha blends triangular regions from img1 and img2 to img
    def warpSwapTriangle(self, img1, img2, t1, t2):

        # Find bounding rectangle for each triangle
        r1 = cv2.boundingRect(np.float32([t1]))
        r2 = cv2.boundingRect(np.float32([t2]))

        # Offset points by left top corner of the respective rectangles
        t1Rect = []
        t2Rect = []
        t2RectInt = []

        for i in xrange(0, 3):
            t1Rect.append(((t1[i][0] - r1[0]), (t1[i][1] - r1[1])))
            t2Rect.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))
            t2RectInt.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))

        # Get mask by filling triangle
        mask = np.zeros((r2[3], r2[2], 3), dtype=np.float32)
        cv2.fillConvexPoly(mask, np.int32(t2RectInt), (1.0, 1.0, 1.0), 16, 0);

        # Apply warpImage to small rectangular patches
        img1Rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
        # img2Rect = np.zeros((r2[3], r2[2]), dtype = img1Rect.dtype)

        size = (r2[2], r2[3])

        img2Rect = self.applyAffineSwapTransform(img1Rect, t1Rect, t2Rect, size)

        img2Rect = img2Rect * mask

        # Copy triangular region of the rectangular patch to the output image
        img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] = img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] * (
                (1.0, 1.0, 1.0) - mask)

        img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] = img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] + img2Rect

    # display swapped faces
    def displaySwappedFaces(self):
        # Make sure OpenCV is version 3.0 or above
        (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

        if int(major_ver) < 3:
            print >> sys.stderr, 'ERROR: Script needs OpenCV 3.0 or higher'
            sys.exit(1)

        # Read images
        filename1 = '/home/ily19/PycharmProjects/imgProcessing/swapImagesNeeded/filename1.jpg'
        filename2 = '/home/ily19/PycharmProjects/imgProcessing/swapImagesNeeded/filename2.jpg'

        img1 = cv2.imread(filename1);
        img2 = cv2.imread(filename2);
        img1Warped = np.copy(img2);

        # Read array of corresponding points
        points1 = self.readSwapPoints(filename1 + '.txt')
        points2 = self.readSwapPoints(filename2 + '.txt')

        # Find convex hull
        hull1 = []
        hull2 = []

        hullIndex = cv2.convexHull(np.array(points2), returnPoints=False)

        for i in xrange(0, len(hullIndex)):
            hull1.append(points1[int(hullIndex[i])])
            hull2.append(points2[int(hullIndex[i])])

        # Find delanauy traingulation for convex hull points
        sizeImg2 = img2.shape
        rect = (0, 0, sizeImg2[1], sizeImg2[0])

        dt = self.calculateSwapDelaunayTriangles(rect, hull2)

        if len(dt) == 0:
            quit()

        # Apply affine transformation to Delaunay triangles
        for i in xrange(0, len(dt)):
            t1 = []
            t2 = []

            # get points for img1, img2 corresponding to the triangles
            for j in xrange(0, 3):
                t1.append(hull1[dt[i][j]])
                t2.append(hull2[dt[i][j]])

            self.warpSwapTriangle(img1, img1Warped, t1, t2)

        # Calculate Mask
        hull8U = []
        for i in xrange(0, len(hull2)):
            hull8U.append((hull2[i][0], hull2[i][1]))

        mask = np.zeros(img2.shape, dtype=img2.dtype)

        cv2.fillConvexPoly(mask, np.int32(hull8U), (255, 255, 255))

        r = cv2.boundingRect(np.float32([hull2]))

        center = ((r[0] + int(r[2] / 2), r[1] + int(r[3] / 2)))

        # Clone seamlessly.
        output = cv2.seamlessClone(np.uint8(img1Warped), img2, mask, center, cv2.NORMAL_CLONE)

        # self.image = (output * 255).round().astype(np.uint8)
        self.image = output
        self.displaySwapImage()

        # cv2.imshow("Face Swapped", output)
        # cv2.waitKey(0)

        # cv2.destroyAllWindows()

    def displaySwapImage(self):
        self.swapImageLabel.setVisible(True)

        qFormat = QtGui.QImage.Format_Indexed8
        if len(self.image.shape) == 3:
            if (self.image.shape[2]) == 4:
                qFormat = QtGui.QImage.Format_RGBA8888
            else:
                qFormat = QtGui.QImage.Format_RGB888

        img = QtGui.QImage(self.image, self.image.shape[1], self.image.shape[0], self.image.strides[0], qFormat)
        # BGR > RGB
        img = img.rgbSwapped()
        self.swapImageLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.swapImageLabel.setPixmap(QtGui.QPixmap.fromImage(img))
        self.swapImageLabel.setScaledContents(True)

    # ----------------------------------------------------face morph functions--------------------------------------------------------------------
    # Read points from text file of morph
    def readMorphPoints(self, path):
        # Create an array of points.
        points = [];
        # Read points
        with open(path) as file:
            for line in file:
                x, y = line.split()
                points.append((int(x), int(y)))

        return points

    # Apply affine transform calculated using srcTri and dstTri to src and for morph images
    # output an image of size.
    def applyMorphAffineTransform(self, src, srcTri, dstTri, size):

        # Given a pair of triangles, find the affine transform.
        warpMat = cv2.getAffineTransform(np.float32(srcTri), np.float32(dstTri))

        # Apply the Affine Transform just found to the src image
        dst = cv2.warpAffine(src, warpMat, (size[0], size[1]), None, flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REFLECT_101)

        return dst

    # Warps and alpha blends triangular regions from img1 and img2 to img
    def morphMorphTriangle(self, img1, img2, img, t1, t2, t, alpha):

        # Find bounding rectangle for each triangle
        r1 = cv2.boundingRect(np.float32([t1]))
        r2 = cv2.boundingRect(np.float32([t2]))
        r = cv2.boundingRect(np.float32([t]))

        # Offset points by left top corner of the respective rectangles
        t1Rect = []
        t2Rect = []
        tRect = []

        for i in xrange(0, 3):
            tRect.append(((t[i][0] - r[0]), (t[i][1] - r[1])))
            t1Rect.append(((t1[i][0] - r1[0]), (t1[i][1] - r1[1])))
            t2Rect.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))

        # Get mask by filling triangle
        mask = np.zeros((r[3], r[2], 3), dtype=np.float32)
        cv2.fillConvexPoly(mask, np.int32(tRect), (1.0, 1.0, 1.0), 16, 0);

        # Apply warpImage to small rectangular patches
        img1Rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
        img2Rect = img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]]

        size = (r[2], r[3])
        warpImage1 = self.applyMorphAffineTransform(img1Rect, t1Rect, tRect, size)
        warpImage2 = self.applyMorphAffineTransform(img2Rect, t2Rect, tRect, size)

        # Alpha blend rectangular patches
        imgRect = (1.0 - alpha) * warpImage1 + alpha * warpImage2

        # Copy triangular region of the rectangular patch to the output image
        img[r[1]:r[1] + r[3], r[0]:r[0] + r[2]] = img[r[1]:r[1] + r[3], r[0]:r[0] + r[2]] * (1 - mask) + imgRect * mask

    def displayMorphImages(self):
        filename1 = '/home/ily19/PycharmProjects/imgProcessing/morphImagesNeeded/filename1.jpg'
        filename2 = '/home/ily19/PycharmProjects/imgProcessing/morphImagesNeeded/filename2.jpg'
        alpha = 0.5

        # Read images
        img1 = cv2.imread(filename1);
        img2 = cv2.imread(filename2);

        # Convert Mat to float data type
        img1 = np.float32(img1)
        img2 = np.float32(img2)

        # Read array of corresponding points
        points1 = self.readMorphPoints(filename1 + '.txt')
        points2 = self.readMorphPoints(filename2 + '.txt')
        points = [];

        # Compute weighted average point coordinates
        for i in xrange(0, len(points1)):
            x = (1 - alpha) * points1[i][0] + alpha * points2[i][0]
            y = (1 - alpha) * points1[i][1] + alpha * points2[i][1]
            points.append((x, y))

        # Allocate space for final output
        imgMorph = np.zeros(img1.shape, dtype=img1.dtype)

        # Read triangles from tri.txt
        with open("/home/ily19/PycharmProjects/imgProcessing/morphImagesNeeded/tri.txt") as file:
            for line in file:
                x, y, z = line.split()

                x = int(x)
                y = int(y)
                z = int(z)

                t1 = [points1[x], points1[y], points1[z]]
                t2 = [points2[x], points2[y], points2[z]]
                t = [points[x], points[y], points[z]]

                # Morph one triangle at a time.
                self.morphMorphTriangle(img1, img2, imgMorph, t1, t2, t, alpha)

        # Display Result
        output = np.uint8(imgMorph)
        self.image = output
        self.displayMorphImage()

    def displayMorphImage(self):
        self.swapImageLabel.setVisible(True)

        qFormat = QtGui.QImage.Format_Indexed8
        if len(self.image.shape) == 3:
            if (self.image.shape[2]) == 4:
                qFormat = QtGui.QImage.Format_RGBA8888
            else:
                qFormat = QtGui.QImage.Format_RGB888

        img = QtGui.QImage(self.image, self.image.shape[1], self.image.shape[0], self.image.strides[0], qFormat)
        # BGR > RGB
        img = img.rgbSwapped()
        self.swapImageLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.swapImageLabel.setPixmap(QtGui.QPixmap.fromImage(img))
        self.swapImageLabel.setScaledContents(True)

    # -------------------------------------------------Delaunary function--------------------------------------------------------------------

    # Check if a point is inside a rectangle
    def rect_containsD(self, rect, point):
        if point[0] < rect[0]:
            return False
        elif point[1] < rect[1]:
            return False
        elif point[0] > rect[2]:
            return False
        elif point[1] > rect[3]:
            return False
        return True

    # Draw a point
    def draw_pointD(self, img, p, color):
        cv2.circle(img, p, 2, color, 1, cv2.LINE_AA, 0)

    # Draw delaunay triangles
    def draw_delaunayD(self, img, subdiv, delaunay_color):
        triangleList = subdiv.getTriangleList();
        size = img.shape
        r = (0, 0, size[1], size[0])

        for t in triangleList:

            pt1 = (t[0], t[1])
            pt2 = (t[2], t[3])
            pt3 = (t[4], t[5])

            if self.rect_containsD(r, pt1) and self.rect_containsD(r, pt2) and self.rect_containsD(r, pt3):
                cv2.line(img, pt1, pt2, delaunay_color, 1, cv2.LINE_AA, 0)
                cv2.line(img, pt2, pt3, delaunay_color, 1, cv2.LINE_AA, 0)
                cv2.line(img, pt3, pt1, delaunay_color, 1, cv2.LINE_AA, 0)

    # Draw voronoi diagram
    def draw_voronoiD(self, img, subdiv):
        (facets, centers) = subdiv.getVoronoiFacetList([])

        for i in xrange(0, len(facets)):
            ifacet_arr = []
            for f in facets[i]:
                ifacet_arr.append(f)

            ifacet = np.array(ifacet_arr, np.int)
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

            cv2.fillConvexPoly(img, ifacet, color, cv2.LINE_AA, 0);
            ifacets = np.array([ifacet])
            cv2.polylines(img, ifacets, True, (0, 0, 0), 1, cv2.LINE_AA, 0)
            cv2.circle(img, (centers[i][0], centers[i][1]), 3, (0, 0, 0), 1, cv2.LINE_AA, 0)

    def displayDelaunary(self):

        # Turn on animation while drawing triangles
        animate = True

        # Define colors for drawing.
        delaunay_color = (255, 255, 255)
        points_color = (0, 0, 255)

        # Read in the image.
        img = cv2.imread("/home/ily19/PycharmProjects/imgProcessing/imgD/filename1.jpg");

        # Keep a copy around
        img_orig = img.copy();

        # Rectangle to be used with Subdiv2D
        size = img.shape
        rect = (0, 0, size[1], size[0])

        # Create an instance of Subdiv2D
        subdiv = cv2.Subdiv2D(rect);

        # Create an array of points.
        points = [];

        # Read in the points from a text file
        with open("/home/ily19/PycharmProjects/imgProcessing/imgD/filename1.txt") as file:
            for line in file:
                x, y = line.split()
                points.append((int(x), int(y)))

        # Insert points into subdiv
        for p in points:
            subdiv.insert(p)

            # Show animation
            if animate:
                img_copy = img_orig.copy()
                # Draw delaunay triangles
                self.draw_delaunayD(img_copy, subdiv, (255, 255, 255));
                cv2.namedWindow('show animation', flags=cv2.WINDOW_GUI_NORMAL)
                cv2.resizeWindow('show animation', 600, 525)
                cv2.imshow("show animation", img_copy)
                cv2.moveWindow("show animation", 235, 85)
                cv2.waitKeyEx(100)

        cv2.destroyWindow('show animation')

        # Draw delaunay triangles
        self.draw_delaunayD(img, subdiv, (255, 255, 255));

        # Draw points
        for p in points:
            self.draw_pointD(img, p, (0, 0, 255))

        # Allocate space for voronoi Diagram
        img_voronoi = np.zeros(img.shape, dtype=img.dtype)

        # Draw voronoi diagram
        self.draw_voronoiD(img_voronoi, subdiv)

        # Show results
        self.image = img
        self.displayOriginalDelaunaryImage()
        self.image = img_voronoi
        self.displayResultDelaunaryImage()
        # cv2.imshow("img", img)
        # cv2.imshow("ggggg", img_voronoi)
        # cv2.waitKey(0)

    def displayOriginalDelaunaryImage(self):
        self.originalDelaunaryImageLabel.setVisible(True)

        qFormat = QtGui.QImage.Format_Indexed8
        if len(self.image.shape) == 3:
            if (self.image.shape[2]) == 4:
                qFormat = QtGui.QImage.Format_RGBA8888
            else:
                qFormat = QtGui.QImage.Format_RGB888

        img = QtGui.QImage(self.image, self.image.shape[1], self.image.shape[0], self.image.strides[0], qFormat)
        # BGR > RGB
        img = img.rgbSwapped()
        self.originalDelaunaryImageLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.originalDelaunaryImageLabel.setPixmap(QtGui.QPixmap.fromImage(img))
        self.originalDelaunaryImageLabel.setScaledContents(True)

    def displayResultDelaunaryImage(self):
        self.resulltDelaunaryImageLabel.setVisible(True)

        qFormat = QtGui.QImage.Format_Indexed8
        if len(self.image.shape) == 3:
            if (self.image.shape[2]) == 4:
                qFormat = QtGui.QImage.Format_RGBA8888
            else:
                qFormat = QtGui.QImage.Format_RGB888

        img = QtGui.QImage(self.image, self.image.shape[1], self.image.shape[0], self.image.strides[0], qFormat)
        # BGR > RGB
        img = img.rgbSwapped()
        self.resulltDelaunaryImageLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.resulltDelaunaryImageLabel.setPixmap(QtGui.QPixmap.fromImage(img))
        self.resulltDelaunaryImageLabel.setScaledContents(True)


# ----------------------------------------------------------------------------------------------------------------------------------------
main1 = MainWindow()



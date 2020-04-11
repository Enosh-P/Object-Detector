import numpy as np
import time, random, cv2, os, sys, logging, imutils

from PyQt5.QtCore import Qt, QBasicTimer
from PyQt5.QtWidgets import (QApplication, QWidget, QToolTip, QPushButton, QLabel,
        QHBoxLayout, QVBoxLayout, QFileDialog, QProgressBar, QPlainTextEdit)
from PyQt5.QtGui import QIcon
from img_detect import detect_image
from live_detect import detect_live
from video_detect import detect_video
from video_detect import check_time

class QTextEditLogger(logging.Handler):
    '''class for setting up logging format'''
    def __init__(self, parent):
        super().__init__()
        self.widget = QPlainTextEdit(parent)
        self.widget.setReadOnly(True)

    def emit(self, record):
        msg = self.format(record)
        self.widget.appendPlainText(msg)


class MyDialog(QWidget):
    '''My Main class for running application'''
    def __init__(self):
        super().__init__()
        #Main Window Setup
        self.setGeometry(300, 300, 700, 700)
        self.setWindowTitle('Multiple Object Detector')
        #to set icon remove the comment and give correct path of the image
        #self.setWindowIcon(QIcon('web.png'))

        #Logging Configuration
        logTextBox  = QTextEditLogger(self)
        logTextBox.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logging.getLogger().addHandler(logTextBox)
        logging.getLogger().setLevel(logging.DEBUG)
        self.loglbl = QLabel('Console output and data loggings:', self)

        #Image Button
        self.img = QPushButton('Image', self)
        self.img.setToolTip('Click to provide <b>IMAGE</b> as input')

        #Video Button
        self.video = QPushButton('Video', self)
        self.video.setToolTip('Click to provide <b>VIDEO</b> as input')

        #Live feed button
        self.live = QPushButton('Live Feed', self)
        self.live.setToolTip('Click to provide <b>LIVE FEED</b> as input')

        #Quit button
        qbtn = QPushButton('Quit', self)
        qbtn.clicked.connect(QApplication.instance().quit)
        qbtn.setToolTip('Click to provide <b>QUIT</b> the application')
        qbtn.resize(qbtn.sizeHint())

        #Progress Bar
        self.lbl = QLabel('Progress of Video processing:', self)
        self.pbar = QProgressBar(self)
        self.timer = QBasicTimer()
        self.step, self.seconds  = 0, 0

        #Creating layout Objects of GUI
        logbox, hbox, layout, hlayout, prog = QVBoxLayout(), QHBoxLayout(), QVBoxLayout(), QHBoxLayout(), QVBoxLayout()

        #Setting space withing layouts
        layout.setSpacing(100)
        hlayout.setSpacing(100)
        prog.setSpacing(20)
        logbox.setSpacing(20)

        #Adding Widgets to layout
        logbox.addWidget(self.loglbl)
        logbox.addWidget(logTextBox.widget)
        hlayout.addWidget(self.img)
        hlayout.addWidget(self.video)
        hlayout.addWidget(self.live)
        prog.addWidget(self.lbl)
        prog.addWidget(self.pbar)
        hbox.addStretch(1)
        hbox.addWidget(qbtn)

        #Adding layouts in layout
        layout.addLayout(logbox)
        layout.addLayout(hlayout)
        layout.addLayout(prog)
        layout.addStretch(1)
        layout.addLayout(hbox)

        #Setting main layout
        self.setLayout(layout)

        # Connect signals to slot
        self.img.clicked.connect(self.image_detection)
        self.video.clicked.connect(self.video_detection)
        self.live.clicked.connect(self.live_detection)
        self.pbar.setValue(0)

    #signal receiver for image Button
    def image_detection(self):
        self.pbar.setValue(0)
        logging.info("Getting Image info...")
        ipath = QFileDialog.getOpenFileName(self, 'Open Image', '/home')[0]
        if ipath: detect_image(ipath)
        else: logging.info("No Image selected")

    #signal receiver for live feed Button
    def live_detection(self):
        self.pbar.setValue(0)
        logging.info("Opening Camera...")
        detect_live()

    #signal receiver for video Button
    def video_detection(self):
        self.pbar.setValue(0)
        logging.info("Getting Video info...")
        vpath = QFileDialog.getOpenFileName(self, 'Open Video', '/home')[0]
        if vpath:
            self.seconds = check_time(vpath)+10
            self.timer.start(self.seconds, self)
            detect_video(vpath)
        else: logging.info("No Video selected")
        self.step = self.seconds
        self.pbar.setValue(self.step)

    #signal receiver for progress bar
    def timerEvent(self, e):
        if self.step >= self.seconds:
            self.timer.stop()
            return
        self.step = self.step + 1
        self.pbar.setValue(self.step)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    dlg = MyDialog()
    dlg.show()
    sys.exit(app.exec_())

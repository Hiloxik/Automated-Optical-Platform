import threading
from threading import Thread
import json
import time
from queue import Queue, Empty
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QLineEdit, QGridLayout, QWidget, QFrame, \
    QMessageBox, QCheckBox, QVBoxLayout
from PyQt5.QtGui import QFont, QDoubleValidator, QPixmap
from PyQt5.QtCore import Qt, pyqtSignal, QThread

import globals
from camera import main as camera_main, get_polygon, get_polygon_tracker, get_shift, get_angle, zoom_in_camera, \
    zoom_out_camera, \
    capture_frame, rgb_plot, rgb_map

# GUI class
class GUI(QMainWindow):

    def __init__(self):
        super().__init__()


        self.initGUI()

    def initGUI(self):
        self.setWindowTitle('Device GUI')
        self.setGeometry(100, 100, 800, 600)

        widget = QWidget(self)
        self.setCentralWidget(widget)

        grid = QGridLayout()
        widget.setLayout(grid)

        # Buttons and labels inside choices_frame with custom colors and font
        label_font = QFont("Arial", 10)
        label_font.setBold(True)

        button_font = QFont("Arial", 10)
        button_size = 50
        radius = button_size // 2


        # Camera label
        camera_label = QLabel(f"FLIR-camera", self)
        font = QFont("Arial")
        font.setPointSize(12)  # Set the font size to 14 points
        camera_label.setFont(font)
        grid.addWidget(camera_label, 0, 0)

        # Camera status
        self.status_label = QLabel("Camera connected successfully.", self)
        font = QFont("Arial")
        font.setPointSize(12)  # Set the font size to 14 points
        self.status_label.setFont(font)
        grid.addWidget(self.status_label, 1, 0)

        # Open camera
        self.open_camera_button = QPushButton('Open Camera')
        self.open_camera_button.setFont(button_font)
        self.open_camera_button.setStyleSheet("background-color: HoneyDew;")
        self.open_camera_button.pressed.connect(self.open_camera)
        self.open_camera_button.setMinimumHeight(70)
        grid.addWidget(self.open_camera_button, 2, 0)

        # Capture screen
        self.capture_button = QPushButton('Capture')
        self.capture_button.setFont(button_font)
        self.capture_button.setStyleSheet("background-color: HoneyDew;")
        self.capture_button.pressed.connect(self.capture)
        self.capture_button.setMinimumHeight(70)
        grid.addWidget(self.capture_button, 3, 0)

        # Tracking mode
        tracker_widget = QWidget()
        tracker_layout = QVBoxLayout()
        self.tracker_image = QLabel()
        pixmap = QPixmap(r"D:\大三下\科研\codes\joystick-transfer\ctypes\2.1.6\pics\tracker.png")
        self.tracker_image.setPixmap(pixmap.scaled(50, 50, Qt.KeepAspectRatio))
        self.tracker_image.mousePressEvent = self.trigger_mode1
        tracker_layout.addWidget(self.tracker_image, alignment=Qt.AlignCenter)
        tracker_widget.setLayout(tracker_layout)
        grid.addWidget(tracker_widget, 0, 1)

        # Drawing mode
        pen_widget = QWidget()
        pen_layout = QVBoxLayout()
        self.pen_image = QLabel()
        pixmap = QPixmap(r"D:\大三下\科研\codes\joystick-transfer\ctypes\2.1.6\pics\pen.png")
        self.pen_image.setPixmap(pixmap.scaled(50, 50, Qt.KeepAspectRatio))
        self.pen_image.mousePressEvent = self.trigger_mode2
        pen_layout.addWidget(self.pen_image, alignment=Qt.AlignCenter)
        pen_widget.setLayout(pen_layout)
        grid.addWidget(pen_widget, 1, 1)

        # Measuring mode
        ruler_widget = QWidget()
        ruler_layout = QVBoxLayout()
        self.ruler_image = QLabel()
        pixmap = QPixmap(r"D:\大三下\科研\codes\joystick-transfer\ctypes\2.1.6\pics\ruler.png")
        self.ruler_image.setPixmap(pixmap.scaled(50, 50, Qt.KeepAspectRatio))
        self.ruler_image.mousePressEvent = self.trigger_mode3
        ruler_layout.addWidget(self.ruler_image, alignment=Qt.AlignCenter)
        ruler_widget.setLayout(ruler_layout)
        grid.addWidget(ruler_widget, 2, 1)

        # Analysis mode
        anal_widget = QWidget()
        anal_layout = QVBoxLayout()
        self.anal_image = QLabel()
        pixmap = QPixmap(r"D:\大三下\科研\codes\joystick-transfer\ctypes\2.1.6\pics\rgb.png")
        self.anal_image.setPixmap(pixmap.scaled(50, 50, Qt.KeepAspectRatio))
        self.anal_image.mousePressEvent = self.trigger_mode4
        anal_layout.addWidget(self.anal_image, alignment=Qt.AlignCenter)
        anal_widget.setLayout(anal_layout)
        grid.addWidget(anal_widget, 3, 4)

        # Zoom in
        self.zoom_in_button = QPushButton('+')
        self.zoom_in_button.setFont(button_font)
        self.zoom_in_button.setFixedSize(button_size, button_size)
        self.zoom_in_button.setStyleSheet((
                "QPushButton {"
                "    background-color: Ivory;"
                "    border-radius: " + str(radius) + "px;"
                                                        "    width: " + str(button_size) + "px;"
                                                                                            "    height: " + str(
            button_size) + "px;"
                            "}"
                            "QPushButton:pressed {"
                            "    background-color: FireBrick;"
                            "    border: 2px solid gray;"
                            "    border-radius: " + str(radius) + "px;"
                                                                    "}"
        ))
        self.zoom_in_button.pressed.connect(self.zoom_in)
        grid.addWidget(self.zoom_in_button, 0, 4)

        # Zoom out
        self.zoom_out_button = QPushButton('-')
        self.zoom_out_button.setFont(button_font)
        self.zoom_out_button.setFixedSize(button_size, button_size)
        self.zoom_out_button.setStyleSheet((
                "QPushButton {"
                "    background-color: Ivory;"
                "    border-radius: " + str(radius) + "px;"
                                                        "    width: " + str(button_size) + "px;"
                                                                                            "    height: " + str(
            button_size) + "px;"
                            "}"
                            "QPushButton:pressed {"
                            "    background-color: LightGray;"
                            "    border: 2px solid gray;"
                            "    border-radius: " + str(radius) + "px;"
                                                                    "}"
        ))
        self.zoom_out_button.pressed.connect(self.zoom_out)
        grid.addWidget(self.zoom_out_button, 1, 4)

        # Fivefold Mirror
        self.fivefold_button = QPushButton('5X')
        self.fivefold_button.setFont(button_font)
        self.fivefold_button.setFixedSize(button_size, button_size)
        self.fivefold_button.setStyleSheet((
                "QPushButton {"
                "    background-color: Crimson;"
                "    border-radius: " + str(radius) + "px;"
                                                        "    width: " + str(button_size) + "px;"
                                                                                            "    height: " + str(
            button_size) + "px;"
                            "}"
                            "QPushButton:pressed {"
                            "    background-color: LightGray;"
                            "    border: 2px solid gray;"
                            "    border-radius: " + str(radius) + "px;"
                                                                    "}"
        ))
        self.fivefold_button.pressed.connect(self.fivefold)
        grid.addWidget(self.fivefold_button, 0, 2)

        # Tenfold Mirror
        self.tenfold_button = QPushButton('10X')
        self.tenfold_button.setFont(button_font)
        self.tenfold_button.setFixedSize(button_size, button_size)
        self.tenfold_button.setStyleSheet((
                "QPushButton {"
                "    background-color: Gold;"
                "    border-radius: " + str(radius) + "px;"
                                                        "    width: " + str(button_size) + "px;"
                                                                                            "    height: " + str(
            button_size) + "px;"
                            "}"
                            "QPushButton:pressed {"
                            "    background-color: GoldenRod;"
                            "    border: 2px solid gray;"
                            "    border-radius: " + str(radius) + "px;"
                                                                    "}"
        ))
        self.tenfold_button.pressed.connect(self.tenfold)
        grid.addWidget(self.tenfold_button, 1, 2)

        # Tenfold Mirror
        self.twentyfold_button = QPushButton('20X')
        self.twentyfold_button.setFont(button_font)
        self.twentyfold_button.setFixedSize(button_size, button_size)
        self.twentyfold_button.setStyleSheet((
                "QPushButton {"
                "    background-color: LawnGreen;"
                "    border-radius: " + str(radius) + "px;"
                                                        "    width: " + str(button_size) + "px;"
                                                                                            "    height: " + str(
            button_size) + "px;"
                            "}"
                            "QPushButton:pressed {"
                            "    background-color: LimeGreen;"
                            "    border: 2px solid gray;"
                            "    border-radius: " + str(radius) + "px;"
                                                                    "}"
        ))
        self.twentyfold_button.pressed.connect(self.twentyfold)
        grid.addWidget(self.twentyfold_button, 2, 2)

        # Retrieve Polygon
        self.retrieve_polygon_button = QPushButton('Retrieve Flake')
        self.retrieve_polygon_button.setFont(button_font)
        self.retrieve_polygon_button.setStyleSheet("background-color: Ivory;")
        self.retrieve_polygon_button.pressed.connect(self.retrieve_polygon)
        grid.addWidget(self.retrieve_polygon_button, 0, 3)

        # Calibrate Polygon
        self.calibrate_button = QPushButton('Calibrate Flake')
        self.calibrate_button.pressed.connect(self.calibrate)
        self.calibrate_button.setFont(button_font)
        self.calibrate_button.setStyleSheet("background-color: Ivory;")
        grid.addWidget(self.calibrate_button, 1, 3)

        # Aiming
        calibrater_widget = QWidget()
        calibrater_layout = QVBoxLayout()
        self.calibrater_image = QLabel()
        pixmap = QPixmap(r"D:\大三下\科研\codes\joystick-transfer\ctypes\2.1.6\pics\calibrator.png")
        self.calibrater_image.setPixmap(pixmap.scaled(50, 50, Qt.KeepAspectRatio))
        self.calibrater_image.mousePressEvent = self.aim
        calibrater_layout.addWidget(self.calibrater_image, alignment=Qt.AlignCenter)
        calibrater_widget.setLayout(calibrater_layout)
        grid.addWidget(calibrater_widget, 3, 1)

        # Align Polygon
        self.align_button = QPushButton('Track Flake')
        self.align_button.pressed.connect(self.align)
        self.align_button.setFont(button_font)
        self.align_button.setStyleSheet("background-color: Ivory;")
        grid.addWidget(self.align_button, 2, 3)

        # Turn on/off alignment
        self.light_button = QPushButton()
        self.light_button.setCheckable(True)
        self.light_button.toggled.connect(self.light_action)
        self.light_button.setFont(button_font)
        self.light_button.setStyleSheet("""
            QPushButton {
                background-color: red;
                border-style: solid;
                border-width: 15px;
                border-radius: 25px;
                min-width: 20px;
                max-width: 20px;
                min-height: 20px;
                max-height: 20px;
            }
            QPushButton:checked {
                background-color: green;
            }
        """)
        # create a layout and add the light_button to it
        light_button_layout = QVBoxLayout()
        light_button_layout.addWidget(self.light_button)
        # add the layout to the grid
        grid.addLayout(light_button_layout, 2, 4)
        
        # RGB linecut
        self.rgb_button = QPushButton('RGB Linecut')
        self.rgb_button.pressed.connect(self.rgb)
        self.rgb_button.setFont(button_font)
        self.rgb_button.setStyleSheet("background-color: Ivory;")
        grid.addWidget(self.rgb_button, 3, 3)

        # RGB map
        self.rgb2_button = QPushButton('RGB Map')
        self.rgb2_button.pressed.connect(self.rgb2)
        self.rgb2_button.setFont(button_font)
        self.rgb2_button.setStyleSheet("background-color: Ivory;")
        grid.addWidget(self.rgb2_button, 4, 3)

    """------------------------------------------------------------------------------------------------------------------------------------------------------------"""

    # Implement open, zooming, polygon manipulation, etc. commands for camera
    def open_camera(self):
        camera_thread = threading.Thread(target=camera_main)
        camera_thread.start()
        status_message = "Camera connected successfully."
        self.update_status(status_message)

    def capture(self):
        status_message = capture_frame(globals.Frame)
        self.update_status(status_message)

    # Trigger tracking mode
    def trigger_mode1(self, event):
        status_message = "Continue tracking..."
        self.update_status(status_message)
        globals.mode = "tracking"

    # Trigger drawing mode
    def trigger_mode2(self, event):
        status_message = "Start design a device..."
        self.update_status(status_message)
        globals.mode = "drawing"

    # Trigger measuring mode
    def trigger_mode3(self, event):
        status_message = "Measuring..."
        self.update_status(status_message)
        globals.mode = "measuring"
    
    # Trigger analysis mode
    def trigger_mode4(self, event):
        status_message = "Analysing..."
        self.update_status(status_message)
        globals.mode = "analysing"

    # Set up a set point for co-rotation
    def aim(self, event):
        pass

    def zoom_in(self):
        zoom_in_camera()

    def zoom_out(self):
        zoom_out_camera()

    def fivefold(self):

        globals.parameters["Camera"]["Rescale"] = (25580, 19400)
        globals.parameters["Camera"]["Scalebar"] = 500
        status_message = "5X Microscope"
        self.update_status(status_message)
        success_dialog = QMessageBox()
        success_dialog.setWindowTitle("Conversion Size")
        success_dialog.setText(f"Conversion size: {globals.parameters['Camera']['Rescale']}")
        success_dialog.exec_()

    def tenfold(self):

        globals.parameters["Camera"]["Rescale"] = (25580 / 2, 19400 / 2)
        globals.parameters["Camera"]["Scalebar"] = 250
        status_message = "10X Microscope"
        self.update_status(status_message)
        success_dialog = QMessageBox()
        success_dialog.setWindowTitle("Conversion Size")
        success_dialog.setText(f"Conversion size: {globals.parameters['Camera']['Rescale']}")
        success_dialog.exec_()

    def twentyfold(self):

        globals.parameters["Camera"]["Rescale"] = (25580 / 4, 19400 / 4)
        globals.parameters["Camera"]["Scalebar"] = 125
        status_message = "20X Microscope"
        self.update_status(status_message)
        success_dialog = QMessageBox()
        success_dialog.setWindowTitle("Conversion Size")
        success_dialog.setText(f"Conversion size: {globals.parameters['Camera']['Rescale']}")
        success_dialog.exec_()

    def retrieve_polygon(self):
        status_message = "Retrieving flake profile..."
        self.update_status(status_message)

        polygon_profile = get_polygon()
        formatted_profile = json.dumps(polygon_profile, indent=4, default=str)
        success_dialog = QMessageBox()
        success_dialog.setWindowTitle("Flake Profile")
        success_dialog.setText(f"Flake profile: \n{formatted_profile}")
        success_dialog.exec_()

        status_message = "Retrieved."
        self.update_status(status_message)

    def calibrate(self):

        pass

    def align(self):

        pass

    def light_action(self, checked):
        if checked:
            # light is on, trigger action A
            globals.disaligning = False
            self.align_button.setText("Align Flake")
            status_message = "Start aligning..."
            self.update_status(status_message)
        else:
            # light is off, trigger action B
            globals.disaligning = True
            self.align_button.setText("Track Flake")
            status_message = "Continue tracking..."
            self.update_status(status_message)
        
    def rgb(self):
        rgb_plot()
    
    def rgb2(self):
        rgb_map(globals.Frame)
    
    def update_status(self, message):
        self.status_label.setText(message)
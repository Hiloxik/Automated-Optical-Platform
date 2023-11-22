from GUI import *
from camera import main as camera_main, capture_frame
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QGroupBox, QTextEdit, QSplashScreen, QMenuBar, QMenu, QAction
from PyQt5.QtGui import QTextCursor, QPixmap, QPainter, QColor
from PyQt5.QtCore import Qt, QThread
import sys
import datetime
import globals
import matplotlib.pyplot as plt

# Output redirection class
class OutputRedirector:
    def __init__(self, console):
        self.console = console
        self.buffer = []
        self.last_update_time = datetime.datetime.now()


    def write(self, text):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.buffer.append(f"[{timestamp}] {text.rstrip()}")


        # Check if enough time has passed since the last update
        if (datetime.datetime.now() - self.last_update_time).seconds >= 0.01:
            self.flush()


    def flush(self):
        # Write all buffered lines to the console
        for line in self.buffer:
            self.console.append(line)
        self.console.moveCursor(QTextCursor.End)
        self.console.ensureCursorVisible()
        self.buffer = []
        self.last_update_time = datetime.datetime.now()

# create a GUI window

app = QApplication([])

# Create a pixmap and draw text on it
# Create a QPixmap object from the image file path
pixmap = QPixmap(r"C:\Users\2dqm\Desktop\microFLAT_picture_dump\f01_2023-09-01-162525_X5_x0.000000y0.000000_linear.png") # Update this with the actual path to your image file

# Scale the pixmap to the desired size
pixmap = pixmap.scaled(3000, 1600, Qt.KeepAspectRatio)  # You can adjust the width and height as needed

# Create a QPainter object to draw on the QPixmap
painter = QPainter(pixmap)

# Set the font for the first text and draw it in the middle region
font = QFont('Gill Sans Bold', 40)
font.setBold(True)  # Make the font bold
painter.setFont(font)

# Set the pen color to white
painter.setPen(QColor(Qt.white))

# Calculate the x and y coordinates for the text to be centered, and convert them to integers
middle_x = int((pixmap.width() - painter.fontMetrics().width("TRANSFER STAGE REMOTE CONTROLLER")) / 2)
middle_y = int(pixmap.height() / 2)

# Draw the text using the integer coordinates
painter.drawText(middle_x, middle_y, "TRANSFER STAGE REMOTE CONTROLLER")

painter.setFont(QFont('Arial', 15))
dn_x = int((pixmap.width() - painter.fontMetrics().width("Initializing. . .")) / 2)
dn_y = int(pixmap.height() - 240)
painter.drawText(dn_x, dn_y, "Initializing. . .")


# Set the font for the second text and draw it in the down-right corner
painter.setFont(QFont('Arial', 15))
right_x = int(pixmap.width() - painter.fontMetrics().width("ver.1 Designed by Jiahui") - 40)
right_y = int(pixmap.height() - 20)
painter.drawText(right_x, right_y, "ver 1.0.0 Designed by Jiahui")

# Finish painting
painter.end()

# Create and show the splash screen as before
splash = QSplashScreen(pixmap)
splash.show()
app.processEvents()


# Create and show the splash screen
splash = QSplashScreen(pixmap)
splash.show()

# Ensures that the splash screen is displayed while the rest of the app initializes
app.processEvents()

"""------------------------------------------------------------------------------------------------------------------------------------------------------------"""

window = QMainWindow()
window.setWindowTitle("Optical Platform Controller")
window.setGeometry(100, 100, 1000, 800)  # Specify the window size

# Set up menu bars
class MenuBarSetup:
    def __init__(self, window):
        self.window = window
        self.menu_bar = QMenuBar(self.window)
        self.window.setMenuBar(self.menu_bar)
        self.mode_groups = {
            "wafer": [],
            "camera": [],
            # ... add other groups as needed
        }
    
        self.setup_file_menu() 
        self.setup_mode_menu() 
    
    def setup_file_menu(self):
        file_menu = QMenu("File", self.menu_bar)
        self.menu_bar.addMenu(file_menu)
        self.add_action_to_menu(file_menu, "Refresh", self.refresh)
        self.add_action_to_menu(file_menu, "Save Image", self.save)
        self.add_action_to_menu(file_menu, "Exist", self.exit)
    
    def setup_mode_menu(self):
        mode_menu = QMenu("Mode", self.menu_bar)
        self.menu_bar.addMenu(mode_menu)
        self.setup_wafer_submenu(mode_menu)
        self.setup_camera_submenu(mode_menu)
    
    def setup_wafer_submenu(self, mode_menu):
        wafer_submenu = QMenu("Wafer", mode_menu)

        self.wafer_actions = {
        "Search": self.add_action_to_menu(wafer_submenu, "Search", self.search_mode, group = "wafer", checkable=True),
        "Adjust": self.add_action_to_menu(wafer_submenu, "Adjust", self.adjust_mode, group = "wafer", checkable=True)
        }
        
        mode_menu.addMenu(wafer_submenu)

    def setup_camera_submenu(self, mode_menu):
        camera_submenu = QMenu("Camera", mode_menu)

        self.camera_actions = {
            "Default": self.add_action_to_menu(camera_submenu, "Default", self.default_mode, group = "camera",checkable=True),
            "Measure": self.add_action_to_menu(camera_submenu, "Measuring", self.measure_mode, group = "camera", checkable=True),
            "Draw": self.add_action_to_menu(camera_submenu, "Drawing", self.draw_mode, group = "camera", checkable=True),
            "Track": self.add_action_to_menu(camera_submenu, "Tracking", self.track_mode, group = "camera",checkable=True)
        }

        self.setup_analysis_subsubmenu(camera_submenu)

        mode_menu.addMenu(camera_submenu)
    
    def setup_analysis_subsubmenu(self, camera_submenu):
        analysis_subsubmenu = QMenu("Analysis", camera_submenu)

        self.analysis_actions = {
            "Graphene":  self.add_action_to_menu(analysis_subsubmenu, "Graphene", self.graphene_mode, group = "camera", checkable=True),
            "hBN": self.add_action_to_menu(analysis_subsubmenu, "hBN", self.hbn_mode, group = "camera", checkable=True),
            "TMD": self.add_action_to_menu(analysis_subsubmenu, "TMD", self.tmd_mode, group = "camera", checkable=True)
        }

        camera_submenu.addMenu(analysis_subsubmenu)



    def add_action_to_menu(self, menu, action_name, function, group=None, checkable=False):
        action = QAction(action_name, self.window)
        action.setCheckable(checkable)
        action.triggered.connect(function)
        menu.addAction(action)
        
        if group and group in self.mode_groups:
            self.mode_groups[group].append(action)

        return action
    
    def refresh(self):
        self.window.repaint()

    def save(self):
        capture_frame(globals.Frame)

    def exit(self):
        self.window.close()
    
    def uncheck_group_modes(self, group_name):
        if group_name in self.mode_groups:
            for action in self.mode_groups[group_name]:
                action.setChecked(False)

    def search_mode(self):
        pass
        

    def adjust_mode(self):
        pass
    
    def default_mode(self):
        self.uncheck_group_modes('camera')
        self.camera_actions["Default"].setChecked(True)
        globals.mode = "default"

    def track_mode(self):
        self.uncheck_group_modes('camera')
        self.camera_actions["Track"].setChecked(True)
        globals.mode = "tracking"

    def draw_mode(self):
        self.uncheck_group_modes('camera')
        self.camera_actions["Draw"].setChecked(True)
        globals.mode = "drawing"

    def measure_mode(self):
        self.uncheck_group_modes('camera')
        self.camera_actions["Measure"].setChecked(True)
        globals.mode = "measuring"

    
    def graphene_mode(self):
        self.uncheck_group_modes('camera')
        self.analysis_actions["Graphene"].setChecked(True)
        globals.mode = "analysing"
        globals.analysismode = "graphene"
        pass
    
    def hbn_mode(self):
        self.uncheck_group_modes('camera')
        self.analysis_actions["hBN"].setChecked(True)
        globals.mode = "analysing"
        globals.analysismode = "hbn"
        pass

    def tmd_mode(self):
        self.uncheck_group_modes('camera')
        self.analysis_actions["TMD"].setChecked(True)
        globals.mode = "analysing"
        globals.analysismode = "tmd"
        pass

# Using the class to set up the menu bar for a window
menu_setup = MenuBarSetup(window)

"""------------------------------------------------------------------------------------------------------------------------------------------------------------"""

# Create a frame inside main_window for each device GUI
central_widget = QWidget()
window.setCentralWidget(central_widget)
layout = QGridLayout()
central_widget.setLayout(layout)  # set layout to central_widget

# Create a QTextEdit for console output and add them to the layout
# console = QTextEdit()
# console.setReadOnly(True)
# layout.addWidget(console, 4, 0, 1, 2)  # Add it to the bottom row, first column

# sys.stdout = OutputRedirector(console)
# sys.stderr = OutputRedirector(console)

group_box_stylesheet = """
QGroupBox {
    font: 30px Cooper Black;/* Set the font size and style of the title */
    border: 2px solid gray; /* Set the width and color of the border */
    border-radius: 9px; /* Set the roundness of the corners */
    margin-top: 0.5em; /* Set the top margin */
}

QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px; /* Set the left margin of the title */
    padding: 0 3px 0 3px; /* Set the padding around the title */
}
"""

widget_gui = GUI()
widget_group = QGroupBox("Camera")
widget_group.setStyleSheet(group_box_stylesheet)
widget_layout = QVBoxLayout()
widget_layout.addWidget(widget_gui)
widget_group.setLayout(widget_layout)
layout.addWidget(widget_group, 0, 0)
# QThread.sleep(10)

# Close the splash screen and show the main window
splash.finish(window)
window.show()

"""------------------------------------------------------------------------------------------------------------------------------------------------------------"""

camera_thread = threading.Thread(target=camera_main)
camera_thread.start()

app.exec_()
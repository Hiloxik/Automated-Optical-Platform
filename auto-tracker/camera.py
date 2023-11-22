""" Camera script. Including camera connection, different modes, and image processing/tracking. """

import cv2
import numpy as np
import globals
# import PySpin
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import QFileDialog
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

""" Initial variabls. """

# camera
camera = 0

# Frame
x_scale, y_scale = 1, 1
zoom_factor = 1.0
top, left = 0, 0
frame = None
r_offset, g_offset, b_offset = 0, 0, 0
brightness, contrast = 0, 1
blur_ksize = 1

# Drawing mode
polygon = []
poly_shifted = []
closed = False
tracking = False
highlighted_point = None
attraction_range = 10  # Attraction range around each point
old_center = None
bbox = None
tracker = None
dragging = False
shift_vector = (0, 0)
polygon_profile = {'center': None, 'points': [], 'edges': [], 'angle': 0}  # New variable
center_shift_vector = (0, 0)  # Initialize the shift vector to (0, 0)
initial_center = None  # Store the initial center when the polygon is closed

# Tracking mode
drawing_polygon = []
drawing_polygons = []
drawing_polygon_profile = {'center': None, 'points': [], 'edges': [], 'angle': 0}  # New variable
drawing_dragging_index = -1
drawing_poly_shifted = []
drawing_closed = False
drawing_highlighted_point = None
drawing_dragging = False
drawing_shift_vector = (0, 0)
colors = [(255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
          (0, 255, 255), (255, 255, 255)]  # Add more colors if needed
color_index = 0

# Measuring mode
ruler_drawing = False
ruler_end = False
ruler_ix, ruler_iy = -1, -1
ruler_line_end = (0, 0)

# Default mode
dragging_frame = False
drag_start_x = 0
drag_start_y = 0

# Analysis mode
drawing = False # True if the mouse is pressed
end = False
ix, iy = -1, -1
line_end = (0, 0)


# Generating function that handling different modes of mouse callback
def mouse_callback(event, x, y, flags, param):
    global x_scale, y_scale, zoom_factor, top, left, frame
    global polygon, tracking, tracker, closed, highlighted_point, bbox, old_center, poly_shifted, dragging, shift_vector, polygon_profile
    global drawing_polygon, drawing_polygons, drawing_dragging_index, drawing_closed, drawing_highlighted_point, \
        drawing_dragging, drawing_poly_shifted, drawing_shift_vector, initial_center, color_index, drawing_polygon_profile
    global ruler_start, ruler_end

    # 1. Reverse the resize transformation
    x_rescaled = x * frame.shape[1] / 1280
    y_rescaled = y * frame.shape[0] / 960

    # 2. Adjust for the zoom
    x_zoomed = int(x_rescaled / zoom_factor)
    y_zoomed = int(y_rescaled / zoom_factor)

    # 3. Account for translation due to zoom
    x_original = x_zoomed + left
    y_original = y_zoomed + top

    # Render point on the original frame
    cv2.circle(frame, (x_original, y_original), 5, (0, 0, 255), -1)
    # Render point on the resized frame exactly where clicked
    cv2.circle(globals.Frame, (x, y), 5, (0, 255, 0), -1)

    # Handle different modes
    if globals.mode == "measuring":
        handle_measuring_mode(event, x_original, y_original, flags, param)
    elif globals.mode == "drawing":
        handle_drawing_mode(event, x_original, y_original, flags, param)
    elif globals.mode == "tracking":
        handle_tracking_mode(event, x_original, y_original, flags, param)
    elif globals.mode == "default":
        handle_default_mode(event, x, y, flags, param)
    elif globals.mode == "analysing":
        handle_analysis_mode(event, x, y, flags, param)

# Default mode handler
""" This mode is the default mode, in which user can use mouse to drag the frame if zoomed in. """
def handle_default_mode(event, x, y, flags, param):
    global dragging_frame, drag_start_x, drag_start_y, left, top, frame, zoom_factor

    if event == cv2.EVENT_LBUTTONDOWN:
        dragging_frame = True
        drag_start_x = x
        drag_start_y = y

    elif event == cv2.EVENT_MOUSEMOVE:
        if dragging_frame and zoom_factor > 1.0:
            dx = int((x - drag_start_x) / zoom_factor)
            dy = int((y - drag_start_y) / zoom_factor)
            left -= dx
            top -= dy
            drag_start_x = x
            drag_start_y = y
            # Here, you should also make sure 'left' and 'top' values don't go out of frame bounds.

    elif event == cv2.EVENT_LBUTTONUP:
        dragging_frame = False

# Drawing mode handler
""" This mode is the drawing mode, in which user can use mouse to draw, drag, rotate, delete different coloful multiple polygons without a tracking algorithm planted in. 
    This mode is used to create targert flake patterns and design device patterns that user aim to assembly and transfer. """
def handle_drawing_mode(event, x, y, flags, param):
    global drawing_polygon, drawing_polygons, drawing_dragging_index, drawing_closed, drawing_highlighted_point, drawing_dragging, drawing_poly_shifted, drawing_shift_vector, \
        color_index, initial_center, drawing_polygon_profile

    if event == cv2.EVENT_LBUTTONDOWN:
        if drawing_closed and drawing_poly_shifted:
            drawing_dragging = False
            drawing_dragging_index = -1
            if drawing_polygon:  # Only add if the polygon is not empty
                drawing_polygons.append((drawing_polygon, colors[color_index]))  # Add current polygon to list
            drawing_polygon = []  # Clear current polygon
            for draw_idx, (draw_polygon, _) in enumerate(drawing_polygons):
                if draw_polygon:  # Only test if the polygon is not empty
                    draw_poly_np = np.array(draw_polygon, dtype=np.int32)
                    if cv2.pointPolygonTest(draw_poly_np, (x, y), False) >= 0:
                        drawing_dragging = True
                        drawing_dragging_index = draw_idx
                        drawing_shift_vector = (x, y)
                        break
            else:
                # If the mouse is clicked outside the polygon, start a new polygon
                drawing_polygon = [(x, y)]  # Start new polygon
                drawing_closed = False
                drawing_highlighted_point = None
                color_index = (color_index + 1) % len(colors)
                drawing_polygon_profile['angle'] = 0


        elif not drawing_polygon:
            drawing_polygon.append((x, y))
        else:
            for draw_i, draw_point in enumerate(drawing_polygon):
                draw_dx, draw_dy = draw_point[0] - x, draw_point[1] - y
                draw_distance = np.sqrt(draw_dx ** 2 + draw_dy ** 2)
                if draw_distance < attraction_range:
                    x, y = draw_point
                    drawing_highlighted_point = draw_point
                    if draw_i == 0 and len(drawing_polygon) > 1:
                        drawing_closed = True
                        drawing_poly_shifted = drawing_polygon.copy()
                        initial_center = (sum(p[0] for p in drawing_poly_shifted) / len(drawing_poly_shifted),
                                          sum(p[1] for p in drawing_poly_shifted) / len(drawing_poly_shifted))

                        drawing_polygon_profile['angle'] = 0
                        break
            if (x, y) not in drawing_polygon:
                drawing_polygon.append((x, y))
                drawing_closed = False
                drawing_highlighted_point = None
                drawing_polygon_profile['angle'] = 0


    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing_dragging:
            draw_dx, draw_dy = x - drawing_shift_vector[0], y - drawing_shift_vector[1]
            drawing_shift_vector = (x, y)
            drawing_poly_shifted = [(draw_p[0] + draw_dx, draw_p[1] + draw_dy) for draw_p in drawing_poly_shifted]
            # Update the polygon_profile with the new coordinates and center
            drawing_center_x = sum(p[0] for p in drawing_poly_shifted) / len(drawing_poly_shifted)
            drawing_center_y = sum(p[1] for p in drawing_poly_shifted) / len(drawing_poly_shifted)
            drawing_polygon_profile['center'] = (drawing_center_x, drawing_center_y)
            drawing_polygon_profile['points'] = drawing_poly_shifted
            drawing_polygon_profile['edges'] = [np.sqrt(
                (drawing_poly_shifted[i][0] - drawing_poly_shifted[i - 1][0]) ** 2 + (
                        drawing_poly_shifted[i][1] - drawing_poly_shifted[i - 1][1]) ** 2)
                for i in range(1, len(drawing_poly_shifted))]
            drawing_polygon_profile['edges'].append(
                np.sqrt((drawing_poly_shifted[0][0] - drawing_poly_shifted[-1][0]) ** 2 + (
                        drawing_poly_shifted[0][1] - drawing_poly_shifted[-1][
                    1]) ** 2))  # Add the edge between the last and first points
            if drawing_dragging_index == len(drawing_polygons):  # Current drawing_polygon
                drawing_polygon = [(draw_p[0] + draw_dx, draw_p[1] + draw_dy) for draw_p in drawing_polygon]
                # Update the polygon_profile with the new coordinates and center

            else:  # Existing polygon in drawing_polygons
                draw_polygon, draw_color = drawing_polygons[drawing_dragging_index]
                drawing_polygons[drawing_dragging_index] = (
                    [(draw_p[0] + draw_dx, draw_p[1] + draw_dy) for draw_p in draw_polygon], draw_color)

        else:
            drawing_highlighted_point = None
            for draw_point in drawing_polygon:
                draw_dx, draw_dy = draw_point[0] - x, draw_point[1] - y
                draw_distance = np.sqrt(draw_dx ** 2 + draw_dy ** 2)
                if draw_distance < attraction_range:
                    drawing_highlighted_point = draw_point
                    break

    elif event == cv2.EVENT_RBUTTONDOWN:
        if drawing_polygon:
            drawing_polygon = []
            drawing_poly_shifted = []
            drawing_closed = False
            drawing_highlighted_point = None
            drawing_polygon_profile['angle'] = 0
        elif drawing_polygons:
            drawing_polygons.pop()

    elif event == cv2.EVENT_LBUTTONUP:
        drawing_dragging = False
        if drawing_closed:
            drawing_center_x = sum(p[0] for p in drawing_poly_shifted) / len(drawing_poly_shifted)
            drawing_center_y = sum(p[1] for p in drawing_poly_shifted) / len(drawing_poly_shifted)
            drawing_polygon_profile['center'] = (drawing_center_x, drawing_center_y)
            drawing_polygon_profile['points'] = drawing_poly_shifted
            drawing_polygon_profile['edges'] = [np.sqrt(
                (drawing_poly_shifted[i][0] - drawing_poly_shifted[i - 1][0]) ** 2 + (
                        drawing_poly_shifted[i][1] - drawing_poly_shifted[i - 1][1]) ** 2)
                for i in range(1, len(drawing_poly_shifted))]
            drawing_polygon_profile['edges'].append(
                np.sqrt((drawing_poly_shifted[0][0] - drawing_poly_shifted[-1][0]) ** 2 + (
                        drawing_poly_shifted[0][1] - drawing_poly_shifted[-1][
                    1]) ** 2))  # Add the edge between the last and first points

    elif event == cv2.EVENT_MOUSEWHEEL:
        if drawing_closed:
            # Determine the direction of rotation
            draw_rotation_angle = 1 if flags > 0 else -1
            # Calculate the center of the polygon
            draw_center_x = sum(p[0] for p in drawing_poly_shifted) / len(drawing_poly_shifted)
            draw_center_y = sum(p[1] for p in drawing_poly_shifted) / len(drawing_poly_shifted)
            draw_center = (draw_center_x, draw_center_y)

            # Define the rotation matrix
            draw_M = cv2.getRotationMatrix2D(draw_center, draw_rotation_angle, 1)

            # Rotate each point of the polygon
            drawing_poly_shifted_rotated = []
            for draw_p in drawing_poly_shifted:
                draw_p_rotated = np.dot(draw_M, (draw_p[0], draw_p[1], 1))
                drawing_poly_shifted_rotated.append((draw_p_rotated[0], draw_p_rotated[1]))
            drawing_poly_shifted = drawing_poly_shifted_rotated

            # Update the polygon_profile with the new coordinates and center
            drawing_polygon_profile['center'] = draw_center
            drawing_polygon_profile['points'] = drawing_poly_shifted
            drawing_polygon_profile['edges'] = [np.sqrt(
                (drawing_poly_shifted[i][0] - drawing_poly_shifted[i - 1][0]) ** 2 + (
                        drawing_poly_shifted[i][1] - drawing_poly_shifted[i - 1][1]) ** 2)
                for i in range(1, len(drawing_poly_shifted))]
            drawing_polygon_profile['edges'].append(np.sqrt((drawing_poly_shifted[0][0] - drawing_poly_shifted[-1][0]) ** 2 + (
                    drawing_poly_shifted[0][1] - drawing_poly_shifted[-1][
                1]) ** 2))  # Add the edge between the last and first points
            drawing_polygon_profile['angle'] += draw_rotation_angle

# Tracking mode handler
""" This mode is the tracking mode, in which users can draw, drag, rotate and delete green single polygon with a tracking algorithm planted in. 
    This mode is used to track current flakes. """
def handle_tracking_mode(event, x, y, flags, param):
    global polygon, tracking, tracker, closed, highlighted_point, bbox, old_center, poly_shifted, dragging, shift_vector, polygon_profile, initial_center

    if event == cv2.EVENT_LBUTTONDOWN:
        if closed and poly_shifted:
            poly_np = np.array(poly_shifted, dtype=np.int32)
            if cv2.pointPolygonTest(poly_np, (x, y), False) >= 0:
                # If the mouse is clicked inside the polygon, start dragging the polygon
                dragging = True
                tracking = False
                shift_vector = (x, y)
            else:
                # If the mouse is clicked outside the polygon, start a new polygon
                polygon = []
                poly_shifted = []
                closed = False
                tracking = False
                highlighted_point = None
                polygon_profile['angle'] = 0

        elif not polygon:
            polygon.append((x, y))
        else:
            for i, point in enumerate(polygon):
                dx, dy = point[0] - x, point[1] - y
                distance = np.sqrt(dx ** 2 + dy ** 2)
                if distance < attraction_range:
                    x, y = point
                    highlighted_point = point
                    if i == 0 and len(polygon) > 1:
                        closed = True
                        poly_shifted = polygon.copy()
                        bbox = cv2.boundingRect(np.array(poly_shifted, dtype=np.int32))
                        old_center = (bbox[0] + bbox[2] // 2, bbox[1] + bbox[3] // 2)
                        initial_center = (sum(p[0] for p in poly_shifted) / len(poly_shifted),
                                          sum(p[1] for p in poly_shifted) / len(poly_shifted))
                        polygon_profile['angle'] = 0

                        break
            if (x, y) not in polygon:
                polygon.append((x, y))
                closed = False
                highlighted_point = None
                polygon_profile['angle'] = 0

    elif event == cv2.EVENT_MOUSEMOVE:
        if dragging:
            dx, dy = x - shift_vector[0], y - shift_vector[1]
            shift_vector = (x, y)
            poly_shifted = [(p[0] + dx, p[1] + dy) for p in poly_shifted]
            bbox = cv2.boundingRect(np.array(poly_shifted, dtype=np.int32))
            old_center = (bbox[0] + bbox[2] // 2, bbox[1] + bbox[3] // 2)

            # Update the polygon_profile with the new coordinates and center
            center_x = sum(p[0] for p in poly_shifted) / len(poly_shifted)
            center_y = sum(p[1] for p in poly_shifted) / len(poly_shifted)
            polygon_profile['center'] = (center_x, center_y)
            polygon_profile['points'] = poly_shifted
            polygon_profile['edges'] = [np.sqrt(
                (poly_shifted[i][0] - poly_shifted[i - 1][0]) ** 2 + (
                        poly_shifted[i][1] - poly_shifted[i - 1][1]) ** 2)
                for i in range(1, len(poly_shifted))]
            polygon_profile['edges'].append(np.sqrt((poly_shifted[0][0] - poly_shifted[-1][0]) ** 2 + (
                    poly_shifted[0][1] - poly_shifted[-1][
                1]) ** 2))  # Add the edge between the last and first points
        else:
            highlighted_point = None
            for point in polygon:
                dx, dy = point[0] - x, point[1] - y
                distance = np.sqrt(dx ** 2 + dy ** 2)
                if distance < attraction_range:
                    highlighted_point = point
                    break

    elif event == cv2.EVENT_RBUTTONDOWN:
        polygon = []
        poly_shifted = []
        closed = False
        tracking = False
        highlighted_point = None
        polygon_profile['angle'] = 0

    elif event == cv2.EVENT_LBUTTONUP:
        dragging = False
        if globals.disaligning:
            if closed:
                tracking = True
                tracker = cv2.legacy.TrackerMOSSE_create()
                print(tracker)
                tracker.init(frame, bbox)

                # Calculate the center and edge lengths of the polygon using poly_shifted
                center_x = sum(p[0] for p in poly_shifted) / len(poly_shifted)
                center_y = sum(p[1] for p in poly_shifted) / len(poly_shifted)
                polygon_profile['center'] = (center_x, center_y)
                polygon_profile['points'] = poly_shifted
                polygon_profile['edges'] = [np.sqrt(
                    (poly_shifted[i][0] - poly_shifted[i - 1][0]) ** 2 + (
                            poly_shifted[i][1] - poly_shifted[i - 1][1]) ** 2)
                    for i in range(1, len(poly_shifted))]
                polygon_profile['edges'].append(np.sqrt((poly_shifted[0][0] - poly_shifted[-1][0]) ** 2 + (
                        poly_shifted[0][1] - poly_shifted[-1][
                    1]) ** 2))  # Add the edge between the last and first points
        else:
            if closed:
                # Calculate the center and edge lengths of the polygon using poly_shifted
                center_x = sum(p[0] for p in poly_shifted) / len(poly_shifted)
                center_y = sum(p[1] for p in poly_shifted) / len(poly_shifted)
                polygon_profile['center'] = (center_x, center_y)
                polygon_profile['points'] = poly_shifted
                polygon_profile['edges'] = [np.sqrt(
                    (poly_shifted[i][0] - poly_shifted[i - 1][0]) ** 2 + (
                            poly_shifted[i][1] - poly_shifted[i - 1][1]) ** 2)
                    for i in range(1, len(poly_shifted))]
                polygon_profile['edges'].append(np.sqrt((poly_shifted[0][0] - poly_shifted[-1][0]) ** 2 + (
                        poly_shifted[0][1] - poly_shifted[-1][
                    1]) ** 2))  # Add the edge between the last and first points
    
    elif event == cv2.EVENT_LBUTTONDBLCLK:
        poly_shifted = drawing_poly_shifted
        polygon_profile = drawing_polygon_profile
        closed = True

        # Extract the details from the original bounding box
        bbox = cv2.boundingRect(np.array(poly_shifted, dtype=np.int32))

        old_center = (bbox[0] + bbox[2] // 2, bbox[1] + bbox[3] // 2)




    elif event == cv2.EVENT_MOUSEWHEEL:
        if closed:
            # Determine the direction of rotation
            rotation_angle = 1 if flags > 0 else -1
            # Calculate the center of the polygon
            center_x = sum(p[0] for p in poly_shifted) / len(poly_shifted)
            center_y = sum(p[1] for p in poly_shifted) / len(poly_shifted)
            center = (center_x, center_y)

            # Define the rotation matrix
            M = cv2.getRotationMatrix2D(center, rotation_angle, 1)

            # Rotate each point of the polygon
            poly_shifted_rotated = []
            for p in poly_shifted:
                p_rotated = np.dot(M, (p[0], p[1], 1))
                poly_shifted_rotated.append((p_rotated[0], p_rotated[1]))
            poly_shifted = poly_shifted_rotated

            # Recalculate the bounding box and center
            # Extract the details from the original bounding box
            bbox = cv2.boundingRect(np.array(poly_shifted, dtype=np.int32))

            old_center = (bbox[0] + bbox[2] // 2, bbox[1] + bbox[3] // 2)

            # Update the polygon_profile with the new coordinates and center
            polygon_profile['center'] = center
            polygon_profile['points'] = poly_shifted
            polygon_profile['edges'] = [np.sqrt(
                (poly_shifted[i][0] - poly_shifted[i - 1][0]) ** 2 + (
                        poly_shifted[i][1] - poly_shifted[i - 1][1]) ** 2)
                for i in range(1, len(poly_shifted))]
            polygon_profile['edges'].append(np.sqrt((poly_shifted[0][0] - poly_shifted[-1][0]) ** 2 + (
                    poly_shifted[0][1] - poly_shifted[-1][
                1]) ** 2))  # Add the edge between the last and first points
            polygon_profile['angle'] += rotation_angle

# Measuring mode handler


def handle_measuring_mode(event, x, y, flags, param):
    global ruler_ix, ruler_iy, ruler_drawing, ruler_line_end, ruler_end

    # Convert coordinates
    x_original = x
    y_original = y

    if event == cv2.EVENT_LBUTTONDOWN:
        ruler_drawing = True
        ruler_end = False
        ruler_ix, ruler_iy = x_original, y_original
        ruler_line_end = (x_original, y_original)

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            ruler_line_end = (x_original, y_original)

    elif event == cv2.EVENT_LBUTTONUP:
        ruler_drawing = False
        ruler_end = True
        ruler_line_end = (x_original, y_original)


def handle_analysis_mode(event, x, y, flags, param):
    global ix, iy, drawing, line_end, end

    # Convert coordinates
    x_rescaled = x * frame.shape[1] / 1280
    y_rescaled = y * frame.shape[0] / 960
    x_original = int(x_rescaled / zoom_factor) + left
    y_original = int(y_rescaled / zoom_factor) + top

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        end = False
        ix, iy = x_original, y_original
        line_end = (x_original, y_original)

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            line_end = (x_original, y_original)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        end = True
        line_end = (x_original, y_original)

def calculate_and_plot_rgb_values():
    global frame, ix, iy, line_end
    
    if globals.analysismode == "plain":

        # Extract RGB values along the line
        x_values = np.linspace(ix, line_end[0], num=1000, dtype=int)
        y_values = np.linspace(iy, line_end[1], num=1000, dtype=int)
        rgb_values = [frame[y, x] for x, y in zip(x_values, y_values)]

        # Calculate the average intensity of the frame
        average_intensity = np.mean(frame, axis=(0, 1))

        # Normalize RGB values based on average intensity
        normalized_rgb_values = [(rgb / average_intensity) * 100 for rgb in rgb_values]
    
    if globals.analysismode == "graphene":
        pass

    if globals.analysismode == "hbn":
        pass

    if globals.analysismode == "tmd":
        pass

    # print(rgb_values)
    globals.rgb = normalized_rgb_values


def get_polygon():
    global drawing_polygon_profile
    return drawing_polygon_profile

def get_polygon_tracker():
    global polygon_profile
    return polygon_profile


def get_shift():
    global center_shift_vector
    return center_shift_vector


def get_angle():
    return drawing_polygon_profile['angle']


def zoom_in_camera():
    global zoom_factor
    MAX_ZOOM = 5
    zoom_factor = min(zoom_factor + 0.1, MAX_ZOOM)


def zoom_out_camera():
    global zoom_factor
    MIN_ZOOM = 0.5
    zoom_factor = max(zoom_factor - 0.1, MIN_ZOOM)

def rgb_plot():
    if globals.rgb is not None:
        plt.plot([np.linalg.norm(rgb) for rgb in globals.rgb])
        plt.title("1D RGB Intensity")
        plt.xlabel("Position")
        plt.ylabel("RGB Intensity")
        plt.show()

def rgb_map(frame, downsample_size=(100, 100)):
    # Downsample the image
    small_frame = cv2.resize(frame, downsample_size, interpolation=cv2.INTER_AREA)

    # Reshape the frame to a 2D array where each row is R, G, B
    reshaped_frame = small_frame.reshape(-1, 3)

    # Extract R, G, B channels
    r = reshaped_frame[:, 0]
    g = reshaped_frame[:, 1]
    b = reshaped_frame[:, 2]

    # Create a 3D plot
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Plot using scatter
    imgplot = ax.scatter(r, g, b, c=reshaped_frame / 255.0, marker='o')

    # Set labels
    ax.set_xlabel('Red Channel')
    ax.set_ylabel('Green Channel')
    ax.set_zlabel('Blue Channel')
    ax.set_title('3D RGB Color Space')

    # Show the plot
    plt.show()


def draw_frame(frame):
    # Draw a sleek frame around the camera view
    thickness = 10  # Adjust the thickness to your preference
    color = (0, 0, 0)  # A light cyan color, but you can customize
    cv2.line(frame, (0, 0), (frame.shape[1], 0), color, thickness)
    cv2.line(frame, (0, 0), (0, frame.shape[0]), color, thickness)
    cv2.line(frame, (frame.shape[1] - 1, 0), (frame.shape[1] - 1, frame.shape[0]), color, thickness)
    cv2.line(frame, (0, frame.shape[0] - 1), (frame.shape[1], frame.shape[0] - 1), color, thickness)


class CaptureThread(QThread):
    finished = pyqtSignal(str)

    def __init__(self, frame, filename):
        super().__init__()
        self.frame = frame
        self.filename = filename

    def run(self):
        if not self.filename.endswith('.png'):
            self.filename += '.png'
        cv2.imwrite(self.filename, self.frame)
        self.finished.emit(f"Captured frame saved to {self.filename}")


def capture_frame(frame, parent_window=None):
    options = QFileDialog.Options()
    filename, _ = QFileDialog.getSaveFileName(parent_window, "Save File", "", "PNG files (*.png);;All files (*.*);;",
                                              options=options)

    if filename:
        capture_thread = CaptureThread(frame, filename)
        capture_thread.finished.connect(lambda message: print(message))
        capture_thread.start()
        return capture_thread
    else:
        return "Save operation cancelled"


def render_drawing(frame):
    for draw_polygon, draw_color in drawing_polygons:
        cv2.polylines(frame, [np.array(draw_polygon, dtype=np.int32)], True, draw_color, 5)

    if drawing_closed:
        cv2.polylines(frame, [np.array(drawing_poly_shifted, dtype=np.int32)], True, colors[color_index], 5)
    else:
        if drawing_polygon:
            for draw_point in drawing_polygon:
                if draw_point == drawing_highlighted_point:
                    cv2.circle(frame, draw_point, attraction_range, (0, 0, 255), 2)
            cv2.polylines(frame, [np.array(drawing_polygon, dtype=np.int32)], False, colors[color_index], 5)

def render_measuring(frame):
    global ruler_ix, ruler_iy, ruler_line_end, ruler_end
    # Draw the line on the image
    cv2.line(frame, (ruler_ix, ruler_iy), ruler_line_end, (0, 255, 0), 5)
    if ruler_end:
        # Calculate distance
        distance = int(np.sqrt((ruler_line_end[0] - ruler_ix) ** 2 + (ruler_line_end[1] - ruler_iy) ** 2))

        # Define the position for the text based on the direction of the line
        text_pos = ((ruler_ix + ruler_line_end[0]) // 2, (ruler_iy + ruler_line_end[1]) // 2 + 20)

        print(distance, text_pos)

        # Put text below the line
        cv2.putText(frame, f"{distance} pixels", text_pos, cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 5)

def render_tracking(display_frame, hidden_frame):
    global poly_shifted, closed, tracking, tracker, old_center
    if closed:
        if tracking:
            success, box = tracker.update(hidden_frame)
            if success:
                new_center = (box[0] + box[2] // 2, box[1] + box[3] // 2)
                dx, dy = (new_center[0] - old_center[0]) / zoom_factor, (
                        new_center[1] - old_center[1]) / zoom_factor
                poly_shifted = [(x + dx, y + dy) for (x, y) in poly_shifted]
                old_center = new_center

                # Update the polygon_profile with the new coordinates and center
                center_x = sum(p[0] for p in poly_shifted) / len(poly_shifted)
                center_y = sum(p[1] for p in poly_shifted) / len(poly_shifted)
                polygon_profile['center'] = (center_x, center_y)
                polygon_profile['points'] = poly_shifted
                polygon_profile['edges'] = [np.sqrt((poly_shifted[i][0] - poly_shifted[i - 1][0]) ** 2 + (
                        poly_shifted[i][1] - poly_shifted[i - 1][1]) ** 2) for i in
                                            range(1, len(poly_shifted))]
                polygon_profile['edges'].append(np.sqrt((poly_shifted[0][0] - poly_shifted[-1][0]) ** 2 + (
                        poly_shifted[0][1] - poly_shifted[-1][
                    1]) ** 2))  # Add the edge between the last and first points
            else:
                cv2.putText(display_frame, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                            (0, 0, 255), 4)

        # poly_shifted_zoomed = [(int(x * zoom_factor), int(y * zoom_factor)) for (x, y) in poly_shifted]
        if poly_shifted:
            cv2.polylines(display_frame, [np.array(poly_shifted, dtype=np.int32)], True, (0, 255, 0), 5)
    else:
        if polygon:
            for point in polygon:
                if point == highlighted_point:
                    cv2.circle(display_frame, point, attraction_range, (0, 0, 255), 2)
            cv2.polylines(display_frame, [np.array(polygon, dtype=np.int32)], False, (0, 255, 0), 5)

def render_analysing(frame):
    global end, ix, iy, line_end
    # Draw the line on the image
    cv2.line(frame, (ix, iy), line_end, (255, 0, 0), 5)
    if end:
        calculate_and_plot_rgb_values()

def main():
    global x_scale, y_scale, zoom_factor, top, left, frame
    global polygon, tracking, tracker, closed, highlighted_point, bbox, old_center, poly_shifted, dragging, shift_vector, bbox, polygon_profile, center_shift_vector
    global drawing_polygon, drawing_polygons, drawing_closed, drawing_highlighted_point, \
        drawing_dragging, drawing_poly_shifted, drawing_shift_vector, initial_center, color_index
    
    
    # if camera == 1:

        # system = PySpin.System.GetInstance()
        # cam_list = system.GetCameras()

        # if cam_list.GetSize() == 0:
        #     cam_list.Clear()
        #     system.ReleaseInstance()
        #     print('Not enough cameras!')
        #     return False

        # cam = cam_list.GetByIndex(0)

        # cv2.namedWindow("Camera")
        # cv2.setMouseCallback("Camera", mouse_callback) # set up mouse callback

        # try:
        #     cam.Init()
        #     # Set pixel format to a color format if it's not already set
        #     # This is an example, you need to adapt it based on your camera's capabilities
        #     nodemap = cam.GetNodeMap()
        #     node_pixel_format = PySpin.CEnumerationPtr(nodemap.GetNode("PixelFormat"))
        #     if PySpin.IsAvailable(node_pixel_format) and PySpin.IsWritable(node_pixel_format):
        #         node_pixel_format_bgr8 = node_pixel_format.GetEntryByName("BGR8")
        #         if PySpin.IsAvailable(node_pixel_format_bgr8) and PySpin.IsReadable(node_pixel_format_bgr8):
        #             pixel_format_bgr8 = node_pixel_format_bgr8.GetValue()
        #             node_pixel_format.SetIntValue(pixel_format_bgr8)

        #     try:
        #         cam.BeginAcquisition()
        #         print('Acquiring images...')
        #         while True:
        #             image_result = cam.GetNextImage()

        #             if image_result.IsIncomplete():
        #                 print('Image incomplete with image status %d ...' % image_result.GetImageStatus())
        #             else:
        #                 # Assuming the camera is in a color format like RGB8 or BGR8
        #                 image_data = image_result.GetNDArray()

        #                 # Resize the image (you can adjust the size as needed)
        #                 resized_image = cv2.resize(image_data, (640, 480))  # Example size: 640x480

        #                 # Display the image using OpenCV
        #                 # cv2.imshow('Camera', resized_image)
        #                 # cv2.setMouseCallback("Camera", mouse_callback) # set up mouse callback

        #                 if cv2.waitKey(1) == ord('q'):
        #                     break

        #                 image_result.Release()
                    
        #             frame = image_data
            
        #             display_frame = frame.copy() # frame being displayed

        #             """--------------------------------------------------------------Create a hidden frame--------------------------------------------------------------"""
                
        #             # Hidden frame being processed and tracked
        #             gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        #             # edges = cv2.Canny(gray, 50, 150)
        #             # enhanced = cv2.addWeighted(gray, 0.8, edges, 0.2, 0)

        #             # Apply brightness and contrast
        #             gray = cv2.convertScaleAbs(gray, alpha=contrast, beta=brightness)

        #             # Apply Gaussian Blur
        #             if blur_ksize > 1:
        #                 gray = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)

        #             # Apply Histogram Equalization for better contrast
        #             # equ = cv2.equalizeHist(gray)

        #             # Convert the equalized grayscale image back to BGR for the tracker
        #             hidden_frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                    
        #             """--------------------------------------------------------------Add zoom factor to the frame--------------------------------------------------------------"""

        #             if globals.original_frame_width is None or globals.original_frame_height is None:
        #                 globals.original_frame_width, globals.original_frame_height = frame.shape[1], frame.shape[0]

        #             # Calculate the dimensions of the zoomed region
        #             zoomed_height = int(frame.shape[0] / zoom_factor)
        #             zoomed_width = int(frame.shape[1] / zoom_factor)

        #             # Check and correct if zoomed region exceeds frame boundaries
        #             if top < 0:
        #                 top = 0
        #             if left < 0:
        #                 left = 0

        #             # If the right or bottom boundaries exceed the frame dimensions, adjust them too.
        #             if left + zoomed_width > frame.shape[1]:
        #                 left = frame.shape[1] - zoomed_width
        #             if top + zoomed_height > frame.shape[0]:
        #                 top = frame.shape[0] - zoomed_height

        #             # Extract the zoomed region
        #             zoomed_region = display_frame[top:top + zoomed_height, left:left + zoomed_width]

        #             x_scale = 1280 / globals.original_frame_width
        #             y_scale = 960 / globals.original_frame_height

        #             """--------------------------------------------------------------Draw different mode handlers--------------------------------------------------------------"""

        #             render_drawing(frame=display_frame)

        #             if globals.mode == "measuring":
        #                 render_measuring(frame=display_frame)

        #             elif globals.mode == "tracking":
        #                 render_tracking(display_frame=display_frame, hidden_frame=hidden_frame)

        #             elif globals.mode == "analysing":
        #                 render_analysing(frame=display_frame)
                    
        #             """--------------------------------------------------------------Fabricate the frame--------------------------------------------------------------"""

        #             draw_frame(display_frame)

        #             # cv2.resizeWindow("Camera", 640, 480)
        #             cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)
                    
        #             resized_frame = cv2.resize(zoomed_region, (1280, 960))

        #             # # Draw scale bar
        #             # draw_scale_bar(resized_frame, zoom_factor)

        #             # Display the combined image
        #             cv2.imshow("Camera", resized_frame)

        #             globals.Frame = resized_frame

        #             if drawing_polygon_profile['center']:  # Make sure the center is not None
        #                 center_shift_vector = (
        #                     drawing_polygon_profile['center'][0] - initial_center[0],
        #                     drawing_polygon_profile['center'][1] - initial_center[1])

        #     finally:
        #         cam.EndAcquisition()

        # except PySpin.SpinnakerException as ex:
        #     print('Error: %s' % ex)
        # finally:
        #     cam.DeInit()
        #     del cam
        #     cam_list.Clear()
        #     system.ReleaseInstance()
        #     cv2.destroyAllWindows()
    
    if camera == 0:
        # Initialize the camera
        # The '0' index is typical for the default camera; adjust as needed for multiple cameras
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("Cannot open camera")
            exit()

        print('Acquiring images...')

        cv2.namedWindow("Camera")
        cv2.setMouseCallback("Camera", mouse_callback) # set up mouse callback

        try:
            while True:
                # Capture frame-by-frame
                ret, frame = cap.read()

                if not ret:
                    print("Can't receive frame (stream end?). Exiting ...")
                    break
                
        
                display_frame = frame.copy() # frame being displayed

                """--------------------------------------------------------------Create a hidden frame--------------------------------------------------------------"""
            
                # Hidden frame being processed and tracked
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # edges = cv2.Canny(gray, 50, 150)
                # enhanced = cv2.addWeighted(gray, 0.8, edges, 0.2, 0)

                # Apply brightness and contrast
                gray = cv2.convertScaleAbs(gray, alpha=contrast, beta=brightness)

                # Apply Gaussian Blur
                if blur_ksize > 1:
                    gray = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)

                # Apply Histogram Equalization for better contrast
                # equ = cv2.equalizeHist(gray)

                # Convert the equalized grayscale image back to BGR for the tracker
                hidden_frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                
                """--------------------------------------------------------------Add zoom factor to the frame--------------------------------------------------------------"""

                if globals.original_frame_width is None or globals.original_frame_height is None:
                    globals.original_frame_width, globals.original_frame_height = frame.shape[1], frame.shape[0]

                # Calculate the dimensions of the zoomed region
                zoomed_height = int(frame.shape[0] / zoom_factor)
                zoomed_width = int(frame.shape[1] / zoom_factor)

                # Check and correct if zoomed region exceeds frame boundaries
                if top < 0:
                    top = 0
                if left < 0:
                    left = 0

                # If the right or bottom boundaries exceed the frame dimensions, adjust them too.
                if left + zoomed_width > frame.shape[1]:
                    left = frame.shape[1] - zoomed_width
                if top + zoomed_height > frame.shape[0]:
                    top = frame.shape[0] - zoomed_height

                # Extract the zoomed region
                zoomed_region = display_frame[top:top + zoomed_height, left:left + zoomed_width]

                x_scale = 1280 / globals.original_frame_width
                y_scale = 960 / globals.original_frame_height

                """--------------------------------------------------------------Draw different mode handlers--------------------------------------------------------------"""

                render_drawing(frame=display_frame)

                if globals.mode == "measuring":
                    render_measuring(frame=display_frame)

                elif globals.mode == "tracking":
                    render_tracking(display_frame=display_frame, hidden_frame=hidden_frame)

                elif globals.mode == "analysing":
                    render_analysing(frame=display_frame)
                
                """--------------------------------------------------------------Fabricate the frame--------------------------------------------------------------"""

                draw_frame(display_frame)

                # cv2.resizeWindow("Camera", 640, 480)
                cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)
                
                resized_frame = cv2.resize(zoomed_region, (1280, 960))

                # # Draw scale bar
                # draw_scale_bar(resized_frame, zoom_factor)

                # Display the combined image
                cv2.imshow("Camera", resized_frame)

                globals.Frame = resized_frame

                if drawing_polygon_profile['center']:  # Make sure the center is not None
                    center_shift_vector = (
                        drawing_polygon_profile['center'][0] - initial_center[0],
                        drawing_polygon_profile['center'][1] - initial_center[1])
                
                if cv2.waitKey(1) == ord('q'):
                    break

        finally:
            cap.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
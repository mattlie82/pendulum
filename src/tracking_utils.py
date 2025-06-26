"""
tracking_utils.py
------------------
Provides utility functions for tracking colored objects in video frames
using HSV color thresholds, and an optional interactive widget for HSV tuning.
"""

import cv2
import numpy as np

# === Core functionality ===

class FrameReadError(Exception):
    pass

def get_frame(input_video, frame = 0):
    cap = cv2.VideoCapture(input_video)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if frame_number < 0 or frame_number >= total_frames:
        raise ValueError(f"Frame number must be between 0 and {total_frames-1}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    
    ret, frame = cap.read()
    if not ret:
        raise FrameReadError(f"Could not read frame {frame_number}")
   
    cap.release()
    return frame    

def detect_color_object(frame, lower=(0, 50, 50), upper=(179, 255, 255)):
    # Convert to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Mask colorful pixels (i.e., not gray)
    mask = cv2.inRange(hsv, lower, upper)

    # Morphological filtering to clean up mask
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print("No colorful object detected.")
        return frame, None, None

    # Largest contour assumed to be pendulum
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)

    # Draw bounding box
    result = frame.copy()
    cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Compute average HSV color inside bounding box
    pendulum_region = hsv[y:y+h, x:x+w]
    region_mask = mask[y:y+h, x:x+w]

    # Mask the HSV region to only colorful parts
    colorful_pixels = pendulum_region[region_mask > 0]
    avg_hsv = np.mean(colorful_pixels, axis=0).astype(int)

    return tuple(avg_hsv), mask
    
def track_colored_object(frame, lower_hsv, upper_hsv, min_contour_area, drawing_params):
    """
    Detects and tracks the largest object within the specified HSV range.

    Args:
        frame (np.ndarray): Input video frame (BGR format).
        lower_hsv (np.ndarray): Lower HSV threshold for object color.
        upper_hsv (np.ndarray): Upper HSV threshold for object color.
        min_contour_area (int): Minimum area to accept a contour as a valid object.
        drawing_params (dict): Dictionary with drawing parameters:
            - 'cross_color' (tuple): BGR color for the cross marker
            - 'contour_color' (tuple): BGR color for the contour outline

    Returns:
        tuple: (processed_frame, (cX, cY), object_found)
            - processed_frame (np.ndarray): Frame with drawn contour and cross if object is found.
            - (cX, cY) (tuple[int, int] or None): Coordinates of the object's centroid.
            - object_found (bool): Whether a valid object was detected.
    """
    cross_color = drawing_params['cross_color']
    contour_color = drawing_params['contour_color']
    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return frame, None, False

    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) <= min_contour_area:
        return frame, None, False

    M = cv2.moments(largest)
    if M["m00"] == 0:
        return frame, None, False

    cX, cY = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
    cv2.drawContours(frame, [largest], -1, contour_color, 2)
    cv2.line(frame, (cX - 20, cY), (cX + 20, cY), cross_color, 2)
    cv2.line(frame, (cX, cY - 20), (cX, cY + 20), cross_color, 2)

    return frame, (cX, cY), True

def process_video(video_path, output_path, lower_hsv, upper_hsv, min_area, rotate, drawing_params={'cross_color': (0, 255, 0), 'contour_color': (0, 0, 0)}):
    """
    Processes a video, tracks a colored object, saves the annotated output video,
    and returns time-series data of object positions.

    Args:
        video_path (str): Path to the input video.
        output_path (str): Path to save the processed video.
        lower_hsv (np.ndarray): Lower HSV threshold for object color.
        upper_hsv (np.ndarray): Upper HSV threshold for object color.
        min_area (int): Minimum area for valid object contour.
        rotate (bool): Whether to rotate frames 90° clockwise before processing.
        drawing_params (dict): Dictionary with drawing parameters passed to track_colored_object.
            Default: {'cross_color': (0, 255, 0), 'contour_color': (0, 0, 0)}

    Returns:
        tuple: (times, x_positions, y_positions)
            - times (list[float]): Time (in seconds) of each detected object.
            - x_positions (list[int]): X coordinates of the object.
            - y_positions (list[int]): Y coordinates of the object.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    out_dim = (height, width) if rotate else (width, height)
    new_w = 640
    new_h = int(out_dim[1] * (new_w / out_dim[0]))
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), int(fps), (new_w, new_h))

    times, xs, ys = [], [], []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if rotate:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

        t_sec = frame_count / fps
        processed, pos, valid = track_colored_object(frame, lower_hsv, upper_hsv, min_area, drawing_params)
        resized = cv2.resize(processed, (new_w, new_h))
        out.write(resized)

        if valid:
            times.append(t_sec)
            xs.append(pos[0])
            ys.append(pos[1])

        frame_count += 1

    cap.release()
    out.release()
    return times, xs, ys

def add_timer(video_path, output_path, drawing_params={'timer_color': (255, 255, 255)}):
    """
    Add a timer overlay to a video showing elapsed time in seconds.
    Uses the same resolution reduction approach as process_video.
    
    Args:
        video_path (str): Path to input video file
        output_path (str): Path for output video file
        drawing_params (dict): Parameters for drawing, expects 'timer_color' as (r,g,b) tuple
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {video_path}")
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Use same resolution reduction as process_video
    new_w = 640
    new_h = int(height * (new_w / width))
    
    # Use same codec as process_video for consistency
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), int(fps), (new_w, new_h))
    
    frame_count = 0
    timer_color = drawing_params.get('timer_color', (255, 255, 255))
    # Convert RGB to BGR for OpenCV
    timer_color_bgr = (timer_color[2], timer_color[1], timer_color[0])
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Calculate time in seconds
        time_seconds = frame_count / fps
        timer_text = f"{time_seconds:.2f}s"
        
        # Add timer text to lower left corner before resizing
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 2
        thickness = 2
        
        # Get text size for background rectangle
        (text_width, text_height), baseline = cv2.getTextSize(timer_text, font, font_scale, thickness)
        
        # Position in lower left corner with some padding
        x = 10
        y = height - 20
        
        # Draw white background rectangle
        cv2.rectangle(frame, (x - 5, y - text_height - 5), (x + text_width + 5, y + baseline + 5), (255, 255, 255), -1)
        
        # Add text to frame
        cv2.putText(frame, timer_text, (x, y), font, font_scale, timer_color_bgr, thickness)
        
        # Resize frame to match output dimensions (same as process_video)
        resized_frame = cv2.resize(frame, (new_w, new_h))
        
        # Write frame to output video
        out.write(resized_frame)
        frame_count += 1
    
    # Release everything
    cap.release()
    out.release()

# === Optional interactive widget interface ===
def launch_hsv_tuning_widget(video_path="INPUT.MOV", max_frames_to_load=100, drawing_params={'cross_color': (0, 255, 0), 'contour_color': (0, 0, 0)}):
    """
    Launches an interactive widget interface to test and adjust HSV bounds
    and minimum area threshold for object detection. Displays the selected
    frame with annotations and a hue gradient bar for visual reference.
    
    Args:
        video_path (str): Path to the input video file.
        max_frames_to_load (int): Maximum number of frames to load from the video.
        drawing_params (dict): Dictionary with drawing parameters for visualization.
    """
    import matplotlib.pyplot as plt
    import ipywidgets as widgets
    from IPython.display import display

    def generate_hue_bar(width=360, height=30):
        """Creates a horizontal hue gradient bar from HSV to RGB."""
        hue = np.linspace(0, 180, width, dtype=np.uint8)
        hsv = np.zeros((height, width, 3), dtype=np.uint8)
        hsv[..., 0] = hue[None, :]
        hsv[..., 1] = 255
        hsv[..., 2] = 255
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        return rgb

    cap = cv2.VideoCapture(video_path)
    frames_bgr = []
    while len(frames_bgr) < max_frames_to_load:
        ret, frame = cap.read()
        if not ret:
            break
        frames_bgr.append(frame)
    cap.release()

    if not frames_bgr:
        print(f"Could not read frames from {video_path}")
        return

    # Widget controls
    lower_h = widgets.IntSlider(value=140, min=0, max=180, description='Lower H')
    lower_s = widgets.IntSlider(value=50, min=0, max=255, description='Lower S')
    lower_v = widgets.IntSlider(value=50, min=0, max=255, description='Lower V')
    upper_h = widgets.IntSlider(value=180, min=0, max=180, description='Upper H')
    upper_s = widgets.IntSlider(value=255, min=0, max=255, description='Upper S')
    upper_v = widgets.IntSlider(value=255, min=0, max=255, description='Upper V')
    min_area_slider = widgets.IntSlider(value=1500, min=0, max=5000, step=100, description='Min Area:')
    frame_idx_slider = widgets.IntSlider(value=0, min=0, max=len(frames_bgr)-1, description='Frame:')

    lb_label = widgets.Label()
    ub_label = widgets.Label()
    output = widgets.Output()

    hue_bar_img = generate_hue_bar()

    def update_display(lower_h, lower_s, lower_v,
                       upper_h, upper_s, upper_v,
                       min_contour_area, frame_index):
        lower = np.minimum([lower_h, lower_s, lower_v], [upper_h, upper_s, upper_v])
        upper = np.maximum([lower_h, lower_s, lower_v], [upper_h, upper_s, upper_v])
        lb_label.value = f"Lower HSV: {lower}"
        ub_label.value = f"Upper HSV: {upper}"

        frame = frames_bgr[frame_index].copy()
        processed, _, _ = track_colored_object(frame, np.array(lower), np.array(upper), min_contour_area, drawing_params)
        rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)

        with output:
            output.clear_output(wait=True)
            fig, axs = plt.subplots(2, 1, figsize=(8, 8), gridspec_kw={"height_ratios": [4, 1]})

            # Image with annotations
            axs[0].imshow(rgb)
            axs[0].set_title(f"Frame {frame_index}")
            axs[0].axis('off')

            # Hue bar with ticks
            axs[1].imshow(hue_bar_img, aspect='auto')
            axs[1].set_title("Hue values (OpenCV scale 0–180)")
            axs[1].set_yticks([])

            ticks = np.linspace(0, hue_bar_img.shape[1] - 1, 10, dtype=int)
            tick_labels = np.linspace(0, 180, 10, dtype=int)
            axs[1].set_xticks(ticks)
            axs[1].set_xticklabels(tick_labels)

            # Optional: show selected hue range
            axs[1].axvline(lower[0] * (hue_bar_img.shape[1] / 180), color='black', linestyle='--')
            axs[1].axvline(upper[0] * (hue_bar_img.shape[1] / 180), color='black', linestyle='--')
            plt.xlim([0,360])
            plt.tight_layout()
            plt.show()

    ui = widgets.VBox([
        lb_label,
        ub_label,
        widgets.HBox([lower_h, lower_s, lower_v]),
        widgets.HBox([upper_h, upper_s, upper_v]),
        min_area_slider,
        frame_idx_slider
    ])

    out = widgets.interactive_output(update_display, {
        "lower_h": lower_h, "lower_s": lower_s, "lower_v": lower_v,
        "upper_h": upper_h, "upper_s": upper_s, "upper_v": upper_v,
        "min_contour_area": min_area_slider,
        "frame_index": frame_idx_slider
    })

    display(ui, output, out)
    update_display(lower_h.value, lower_s.value, lower_v.value,
                   upper_h.value, upper_s.value, upper_v.value,
                   min_area_slider.value, frame_idx_slider.value)

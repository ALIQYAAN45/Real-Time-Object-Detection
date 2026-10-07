"""
webcam.py
---------
Modular webcam management module using OpenCV.
Handles camera initialization, frame capture, on-screen text, and safe cleanup.
"""

import cv2
import sys


def open_webcam(camera_index=0):
    """
    Safely opens the specified webcam index.
    
    Parameters:
        camera_index (int): Index of the camera (default: 0).
        
    Returns:
        cv2.VideoCapture: Initialized video capture object.
        
    Raises:
        RuntimeError: If the webcam cannot be accessed or opened.
    """
    # Initialize OpenCV VideoCapture with the requested camera index
    cap = cv2.VideoCapture(camera_index)

    # On Windows, if default backend takes long or fails, DirectShow (CAP_DSHOW) helps
    if not cap.isOpened() and sys.platform.startswith("win"):
        cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)

    # Verify if camera was successfully opened
    if not cap.isOpened():
        raise RuntimeError(
            f"Error: Unable to open webcam at index {camera_index}.\n"
            "Possible causes:\n"
            "  1. No webcam is connected to this computer.\n"
            "  2. Another application (Zoom, Teams, Camera app) is currently using the webcam.\n"
            "  3. Camera permissions are blocked in Windows Settings (Privacy & Security -> Camera)."
        )

    return cap


def draw_overlay(frame, text="Live Feed | Press 'Q' to quit"):
    """
    Draws a clean instructional banner on top of the webcam frame.
    
    Parameters:
        frame: The OpenCV BGR image frame.
        text (str): The text message to display on the overlay.
        
    Returns:
        frame: The frame with overlay drawn on it.
    """
    # Frame dimensions
    h, w = frame.shape[:2]

    # Draw a top background rectangle for high contrast readability
    cv2.rectangle(frame, (0, 0), (w, 40), (20, 20, 20), -1)

    # Put the instructional text on top of the dark rectangle
    cv2.putText(
        frame,
        text,
        (15, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 0),  # Bright green text
        2,
        cv2.LINE_AA
    )
    return frame


def run_webcam_stream(camera_index=0, detector=None, logger=None, max_frames=None, window_name="Real-Time Detection Feed"):
    """
    Runs the continuous webcam feed loop with optional YOLO object detection
    and optional MySQL detection logging with cooldown debouncing.
    
    Parameters:
        camera_index (int): The webcam device index to open.
        detector (ObjectDetector, optional): Detector instance to detect and draw objects.
        logger (DetectionLogger, optional): Database logger with cooldown debouncing.
        max_frames (int, optional): Number of frames to process before auto-stopping.
        window_name (str): Title of the OpenCV display window.
        
    Returns:
        bool: True if the stream completed successfully.
    """
    print(f"[INFO] Initializing webcam (index: {camera_index})...")
    cap = open_webcam(camera_index)
    print(f"[INFO] Webcam successfully opened. Press 'Q' in the video window to exit.")

    frame_count = 0

    try:
        while True:
            # Read frame from webcam
            ret, frame = cap.read()

            # Check if frame was read correctly
            if not ret or frame is None:
                print("[WARNING] Failed to grab frame from webcam. Exiting stream loop.")
                break

            frame_count += 1

            if detector is not None:
                # Run YOLO inference through modular detector
                detections = detector.detect(frame)

                # If a database logger is provided, log valid detections with debouncing
                if logger is not None and detections:
                    saved_ids = logger.log_detections(detections)
                    if saved_ids:
                        logged_classes = [d["class_name"] for d in detections if d.get("class_name")]
                        print(f"[MYSQL LOG] Saved {len(saved_ids)} event(s) to database: {logged_classes}")

                # Draw bounding boxes and labels on the frame
                frame = detector.draw_detections(frame, detections)

                # Banner with active detection stats
                banner_text = f"YOLO Live Feed | Detected: {len(detections)} | Press 'Q' to quit"
                frame = draw_overlay(frame, banner_text)
            else:
                # Default overlay without detection
                frame = draw_overlay(frame, "Live Feed | Press 'Q' to quit")

            # Display the video frame in an OpenCV window
            cv2.imshow(window_name, frame)

            # Wait 1ms for keyboard input; check if user pressed 'q' or 'Q' or ESC (27)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q') or key == 27:
                print("[INFO] 'Q' pressed. Exiting webcam stream.")
                break

            # If a frame limit was set (e.g. for self-test verification), break after reached
            if max_frames is not None and frame_count >= max_frames:
                print(f"[INFO] Reached requested test limit of {max_frames} frames.")
                break

        return True

    finally:
        # Always release the camera and close all OpenCV windows cleanly
        print("[INFO] Releasing webcam and closing OpenCV display windows...")
        cap.release()
        cv2.destroyAllWindows()


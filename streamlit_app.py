"""
streamlit_app.py
----------------
Main Streamlit Web Application for the Real-Time Object Detection & Logging Platform.
CEP Assignment 5: Complete Dashboard with Live OpenCV Feed, YOLOv8 Detection, and MySQL Logging.
"""

import time
import cv2
import streamlit as st
import pandas as pd

import config
from detector import ObjectDetector
from database import (
    initialize_database,
    test_connection,
    get_recent_logs,
    get_detection_count,
    get_unique_object_count,
    get_class_detection_counts,
    clear_all_logs,
    DetectionLogger,
)
from ui import inject_custom_css, render_header, format_logs_dataframe

# 1. Page Configuration
st.set_page_config(
    page_title="Real-Time Object Detection & Logging Platform",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject custom modern styling
inject_custom_css()

# Render title and subtitle
render_header()


# 2. Cached YOLO Model Initializer (Loads weights once into memory)
@st.cache_resource
def load_detector():
    """Initializes and caches the YOLO object detector."""
    return ObjectDetector(model_name=config.MODEL_NAME)


detector = load_detector()
all_coco_classes = detector.get_all_classes()


# 3. Sidebar Configuration Controls
st.sidebar.markdown("### ⚙️ Detection Controls")

# Confidence threshold slider (Default from config: 0.70)
confidence_threshold = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.20,
    max_value=1.00,
    value=float(config.DEFAULT_CONFIDENCE_THRESHOLD),
    step=0.05,
    format="%.2f",
    help="Only detections with confidence greater than or equal to this threshold will be shown and logged."
)

# Object selection multi-select filter (Defaults from config)
valid_defaults = [c for c in config.DEFAULT_SELECTED_OBJECTS if c in all_coco_classes]
selected_objects = st.sidebar.multiselect(
    "Filter Objects to Detect & Log",
    options=all_coco_classes,
    default=valid_defaults,
    help="Select the specific object classes you want to detect and record in MySQL."
)

# Logging cooldown slider
cooldown_seconds = st.sidebar.slider(
    "Logging Cooldown (seconds)",
    min_value=0.5,
    max_value=10.0,
    value=float(config.DETECTION_COOLDOWN_SECONDS),
    step=0.5,
    help="Prevents database spamming by waiting this many seconds before saving the same object class again."
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📹 Camera Settings")
camera_index = st.sidebar.number_input(
    "Camera Device Index",
    min_value=0,
    max_value=5,
    value=config.WEBCAM_INDEX,
    step=1,
    help="0 is the default laptop webcam, 1 is typically an external USB webcam."
)

# Main toggle switch for starting/stopping the webcam feed
run_webcam = st.sidebar.toggle("▶️ Start Live Webcam Feed", value=False)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🗄️ Database Status")

# Check MySQL connection status
db_connected, db_msg = test_connection()
if db_connected:
    try:
        initialize_database()
        st.sidebar.markdown(
            f'<div class="status-badge-ok">🟢 MySQL Connected ({config.DB_NAME})</div>',
            unsafe_allow_html=True
        )
    except Exception as e:
        st.sidebar.markdown(
            f'<div class="status-badge-err">⚠️ Init Error: {e}</div>',
            unsafe_allow_html=True
        )
else:
    st.sidebar.markdown(
        '<div class="status-badge-err">🔴 MySQL Offline (Start XAMPP)</div>',
        unsafe_allow_html=True
    )
    st.sidebar.caption(db_msg)

if st.sidebar.button("🔄 Refresh Database Logs"):
    st.rerun()

if st.sidebar.button("🗑️ Clear All Logs"):
    if db_connected:
        clear_all_logs()
        st.sidebar.success("Database logs cleared.")
        st.rerun()


# 4. Top KPI Metric Cards
total_logs = get_detection_count() if db_connected else 0
unique_classes = get_unique_object_count() if db_connected else 0

col_m1, col_m2, col_m3, col_m4 = st.columns(4)
with col_m1:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">📊 Total Detections Logged</div>
            <div class="metric-value">{total_logs}</div>
        </div>
        """,
        unsafe_allow_html=True
    )
with col_m2:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">🏷️ Unique Classes Recorded</div>
            <div class="metric-value">{unique_classes}</div>
        </div>
        """,
        unsafe_allow_html=True
    )
with col_m3:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">🎯 Active Confidence Threshold</div>
            <div class="metric-value">{int(confidence_threshold * 100)}%</div>
        </div>
        """,
        unsafe_allow_html=True
    )
with col_m4:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">⏱️ Logging Cooldown</div>
            <div class="metric-value">{cooldown_seconds:.1f}s</div>
        </div>
        """,
        unsafe_allow_html=True
    )


# 5. Main Content: Video Feed & Real-Time Stats
col_video, col_stats = st.columns([7, 5])

with col_video:
    st.markdown("### 🎥 Live Detection Feed")
    video_placeholder = st.empty()
    status_placeholder = st.empty()

    if not run_webcam:
        video_placeholder.info(
            "📷 Webcam is currently **stopped**.\n\n"
            "Toggle **'▶️ Start Live Webcam Feed'** in the sidebar to begin real-time YOLO object detection and MySQL logging."
        )

with col_stats:
    st.markdown("### 📈 Detection Distribution")
    chart_placeholder = st.empty()
    
    if db_connected:
        class_dist = get_class_detection_counts()
        if class_dist:
            dist_df = pd.DataFrame(list(class_dist.items()), columns=["Object Class", "Count"]).set_index("Object Class")
            chart_placeholder.bar_chart(dist_df)
        else:
            chart_placeholder.info("No detections logged in the database yet. Point your camera at selected objects!")
    else:
        chart_placeholder.warning("MySQL is offline. Start MySQL in XAMPP to record detection logs.")


# 6. Webcam Capture & Inference Loop (Active when toggle is ON)
if run_webcam:
    if not selected_objects:
        st.warning("⚠️ No object classes selected. Please pick at least one object class in the sidebar.")
    else:
        # Update detector dynamically with sidebar settings
        detector.set_confidence_threshold(confidence_threshold)
        detector.set_selected_classes(selected_objects)

        # Initialize debounced database logger
        db_logger = DetectionLogger(cooldown_seconds=cooldown_seconds) if db_connected else None

        # Open webcam device
        cap = cv2.VideoCapture(int(camera_index))
        # Windows DirectShow fallback if needed
        if not cap.isOpened() and cv2.__file__:
            cap = cv2.VideoCapture(int(camera_index), cv2.CAP_DSHOW)

        if not cap.isOpened():
            st.error(
                f"❌ Unable to open camera at index {camera_index}.\n"
                "Please verify camera permissions in Windows Settings and ensure other video apps are closed."
            )
        else:
            status_placeholder.success("🟢 Camera streaming actively. Real-time YOLO detection in progress.")

            try:
                # Continuous streaming loop
                while run_webcam:
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        st.warning("Failed to grab video frame from camera.")
                        break

                    # 1. Run YOLO inference and apply filters
                    detections = detector.detect(frame)

                    # 2. Log valid detections to MySQL with debouncing
                    if db_logger and detections:
                        db_logger.log_detections(detections)

                    # 3. Draw bounding boxes & labels
                    annotated_frame = detector.draw_detections(frame.copy(), detections)

                    # 4. Convert BGR to RGB for Streamlit rendering
                    rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

                    # 5. Display the frame in the Streamlit placeholder
                    video_placeholder.image(
                        rgb_frame,
                        caption=f"Live Detection | Active detections in frame: {len(detections)}",
                        use_container_width=True
                    )

                    # Short sleep to yield control and keep CPU usage balanced
                    time.sleep(0.03)

            finally:
                cap.release()
                status_placeholder.info("Webcam stopped and camera device released cleanly.")


# 7. Recent Database Logs Table
st.markdown("---")
st.markdown("### 📋 Recent Detection Logs (MySQL: `vision_platform.detection_logs`)")

if db_connected:
    recent_logs = get_recent_logs(limit=15)
    logs_df = format_logs_dataframe(recent_logs)
    if not logs_df.empty:
        st.dataframe(logs_df, use_container_width=True, hide_index=True)
    else:
        st.info("No detections recorded in `detection_logs` table yet.")
else:
    st.error("Cannot display logs: MySQL server is not connected.")

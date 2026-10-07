"""
config.py
---------
Configuration settings for the Real-Time Object Detection & Logging Platform.
Loads environment variables from .env using python-dotenv.
"""

import os
from dotenv import load_dotenv

# Load variables from .env file into environment
load_dotenv()

# --- Webcam Configuration ---
# 0 is usually the built-in laptop camera, 1 is typically an external USB webcam
WEBCAM_INDEX = int(os.getenv("WEBCAM_INDEX", 0))

# --- Database Configuration ---
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", 3306))
DB_USER = os.getenv("DB_USER", "root")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")
DB_NAME = os.getenv("DB_NAME", "vision_platform")

# --- Detection Settings ---
# Ultralytics lightweight model (yolov8n is ~6MB and fast on standard laptop CPU)
MODEL_NAME = os.getenv("MODEL_NAME", "yolov8n.pt")

# Cooldown in seconds before the same object class can be logged again to MySQL
DETECTION_COOLDOWN_SECONDS = float(os.getenv("DETECTION_COOLDOWN_SECONDS", 3.0))

# Default minimum confidence score (e.g. 0.70 = 70%)
DEFAULT_CONFIDENCE_THRESHOLD = float(os.getenv("DEFAULT_CONFIDENCE_THRESHOLD", 0.70))

# Default list of object classes to detect/log
_default_objects_str = os.getenv(
    "SELECTED_OBJECTS",
    "person,cell phone,laptop,bottle,chair"
)
DEFAULT_SELECTED_OBJECTS = [
    obj.strip().lower() for obj in _default_objects_str.split(",") if obj.strip()
]

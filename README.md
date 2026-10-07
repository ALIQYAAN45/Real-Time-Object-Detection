# Real-Time Object Detection & Logging Platform
### CEP Assignment 5: Computer Vision & Database Engineering

A modular, real-time computer vision platform combining **OpenCV**, **Ultralytics YOLOv8**, **MySQL**, and **Streamlit**. The system captures live video from a webcam, performs real-time object detection with bounding boxes, applies confidence and object-class filtering, logs detection events into a MySQL database with cooldown debouncing, and displays metrics on a Streamlit dashboard.

---

## 🌟 Key Features

- **Live Webcam Capture:** Real-time continuous video frame acquisition using OpenCV.
- **YOLOv8 Object Detection:** High-speed, lightweight inference on standard CPU laptops using pretrained `yolov8n.pt`.
- **Bounding Box & Label Overlays:** Visualizes detected objects with class labels and confidence percentages directly on video frames.
- **Configurable Confidence Threshold:** Adjustable minimum confidence (default: `70%` / `0.70`) to eliminate false positives.
- **Selective Object Filtering:** Configurable target object classes (`person`, `cell phone`, `laptop`, `bottle`, `chair`, or any of the 80 COCO classes).
- **Automated MySQL Logging:** Automatically saves timestamps, detected classes, confidence scores, and bounding box coordinates (`x, y, w, h`) to MySQL.
- **Smart Debouncing / Cooldown:** Prevents database flooding by enforcing a configurable time window between repeat logs of the same object class.
- **Interactive Streamlit Dashboard:** Complete web dashboard featuring live webcam feed controls, interactive sliders, dynamic KPI metrics, and recent detection logs.
- **Modular Clean Architecture:** Separation of concerns across configuration, camera capture, detection, database operations, and presentation.

---

## 🛠️ Technologies Used

| Technology | Purpose |
| :--- | :--- |
| **Python 3.11** | Core programming language |
| **OpenCV (`cv2`)** | Video frame capture, image color conversion, and graphics drawing |
| **Ultralytics YOLOv8** | Lightweight deep learning model for real-time object detection |
| **MySQL (via XAMPP)** | Relational database for persistent detection logging |
| **mysql-connector-python** | Official MySQL driver for Python |
| **Streamlit** | Interactive web dashboard and user interface |
| **python-dotenv** | Secure environment variable management |
| **Pandas** | Tabular data presentation in the UI |

---

## 🏗️ System Architecture

```text
               +------------------------------------+
               |         Webcam Device (0)          |
               +-----------------+------------------+
                                 |
                                 v
               +-----------------+------------------+
               |        webcam.py (OpenCV)          |
               +-----------------+------------------+
                                 | (Raw Frame)
                                 v
               +-----------------+------------------+
               |       detector.py (YOLOv8)         |
               |   - Confidence Filter (>= 70%)     |
               |   - Object Selection Filter        |
               +-----------------+------------------+
                                 |
                 +---------------+---------------+
                 | (Filtered Detections)         | (Annotated Frame)
                 v                               v
+----------------+----------------+    +---------+----------+
|  database.py (MySQL Connector)  |    |  streamlit_app.py  |
|   - Cooldown Debouncing (3.0s)  |    |  - Live Video View |
|   - vision_platform Database    |    |  - Real-Time KPIs  |
|   - detection_logs Table        |    |  - Log Table View  |
+---------------------------------+    +--------------------+
```

---

## 📁 Project Structure

```text
Real-Time-Object-Detection/
│
├── venv/                         # Python virtual environment (ignored by Git)
├── models/
│   └── yolov8n.pt                # Pretrained YOLOv8 Nano model weights (~6MB)
│
├── main.py                       # CLI entry point for standalone OpenCV detection
├── detector.py                   # Modular YOLOv8 object detector with filtering
├── webcam.py                     # OpenCV camera capture, overlay, and loop
├── database.py                   # MySQL connection, table creation, and queries
├── config.py                     # Centralized settings loaded from .env
├── ui.py                         # Streamlit CSS styles, theme tokens, and formatters
├── streamlit_app.py              # Interactive Streamlit dashboard application
│
├── database_schema.sql           # SQL script to create database and tables
├── requirements.txt              # Project Python dependencies
├── .env                          # Local credentials (ignored by Git)
├── .env.example                  # Template environment variables for GitHub
├── .gitignore                    # Git ignore file protecting credentials and caches
├── README.md                     # Documentation and student guide
│
└── tests/
    └── test_suite.py             # Automated 15-point verification suite
```

---

## 🚀 Installation & Setup

### 1. Clone or Open the Project
Open PowerShell or your terminal in the project directory:
```powershell
cd Real-Time-Object-Detection
```

### 2. Activate the Virtual Environment
Activate the existing virtual environment:
```powershell
.\venv\Scripts\Activate.ps1
```
*(Or create a new one using `py -3.11 -m venv venv` if setting up on a new computer).*

### 3. Install Dependencies
```powershell
.\venv\Scripts\pip.exe install -r requirements.txt
```

---

## 🗄️ XAMPP & MySQL Setup

1. Open the **XAMPP Control Panel**.
2. Click **Start** next to **MySQL** (and optionally **Apache**).
3. Ensure MySQL shows port **3306** active.
4. Open your browser to `http://localhost/phpmyadmin` to verify phpMyAdmin is accessible.
5. *(Optional)* You can import `database_schema.sql` directly into phpMyAdmin, or let the Python application create the database and table automatically when it runs.

---

## ⚙️ Environment Variables Configuration

Copy `.env.example` to `.env`:
```powershell
Copy-Item .env.example .env
```

Ensure your `.env` contains your local MySQL settings:
```ini
# MySQL Database Configuration
DB_HOST=localhost
DB_PORT=3306
DB_USER=root
DB_PASSWORD=
DB_NAME=vision_platform

# Detection & Logging Configuration
MODEL_NAME=yolov8n.pt
DEFAULT_CONFIDENCE_THRESHOLD=0.70
SELECTED_OBJECTS=person,cell phone,laptop,bottle,chair
DETECTION_COOLDOWN_SECONDS=3.0
WEBCAM_INDEX=0
```

> **Security Note:** Never commit your real `.env` file to GitHub. The `.gitignore` file is already configured to keep `.env` private.

---

## 🎮 How to Run the Application

### Option A: Interactive Streamlit Web Dashboard (Recommended)
To launch the full web dashboard:
```powershell
.\venv\Scripts\streamlit.exe run streamlit_app.py
```
Open your browser at `http://localhost:8501`.
- Toggle **"▶️ Start Live Webcam Feed"** in the sidebar.
- Adjust confidence slider and select objects in real time.
- View live detection metrics and the MySQL log table update dynamically.

### Option B: Standalone OpenCV Window
To run the lightweight OpenCV desktop window directly:
```powershell
.\venv\Scripts\python.exe main.py
```
- Press **`Q`** or **`Esc`** in the video window to quit safely.

---

## 🧪 Verification & Testing

To run the automated **15-point verification test suite**:
```powershell
.\venv\Scripts\python.exe tests/test_suite.py
```
This tests:
1. Python environment version
2. Library imports (`cv2`, `ultralytics`, `streamlit`, `mysql.connector`, `dotenv`)
3. YOLOv8 model loading
4. Webcam frame capture
5. YOLO inference execution
6. Selected-object filtering
7. Confidence threshold filtering
8. MySQL database connectivity
9. Table existence (`vision_platform.detection_logs`)
10. Detection record insertion
11. Detection record retrieval
12. Streamlit code compilation
13. Video pipeline transformation
14. Valid detection logging with cooldown
15. Invalid detection rejection

---

## 📊 Database Schema Details

**Database:** `vision_platform`  
**Table:** `detection_logs`

| Column | Type | Description |
| :--- | :--- | :--- |
| `log_id` | `INT AUTO_INCREMENT PRIMARY KEY` | Unique log entry identifier |
| `timestamp` | `DATETIME DEFAULT CURRENT_TIMESTAMP` | Time of the detection event |
| `object_class` | `VARCHAR(100)` | Name of the detected object (e.g. `person`) |
| `confidence` | `FLOAT` | Detection confidence score (0.0 to 1.0) |
| `bbox_x` | `INT` | Bounding box top-left X coordinate |
| `bbox_y` | `INT` | Bounding box top-left Y coordinate |
| `bbox_w` | `INT` | Bounding box width in pixels |
| `bbox_h` | `INT` | Bounding box height in pixels |

---

## 📸 Screenshots & Demonstration

*(Place your project presentation screenshots here when preparing your college report)*
- **Dashboard Overview:** `docs/screenshots/dashboard_view.png`
- **Live Detection in Action:** `docs/screenshots/live_webcam_detection.png`
- **phpMyAdmin MySQL Table:** `docs/screenshots/mysql_table_logs.png`

---

## 🔮 Future Improvements

- Add support for recording short video clips of detected security events.
- Implement Telegram or Email alerts when specific critical objects (e.g., unauthorized persons) are detected.
- Add FPS (frames-per-second) counter overlay on video streams.
- Support multi-camera switching for multi-room surveillance.

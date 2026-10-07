"""
database.py
-----------
Modular MySQL database management module for the Real-Time Object Detection Platform.
Handles connection pooling, database initialization, detection logging, and queries.
Uses mysql-connector-python and configuration from config.py (.env).
"""

import time
import mysql.connector
from mysql.connector import Error
import config


def get_connection(use_database=True):
    """
    Establishes and returns a connection to the MySQL database.
    
    Parameters:
        use_database (bool): If True, connects directly to DB_NAME.
                             If False, connects to MySQL server without selecting a DB
                             (used for initial database creation).
                             
    Returns:
        mysql.connector.connection.MySQLConnection: Active MySQL connection.
        
    Raises:
        mysql.connector.Error: If the connection fails.
    """
    conn_params = {
        "host": config.DB_HOST,
        "port": config.DB_PORT,
        "user": config.DB_USER,
        "password": config.DB_PASSWORD,
    }

    if use_database:
        conn_params["database"] = config.DB_NAME

    return mysql.connector.connect(**conn_params)


def test_connection():
    """
    Tests whether the MySQL database server is reachable and responsive.
    
    Returns:
        tuple[bool, str]: (Success boolean, Status message)
    """
    try:
        conn = get_connection(use_database=False)
        if conn.is_connected():
            server_info = conn.get_server_info()
            conn.close()
            return True, f"Connected successfully to MySQL Server (v{server_info}) on {config.DB_HOST}:{config.DB_PORT}."
    except Error as e:
        return False, f"MySQL connection failed: {e}"
    except Exception as e:
        return False, f"Unexpected connection error: {e}"

    return False, "Unable to establish MySQL connection."


def initialize_database():
    """
    Initializes the MySQL database and table.
    Safe to execute multiple times (uses CREATE DATABASE/TABLE IF NOT EXISTS).
    
    Returns:
        bool: True if initialization succeeded.
    """
    try:
        # Step 1: Connect to server without database selected to create database
        conn = get_connection(use_database=False)
        cursor = conn.cursor()

        # Create database if not already created
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS `{config.DB_NAME}`;")
        cursor.close()
        conn.close()

        # Step 2: Connect to the newly ensured database to create table
        conn = get_connection(use_database=True)
        cursor = conn.cursor()

        create_table_query = """
        CREATE TABLE IF NOT EXISTS detection_logs (
            log_id INT AUTO_INCREMENT PRIMARY KEY,
            timestamp DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
            object_class VARCHAR(100) NOT NULL,
            confidence FLOAT NOT NULL,
            bbox_x INT NOT NULL,
            bbox_y INT NOT NULL,
            bbox_w INT NOT NULL,
            bbox_h INT NOT NULL
        );
        """
        cursor.execute(create_table_query)
        conn.commit()

        cursor.close()
        conn.close()
        print(f"[INFO] Database '{config.DB_NAME}' and table 'detection_logs' initialized successfully.")
        return True

    except Error as e:
        print(f"[ERROR] Failed to initialize database: {e}")
        raise


def log_detection(object_class, confidence, bbox_x, bbox_y, bbox_w, bbox_h):
    """
    Logs a single detection record into the detection_logs table.
    
    Parameters:
        object_class (str): Name of detected object (e.g. 'person').
        confidence (float): Confidence score (0.0 to 1.0).
        bbox_x (int): Top-left X coordinate of bounding box.
        bbox_y (int): Top-left Y coordinate of bounding box.
        bbox_w (int): Width of bounding box.
        bbox_h (int): Height of bounding box.
        
    Returns:
        int: The log_id of the newly inserted row, or None on failure.
    """
    conn = None
    try:
        conn = get_connection(use_database=True)
        cursor = conn.cursor()

        query = """
        INSERT INTO detection_logs (object_class, confidence, bbox_x, bbox_y, bbox_w, bbox_h)
        VALUES (%s, %s, %s, %s, %s, %s)
        """
        values = (
            str(object_class),
            float(confidence),
            int(bbox_x),
            int(bbox_y),
            int(bbox_w),
            int(bbox_h),
        )

        cursor.execute(query, values)
        conn.commit()
        log_id = cursor.lastrowid

        cursor.close()
        return log_id

    except Error as e:
        print(f"[ERROR] Failed to log detection to MySQL: {e}")
        return None
    finally:
        if conn and conn.is_connected():
            conn.close()


def get_recent_logs(limit=20):
    """
    Retrieves the most recent detection logs from the database.
    
    Parameters:
        limit (int): Maximum number of log rows to retrieve (default: 20).
        
    Returns:
        list[dict]: List of detection log dictionaries.
    """
    conn = None
    try:
        conn = get_connection(use_database=True)
        cursor = conn.cursor(dictionary=True)

        query = """
        SELECT log_id, timestamp, object_class, confidence, bbox_x, bbox_y, bbox_w, bbox_h
        FROM detection_logs
        ORDER BY log_id DESC
        LIMIT %s
        """
        cursor.execute(query, (int(limit),))
        rows = cursor.fetchall()

        cursor.close()
        return rows

    except Error as e:
        print(f"[ERROR] Failed to fetch detection logs: {e}")
        return []
    finally:
        if conn and conn.is_connected():
            conn.close()


def get_detection_count():
    """
    Returns the total number of detections logged in the database.
    
    Returns:
        int: Total detection count.
    """
    conn = None
    try:
        conn = get_connection(use_database=True)
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM detection_logs")
        count = cursor.fetchone()[0]

        cursor.close()
        return int(count)

    except Error as e:
        print(f"[ERROR] Failed to count detections: {e}")
        return 0
    finally:
        if conn and conn.is_connected():
            conn.close()


def get_unique_object_count():
    """
    Returns the count of distinct object classes recorded in the database.
    
    Returns:
        int: Number of unique object classes.
    """
    conn = None
    try:
        conn = get_connection(use_database=True)
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(DISTINCT object_class) FROM detection_logs")
        count = cursor.fetchone()[0]

        cursor.close()
        return int(count)

    except Error as e:
        print(f"[ERROR] Failed to count unique object classes: {e}")
        return 0
    finally:
        if conn and conn.is_connected():
            conn.close()


def get_class_detection_counts():
    """
    Returns the count of detections grouped by object class.
    Useful for Streamlit charts and distributions.
    
    Returns:
        dict[str, int]: Mapping from object_class to total count.
    """
    conn = None
    try:
        conn = get_connection(use_database=True)
        cursor = conn.cursor()

        cursor.execute("""
        SELECT object_class, COUNT(*) 
        FROM detection_logs 
        GROUP BY object_class 
        ORDER BY COUNT(*) DESC
        """)
        rows = cursor.fetchall()
        cursor.close()
        return {row[0]: row[1] for row in rows}

    except Error as e:
        print(f"[ERROR] Failed to get class counts: {e}")
        return {}
    finally:
        if conn and conn.is_connected():
            conn.close()


def clear_all_logs():
    """
    Deletes all records from detection_logs (useful for reset/testing).
    
    Returns:
        bool: True on success.
    """
    conn = None
    try:
        conn = get_connection(use_database=True)
        cursor = conn.cursor()
        cursor.execute("TRUNCATE TABLE detection_logs")
        conn.commit()
        cursor.close()
        return True
    except Error as e:
        print(f"[ERROR] Failed to truncate detection_logs: {e}")
        return False
    finally:
        if conn and conn.is_connected():
            conn.close()


class DetectionLogger:
    """
    Handles logging valid detections to MySQL with per-class cooldown debouncing.
    Prevents flooding the database with identical detections on every video frame.
    """

    def __init__(self, cooldown_seconds=None):
        """
        Parameters:
            cooldown_seconds (float, optional): Cooldown interval in seconds.
        """
        self.cooldown_seconds = (
            float(cooldown_seconds)
            if cooldown_seconds is not None
            else float(config.DETECTION_COOLDOWN_SECONDS)
        )
        # Dictionary tracking the timestamp when each object class was last logged:
        # e.g. {"person": 1696700000.5, "laptop": 1696700004.2}
        self.last_logged_times = {}

    def set_cooldown(self, new_cooldown):
        """Update cooldown interval dynamically."""
        self.cooldown_seconds = float(new_cooldown)

    def log_if_ready(self, detection):
        """
        Logs a single detection dictionary if its cooldown interval has elapsed.
        
        Parameters:
            detection (dict): Detection containing class_name, confidence, bbox coordinates.
            
        Returns:
            int or None: The log_id if logged, or None if skipped due to cooldown.
        """
        class_name = detection["class_name"].lower()
        now = time.time()
        last_logged = self.last_logged_times.get(class_name, 0.0)

        # Check if enough time has passed since this object class was last saved
        if (now - last_logged) >= self.cooldown_seconds:
            log_id = log_detection(
                object_class=detection["class_name"],
                confidence=detection["confidence"],
                bbox_x=detection["bbox_x"],
                bbox_y=detection["bbox_y"],
                bbox_w=detection["bbox_w"],
                bbox_h=detection["bbox_h"]
            )
            if log_id is not None:
                self.last_logged_times[class_name] = now
                return log_id

        return None

    def log_detections(self, detections):
        """
        Logs all detections in the list that satisfy cooldown constraints.
        
        Parameters:
            detections (list[dict]): Filtered detections for current frame.
            
        Returns:
            list[int]: List of generated log_ids for objects inserted in this call.
        """
        inserted_ids = []
        for det in detections:
            log_id = self.log_if_ready(det)
            if log_id:
                inserted_ids.append(log_id)
        return inserted_ids

    def reset_cooldowns(self):
        """Resets the cooldown history."""
        self.last_logged_times.clear()


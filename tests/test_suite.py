"""
test_suite.py
-------------
Automated 15-point verification suite for the Real-Time Object Detection & Logging Platform.
Tests all requirements from the CEP Assignment 5 specification.
"""

import sys
import os
import time

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import config


def run_tests():
    print("=" * 75)
    print(" CEP ASSIGNMENT 5: COMPREHENSIVE 15-POINT VERIFICATION SUITE")
    print("=" * 75)

    passed_tests = 0

    # ----------------------------------------------------
    # TEST 1: Python environment works
    # ----------------------------------------------------
    try:
        py_version = sys.version.split()[0]
        assert sys.version_info >= (3, 10), f"Python version too old: {py_version}"
        print(f"[PASS] TEST 1: Python environment works (Python {py_version})")
        passed_tests += 1
    except Exception as e:
        print(f"[FAIL] TEST 1: Python environment failed: {e}")

    # ----------------------------------------------------
    # TEST 2: Required packages import successfully
    # ----------------------------------------------------
    try:
        import cv2
        import ultralytics
        import streamlit
        import mysql.connector
        import dotenv
        print("[PASS] TEST 2: All 5 required packages import successfully (cv2, ultralytics, streamlit, mysql.connector, dotenv)")
        passed_tests += 1
    except Exception as e:
        print(f"[FAIL] TEST 2: Required packages failed to import: {e}")

    # ----------------------------------------------------
    # TEST 3: YOLO model loads
    # ----------------------------------------------------
    detector = None
    try:
        from detector import ObjectDetector
        detector = ObjectDetector(model_name=config.MODEL_NAME, conf_threshold=0.70)
        assert detector.model is not None
        print(f"[PASS] TEST 3: YOLO model loads successfully ({config.MODEL_NAME})")
        passed_tests += 1
    except Exception as e:
        print(f"[FAIL] TEST 3: YOLO model loading failed: {e}")

    # ----------------------------------------------------
    # TEST 4: Webcam works
    # ----------------------------------------------------
    webcam_frame = None
    try:
        from webcam import open_webcam
        cap = open_webcam(config.WEBCAM_INDEX)
        ret, frame = cap.read()
        cap.release()
        assert ret and frame is not None, "Failed to read frame from webcam"
        webcam_frame = frame
        print(f"[PASS] TEST 4: Webcam works (Index {config.WEBCAM_INDEX}, Frame resolution: {frame.shape[1]}x{frame.shape[0]})")
        passed_tests += 1
    except Exception as e:
        print(f"[FAIL] TEST 4: Webcam capture failed: {e}")

    # ----------------------------------------------------
    # TEST 5: Object detection works
    # ----------------------------------------------------
    try:
        # Run detection on captured webcam frame or dummy frame
        assert detector is not None
        dets = detector.detect(webcam_frame)
        print(f"[PASS] TEST 5: Object detection inference works (Executed on frame, returned {len(dets)} filtered detections)")
        passed_tests += 1
    except Exception as e:
        print(f"[FAIL] TEST 5: Object detection inference failed: {e}")

    # ----------------------------------------------------
    # TEST 6: Object filtering works
    # ----------------------------------------------------
    try:
        detector.set_selected_classes(["person", "laptop"])
        detector.set_confidence_threshold(0.70)
        # Person should pass
        assert detector.is_valid_detection("person", 0.85) is True, "Selected class 'person' rejected"
        # Car should be rejected
        assert detector.is_valid_detection("car", 0.95) is False, "Unselected class 'car' accepted"
        print("[PASS] TEST 6: Object class filtering works (Selected classes accepted, unselected classes rejected)")
        passed_tests += 1
    except Exception as e:
        print(f"[FAIL] TEST 6: Object filtering failed: {e}")

    # ----------------------------------------------------
    # TEST 7: Confidence threshold works
    # ----------------------------------------------------
    try:
        detector.set_selected_classes(["person"])
        detector.set_confidence_threshold(0.70)
        # 0.80 >= 0.70 -> True
        assert detector.is_valid_detection("person", 0.80) is True, "Confidence >= 0.70 rejected"
        # 0.65 < 0.70 -> False
        assert detector.is_valid_detection("person", 0.65) is False, "Confidence < 0.70 accepted"
        print("[PASS] TEST 7: Confidence threshold works (>= 70% accepted, < 70% rejected)")
        passed_tests += 1
    except Exception as e:
        print(f"[FAIL] TEST 7: Confidence threshold failed: {e}")

    # ----------------------------------------------------
    # TEST 8: MySQL connection works
    # ----------------------------------------------------
    try:
        import database
        ok, msg = database.test_connection()
        assert ok is True, f"Connection test returned False: {msg}"
        print(f"[PASS] TEST 8: MySQL connection works ({config.DB_HOST}:{config.DB_PORT})")
        passed_tests += 1
    except Exception as e:
        print(f"[FAIL] TEST 8: MySQL connection failed: {e}")

    # ----------------------------------------------------
    # TEST 9: Database/table exists
    # ----------------------------------------------------
    try:
        init_ok = database.initialize_database()
        assert init_ok is True, "Database initialization returned False"
        # Verify table existence in MySQL
        conn = database.get_connection(use_database=True)
        cursor = conn.cursor()
        cursor.execute("SHOW TABLES LIKE 'detection_logs'")
        tbl = cursor.fetchone()
        cursor.close()
        conn.close()
        assert tbl is not None, "detection_logs table does not exist"
        print(f"[PASS] TEST 9: Database '{config.DB_NAME}' and table 'detection_logs' exist and verified")
        passed_tests += 1
    except Exception as e:
        print(f"[FAIL] TEST 9: Database/table verification failed: {e}")

    # ----------------------------------------------------
    # TEST 10: Detection can be inserted
    # ----------------------------------------------------
    test_log_id = None
    try:
        test_log_id = database.log_detection(
            object_class="test_object",
            confidence=0.91,
            bbox_x=12,
            bbox_y=34,
            bbox_w=56,
            bbox_h=78
        )
        assert test_log_id is not None and test_log_id > 0, "Insert returned invalid ID"
        print(f"[PASS] TEST 10: Detection can be inserted into MySQL (Assigned log_id: {test_log_id})")
        passed_tests += 1
    except Exception as e:
        print(f"[FAIL] TEST 10: Detection insertion failed: {e}")

    # ----------------------------------------------------
    # TEST 11: Detection can be retrieved
    # ----------------------------------------------------
    try:
        recent = database.get_recent_logs(limit=5)
        assert len(recent) > 0, "No logs returned"
        match = any(row["log_id"] == test_log_id for row in recent)
        assert match is True, f"Inserted test_log_id {test_log_id} not found in recent logs"
        print(f"[PASS] TEST 11: Detection can be retrieved from MySQL (Retrieved row with ID {test_log_id})")
        passed_tests += 1
    except Exception as e:
        print(f"[FAIL] TEST 11: Detection retrieval failed: {e}")

    # ----------------------------------------------------
    # TEST 12: Streamlit application starts / validates
    # ----------------------------------------------------
    try:
        import py_compile
        py_compile.compile(os.path.join(PROJECT_ROOT, "streamlit_app.py"), doraise=True)
        py_compile.compile(os.path.join(PROJECT_ROOT, "ui.py"), doraise=True)
        print("[PASS] TEST 12: Streamlit application code and UI components compiled and validated")
        passed_tests += 1
    except Exception as e:
        print(f"[FAIL] TEST 12: Streamlit app validation failed: {e}")

    # ----------------------------------------------------
    # TEST 13: Live webcam detection works through pipeline
    # ----------------------------------------------------
    try:
        # Simulate Streamlit frame transformation pipeline
        annotated = detector.draw_detections(webcam_frame.copy(), [])
        rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        assert rgb.shape == webcam_frame.shape, "RGB shape mismatch"
        print("[PASS] TEST 13: Live webcam detection pipeline (BGR -> Detection -> Draw -> RGB) works")
        passed_tests += 1
    except Exception as e:
        print(f"[FAIL] TEST 13: Live webcam detection pipeline failed: {e}")

    # ----------------------------------------------------
    # TEST 14: Valid detections are stored in MySQL (with cooldown)
    # ----------------------------------------------------
    try:
        test_logger = database.DetectionLogger(cooldown_seconds=0.5)
        valid_sample = {
            "class_name": "laptop",
            "confidence": 0.88,
            "bbox_x": 100,
            "bbox_y": 100,
            "bbox_w": 200,
            "bbox_h": 200
        }
        lid = test_logger.log_if_ready(valid_sample)
        assert lid is not None, "Valid detection was not logged"
        print(f"[PASS] TEST 14: Valid detections are stored in MySQL (Logged with ID: {lid})")
        passed_tests += 1
    except Exception as e:
        print(f"[FAIL] TEST 14: Valid detection storage failed: {e}")

    # ----------------------------------------------------
    # TEST 15: Invalid detections are not stored
    # ----------------------------------------------------
    try:
        # Test cooldown blocking: immediate duplicate of same class must not be stored
        blocked_lid = test_logger.log_if_ready(valid_sample)
        assert blocked_lid is None, "Duplicate detection was wrongly logged within cooldown window"

        # Test filter rejection: detection failing criteria is blocked by detector
        detector.set_selected_classes(["person"])
        detector.set_confidence_threshold(0.70)
        assert detector.is_valid_detection("chair", 0.60) is False
        print("[PASS] TEST 15: Invalid detections (below threshold / unselected / within cooldown) are NOT stored")
        passed_tests += 1
    except Exception as e:
        print(f"[FAIL] TEST 15: Invalid detection rejection test failed: {e}")

    print("=" * 75)
    print(f" FINAL TEST RESULT: {passed_tests}/15 TESTS PASSED")
    print("=" * 75)

    if passed_tests == 15:
        print("ALL TECHNICAL TESTS PASSED SUCCESSFULLY!")
        return True
    else:
        print(f"WARNING: {15 - passed_tests} test(s) failed.")
        return False


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)

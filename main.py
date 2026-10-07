"""
main.py
-------
Main entry point for the Real-Time Object Detection & Logging Platform.
Orchestrates webcam capture, YOLO detection, filtering, and MySQL logging.
"""

import sys
import argparse
import config
from webcam import run_webcam_stream
from detector import ObjectDetector
import database
from database import DetectionLogger


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="CEP Assignment 5: Real-Time Object Detection & Logging Platform"
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=config.WEBCAM_INDEX,
        help=f"Index of the camera device to use (default: {config.WEBCAM_INDEX})"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=config.DEFAULT_CONFIDENCE_THRESHOLD,
        help=f"Confidence threshold between 0.0 and 1.0 (default: {config.DEFAULT_CONFIDENCE_THRESHOLD})"
    )
    parser.add_argument(
        "--objects",
        type=str,
        default=",".join(config.DEFAULT_SELECTED_OBJECTS),
        help=f"Comma-separated list of object classes to detect/log (default: {','.join(config.DEFAULT_SELECTED_OBJECTS)})"
    )
    parser.add_argument(
        "--no-db",
        action="store_true",
        help="Disable MySQL database logging (detection-only mode)"
    )
    parser.add_argument(
        "--test-frames",
        type=int,
        default=None,
        help="Optional: Run for a fixed number of frames and auto-exit (useful for testing)"
    )
    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_arguments()

    # Parse selected classes list
    selected_classes = [obj.strip().lower() for obj in args.objects.split(",") if obj.strip()]

    print("=" * 72)
    print(" CEP Assignment 5: Real-Time Object Detection & Logging Platform")
    print(" Full Pipeline: Webcam + YOLOv8 + Object Filter + MySQL Logging")
    print("=" * 72)
    print(f"[CONFIG] Target Camera Index  : {args.camera_index}")
    print(f"[CONFIG] Confidence Threshold : {args.conf * 100:.0f}% ({args.conf:.2f})")
    print(f"[CONFIG] Selected Objects     : {selected_classes}")
    print(f"[CONFIG] Model Weights        : {config.MODEL_NAME}")
    print(f"[CONFIG] Database Logging     : {'Disabled' if args.no_db else 'Enabled'}")
    if args.test_frames:
        print(f"[CONFIG] Auto-Stop Limit      : {args.test_frames} frames")
    print("=" * 72)

    # 1. Initialize MySQL database and detection logger (if enabled)
    logger = None
    if not args.no_db:
        try:
            database.initialize_database()
            logger = DetectionLogger(cooldown_seconds=config.DETECTION_COOLDOWN_SECONDS)
            print(f"[INFO] MySQL logging active (Cooldown: {config.DETECTION_COOLDOWN_SECONDS}s per object).")
        except Exception as e:
            print(f"[WARNING] MySQL unavailable ({e}). Continuing in detection-only mode.")

    try:
        # 2. Initialize detector with configurable threshold & class filter
        detector = ObjectDetector(
            model_name=config.MODEL_NAME,
            conf_threshold=args.conf,
            selected_classes=selected_classes
        )

        # 3. Launch the live detection and logging stream
        run_webcam_stream(
            camera_index=args.camera_index,
            detector=detector,
            logger=logger,
            max_frames=args.test_frames,
            window_name="CEP Assignment 5 - Object Detection & Logging"
        )
        print("[SUCCESS] Real-time detection session ended cleanly.")

    except RuntimeError as err:
        print(f"\n[ERROR] {err}")
        print("\n[TROUBLESHOOTING TIPS]")
        print("1. If using Windows, ensure Camera access is enabled in:")
        print("   Settings -> Privacy & Security -> Camera -> 'Let desktop apps access your camera'.")
        print("2. Ensure other apps (Teams, Zoom, Discord, Windows Camera) are completely closed.")
        print("3. If you have an external webcam, try running with: python main.py --camera-index 1")
        sys.exit(1)

    except KeyboardInterrupt:
        print("\n[INFO] Program interrupted by user (Ctrl+C). Exiting cleanly.")


if __name__ == "__main__":
    main()

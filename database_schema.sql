-- ===================================================================
-- CEP Assignment 5: Real-Time Object Detection & Logging Platform
-- Database Setup Script (MySQL / XAMPP)
-- ===================================================================

-- Step 1: Create the database if it doesn't already exist
CREATE DATABASE IF NOT EXISTS vision_platform;

-- Step 2: Switch to the database
USE vision_platform;

-- Step 3: Create the detection_logs table
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

-- Optional: Verify table schema
-- DESCRIBE detection_logs;

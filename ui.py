"""
ui.py
-----
UI styling, theme tokens, and component renderers for the Streamlit dashboard.
Provides clean aesthetics, metric cards, and table formatters.
"""

import streamlit as st
import pandas as pd


def inject_custom_css():
    """Injects modern, polished CSS for a professional dashboard look."""
    st.markdown(
        """
        <style>
            /* Main container padding */
            .block-container {
                padding-top: 1.8rem;
                padding-bottom: 2rem;
                padding-left: 2.5rem;
                padding-right: 2.5rem;
            }

            /* Header Title styling */
            .main-header {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                font-size: 2.2rem;
                font-weight: 700;
                background: linear-gradient(135deg, #00C9FF 0%, #92FE9D 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                margin-bottom: 0.2rem;
            }

            .sub-header {
                color: #A0AEC0;
                font-size: 0.95rem;
                margin-bottom: 1.5rem;
                font-weight: 400;
            }

            /* Card styling for metrics and feeds */
            .metric-card {
                background: rgba(30, 41, 59, 0.7);
                border: 1px solid rgba(255, 255, 255, 0.08);
                border-radius: 12px;
                padding: 16px 20px;
                margin-bottom: 12px;
                box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
            }

            .metric-title {
                font-size: 0.82rem;
                color: #94A3B8;
                text-transform: uppercase;
                letter-spacing: 0.05em;
                margin-bottom: 4px;
            }

            .metric-value {
                font-size: 1.8rem;
                font-weight: 700;
                color: #F8FAFC;
            }

            /* Status badge */
            .status-badge-ok {
                display: inline-block;
                padding: 4px 10px;
                border-radius: 9999px;
                font-size: 0.8rem;
                font-weight: 600;
                background-color: rgba(16, 185, 129, 0.2);
                color: #10B981;
                border: 1px solid rgba(16, 185, 129, 0.3);
            }

            .status-badge-err {
                display: inline-block;
                padding: 4px 10px;
                border-radius: 9999px;
                font-size: 0.8rem;
                font-weight: 600;
                background-color: rgba(239, 68, 68, 0.2);
                color: #EF4444;
                border: 1px solid rgba(239, 68, 68, 0.3);
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header():
    """Renders the top branding header."""
    st.markdown('<div class="main-header">Real-Time Object Detection & Logging Platform</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">CEP Assignment 5 • Real-time computer vision with Ultralytics YOLOv8, OpenCV, and MySQL logging</div>',
        unsafe_allow_html=True,
    )


def format_logs_dataframe(logs):
    """
    Formats database query logs into a clean, display-ready Pandas DataFrame.
    
    Parameters:
        logs (list[dict]): List of log records from database.get_recent_logs().
        
    Returns:
        pd.DataFrame: Formatted DataFrame.
    """
    if not logs:
        return pd.DataFrame(
            columns=["Log ID", "Timestamp", "Object Class", "Confidence", "Bounding Box (X, Y, W, H)"]
        )

    formatted_data = []
    for row in logs:
        conf_val = row.get("confidence", 0.0)
        formatted_data.append({
            "Log ID": row.get("log_id"),
            "Timestamp": str(row.get("timestamp")),
            "Object Class": row.get("object_class"),
            "Confidence": f"{conf_val * 100:.1f}%",
            "Bounding Box (X, Y, W, H)": f"({row.get('bbox_x')}, {row.get('bbox_y')}, {row.get('bbox_w')}, {row.get('bbox_h')})",
        })

    df = pd.DataFrame(formatted_data)
    return df

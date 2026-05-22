import sqlite3
from datetime import datetime
from pathlib import Path


DB_PATH = Path("skin_cancer_app.db")


def get_connection():
    return sqlite3.connect(DB_PATH)


def init_db():
    with get_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                prediction_score REAL NOT NULL,
                threshold REAL NOT NULL,
                result TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )


def save_prediction(filename, prediction_score, threshold, result):
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO predictions (
                filename,
                prediction_score,
                threshold,
                result,
                created_at
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                filename,
                float(prediction_score),
                float(threshold),
                result,
                datetime.utcnow().isoformat(timespec="seconds"),
            ),
        )


def get_recent_predictions(limit=10):
    with get_connection() as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT filename, prediction_score, threshold, result, created_at
            FROM predictions
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

    return [dict(row) for row in rows]

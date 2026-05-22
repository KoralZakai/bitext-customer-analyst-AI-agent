"""SQLite conversation checkpoints."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager

from langgraph.checkpoint.sqlite import SqliteSaver

from src.config import CHECKPOINT_DB_PATH


@contextmanager
def get_checkpointer():
    """
    Name: get_checkpointer
    Input: None (uses CHECKPOINT_DB_PATH from config)
    Output: context manager yielding SqliteSaver
    Purpose: Persist multi-turn chat state per --session thread id.
    """
    CHECKPOINT_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(CHECKPOINT_DB_PATH), check_same_thread=False)
    try:
        saver = SqliteSaver(conn)
        yield saver
    finally:
        conn.close()

"""Load configuration from environment (.env)."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


def _path(key: str, default: str) -> Path:
    """
    Name: _path
    Input: key — environment variable name; default — fallback path string
    Output: Path — absolute path (project-relative if env value is relative)
    Purpose: Resolve data and runtime paths from .env with sensible defaults.
    """
    raw = os.getenv(key, default)
    p = Path(raw)
    return p if p.is_absolute() else PROJECT_ROOT / p


NEBIUS_API_KEY: str = os.getenv("NEBIUS_API_KEY", "")
NEBIUS_BASE_URL: str = os.getenv(
    "NEBIUS_BASE_URL", "https://api.tokenfactory.nebius.com/v1/"
)
ROUTER_MODEL: str = os.getenv("ROUTER_MODEL", "google/gemma-3-27b-it")
AGENT_MODEL: str = os.getenv("AGENT_MODEL", "meta-llama/Llama-3.3-70B-Instruct")
MAX_ITERATIONS: int = int(os.getenv("MAX_ITERATIONS", "12"))

DATA_RAW_DIR: Path = _path("DATA_RAW_DIR", "data/raw")
DATA_PROCESSED_PATH: Path = _path("DATA_PROCESSED_PATH", "data/processed/bitext_clean.parquet")
DATA_AUGMENTED_PATH: Path = _path(
    "DATA_AUGMENTED_PATH", "data/processed/bitext_augmented.parquet"
)
DATA_METADATA_PATH: Path = _path("DATA_METADATA_PATH", "data/processed/metadata.json")
DATA_QUALITY_REPORT_PATH: Path = _path(
    "DATA_QUALITY_REPORT_PATH", "data/processed/quality_report.json"
)
CHECKPOINT_DB_PATH: Path = _path("CHECKPOINT_DB_PATH", "data/checkpoints.sqlite")
MIN_EXPECTED_ROWS: int = int(os.getenv("MIN_EXPECTED_ROWS", "20000"))

DATASET_HF_ID = "bitext/Bitext-customer-support-llm-chatbot-training-dataset"
DATASET_HF_CSV = "Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv"

REQUIRED_COLUMNS = ("instruction", "category", "intent", "response", "flags")

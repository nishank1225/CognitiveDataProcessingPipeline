"""
Central configuration module for the Cognitive Data Processing Pipeline.
Handles environment variables, data directory paths, processing thresholds,
and ML feature extraction parameters.
"""

import os
from pathlib import Path
from typing import Dict, Any

# Root workspace directory
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Fallback .env file loader if python-dotenv is not installed
def _load_env_file(env_path: Path) -> None:
    if not env_path.exists():
        return
    try:
        from dotenv import load_dotenv
        load_dotenv(env_path)
    except ImportError:
        # Simple manual fallback parser
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, val = line.split("=", 1)
                key = key.strip()
                val = val.strip().strip("'\"")
                if key not in os.environ:
                    os.environ[key] = val

# Load .env if present
_load_env_file(BASE_DIR / ".env")


class Config:
    """Global configuration settings container."""

    def __init__(self):
        # Base paths
        self.BASE_DIR: Path = BASE_DIR
        
        # Server & System Settings
        self.ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
        self.LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
        self.HOST: str = os.getenv("HOST", "0.0.0.0")
        self.PORT: int = int(os.getenv("PORT", "8000"))

        # Data Directory Settings
        raw_data_dir = os.getenv("DATA_DIR", "data")
        self.DATA_DIR: Path = self.BASE_DIR / raw_data_dir
        self.RAW_DATA_DIR: Path = self.BASE_DIR / os.getenv("RAW_DATA_DIR", "data/raw")
        self.PROCESSED_DATA_DIR: Path = self.BASE_DIR / os.getenv("PROCESSED_DATA_DIR", "data/processed")
        self.OUTPUT_DATA_DIR: Path = self.BASE_DIR / os.getenv("OUTPUT_DATA_DIR", "data/output")

        # Specific Data File Paths
        # Raw Data File Paths
        self.RAW_PRODUCTS_FILE: Path = self.RAW_DATA_DIR / "products_raw.json"
        self.RAW_STARTUPS_FILE: Path = self.RAW_DATA_DIR / "startups_raw.json"

        # Processed Data File Paths
        self.PROCESSED_PRODUCTS_FILE: Path = self.PROCESSED_DATA_DIR / "products_cleaned.json"
        self.PROCESSED_STARTUPS_FILE: Path = self.PROCESSED_DATA_DIR / "startups_cleaned.json"

        # Output / ML Feature Enriched File Paths
        self.OUTPUT_PRODUCTS_FILE: Path = self.OUTPUT_DATA_DIR / "products.json"
        self.OUTPUT_STARTUPS_FILE: Path = self.OUTPUT_DATA_DIR / "startups.json"
        self.OUTPUT_FEATURE_MATRIX_FILE: Path = self.OUTPUT_DATA_DIR / "feature_matrix.json"
        self.OUTPUT_UNIFIED_SUMMARY_FILE: Path = self.OUTPUT_DATA_DIR / "unified_summary.json"

        # Preprocessing & Data Cleaning Thresholds
        self.MIN_TEXT_LENGTH: int = 3
        self.STRIP_HTML: bool = True
        self.DEDUPLICATE_ENTRIES: bool = True
        self.DEFAULT_IMPUTE_RATING: float = 0.0

        # AI / ML Feature Extraction Settings
        self.TFIDF_MAX_FEATURES: int = 100
        self.SENTIMENT_POSITIVE_THRESHOLD: float = 0.1
        self.SENTIMENT_NEGATIVE_THRESHOLD: float = -0.1

        # Innovation & Growth Scoring Weights
        self.GROWTH_WEIGHTS: Dict[str, float] = {
            "funding": 0.4,
            "team_growth": 0.3,
            "sentiment": 0.2,
            "rating": 0.1
        }

    def ensure_directories(self) -> None:
        """Create necessary raw, processed, and output data directories if they do not exist."""
        for path in [self.DATA_DIR, self.RAW_DATA_DIR, self.PROCESSED_DATA_DIR, self.OUTPUT_DATA_DIR]:
            path.mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> Dict[str, Any]:
        """Return configuration settings as a dictionary for API/debugging inspection."""
        return {
            "environment": self.ENVIRONMENT,
            "log_level": self.LOG_LEVEL,
            "host": self.HOST,
            "port": self.PORT,
            "base_dir": str(self.BASE_DIR),
            "data_dir": str(self.DATA_DIR),
            "raw_data_dir": str(self.RAW_DATA_DIR),
            "processed_data_dir": str(self.PROCESSED_DATA_DIR),
            "output_data_dir": str(self.OUTPUT_DATA_DIR),
            "tfidf_max_features": self.TFIDF_MAX_FEATURES,
            "sentiment_positive_threshold": self.SENTIMENT_POSITIVE_THRESHOLD,
            "sentiment_negative_threshold": self.SENTIMENT_NEGATIVE_THRESHOLD,
        }


# Singleton config instance
config = Config()

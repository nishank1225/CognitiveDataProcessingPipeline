"""
Product Intelligence Collector module for Cognitive Data Processing Pipeline.
Ingests tech product specifications, user reviews, telemetry metrics, and market feedback.
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import requests

from src.utils.config import config
from src.utils.logger import logger


class ProductCollector:
    """
    Collector responsible for ingesting, retrieving, and serializing
    product intelligence datasets.
    """

    def __init__(self, raw_dir: Optional[Path] = None):
        self.raw_dir = raw_dir or config.RAW_DATA_DIR
        self.output_file = config.RAW_PRODUCTS_FILE

    def collect_sample_data(self) -> List[Dict[str, Any]]:
        """
        Generates realistic sample raw product dataset for testing and execution.
        Includes tech specs, customer reviews with raw HTML, ratings, and usage metrics.

        Returns:
            List[Dict[str, Any]]: List of raw product record dictionaries.
        """
        logger.info("Generating sample product intelligence data payload...")
        sample_products = [
            {
                "id": "prod-101",
                "name": "CognitiveAnalytics AI",
                "category": "AI & Machine Learning",
                "description": "<p>An advanced <b>real-time predictive analytics platform</b> for enterprise telemetry.</p>",
                "tech_specs": {
                    "architecture": "Distributed Microservices",
                    "cloud_provider": "AWS / Multi-cloud",
                    "api_protocol": "gRPC / REST",
                    "max_throughput_qps": 50000,
                    "ram_required_gb": 16,
                },
                "rating": 4.8,
                "reviews": [
                    "<p>Outstanding <b>throughput</b> and seamless model serving integration!</p>",
                    "Excellent SDK performance, cut our inference latency by 40%.",
                    "Documentation could be better, but the backend engine is top tier."
                ],
                "active_users": 14200,
                "monthly_active_users": 85000,
                "pricing_tier": "Enterprise",
                "release_date": "2023-04-15"
            },
            {
                "id": "prod-102",
                "name": "CloudVault Pro",
                "category": "Cloud Infrastructure",
                "description": "<div>Zero-trust <i>encrypted cloud storage solution</i> with automated backup pipelines.</div>",
                "tech_specs": {
                    "encryption": "AES-256-GCM",
                    "compliance": ["SOC2", "HIPAA", "GDPR"],
                    "storage_backend": "Distributed NVMe",
                    "api_protocol": "REST / S3-compatible"
                },
                "rating": 4.5,
                "reviews": [
                    "Rock-solid security features. High uptime across all regions.",
                    "The automatic key rotation feature saves us hours of devops overhead."
                ],
                "active_users": 38900,
                "monthly_active_users": 210000,
                "pricing_tier": "Business",
                "release_date": "2022-11-01"
            },
            {
                "id": "prod-103",
                "name": "DevFlow Studio",
                "category": "Developer Tools",
                "description": "<span>Cloud-native <b>CI/CD automation workspace</b> designed for microservice architectures.</span>",
                "tech_specs": {
                    "container_engine": "Docker / Containerd",
                    "orchestration": "Kubernetes Native",
                    "build_runners": "Parallel Distributed Workers"
                },
                "rating": 3.9,
                "reviews": [
                    "Great visual pipeline builder, but memory usage is quite high during parallel builds.",
                    "Needs better integration with custom plugin hooks."
                ],
                "active_users": 9500,
                "monthly_active_users": 43000,
                "pricing_tier": "Pro",
                "release_date": "2023-01-20"
            },
            {
                "id": "prod-104",
                "name": "CyberShield Guard",
                "category": "Cybersecurity",
                "description": "<p>Autonomous threat detection engine using <b>anomaly detection algorithms</b>.</p>",
                "tech_specs": {
                    "detection_model": "RandomForest + LSTM Anomaly Engine",
                    "latency_ms": 12,
                    "agent_footprint_mb": 45
                },
                "rating": None,  # Intentionally missing value for testing cleaner imputation
                "reviews": [
                    "Caught two credential stuffing attacks on day one.",
                    "False positive rate is impressively low."
                ],
                "active_users": 6200,
                "monthly_active_users": 31000,
                "pricing_tier": "Enterprise",
                "release_date": "2023-08-10"
            },
            {
                "id": "prod-105",
                "name": "NeuralData Mesh",
                "category": "Data Engineering",
                "description": "<div>Decentralized <b>data virtualization layer</b> for federated SQL querying.</div>",
                "tech_specs": {
                    "query_engine": "Apache Arrow / Rust",
                    "supported_connectors": ["Snowflake", "PostgreSQL", "BigQuery", "Mongo"],
                    "cache_layer": "Redis Cluster"
                },
                "rating": 4.7,
                "reviews": [
                    "Querying across 4 different databases in a single SQL statement feels like magic!",
                    "Blazing fast sub-second aggregations."
                ],
                "active_users": 18400,
                "monthly_active_users": 92000,
                "pricing_tier": "Enterprise",
                "release_date": "2023-06-05"
            },
            # Duplicate entry to test cleaner deduplication
            {
                "id": "prod-101",
                "name": "CognitiveAnalytics AI",
                "category": "AI & Machine Learning",
                "description": "<p>An advanced <b>real-time predictive analytics platform</b> for enterprise telemetry.</p>",
                "tech_specs": {
                    "architecture": "Distributed Microservices",
                    "cloud_provider": "AWS / Multi-cloud",
                    "api_protocol": "gRPC / REST",
                    "max_throughput_qps": 50000,
                    "ram_required_gb": 16,
                },
                "rating": 4.8,
                "reviews": [
                    "<p>Outstanding <b>throughput</b> and seamless model serving integration!</p>",
                    "Excellent SDK performance, cut our inference latency by 40%."
                ],
                "active_users": 14200,
                "monthly_active_users": 85000,
                "pricing_tier": "Enterprise",
                "release_date": "2023-04-15"
            }
        ]
        return sample_products

    def load_from_file(self, filepath: Union[str, Path]) -> List[Dict[str, Any]]:
        """
        Loads product records from a local JSON file.

        Args:
            filepath: Path to the JSON data file.

        Returns:
            List[Dict[str, Any]]: Parsed product items.
        """
        path = Path(filepath)
        if not path.exists():
            logger.error(f"Product data file not found: {path}")
            raise FileNotFoundError(f"File not found: {path}")

        logger.info(f"Loading raw product data from file: {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        if isinstance(data, dict) and "products" in data:
            data = data["products"]

        if not isinstance(data, list):
            raise ValueError(f"Expected list of records in {path}, got {type(data)}")

        logger.info(f"Successfully loaded {len(data)} product records from {path}")
        return data

    def fetch_from_api(self, url: str, timeout: int = 10) -> List[Dict[str, Any]]:
        """
        Fetches raw product data from an external API endpoint.

        Args:
            url: Target endpoint URL.
            timeout: Request timeout in seconds.

        Returns:
            List[Dict[str, Any]]: List of product data records.
        """
        logger.info(f"Fetching raw product data from endpoint: {url}")
        try:
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()
            payload = response.json()
            if isinstance(payload, dict) and "data" in payload:
                payload = payload["data"]
            logger.info(f"Successfully fetched {len(payload)} products from API.")
            return payload
        except Exception as e:
            logger.error(f"Failed to fetch product data from API ({url}): {e}")
            raise

    def collect_all(self, source_file: Optional[Union[str, Path]] = None) -> List[Dict[str, Any]]:
        """
        Primary entrypoint for collecting product intelligence data.
        Pulls from a source file if provided, otherwise falls back to sample generation.

        Args:
            source_file: Optional input JSON file path.

        Returns:
            List[Dict[str, Any]]: Raw product datasets.
        """
        if source_file and Path(source_file).exists():
            logger.info(f"Collecting product data from source file: {source_file}")
            return self.load_from_file(source_file)
        
        logger.info("No external source specified or file missing. Generating sample product dataset.")
        return self.collect_sample_data()

    def save_raw_data(self, data: List[Dict[str, Any]], output_path: Optional[Union[str, Path]] = None) -> Path:
        """
        Persists raw collected product records to the designated raw data JSON file.

        Args:
            data: Raw product dictionary records.
            output_path: Target filepath. Defaults to config.RAW_PRODUCTS_FILE.

        Returns:
            Path: The resolved file path where data was written.
        """
        target_path = Path(output_path) if output_path else self.output_file
        target_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving {len(data)} raw product records to {target_path}...")
        with open(target_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(f"Successfully written raw product dataset to {target_path}")
        return target_path


if __name__ == "__main__":
    collector = ProductCollector()
    raw_products = collector.collect_all()
    collector.save_raw_data(raw_products)

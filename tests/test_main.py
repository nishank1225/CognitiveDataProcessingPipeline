"""
Unit tests for main pipeline orchestrator (src/main.py).
"""

import json
import shutil
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from src.main import run_pipeline, main, print_pipeline_summary


class TestMainOrchestrator(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.output_dir = Path(self.temp_dir) / "output"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_run_pipeline_default(self):
        result = run_pipeline(
            output_dir=self.output_dir,
            export_csv=True,
            export_sqlite=True,
            verbose=False
        )

        self.assertIn("products", result)
        self.assertIn("startups", result)
        self.assertIn("unified_summary", result)
        self.assertIn("saved_paths", result)

        saved_paths = result["saved_paths"]
        self.assertIn("processed_products", saved_paths)
        self.assertIn("output_products", saved_paths)
        self.assertIn("unified_summary", saved_paths)
        self.assertIn("csv_products", saved_paths)
        self.assertIn("sqlite_products", saved_paths)

        # Verify files actually exist
        for key, path in saved_paths.items():
            self.assertTrue(Path(path).exists(), f"File for {key} does not exist: {path}")

    def test_run_pipeline_with_custom_files(self):
        # Create temporary custom input JSON files
        prod_input = Path(self.temp_dir) / "custom_products.json"
        start_input = Path(self.temp_dir) / "custom_startups.json"

        sample_prods = [
            {
                "id": "cp-1",
                "name": "Custom Product",
                "category": "Testing",
                "description": "<p>Test desc</p>",
                "rating": 4.5
            }
        ]
        sample_starts = [
            {
                "id": "cs-1",
                "name": "Custom Startup",
                "industry_vertical": "Testing",
                "total_funding_usd": 1000000.0,
                "team_growth_pct": 20.0
            }
        ]

        with open(prod_input, "w", encoding="utf-8") as f:
            json.dump(sample_prods, f)

        with open(start_input, "w", encoding="utf-8") as f:
            json.dump(sample_starts, f)

        result = run_pipeline(
            products_source=prod_input,
            startups_source=start_input,
            output_dir=self.output_dir,
            export_csv=False,
            export_sqlite=False,
            verbose=False
        )

        self.assertEqual(len(result["products"]["cleaned_records"]), 1)
        self.assertEqual(result["products"]["cleaned_records"][0]["name"], "Custom Product")
        self.assertEqual(len(result["startups"]["cleaned_records"]), 1)
        self.assertEqual(result["startups"]["cleaned_records"][0]["name"], "Custom Startup")

    def test_print_pipeline_summary(self):
        pipeline_result = {
            "products": {
                "summary": {"total_processed": 5, "duplicates_removed": 1}
            },
            "startups": {
                "summary": {"total_processed": 5, "duplicates_removed": 1}
            },
            "unified_summary": {
                "total_records_processed": 10,
                "average_growth_score": 75.5,
                "average_sentiment_score": 0.45,
                "total_startup_funding_usd": 50000000.0,
                "top_tech_entities": [{"tech": "python", "count": 5}],
                "sentiment_distribution": {"positive": 8, "neutral": 2, "negative": 0}
            },
            "saved_paths": {
                "output_products": self.output_dir / "products.json"
            }
        }
        # Call print_pipeline_summary to verify no exception occurs
        print_pipeline_summary(pipeline_result)

    @patch("sys.argv", ["main.py", "--quiet", "--no-csv", "--no-sqlite"])
    def test_main_cli(self):
        with patch("src.main.run_pipeline") as mock_run:
            mock_run.return_value = {}
            main()
            mock_run.assert_called_once_with(
                products_source=None,
                startups_source=None,
                export_csv=False,
                export_sqlite=False,
                output_dir=None,
                verbose=False
            )


if __name__ == "__main__":
    unittest.main()

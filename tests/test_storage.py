"""
Unit tests for OutputManager storage module.
"""

import os
import shutil
import tempfile
import unittest
from pathlib import Path

from src.storage.output_manager import OutputManager, CustomJSONEncoder


class TestOutputManager(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.raw_dir = Path(self.temp_dir) / "raw"
        self.processed_dir = Path(self.temp_dir) / "processed"
        self.output_dir = Path(self.temp_dir) / "output"
        self.db_path = self.output_dir / "test_pipeline.db"

        self.storage = OutputManager(
            raw_dir=self.raw_dir,
            processed_dir=self.processed_dir,
            output_dir=self.output_dir,
            db_path=self.db_path
        )

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_ensure_directories(self):
        self.assertTrue(self.raw_dir.exists())
        self.assertTrue(self.processed_dir.exists())
        self.assertTrue(self.output_dir.exists())

    def test_save_and_load_json(self):
        data = [{"id": "1", "name": "Item 1", "score": 9.5}]
        target_path = self.output_dir / "test_data.json"

        saved_path = self.storage.save_json(data, target_path)
        self.assertTrue(saved_path.exists())

        loaded_data = self.storage.load_json(target_path)
        self.assertEqual(len(loaded_data), 1)
        self.assertEqual(loaded_data[0]["name"], "Item 1")

    def test_load_json_nonexistent(self):
        result = self.storage.load_json(self.output_dir / "nonexistent.json", default=[])
        self.assertEqual(result, [])

    def test_save_and_load_csv(self):
        records = [
            {"id": "p1", "name": "Prod 1", "tags": ["ai", "cloud"], "rating": 4.8},
            {"id": "p2", "name": "Prod 2", "tags": ["db"], "rating": 4.2}
        ]
        csv_path = self.output_dir / "test_records.csv"

        saved_path = self.storage.save_csv(records, csv_path)
        self.assertTrue(saved_path.exists())

        loaded = self.storage.load_csv(csv_path)
        self.assertEqual(len(loaded), 2)
        self.assertEqual(loaded[0]["name"], "Prod 1")
        self.assertEqual(loaded[0]["tags"], ["ai", "cloud"])

    def test_save_and_load_sqlite(self):
        records = [
            {"id": "s1", "name": "Startup Alpha", "funding_usd": 1000000.0},
            {"id": "s2", "name": "Startup Beta", "funding_usd": 5000000.0}
        ]

        db_file = self.storage.save_sqlite(records, table_name="test_startups")
        self.assertTrue(db_file.exists())

        loaded = self.storage.load_sqlite("test_startups")
        self.assertEqual(len(loaded), 2)
        self.assertEqual(loaded[0]["name"], "Startup Alpha")

    def test_domain_tier_methods(self):
        raw_prods = [{"id": "r1", "name": "Raw Product"}]
        proc_prods = [{"id": "r1", "name": "Cleaned Product"}]
        out_prods = [{"id": "r1", "name": "Enriched Product", "growth_score": 8.5}]

        # Custom file paths in temp dir
        raw_file = self.raw_dir / "products_raw.json"
        proc_file = self.processed_dir / "products_cleaned.json"
        out_file = self.output_dir / "products.json"

        self.storage.save_raw_products(raw_prods, file_path=raw_file)
        self.storage.save_processed_products(proc_prods, file_path=proc_file)
        self.storage.save_enriched_products(out_prods, file_path=out_file)

        self.assertEqual(self.storage.load_raw_products(raw_file)[0]["name"], "Raw Product")
        self.assertEqual(self.storage.load_processed_products(proc_file)[0]["name"], "Cleaned Product")
        self.assertEqual(self.storage.load_enriched_products(out_file)[0]["name"], "Enriched Product")

    def test_save_pipeline_outputs(self):
        pipeline_result = {
            "products": {
                "cleaned_records": [{"id": "p1", "name": "Prod Clean"}],
                "enriched_records": [{"id": "p1", "name": "Prod Clean", "growth_score": 90.0, "extracted_entities": ["ai"]}],
                "summary": {"total_processed": 1}
            },
            "startups": {
                "cleaned_records": [{"id": "s1", "name": "Start Clean"}],
                "enriched_records": [{"id": "s1", "name": "Start Clean", "growth_score": 85.0, "extracted_entities": ["cloud"]}],
                "summary": {"total_processed": 1}
            },
            "unified_summary": {
                "total_records_processed": 2,
                "average_growth_score": 87.5
            }
        }

        saved_paths = self.storage.save_pipeline_outputs(pipeline_result, export_csv=True, export_sqlite=True)
        self.assertIn("processed_products", saved_paths)
        self.assertIn("output_products", saved_paths)
        self.assertIn("unified_summary", saved_paths)
        self.assertIn("csv_products", saved_paths)
        self.assertIn("sqlite_products", saved_paths)

        # Verify contents
        loaded_summary = self.storage.load_json(saved_paths["unified_summary"])
        self.assertEqual(loaded_summary["total_records_processed"], 2)


if __name__ == "__main__":
    unittest.main()

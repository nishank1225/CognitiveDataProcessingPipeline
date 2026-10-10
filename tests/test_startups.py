"""
Unit tests for StartupCollector module.
"""

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from src.startups.collector import StartupCollector
from src.utils.config import config


class TestStartupCollector(unittest.TestCase):

    def setUp(self):
        self.collector = StartupCollector()

    def test_collect_sample_data(self):
        data = self.collector.collect_sample_data()
        self.assertIsInstance(data, list)
        self.assertGreater(len(data), 0)
        first_item = data[0]
        self.assertIn("id", first_item)
        self.assertIn("name", first_item)
        self.assertIn("funding_stage", first_item)
        self.assertIn("tech_stack", first_item)

    def test_collect_all_default(self):
        data = self.collector.collect_all()
        self.assertIsInstance(data, list)
        self.assertGreater(len(data), 0)

    def test_save_raw_data(self):
        with TemporaryDirectory() as tmpdir:
            out_file = Path(tmpdir) / "test_startups_raw.json"
            sample_data = self.collector.collect_sample_data()
            saved_path = self.collector.save_raw_data(sample_data, output_path=out_file)
            self.assertTrue(saved_path.exists())
            
            with open(saved_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            self.assertEqual(len(loaded), len(sample_data))

    def test_load_from_file(self):
        with TemporaryDirectory() as tmpdir:
            out_file = Path(tmpdir) / "sample_startup_input.json"
            sample_data = [{"id": "start-999", "name": "Test Startup", "funding_stage": "Seed"}]
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(sample_data, f)

            loaded = self.collector.load_from_file(out_file)
            self.assertEqual(len(loaded), 1)
            self.assertEqual(loaded[0]["id"], "start-999")


if __name__ == "__main__":
    unittest.main()

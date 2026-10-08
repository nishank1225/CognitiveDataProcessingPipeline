"""
Unit tests for REST API server (src/server.py).
Tests API endpoints directly without requiring external HTTP libraries.
"""

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

from fastapi import HTTPException
from src.server import (
    app,
    root,
    health_check,
    get_configuration,
    get_unified_metrics,
    list_products,
    get_product_by_id,
    list_startups,
    get_startup_by_id,
    get_feature_matrix,
    trigger_pipeline_execution,
    PipelineRunRequest
)
from src.main import run_pipeline


class TestServerAPI(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.output_dir = Path(self.temp_dir) / "output"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Run pipeline once to populate storage output files for API testing
        run_pipeline(
            output_dir=self.output_dir,
            export_csv=True,
            export_sqlite=True,
            verbose=False
        )

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_root_endpoint(self):
        res = root()
        self.assertEqual(res["status"], "online")
        self.assertIn("endpoints", res)

    def test_health_endpoint(self):
        res = health_check()
        self.assertEqual(res.status, "healthy")
        self.assertIsNotNone(res.timestamp)

    def test_config_endpoint(self):
        res = get_configuration()
        self.assertIn("environment", res)
        self.assertIn("port", res)

    def test_metrics_endpoint(self):
        res = get_unified_metrics()
        self.assertIn("total_records_processed", res)
        self.assertIn("average_growth_score", res)

    def test_list_products_endpoint(self):
        res = list_products()
        self.assertIn("total", res)
        self.assertIn("products", res)
        self.assertGreater(res["total"], 0)

    def test_list_products_filtering(self):
        # Category filter
        res_cat = list_products(category="AI")
        self.assertTrue(all("AI" in p.get("category", "") for p in res_cat["products"]))

        # Search term filter
        res_search = list_products(search="CognitiveAnalytics")
        self.assertGreater(res_search["total"], 0)

        # Min rating filter
        res_rating = list_products(min_rating=4.0)
        self.assertTrue(all(p.get("rating", 0) >= 4.0 for p in res_rating["products"]))

    def test_get_product_by_id(self):
        res_list = list_products()
        first_id = res_list["products"][0]["id"]

        product = get_product_by_id(product_id=first_id)
        self.assertEqual(product["id"], first_id)

    def test_get_product_not_found(self):
        with self.assertRaises(HTTPException) as ctx:
            get_product_by_id(product_id="nonexistent-id-999")
        self.assertEqual(ctx.exception.status_code, 404)

    def test_list_startups_endpoint(self):
        res = list_startups()
        self.assertIn("total", res)
        self.assertIn("startups", res)
        self.assertGreater(res["total"], 0)

    def test_list_startups_filtering(self):
        res_ind = list_startups(industry="DeepTech")
        self.assertTrue(all("DeepTech" in s.get("industry_vertical", "") for s in res_ind["startups"]))

        res_funding = list_startups(min_funding=1000000.0)
        self.assertTrue(all(s.get("total_funding_usd", 0) >= 1000000.0 for s in res_funding["startups"]))

    def test_get_startup_by_id(self):
        res_list = list_startups()
        first_id = res_list["startups"][0]["id"]

        startup = get_startup_by_id(startup_id=first_id)
        self.assertEqual(startup["id"], first_id)

    def test_get_startup_not_found(self):
        with self.assertRaises(HTTPException) as ctx:
            get_startup_by_id(startup_id="nonexistent-startup-999")
        self.assertEqual(ctx.exception.status_code, 404)

    def test_feature_matrix_endpoint(self):
        res = get_feature_matrix()
        self.assertIn("feature_matrix", res)

    def test_trigger_pipeline_sync(self):
        req = PipelineRunRequest(
            export_csv=True,
            export_sqlite=False,
            async_run=False
        )
        bg_tasks = MagicMock()
        res = trigger_pipeline_execution(payload=req, background_tasks=bg_tasks)
        self.assertEqual(res["status"], "completed")
        self.assertIn("summary", res)

    def test_trigger_pipeline_async(self):
        req = PipelineRunRequest(
            export_csv=False,
            export_sqlite=False,
            async_run=True
        )
        bg_tasks = MagicMock()
        res = trigger_pipeline_execution(payload=req, background_tasks=bg_tasks)
        self.assertEqual(res["status"], "accepted")
        bg_tasks.add_task.assert_called_once()


if __name__ == "__main__":
    unittest.main()

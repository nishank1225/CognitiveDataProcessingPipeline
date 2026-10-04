"""
Unit tests for DataCleaner module.
"""

import unittest
from src.processing.cleaner import DataCleaner
from src.processing.feature_extractor import FeatureExtractor
from src.processing.processor import DataProcessor, CognitiveProcessor


class TestDataCleaner(unittest.TestCase):


    def setUp(self):
        self.cleaner = DataCleaner(default_rating=4.0)

    def test_strip_html(self):
        html_input = "<p>Hello <b>World</b>! &amp; welcome.</p>"
        expected = "Hello World! & welcome."
        self.assertEqual(self.cleaner.strip_html(html_input), expected)

    def test_strip_html_none(self):
        self.assertEqual(self.cleaner.strip_html(None), "")

    def test_deduplicate(self):
        records = [
            {"id": "1", "name": "Item 1"},
            {"id": "2", "name": "Item 2"},
            {"id": "1", "name": "Item 1 Duplicate"}
        ]
        deduped = self.cleaner.deduplicate(records, key="id")
        self.assertEqual(len(deduped), 2)
        self.assertEqual(deduped[0]["name"], "Item 1")

    def test_clean_product_record_imputation(self):
        raw_product = {
            "id": "prod-99",
            "name": "<span>Test Product</span>",
            "category": "Testing",
            "description": "<p>Raw HTML desc</p>",
            "rating": None,  # should be imputed
            "reviews": ["<p>Great product!</p>"]
        }
        cleaned = self.cleaner.clean_product_record(raw_product)
        self.assertEqual(cleaned["name"], "Test Product")
        self.assertEqual(cleaned["description"], "Raw HTML desc")
        self.assertEqual(cleaned["rating"], 4.0)
        self.assertEqual(cleaned["reviews"], ["Great product!"])

    def test_clean_startup_record_imputation(self):
        raw_startup = {
            "id": "start-99",
            "name": "<b>Startup X</b>",
            "customer_rating": None,
            "tech_stack": ["<b>Python</b>", "FastAPI"]
        }
        cleaned = self.cleaner.clean_startup_record(raw_startup)
        self.assertEqual(cleaned["name"], "Startup X")
        self.assertEqual(cleaned["customer_rating"], 4.0)
        self.assertEqual(cleaned["tech_stack"], ["Python", "FastAPI"])


class TestFeatureExtractor(unittest.TestCase):

    def setUp(self):
        self.extractor = FeatureExtractor()

    def test_analyze_sentiment_positive(self):
        res = self.extractor.analyze_sentiment("Outstanding platform with blazing fast performance!")
        self.assertGreater(res["score"], 0)
        self.assertEqual(res["label"], "positive")

    def test_analyze_sentiment_negative(self):
        res = self.extractor.analyze_sentiment("Slow response with high latency and bad bug issues.")
        self.assertLess(res["score"], 0)
        self.assertEqual(res["label"], "negative")

    def test_extract_entities(self):
        text = "Building Python and Docker microservices for Sequoia Capital."
        res = self.extractor.extract_entities(text)
        self.assertIn("python", res["tech_entities"])
        self.assertIn("docker", res["tech_entities"])

    def test_calculate_growth_score(self):
        record = {
            "total_funding_usd": 10000000.0,
            "team_growth_pct": 50.0,
            "summary_text": "Outstanding AI growth startup",
            "customer_rating": 4.8
        }
        score = self.extractor.calculate_startup_growth_score(record)
        self.assertIsInstance(score, float)
        self.assertGreater(score, 0.0)

    def test_process_product_batch(self):
        records = [
            {
                "id": "p1",
                "name": "AI Platform",
                "category": "AI",
                "description": "Enterprise AI search engine",
                "rating": 4.5,
                "reviews": ["Great tool!"]
            },
            {
                "id": "p2",
                "name": "Cloud Storage",
                "category": "Cloud",
                "description": "Secure encrypted storage solution",
                "rating": 4.0,
                "reviews": ["Solid security."]
            }
        ]
        enriched = self.extractor.process_product_batch(records)
        self.assertEqual(len(enriched), 2)
        self.assertIn("sentiment_score", enriched[0])
        self.assertIn("growth_score", enriched[0])
        self.assertIn("similar_item_ids", enriched[0])


class TestDataProcessor(unittest.TestCase):

    def setUp(self):
        self.processor = DataProcessor()

    def test_process_products(self):
        raw_products = [
            {"id": "p1", "name": "<b>Prod 1</b>", "category": "AI", "description": "Great AI tool"},
            {"id": "p2", "name": "Prod 2", "category": "Cloud", "description": "Cloud storage solution"}
        ]
        result = self.processor.process_products(raw_products)
        self.assertIn("cleaned_records", result)
        self.assertIn("enriched_records", result)
        self.assertIn("summary", result)
        self.assertEqual(result["summary"]["total_processed"], 2)

    def test_process_startups(self):
        raw_startups = [
            {"id": "s1", "name": "Startup Alpha", "industry_vertical": "FinTech", "total_funding_usd": 5000000},
            {"id": "s2", "name": "Startup Beta", "industry_vertical": "HealthTech", "total_funding_usd": 12000000}
        ]
        result = self.processor.process_startups(raw_startups)
        self.assertIn("cleaned_records", result)
        self.assertIn("enriched_records", result)
        self.assertIn("summary", result)
        self.assertEqual(result["summary"]["total_processed"], 2)

    def test_process_pipeline(self):
        raw_products = [{"id": "p1", "name": "Prod 1", "category": "AI"}]
        raw_startups = [{"id": "s1", "name": "Startup Alpha", "industry_vertical": "AI"}]
        result = self.processor.process_pipeline(raw_products, raw_startups)
        self.assertIn("products", result)
        self.assertIn("startups", result)
        self.assertIn("unified_summary", result)
        self.assertEqual(result["unified_summary"]["total_records_processed"], 2)

    def test_single_record_processing(self):
        single_prod = self.processor.process_single_product({"id": "p-single", "name": "Single Product"})
        self.assertEqual(single_prod["id"], "p-single")
        self.assertIn("growth_score", single_prod)

        single_start = self.processor.process_single_startup({"id": "s-single", "name": "Single Startup"})
        self.assertEqual(single_start["id"], "s-single")
        self.assertIn("growth_score", single_start)


if __name__ == "__main__":
    unittest.main()



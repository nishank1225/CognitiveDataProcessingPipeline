"""
Main processing orchestrator module for Cognitive Data Processing Pipeline.
Combines DataCleaner (sanitization, missing value imputation, deduplication)
and FeatureExtractor (sentiment scoring, entity extraction, TF-IDF vector embeddings, growth scoring)
into an integrated batch and record processing engine.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Union

from src.processing.cleaner import DataCleaner
from src.processing.feature_extractor import FeatureExtractor
from src.utils.config import config
from src.utils.logger import logger


class DataProcessor:
    """
    Main processing engine orchestrator that coordinates data sanitization,
    imputation, feature extraction, similarity vector computation, and metric aggregation
    for products and startups datasets.
    """

    def __init__(
        self,
        cleaner: Optional[DataCleaner] = None,
        feature_extractor: Optional[FeatureExtractor] = None
    ):
        """
        Initialize DataProcessor with custom or default DataCleaner and FeatureExtractor instances.

        Args:
            cleaner: Optional DataCleaner instance.
            feature_extractor: Optional FeatureExtractor instance.
        """
        self.cleaner = cleaner or DataCleaner(default_rating=config.DEFAULT_IMPUTE_RATING or 4.0)
        self.feature_extractor = feature_extractor or FeatureExtractor()

    def process_products(self, raw_products: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Orchestrates full processing pipeline for a batch of raw product records:
        1. Sanitizes, cleans text fields, and deduplicates input records.
        2. Extracts sentiment scores, entities, growth scores, and TF-IDF similarity vectors.
        3. Computes batch summary statistics.

        Args:
            raw_products: Raw list of product dictionaries.

        Returns:
            Dict[str, Any]: Contains 'cleaned_records', 'enriched_records', and 'summary' metrics.
        """
        logger.info(f"Starting product pipeline processing for {len(raw_products)} raw records...")

        # 1. Cleaning stage
        cleaned_records = self.cleaner.clean_products(raw_products)

        # 2. Feature extraction stage
        enriched_records = self.feature_extractor.process_product_batch(cleaned_records)

        # 3. Compute summary metrics
        summary = self._summarize_products(raw_products, cleaned_records, enriched_records)

        logger.info(
            f"Product pipeline completed: {summary['total_processed']} items enriched "
            f"(Avg Growth Score: {summary['avg_growth_score']}, Avg Sentiment: {summary['avg_sentiment']})."
        )

        return {
            "cleaned_records": cleaned_records,
            "enriched_records": enriched_records,
            "summary": summary
        }

    def process_startups(self, raw_startups: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Orchestrates full processing pipeline for a batch of raw startup records:
        1. Sanitizes, cleans text fields, and deduplicates input records.
        2. Extracts sentiment scores, tech entities, growth index, and TF-IDF similarity vectors.
        3. Computes batch summary statistics.

        Args:
            raw_startups: Raw list of startup dictionaries.

        Returns:
            Dict[str, Any]: Contains 'cleaned_records', 'enriched_records', and 'summary' metrics.
        """
        logger.info(f"Starting startup pipeline processing for {len(raw_startups)} raw records...")

        # 1. Cleaning stage
        cleaned_records = self.cleaner.clean_startups(raw_startups)

        # 2. Feature extraction stage
        enriched_records = self.feature_extractor.process_startup_batch(cleaned_records)

        # 3. Compute summary metrics
        summary = self._summarize_startups(raw_startups, cleaned_records, enriched_records)

        logger.info(
            f"Startup pipeline completed: {summary['total_processed']} items enriched "
            f"(Avg Growth Score: {summary['avg_growth_score']}, Total Funding: ${summary['total_funding_usd']:,.2f})."
        )

        return {
            "cleaned_records": cleaned_records,
            "enriched_records": enriched_records,
            "summary": summary
        }

    def process_pipeline(
        self,
        raw_products: List[Dict[str, Any]],
        raw_startups: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Executes end-to-end processing pipeline for both product and startup datasets concurrently.

        Args:
            raw_products: Raw product record list.
            raw_startups: Raw startup record list.

        Returns:
            Dict[str, Any]: Unified pipeline payload with cleaned records, enriched records,
                           individual summaries, and unified cognitive insights summary.
        """
        start_time = datetime.now()
        logger.info("Executing unified Cognitive Data Processing Pipeline...")

        products_result = self.process_products(raw_products)
        startups_result = self.process_startups(raw_startups)

        unified_summary = self.generate_unified_summary(
            products_result["enriched_records"],
            startups_result["enriched_records"]
        )

        execution_duration = (datetime.now() - start_time).total_seconds()
        unified_summary["execution_duration_seconds"] = round(execution_duration, 4)
        unified_summary["timestamp"] = datetime.now().isoformat()

        logger.info(f"Unified processing pipeline finished in {execution_duration:.3f} seconds.")

        return {
            "products": products_result,
            "startups": startups_result,
            "unified_summary": unified_summary
        }

    def process_single_product(self, raw_record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes a single raw product record through cleaning and feature extraction.

        Args:
            raw_record: Raw product record dictionary.

        Returns:
            Dict[str, Any]: Enriched product dictionary.
        """
        cleaned = self.cleaner.clean_product_record(raw_record)
        enriched = self.feature_extractor.extract_product_features(cleaned)
        enriched["similar_item_ids"] = []
        return enriched

    def process_single_startup(self, raw_record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes a single raw startup record through cleaning and feature extraction.

        Args:
            raw_record: Raw startup profile dictionary.

        Returns:
            Dict[str, Any]: Enriched startup profile dictionary.
        """
        cleaned = self.cleaner.clean_startup_record(raw_record)
        enriched = self.feature_extractor.extract_startup_features(cleaned)
        enriched["similar_item_ids"] = []
        return enriched

    def process_files(
        self,
        raw_products_path: Optional[Union[str, Path]] = None,
        raw_startups_path: Optional[Union[str, Path]] = None
    ) -> Dict[str, Any]:
        """
        Loads raw JSON datasets from disk, executes full processing pipeline,
        and returns enriched results.

        Args:
            raw_products_path: File path to raw products JSON. Defaults to config.RAW_PRODUCTS_FILE.
            raw_startups_path: File path to raw startups JSON. Defaults to config.RAW_STARTUPS_FILE.

        Returns:
            Dict[str, Any]: Unified pipeline payload.
        """
        prod_path = Path(raw_products_path or config.RAW_PRODUCTS_FILE)
        start_path = Path(raw_startups_path or config.RAW_STARTUPS_FILE)

        raw_products = []
        if prod_path.exists():
            with open(prod_path, "r", encoding="utf-8") as f:
                raw_products = json.load(f)
            logger.info(f"Loaded {len(raw_products)} raw product records from {prod_path}")
        else:
            logger.warning(f"Raw products file not found at {prod_path}")

        raw_startups = []
        if start_path.exists():
            with open(start_path, "r", encoding="utf-8") as f:
                raw_startups = json.load(f)
            logger.info(f"Loaded {len(raw_startups)} raw startup records from {start_path}")
        else:
            logger.warning(f"Raw startups file not found at {start_path}")

        return self.process_pipeline(raw_products, raw_startups)

    def generate_unified_summary(
        self,
        enriched_products: List[Dict[str, Any]],
        enriched_startups: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Generates global cognitive intelligence metrics across both product and startup domains.

        Args:
            enriched_products: List of enriched product records.
            enriched_startups: List of enriched startup records.

        Returns:
            Dict[str, Any]: Comprehensive unified insights summary.
        """
        total_items = len(enriched_products) + len(enriched_startups)

        # Average metrics
        prod_growth_scores = [p.get("growth_score", 0.0) for p in enriched_products]
        start_growth_scores = [s.get("growth_score", 0.0) for s in enriched_startups]
        all_growth_scores = prod_growth_scores + start_growth_scores
        avg_growth = round(float(sum(all_growth_scores) / len(all_growth_scores)), 2) if all_growth_scores else 0.0

        prod_sentiments = [p.get("sentiment_score", 0.0) for p in enriched_products]
        start_sentiments = [s.get("sentiment_score", 0.0) for s in enriched_startups]
        all_sentiments = prod_sentiments + start_sentiments
        avg_sentiment = round(float(sum(all_sentiments) / len(all_sentiments)), 3) if all_sentiments else 0.0

        # Sentiment breakdown
        positive_count = sum(1 for s in all_sentiments if s >= config.SENTIMENT_POSITIVE_THRESHOLD)
        negative_count = sum(1 for s in all_sentiments if s <= config.SENTIMENT_NEGATIVE_THRESHOLD)
        neutral_count = len(all_sentiments) - (positive_count + negative_count)

        # Aggregate tech stack entities & keywords
        tech_entities: Dict[str, int] = {}
        for item in enriched_products + enriched_startups:
            entities = item.get("extracted_entities", [])
            for entity in entities:
                tech_entities[entity] = tech_entities.get(entity, 0) + 1

        sorted_tech = sorted(tech_entities.items(), key=lambda x: x[1], reverse=True)[:10]

        total_funding = sum(float(s.get("total_funding_usd", 0.0)) for s in enriched_startups)

        return {
            "total_records_processed": total_items,
            "total_products": len(enriched_products),
            "total_startups": len(enriched_startups),
            "average_growth_score": avg_growth,
            "average_sentiment_score": avg_sentiment,
            "sentiment_distribution": {
                "positive": positive_count,
                "neutral": neutral_count,
                "negative": negative_count
            },
            "top_tech_entities": [{"tech": t[0], "count": t[1]} for t in sorted_tech],
            "total_startup_funding_usd": round(total_funding, 2)
        }

    def _summarize_products(
        self,
        raw_products: List[Dict[str, Any]],
        cleaned_records: List[Dict[str, Any]],
        enriched_records: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calculates batch statistical summary for products."""
        total_input = len(raw_products)
        total_processed = len(enriched_records)
        duplicates_removed = total_input - len(cleaned_records)

        categories: Dict[str, int] = {}
        for rec in enriched_records:
            cat = rec.get("category", "Uncategorized")
            categories[cat] = categories.get(cat, 0) + 1

        growth_scores = [r.get("growth_score", 0.0) for r in enriched_records]
        avg_growth = round(float(sum(growth_scores) / len(growth_scores)), 2) if growth_scores else 0.0

        sentiments = [r.get("sentiment_score", 0.0) for r in enriched_records]
        avg_sentiment = round(float(sum(sentiments) / len(sentiments)), 3) if sentiments else 0.0

        ratings = [r.get("rating", 0.0) for r in enriched_records]
        avg_rating = round(float(sum(ratings) / len(ratings)), 2) if ratings else 0.0

        return {
            "total_input": total_input,
            "total_processed": total_processed,
            "duplicates_removed": max(0, duplicates_removed),
            "avg_growth_score": avg_growth,
            "avg_sentiment": avg_sentiment,
            "avg_rating": avg_rating,
            "category_counts": categories
        }

    def _summarize_startups(
        self,
        raw_startups: List[Dict[str, Any]],
        cleaned_records: List[Dict[str, Any]],
        enriched_records: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calculates batch statistical summary for startups."""
        total_input = len(raw_startups)
        total_processed = len(enriched_records)
        duplicates_removed = total_input - len(cleaned_records)

        industries: Dict[str, int] = {}
        for rec in enriched_records:
            ind = rec.get("industry_vertical", "General Tech")
            industries[ind] = industries.get(ind, 0) + 1

        funding_stages: Dict[str, int] = {}
        for rec in enriched_records:
            stage = rec.get("funding_stage", "Unfunded")
            funding_stages[stage] = funding_stages.get(stage, 0) + 1

        total_funding = sum(float(r.get("total_funding_usd", 0.0)) for r in enriched_records)
        avg_funding = round(total_funding / len(enriched_records), 2) if enriched_records else 0.0

        growth_scores = [r.get("growth_score", 0.0) for r in enriched_records]
        avg_growth = round(float(sum(growth_scores) / len(growth_scores)), 2) if growth_scores else 0.0

        sentiments = [r.get("sentiment_score", 0.0) for r in enriched_records]
        avg_sentiment = round(float(sum(sentiments) / len(sentiments)), 3) if sentiments else 0.0

        return {
            "total_input": total_input,
            "total_processed": total_processed,
            "duplicates_removed": max(0, duplicates_removed),
            "total_funding_usd": round(total_funding, 2),
            "avg_funding_usd": avg_funding,
            "avg_growth_score": avg_growth,
            "avg_sentiment": avg_sentiment,
            "industry_counts": industries,
            "funding_stage_counts": funding_stages
        }


# Alias for backward compatibility / domain semantics
CognitiveProcessor = DataProcessor


if __name__ == "__main__":
    from src.products.collector import ProductCollector
    from src.startups.collector import StartupCollector

    processor = DataProcessor()

    prod_collector = ProductCollector()
    start_collector = StartupCollector()

    raw_prods = prod_collector.collect_sample_data()
    raw_starts = start_collector.collect_sample_data()

    pipeline_result = processor.process_pipeline(raw_prods, raw_starts)
    
    print("=== Pipeline Execution Completed Successfully ===")
    print(f"Products Processed: {pipeline_result['products']['summary']['total_processed']}")
    print(f"Startups Processed: {pipeline_result['startups']['summary']['total_processed']}")
    print(f"Execution Duration: {pipeline_result['unified_summary']['execution_duration_seconds']} sec")
    print("\nUnified Insights Summary:")
    print(json.dumps(pipeline_result['unified_summary'], indent=2))

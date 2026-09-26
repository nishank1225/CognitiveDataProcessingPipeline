"""
Main Entry Point and Pipeline Orchestrator for Cognitive Data Processing Pipeline.

Architecture / Data Flow:
    Products Collector  ──┐
                          ↓
    Startups Collector  ──┤
                          ↓
                       Cleaner
                          ↓
                  Feature Extractor
                          ↓
                      Processor
                          ↓
                  Output Manager
                          ↓
                    Final Output
"""

import sys
import argparse
import json
from pathlib import Path
from typing import Dict, Any, Optional, Union

# Ensure workspace root is in sys.path when script is executed directly
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.products.collector import ProductCollector
from src.startups.collector import StartupCollector
from src.processing.cleaner import DataCleaner
from src.processing.feature_extractor import FeatureExtractor
from src.processing.processor import DataProcessor, CognitiveProcessor
from src.storage.output_manager import OutputManager, StorageManager
from src.utils.config import config
from src.utils.logger import logger


def run_pipeline(
    products_source: Optional[Union[str, Path]] = None,
    startups_source: Optional[Union[str, Path]] = None,
    export_csv: bool = True,
    export_sqlite: bool = True,
    output_dir: Optional[Union[str, Path]] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Executes the complete Cognitive Data Processing Pipeline end-to-end.

    Args:
        products_source: Optional path to raw products input JSON file.
        startups_source: Optional path to raw startups input JSON file.
        export_csv: Whether to export datasets as CSV files.
        export_sqlite: Whether to persist records into SQLite database.
        output_dir: Optional path to custom output directory.
        verbose: Whether to log detailed execution information.

    Returns:
        Dict[str, Any]: Pipeline execution result containing raw, cleaned, enriched records,
                       summaries, and saved file paths.
    """
    logger.info(" Starting Cognitive Data Processing Pipeline Orchestration")
    config.ensure_directories()

    # Instantiate storage manager if custom output dir is provided
    output_mgr = OutputManager(output_dir=output_dir) if output_dir else OutputManager()

    # -------------------------------------------------------------------------
    # Step 1: Products Collector
    # -------------------------------------------------------------------------
    logger.info("[Step 1/6] Ingesting products intelligence data via ProductCollector...")
    prod_collector = ProductCollector()
    raw_products = prod_collector.collect_all(source_file=products_source)
    output_mgr.save_raw_products(raw_products)
    logger.info(f"Products Collector completed: Ingested {len(raw_products)} raw product records.")

    # -------------------------------------------------------------------------
    # Step 2: Startups Collector
    # -------------------------------------------------------------------------
    logger.info("[Step 2/6] Ingesting startups intelligence data via StartupCollector...")
    start_collector = StartupCollector()
    raw_startups = start_collector.collect_all(source_file=startups_source)
    output_mgr.save_raw_startups(raw_startups)
    logger.info(f"Startups Collector completed: Ingested {len(raw_startups)} raw startup records.")

    # -------------------------------------------------------------------------
    # Step 3: Cleaner
    # -------------------------------------------------------------------------
    logger.info("[Step 3/6] Cleaning, sanitizing, and deduplicating data via DataCleaner...")
    cleaner = DataCleaner(default_rating=config.DEFAULT_IMPUTE_RATING or 4.0)
    cleaned_products = cleaner.clean_products(raw_products)
    cleaned_startups = cleaner.clean_startups(raw_startups)
    logger.info(
        f"Cleaner stage completed: Products ({len(raw_products)} -> {len(cleaned_products)}), "
        f"Startups ({len(raw_startups)} -> {len(cleaned_startups)})."
    )

    # -------------------------------------------------------------------------
    # Step 4: Feature Extractor
    # -------------------------------------------------------------------------
    logger.info("[Step 4/6] Extracting AI/ML features & similarity vectors via FeatureExtractor...")
    feature_extractor = FeatureExtractor(max_tfidf_features=config.TFIDF_MAX_FEATURES)
    enriched_products = feature_extractor.process_product_batch(cleaned_products)
    enriched_startups = feature_extractor.process_startup_batch(cleaned_startups)
    logger.info(
        f"Feature Extractor completed: Enriched {len(enriched_products)} products "
        f"and {len(enriched_startups)} startups with sentiment, entities, and growth index."
    )

    # -------------------------------------------------------------------------
    # Step 5: Processor
    # -------------------------------------------------------------------------
    logger.info("[Step 5/6] Aggregating cognitive metrics via DataProcessor...")
    processor = DataProcessor(cleaner=cleaner, feature_extractor=feature_extractor)
    
    prod_summary = processor._summarize_products(raw_products, cleaned_products, enriched_products)
    start_summary = processor._summarize_startups(raw_startups, cleaned_startups, enriched_startups)
    unified_summary = processor.generate_unified_summary(enriched_products, enriched_startups)

    pipeline_result = {
        "products": {
            "raw_records": raw_products,
            "cleaned_records": cleaned_products,
            "enriched_records": enriched_products,
            "summary": prod_summary
        },
        "startups": {
            "raw_records": raw_startups,
            "cleaned_records": cleaned_startups,
            "enriched_records": enriched_startups,
            "summary": start_summary
        },
        "unified_summary": unified_summary
    }

    # -------------------------------------------------------------------------
    # Step 6: Output Manager
    # -------------------------------------------------------------------------
    logger.info("[Step 6/6] Persisting pipeline outputs across storage tiers via OutputManager...")
    saved_paths = output_mgr.save_pipeline_outputs(
        pipeline_result,
        export_csv=export_csv,
        export_sqlite=export_sqlite
    )
    pipeline_result["saved_paths"] = saved_paths

    # -------------------------------------------------------------------------
    # Final Output Display
    # -------------------------------------------------------------------------
    if verbose:
        print_pipeline_summary(pipeline_result)

    logger.info(" Pipeline execution completed successfully!")
    return pipeline_result


def print_pipeline_summary(pipeline_result: Dict[str, Any]) -> None:
    """Prints a formatted summary banner of the pipeline results to stdout."""
    summary = pipeline_result.get("unified_summary", {})
    prod_summary = pipeline_result.get("products", {}).get("summary", {})
    start_summary = pipeline_result.get("startups", {}).get("summary", {})
    saved_paths = pipeline_result.get("saved_paths", {})

    print("\n" + "=" * 70)
    print("      COGNITIVE DATA PROCESSING PIPELINE - EXECUTION SUMMARY      ")
    print("=" * 70)
    print(f" Total Records Processed : {summary.get('total_records_processed', 0)}")
    print(f" Products Processed      : {prod_summary.get('total_processed', 0)} (Duplicates Removed: {prod_summary.get('duplicates_removed', 0)})")
    print(f" Startups Processed      : {start_summary.get('total_processed', 0)} (Duplicates Removed: {start_summary.get('duplicates_removed', 0)})")
    print("-" * 70)
    print(f" Average Growth Score    : {summary.get('average_growth_score', 0.0)} / 100")
    print(f" Average Sentiment Score : {summary.get('average_sentiment_score', 0.0)}")
    print(f" Total Startup Funding   : ${summary.get('total_startup_funding_usd', 0.0):,.2f}")
    print("-" * 70)
    
    top_tech = summary.get("top_tech_entities", [])
    if top_tech:
        tech_list_str = ", ".join([f"{item['tech']} ({item['count']})" for item in top_tech[:5]])
        print(f" Top Tech Stack Entities : {tech_list_str}")

    sentiment_dist = summary.get("sentiment_distribution", {})
    print(f" Sentiment Breakdown     : Positive={sentiment_dist.get('positive', 0)}, Neutral={sentiment_dist.get('neutral', 0)}, Negative={sentiment_dist.get('negative', 0)}")

    print("-" * 70)
    print(" Saved Storage Artifacts:")
    for key, path in saved_paths.items():
        print(f"   * {key:<20}: {path}")
    print("=" * 70 + "\n")


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Cognitive Data Processing Pipeline Orchestrator"
    )
    parser.add_argument(
        "-p", "--products-file",
        type=str,
        default=None,
        help="Path to input products JSON file (optional)"
    )
    parser.add_argument(
        "-s", "--startups-file",
        type=str,
        default=None,
        help="Path to input startups JSON file (optional)"
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=str,
        default=None,
        help="Path to custom output directory (optional)"
    )
    parser.add_argument(
        "--no-csv",
        action="store_true",
        help="Disable CSV dataset exports"
    )
    parser.add_argument(
        "--no-sqlite",
        action="store_true",
        help="Disable SQLite database exports"
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress summary output banner"
    )
    return parser.parse_args()


def main() -> None:
    """Main CLI entry point."""
    args = parse_args()
    try:
        run_pipeline(
            products_source=args.products_file,
            startups_source=args.startups_file,
            export_csv=not args.no_csv,
            export_sqlite=not args.no_sqlite,
            output_dir=args.output_dir,
            verbose=not args.quiet
        )
    except Exception as e:
        logger.critical(f"Pipeline execution failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

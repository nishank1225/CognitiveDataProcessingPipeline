"""
Data Cleaner and Sanitizer module for Cognitive Data Processing Pipeline.
Handles data validation, missing value imputation, HTML stripping, text normalization, and deduplication.
"""

import html
import re
from typing import List, Dict, Any, Optional, Union
from src.utils.logger import logger


class DataCleaner:
    """
    Data cleaner utility responsible for sanitizing, imputing missing fields,
    stripping HTML tags, and deduplicating product and startup intelligence records.
    """

    def __init__(self, default_rating: float = 4.0):
        """
        Initialize DataCleaner.

        Args:
            default_rating: Default rating value to use when imputing missing ratings.
        """
        self.default_rating = default_rating

    @staticmethod
    def strip_html(text: Optional[str]) -> str:
        """
        Strips HTML tags and unescapes HTML entities from text string.

        Args:
            text: Raw string containing potential HTML markup.

        Returns:
            str: Clean text stripped of HTML tags and normalized.
        """
        if not text or not isinstance(text, str):
            return ""
        
        # Remove HTML tags using regular expression
        clean_text = re.sub(r"<[^>]+>", "", text)
        # Unescape HTML entities (e.g. &amp; -> &)
        clean_text = html.unescape(clean_text)
        # Normalize excessive whitespace
        clean_text = re.sub(r"\s+", " ", clean_text).strip()
        return clean_text

    @staticmethod
    def clean_text_list(items: Optional[List[str]]) -> List[str]:
        """
        Strips HTML tags and cleans all strings in a list.

        Args:
            items: List of raw strings.

        Returns:
            List[str]: List of cleaned, non-empty text strings.
        """
        if not items or not isinstance(items, list):
            return []
        
        cleaned = []
        for item in items:
            if isinstance(item, str):
                cleaned_item = DataCleaner.strip_html(item)
                if cleaned_item:
                    cleaned.append(cleaned_item)
        return cleaned

    @staticmethod
    def deduplicate(records: List[Dict[str, Any]], key: str = "id") -> List[Dict[str, Any]]:
        """
        Deduplicates a list of dictionaries based on a unique identifier key.
        Preserves original order of appearance.

        Args:
            records: List of record dictionaries.
            key: Primary key field name to check uniqueness against.

        Returns:
            List[Dict[str, Any]]: Deduplicated list of records.
        """
        seen_keys = set()
        deduped = []
        for record in records:
            record_id = record.get(key)
            if record_id is None:
                # If key is missing, retain record
                deduped.append(record)
            elif record_id not in seen_keys:
                seen_keys.add(record_id)
                deduped.append(record)
        return deduped

    def clean_product_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sanitizes and imputes missing fields in a single product record.

        Args:
            record: Raw product dictionary.

        Returns:
            Dict[str, Any]: Cleaned product record dictionary.
        """
        cleaned = dict(record)

        # 1. Clean string fields and strip HTML
        cleaned["id"] = str(cleaned.get("id", "")).strip()
        cleaned["name"] = self.strip_html(cleaned.get("name", "Unknown Product"))
        cleaned["category"] = self.strip_html(cleaned.get("category", "Uncategorized"))
        cleaned["description"] = self.strip_html(cleaned.get("description", ""))

        # 2. Clean reviews list
        reviews = cleaned.get("reviews", [])
        cleaned["reviews"] = self.clean_text_list(reviews)

        # 3. Handle ratings imputation
        rating = cleaned.get("rating")
        if rating is None or not isinstance(rating, (int, float)):
            cleaned["rating"] = self.default_rating
        else:
            cleaned["rating"] = round(float(rating), 2)

        # 4. Handle numerical metric defaults
        cleaned["active_users"] = int(cleaned.get("active_users") or 0)
        cleaned["monthly_active_users"] = int(cleaned.get("monthly_active_users") or 0)
        cleaned["pricing_tier"] = str(cleaned.get("pricing_tier") or "Standard").strip()
        cleaned["release_date"] = str(cleaned.get("release_date") or "").strip()

        # 5. Ensure tech_specs dictionary is valid
        tech_specs = cleaned.get("tech_specs")
        if not isinstance(tech_specs, dict):
            cleaned["tech_specs"] = {}

        return cleaned

    def clean_startup_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sanitizes and imputes missing fields in a single startup profile record.

        Args:
            record: Raw startup dictionary.

        Returns:
            Dict[str, Any]: Cleaned startup record dictionary.
        """
        cleaned = dict(record)

        # 1. Clean string fields and strip HTML
        cleaned["id"] = str(cleaned.get("id", "")).strip()
        cleaned["name"] = self.strip_html(cleaned.get("name", "Unknown Startup"))
        cleaned["industry_vertical"] = self.strip_html(cleaned.get("industry_vertical", "General Tech"))
        cleaned["description"] = self.strip_html(cleaned.get("description", ""))
        cleaned["summary_text"] = self.strip_html(cleaned.get("summary_text", ""))
        cleaned["location"] = self.strip_html(cleaned.get("location", "Unknown Location"))
        cleaned["funding_stage"] = str(cleaned.get("funding_stage") or "Unfunded").strip()

        # 2. Clean tag & list fields
        cleaned["tech_stack"] = self.clean_text_list(cleaned.get("tech_stack", []))
        cleaned["domain_tags"] = self.clean_text_list(cleaned.get("domain_tags", []))
        cleaned["investors"] = self.clean_text_list(cleaned.get("investors", []))

        # 3. Handle ratings & numeric fields
        rating = cleaned.get("customer_rating")
        if rating is None or not isinstance(rating, (int, float)):
            cleaned["customer_rating"] = self.default_rating
        else:
            cleaned["customer_rating"] = round(float(rating), 2)

        cleaned["total_funding_usd"] = float(cleaned.get("total_funding_usd") or 0.0)
        cleaned["valuation_usd"] = float(cleaned.get("valuation_usd") or 0.0)
        cleaned["team_size"] = int(cleaned.get("team_size") or 0)
        cleaned["team_growth_pct"] = float(cleaned.get("team_growth_pct") or 0.0)
        cleaned["founded_year"] = int(cleaned.get("founded_year") or 2020)

        return cleaned

    def clean_products(self, products: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Batch cleans and deduplicates a list of product records.

        Args:
            products: Raw list of product records.

        Returns:
            List[Dict[str, Any]]: Cleaned and deduplicated product records.
        """
        logger.info(f"Cleaning batch of {len(products)} product records...")
        deduped = self.deduplicate(products, key="id")
        initial_count = len(products)
        deduped_count = len(deduped)
        
        cleaned_records = [self.clean_product_record(rec) for rec in deduped]
        logger.info(
            f"Product cleaning complete: {initial_count} raw -> "
            f"{deduped_count} after deduplication ({initial_count - deduped_count} duplicates removed)."
        )
        return cleaned_records

    def clean_startups(self, startups: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Batch cleans and deduplicates a list of startup records.

        Args:
            startups: Raw list of startup records.

        Returns:
            List[Dict[str, Any]]: Cleaned and deduplicated startup records.
        """
        logger.info(f"Cleaning batch of {len(startups)} startup records...")
        deduped = self.deduplicate(startups, key="id")
        initial_count = len(startups)
        deduped_count = len(deduped)

        cleaned_records = [self.clean_startup_record(rec) for rec in deduped]
        logger.info(
            f"Startup cleaning complete: {initial_count} raw -> "
            f"{deduped_count} after deduplication ({initial_count - deduped_count} duplicates removed)."
        )
        return cleaned_records


if __name__ == "__main__":
    from src.products.collector import ProductCollector
    from src.startups.collector import StartupCollector

    cleaner = DataCleaner()
    
    prod_coll = ProductCollector()
    raw_prods = prod_coll.collect_sample_data()
    cleaned_prods = cleaner.clean_products(raw_prods)
    print(f"Cleaned {len(cleaned_prods)} products successfully.")

    start_coll = StartupCollector()
    raw_starts = start_coll.collect_sample_data()
    cleaned_starts = cleaner.clean_startups(raw_starts)
    print(f"Cleaned {len(cleaned_starts)} startups successfully.")

"""
Storage and Output Management module for Cognitive Data Processing Pipeline.
Handles persisting, loading, exporting, and managing raw, processed, and ML feature enriched datasets
across JSON, CSV, and SQLite database storage tiers.
"""

import csv
import json
import sqlite3
from datetime import datetime, date
from pathlib import Path
from typing import List, Dict, Any, Optional, Union

from src.utils.config import config
from src.utils.logger import logger


class CustomJSONEncoder(json.JSONEncoder):
    """
    JSON encoder extension to handle Path, datetime, set, and NumPy object types.
    """
    def default(self, obj: Any) -> Any:
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        elif isinstance(obj, Path):
            return str(obj)
        elif isinstance(obj, set):
            return list(obj)
        try:
            import numpy as np
            if isinstance(obj, (np.integer, np.floating)):
                return obj.item()
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
        except ImportError:
            pass
        return super().default(obj)


class OutputManager:
    """
    Data Storage Manager for reading and writing raw, processed, and enriched cognitive datasets.
    Supports JSON, CSV, and SQLite database persistence.
    """

    def __init__(
        self,
        raw_dir: Optional[Union[str, Path]] = None,
        processed_dir: Optional[Union[str, Path]] = None,
        output_dir: Optional[Union[str, Path]] = None,
        db_path: Optional[Union[str, Path]] = None
    ):
        """
        Initialize OutputManager with custom or default directory paths.

        Args:
            raw_dir: Path to raw data directory. Defaults to config.RAW_DATA_DIR.
            processed_dir: Path to processed data directory. Defaults to config.PROCESSED_DATA_DIR.
            output_dir: Path to output directory. Defaults to config.OUTPUT_DATA_DIR.
            db_path: Path to SQLite database file. Defaults to config.OUTPUT_DATA_DIR / 'cognitive_pipeline.db'.
        """
        self.raw_dir = Path(raw_dir or config.RAW_DATA_DIR)
        self.processed_dir = Path(processed_dir or config.PROCESSED_DATA_DIR)
        self.output_dir = Path(output_dir or config.OUTPUT_DATA_DIR)
        self.db_path = Path(db_path or (self.output_dir / "cognitive_pipeline.db"))

        self.ensure_directories()

    def ensure_directories(self) -> None:
        """Ensure all required storage directories exist on disk."""
        for directory in [self.raw_dir, self.processed_dir, self.output_dir]:
            directory.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # Core JSON File I/O
    # -------------------------------------------------------------------------

    def save_json(
        self,
        data: Any,
        file_path: Union[str, Path],
        indent: int = 2
    ) -> Path:
        """
        Saves Python dictionary or list data to a JSON file.

        Args:
            data: Data object (dict or list) to serialize.
            file_path: Destination JSON file path.
            indent: JSON indentation formatting level.

        Returns:
            Path: Absolute Path object of the saved file.
        """
        target_path = Path(file_path).resolve()
        target_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(target_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=indent, cls=CustomJSONEncoder, ensure_ascii=False)
            logger.info(f"Successfully saved JSON data to {target_path}")
            return target_path
        except Exception as e:
            logger.error(f"Failed to save JSON data to {target_path}: {e}")
            raise

    def load_json(
        self,
        file_path: Union[str, Path],
        default: Any = None
    ) -> Any:
        """
        Reads and parses a JSON file. Returns default if file does not exist or fails to parse.

        Args:
            file_path: Source JSON file path.
            default: Fallback object to return if file is missing or invalid.

        Returns:
            Any: Parsed JSON content or default value.
        """
        target_path = Path(file_path).resolve()
        if not target_path.exists():
            logger.warning(f"File not found: {target_path}. Returning default fallback.")
            return default if default is not None else []

        try:
            with open(target_path, "r", encoding="utf-8") as f:
                content = json.load(f)
            logger.info(f"Successfully loaded JSON data from {target_path}")
            return content
        except Exception as e:
            logger.error(f"Error reading JSON file {target_path}: {e}")
            return default if default is not None else []

    # -------------------------------------------------------------------------
    # CSV Data Export / Import
    # -------------------------------------------------------------------------

    def save_csv(
        self,
        records: List[Dict[str, Any]],
        file_path: Union[str, Path]
    ) -> Path:
        """
        Exports a list of record dictionaries to a CSV file.
        Complex nested values (lists, dicts) are serialized to JSON strings.

        Args:
            records: List of record dictionaries.
            file_path: Destination CSV file path.

        Returns:
            Path: Path of the saved CSV file.
        """
        target_path = Path(file_path).resolve()
        target_path.parent.mkdir(parents=True, exist_ok=True)

        if not records:
            logger.warning(f"No records provided to write CSV to {target_path}")
            # Touch empty file with header if possible
            with open(target_path, "w", encoding="utf-8", newline="") as f:
                f.write("")
            return target_path

        # Gather all field names dynamically
        fieldnames: List[str] = []
        for r in records:
            for k in r.keys():
                if k not in fieldnames:
                    fieldnames.append(k)

        try:
            with open(target_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for record in records:
                    row = {}
                    for k in fieldnames:
                        val = record.get(k, "")
                        if isinstance(val, (list, dict)):
                            val = json.dumps(val, cls=CustomJSONEncoder)
                        elif isinstance(val, (datetime, date)):
                            val = val.isoformat()
                        elif val is None:
                            val = ""
                        row[k] = val
                    writer.writerow(row)
            logger.info(f"Successfully saved {len(records)} records to CSV at {target_path}")
            return target_path
        except Exception as e:
            logger.error(f"Failed to export CSV to {target_path}: {e}")
            raise

    def load_csv(
        self,
        file_path: Union[str, Path]
    ) -> List[Dict[str, Any]]:
        """
        Reads records from a CSV file into a list of dictionaries.

        Args:
            file_path: Source CSV file path.

        Returns:
            List[Dict[str, Any]]: List of dictionary records.
        """
        target_path = Path(file_path).resolve()
        if not target_path.exists():
            logger.warning(f"CSV file not found: {target_path}")
            return []

        records = []
        try:
            with open(target_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    cleaned_row = {}
                    for k, v in row.items():
                        # Try parsing JSON string structures back to Python objects
                        if v.startswith("[") or v.startswith("{"):
                            try:
                                cleaned_row[k] = json.loads(v)
                            except Exception:
                                cleaned_row[k] = v
                        else:
                            cleaned_row[k] = v
                    records.append(cleaned_row)
            logger.info(f"Loaded {len(records)} records from CSV at {target_path}")
            return records
        except Exception as e:
            logger.error(f"Failed to load CSV from {target_path}: {e}")
            return []

    # -------------------------------------------------------------------------
    # SQLite Database Persistence
    # -------------------------------------------------------------------------

    def save_sqlite(
        self,
        records: List[Dict[str, Any]],
        table_name: str,
        db_path: Optional[Union[str, Path]] = None,
        if_exists: str = "replace"
    ) -> Path:
        """
        Persists a list of dictionary records to a SQLite database table.

        Args:
            records: List of dictionaries to store.
            table_name: SQLite database table name.
            db_path: Optional path to SQLite file. Defaults to self.db_path.
            if_exists: Strategy if table exists ('replace' or 'append').

        Returns:
            Path: Path of the SQLite database file.
        """
        target_db = Path(db_path or self.db_path).resolve()
        target_db.parent.mkdir(parents=True, exist_ok=True)

        if not records:
            logger.warning(f"No records to save to SQLite table '{table_name}' in {target_db}")
            return target_db

        # Extract columns
        columns: List[str] = []
        for r in records:
            for k in r.keys():
                if k not in columns:
                    columns.append(k)

        conn = sqlite3.connect(target_db)
        cursor = conn.cursor()

        try:
            if if_exists == "replace":
                cursor.execute(f"DROP TABLE IF EXISTS {table_name}")

            # Create table schema dynamically
            col_defs = []
            for col in columns:
                col_defs.append(f'"{col}" TEXT')
            create_query = f"CREATE TABLE IF NOT EXISTS {table_name} ({', '.join(col_defs)})"
            cursor.execute(create_query)

            # Insert records
            placeholders = ", ".join(["?"] * len(columns))
            insert_query = f"INSERT INTO {table_name} ({', '.join([f'\"{c}\"' for c in columns])}) VALUES ({placeholders})"

            rows_to_insert = []
            for rec in records:
                row_vals = []
                for col in columns:
                    val = rec.get(col, None)
                    if isinstance(val, (list, dict)):
                        val = json.dumps(val, cls=CustomJSONEncoder)
                    elif isinstance(val, (datetime, date)):
                        val = val.isoformat()
                    elif val is None:
                        val = ""
                    else:
                        val = str(val)
                    row_vals.append(val)
                rows_to_insert.append(row_vals)

            cursor.executemany(insert_query, rows_to_insert)
            conn.commit()
            logger.info(f"Saved {len(records)} records to SQLite table '{table_name}' at {target_db}")
            return target_db
        except Exception as e:
            conn.rollback()
            logger.error(f"Error saving records to SQLite table '{table_name}': {e}")
            raise
        finally:
            conn.close()

    def load_sqlite(
        self,
        table_name: str,
        db_path: Optional[Union[str, Path]] = None
    ) -> List[Dict[str, Any]]:
        """
        Queries and returns all records from a SQLite table as a list of dictionaries.

        Args:
            table_name: SQLite database table name.
            db_path: Optional path to SQLite database. Defaults to self.db_path.

        Returns:
            List[Dict[str, Any]]: List of dictionary records.
        """
        target_db = Path(db_path or self.db_path).resolve()
        if not target_db.exists():
            logger.warning(f"Database file not found: {target_db}")
            return []

        conn = sqlite3.connect(target_db)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        try:
            cursor.execute(f"SELECT * FROM {table_name}")
            rows = cursor.fetchall()
            records = []
            for row in rows:
                rec = dict(row)
                cleaned_rec = {}
                for k, v in rec.items():
                    if isinstance(v, str) and (v.startswith("[") or v.startswith("{")):
                        try:
                            cleaned_rec[k] = json.loads(v)
                        except Exception:
                            cleaned_rec[k] = v
                    else:
                        cleaned_rec[k] = v
                records.append(cleaned_rec)
            logger.info(f"Loaded {len(records)} records from SQLite table '{table_name}' at {target_db}")
            return records
        except Exception as e:
            logger.error(f"Failed to query SQLite table '{table_name}' from {target_db}: {e}")
            return []
        finally:
            conn.close()

    # -------------------------------------------------------------------------
    # Domain-Specific Methods for Raw, Processed, and Output Tiers
    # -------------------------------------------------------------------------

    # Raw Tier
    def save_raw_products(self, records: List[Dict[str, Any]], file_path: Optional[Union[str, Path]] = None) -> Path:
        """Saves raw products dataset to JSON."""
        path = file_path or (self.raw_dir / "products_raw.json")
        return self.save_json(records, path)

    def load_raw_products(self, file_path: Optional[Union[str, Path]] = None) -> List[Dict[str, Any]]:
        """Loads raw products dataset from JSON."""
        path = file_path or (self.raw_dir / "products_raw.json")
        return self.load_json(path, default=[])

    def save_raw_startups(self, records: List[Dict[str, Any]], file_path: Optional[Union[str, Path]] = None) -> Path:
        """Saves raw startups dataset to JSON."""
        path = file_path or (self.raw_dir / "startups_raw.json")
        return self.save_json(records, path)

    def load_raw_startups(self, file_path: Optional[Union[str, Path]] = None) -> List[Dict[str, Any]]:
        """Loads raw startups dataset from JSON."""
        path = file_path or (self.raw_dir / "startups_raw.json")
        return self.load_json(path, default=[])

    # Processed Tier
    def save_processed_products(self, records: List[Dict[str, Any]], file_path: Optional[Union[str, Path]] = None) -> Path:
        """Saves cleaned/processed products dataset to JSON."""
        path = file_path or (self.processed_dir / "products_cleaned.json")
        return self.save_json(records, path)

    def load_processed_products(self, file_path: Optional[Union[str, Path]] = None) -> List[Dict[str, Any]]:
        """Loads cleaned/processed products dataset from JSON."""
        path = file_path or (self.processed_dir / "products_cleaned.json")
        return self.load_json(path, default=[])

    def save_processed_startups(self, records: List[Dict[str, Any]], file_path: Optional[Union[str, Path]] = None) -> Path:
        """Saves cleaned/processed startups dataset to JSON."""
        path = file_path or (self.processed_dir / "startups_cleaned.json")
        return self.save_json(records, path)

    def load_processed_startups(self, file_path: Optional[Union[str, Path]] = None) -> List[Dict[str, Any]]:
        """Loads cleaned/processed startups dataset from JSON."""
        path = file_path or (self.processed_dir / "startups_cleaned.json")
        return self.load_json(path, default=[])

    # Output Tier
    def save_enriched_products(self, records: List[Dict[str, Any]], file_path: Optional[Union[str, Path]] = None) -> Path:
        """Saves feature-enriched products dataset to JSON."""
        path = file_path or (self.output_dir / "products.json")
        return self.save_json(records, path)

    def load_enriched_products(self, file_path: Optional[Union[str, Path]] = None) -> List[Dict[str, Any]]:
        """Loads feature-enriched products dataset from JSON."""
        path = file_path or (self.output_dir / "products.json")
        return self.load_json(path, default=[])

    def save_enriched_startups(self, records: List[Dict[str, Any]], file_path: Optional[Union[str, Path]] = None) -> Path:
        """Saves feature-enriched startups dataset to JSON."""
        path = file_path or (self.output_dir / "startups.json")
        return self.save_json(records, path)

    def load_enriched_startups(self, file_path: Optional[Union[str, Path]] = None) -> List[Dict[str, Any]]:
        """Loads feature-enriched startups dataset from JSON."""
        path = file_path or (self.output_dir / "startups.json")
        return self.load_json(path, default=[])

    def save_unified_summary(self, summary: Dict[str, Any], file_path: Optional[Union[str, Path]] = None) -> Path:
        """Saves unified pipeline insights summary to JSON."""
        path = file_path or (self.output_dir / "unified_summary.json")
        return self.save_json(summary, path)

    def load_unified_summary(self, file_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
        """Loads unified pipeline insights summary from JSON."""
        path = file_path or (self.output_dir / "unified_summary.json")
        return self.load_json(path, default={})

    def save_feature_matrix(self, feature_matrix: Dict[str, Any], file_path: Optional[Union[str, Path]] = None) -> Path:
        """Saves TF-IDF feature matrix metadata and vector representations to JSON."""
        path = file_path or (self.output_dir / "feature_matrix.json")
        return self.save_json(feature_matrix, path)

    def load_feature_matrix(self, file_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
        """Loads TF-IDF feature matrix metadata from JSON."""
        path = file_path or (self.output_dir / "feature_matrix.json")
        return self.load_json(path, default={})

    # -------------------------------------------------------------------------
    # End-to-End Pipeline Output Saver Orchestration
    # -------------------------------------------------------------------------

    def save_pipeline_outputs(
        self,
        pipeline_result: Dict[str, Any],
        export_csv: bool = True,
        export_sqlite: bool = True
    ) -> Dict[str, Path]:
        """
        Orchestrates saving all pipeline outputs (cleaned records, enriched records,
        unified summary, feature matrix, CSV files, SQLite tables) to disk.

        Args:
            pipeline_result: Unified pipeline execution dictionary returned by DataProcessor.process_pipeline().
            export_csv: Whether to export products and startups as CSV.
            export_sqlite: Whether to save products and startups to SQLite database.

        Returns:
            Dict[str, Path]: Mapping of dataset keys to saved file paths.
        """
        logger.info("Saving complete pipeline outputs across storage tiers...")

        saved_paths: Dict[str, Path] = {}

        # 1. Processed Data Tier (Cleaned Records)
        products_payload = pipeline_result.get("products", {})
        startups_payload = pipeline_result.get("startups", {})

        cleaned_prods = products_payload.get("cleaned_records", [])
        cleaned_starts = startups_payload.get("cleaned_records", [])

        if cleaned_prods:
            saved_paths["processed_products"] = self.save_processed_products(cleaned_prods)
        if cleaned_starts:
            saved_paths["processed_startups"] = self.save_processed_startups(cleaned_starts)

        # 2. Output Tier (Enriched Records)
        enriched_prods = products_payload.get("enriched_records", [])
        enriched_starts = startups_payload.get("enriched_records", [])

        if enriched_prods:
            saved_paths["output_products"] = self.save_enriched_products(enriched_prods)
        if enriched_starts:
            saved_paths["output_startups"] = self.save_enriched_startups(enriched_starts)

        # 3. Unified Insights Summary
        unified_summary = pipeline_result.get("unified_summary", {})
        if unified_summary:
            saved_paths["unified_summary"] = self.save_unified_summary(unified_summary)

        # 4. Extract & Save Feature Matrix
        feature_matrix = self._extract_feature_matrix(enriched_prods, enriched_starts)
        saved_paths["feature_matrix"] = self.save_feature_matrix(feature_matrix)

        # 5. CSV Exports
        if export_csv:
            if enriched_prods:
                saved_paths["csv_products"] = self.save_csv(
                    enriched_prods, self.output_dir / "products.csv"
                )
            if enriched_starts:
                saved_paths["csv_startups"] = self.save_csv(
                    enriched_starts, self.output_dir / "startups.csv"
                )

        # 6. SQLite Exports
        if export_sqlite:
            if enriched_prods:
                saved_paths["sqlite_products"] = self.save_sqlite(
                    enriched_prods, table_name="products", db_path=self.db_path
                )
            if enriched_starts:
                saved_paths["sqlite_startups"] = self.save_sqlite(
                    enriched_starts, table_name="startups", db_path=self.db_path
                )
            if unified_summary:
                saved_paths["sqlite_summary"] = self.save_sqlite(
                    [unified_summary], table_name="unified_summary", db_path=self.db_path
                )

        logger.info(f"Pipeline output storage completed successfully. Saved {len(saved_paths)} target files.")
        return saved_paths

    def _extract_feature_matrix(
        self,
        products: List[Dict[str, Any]],
        startups: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Builds combined feature matrix mapping item IDs to extracted feature vectors & tags."""
        item_vectors = {}

        for p in products:
            item_vectors[p.get("id", f"prod_{len(item_vectors)}")] = {
                "type": "product",
                "name": p.get("name", ""),
                "category": p.get("category", ""),
                "growth_score": p.get("growth_score", 0.0),
                "sentiment_score": p.get("sentiment_score", 0.0),
                "entities": p.get("extracted_entities", []),
                "vector_embedding": p.get("vector_embedding", [])
            }

        for s in startups:
            item_vectors[s.get("id", f"start_{len(item_vectors)}")] = {
                "type": "startup",
                "name": s.get("name", ""),
                "industry_vertical": s.get("industry_vertical", ""),
                "growth_score": s.get("growth_score", 0.0),
                "sentiment_score": s.get("sentiment_score", 0.0),
                "entities": s.get("extracted_entities", []),
                "vector_embedding": s.get("vector_embedding", [])
            }

        return {
            "total_items": len(item_vectors),
            "generated_at": datetime.now().isoformat(),
            "feature_matrix": item_vectors
        }


# Alias for backward compatibility / domain semantics
StorageManager = OutputManager


if __name__ == "__main__":
    out_mgr = OutputManager()
    
    # Quick self-test
    sample_products = [
        {
            "id": "p-101",
            "name": "Neural DB",
            "category": "Database",
            "growth_score": 88.5,
            "sentiment_score": 0.82,
            "extracted_entities": ["python", "c++", "vector-db"]
        }
    ]
    
    saved_path = out_mgr.save_enriched_products(sample_products)
    loaded_data = out_mgr.load_enriched_products()
    
    print(f"Saved to: {saved_path}")
    print(f"Loaded {len(loaded_data)} items back successfully.")

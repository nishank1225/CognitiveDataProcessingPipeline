"""
REST API Server for Cognitive Data Processing Pipeline.
Exposes endpoints for querying intelligence datasets (Products & Startups),
pipeline execution summaries, feature matrices, and triggering pipeline orchestration.

Usage:
    Direct execution:
        python -m src.server [--host HOST] [--port PORT] [--reload]

    FastAPI / Uvicorn server invocation:
        uvicorn src.server:app --host 0.0.0.0 --port 8000
"""

import argparse
from datetime import datetime
from pathlib import Path
import sys
from typing import Dict, Any, List, Optional, Union

from fastapi import FastAPI, HTTPException, Query, Path as APIPath, BackgroundTasks, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# Ensure workspace root is in sys.path when script is executed directly
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.utils.config import config
from src.utils.logger import logger
from src.storage.output_manager import OutputManager
from src.main import run_pipeline

# Initialize FastAPI application
app = FastAPI(
    title="Cognitive Data Processing Pipeline REST API",
    description=(
        "RESTful API serving product intelligence, startup intelligence, "
        "AI/ML cognitive feature matrices, summary metrics, and pipeline orchestration triggers."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS for dashboard UI and external frontend integrations
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Output Manager instance for reading storage tiers
output_manager = OutputManager()


# -----------------------------------------------------------------------------
# Parameter Sanitization Helpers (handles direct python calls & FastAPI DI)
# -----------------------------------------------------------------------------

def _str_val(val: Any) -> Optional[str]:
    return val if isinstance(val, str) else None


def _float_val(val: Any) -> Optional[float]:
    return float(val) if isinstance(val, (int, float)) else None


def _int_val(val: Any, default: int) -> int:
    return int(val) if isinstance(val, int) else default


# -----------------------------------------------------------------------------
# Pydantic Models & Schemas
# -----------------------------------------------------------------------------

class PipelineRunRequest(BaseModel):
    """Payload schema for triggering end-to-end pipeline execution."""
    products_file: Optional[str] = Field(
        default=None, description="Path to raw products input JSON file"
    )
    startups_file: Optional[str] = Field(
        default=None, description="Path to raw startups input JSON file"
    )
    export_csv: bool = Field(
        default=True, description="Whether to export CSV files"
    )
    export_sqlite: bool = Field(
        default=True, description="Whether to update SQLite database tables"
    )
    async_run: bool = Field(
        default=False, description="Run pipeline asynchronously in background task"
    )


class HealthResponse(BaseModel):
    """Schema for system health check status."""
    status: str
    timestamp: str
    environment: str
    storage_directories_exist: bool
    version: str


# -----------------------------------------------------------------------------
# API Endpoints
# -----------------------------------------------------------------------------

@app.get("/", tags=["System"])
def root() -> Dict[str, Any]:
    """Root endpoint returning API overview and available endpoints."""
    return {
        "name": "Cognitive Data Processing Pipeline REST API",
        "version": "1.0.0",
        "status": "online",
        "documentation": "/docs",
        "endpoints": {
            "health": "/health",
            "config": "/api/config",
            "metrics": "/api/metrics",
            "products": "/api/products",
            "startups": "/api/startups",
            "feature_matrix": "/api/feature-matrix",
            "run_pipeline": "/api/pipeline/run"
        }
    }


@app.get("/health", tags=["System"], response_model=HealthResponse)
@app.get("/api/health", tags=["System"], response_model=HealthResponse)
def health_check() -> HealthResponse:
    """Returns system operational health, timestamp, and storage state."""
    dirs_exist = all([
        config.RAW_DATA_DIR.exists(),
        config.PROCESSED_DATA_DIR.exists(),
        config.OUTPUT_DATA_DIR.exists()
    ])
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        environment=config.ENVIRONMENT,
        storage_directories_exist=dirs_exist,
        version="1.0.0"
    )


@app.get("/api/config", tags=["System"])
def get_configuration() -> Dict[str, Any]:
    """Returns global application configuration settings."""
    return config.to_dict()


@app.get("/api/metrics", tags=["Intelligence & Analytics"])
@app.get("/api/summary", tags=["Intelligence & Analytics"])
def get_unified_metrics() -> Dict[str, Any]:
    """
    Returns aggregated metrics, average growth score, total startup funding,
    sentiment breakdown, and top tech entities.
    """
    summary = output_manager.load_unified_summary()
    if not summary:
        logger.info("Unified summary file empty or missing. Loading default/raw data fallback.")
        return {
            "status": "empty",
            "message": "No pipeline run output found. Execute pipeline via /api/pipeline/run to generate summary.",
            "total_records_processed": 0,
            "average_growth_score": 0.0,
            "average_sentiment_score": 0.0,
            "total_startup_funding_usd": 0.0
        }
    return summary


@app.get("/api/products", tags=["Products Intelligence"])
def list_products(
    category: Optional[str] = Query(None, description="Filter products by category keyword"),
    search: Optional[str] = Query(None, description="Search term across name, category, and description"),
    min_rating: Optional[float] = Query(None, description="Filter by minimum rating (0.0 - 5.0)"),
    min_growth_score: Optional[float] = Query(None, description="Filter by minimum AI growth score (0.0 - 100.0)"),
    limit: int = Query(50, ge=1, le=500, description="Max number of items to return"),
    offset: int = Query(0, ge=0, description="Offset index for pagination")
) -> Dict[str, Any]:
    """
    Queries products intelligence dataset with filtering and pagination.
    Pulls enriched product records if available, otherwise falls back to cleaned or raw records.
    """
    category_str = _str_val(category)
    search_str = _str_val(search)
    min_rating_flt = _float_val(min_rating)
    min_growth_flt = _float_val(min_growth_score)
    limit_int = _int_val(limit, 50)
    offset_int = _int_val(offset, 0)

    products = output_manager.load_enriched_products()
    if not products:
        products = output_manager.load_processed_products()
    if not products:
        products = output_manager.load_raw_products()

    filtered = products

    # Filter by category
    if category_str:
        cat_lower = category_str.lower()
        filtered = [
            p for p in filtered
            if cat_lower in str(p.get("category", "")).lower()
        ]

    # Filter by search term
    if search_str:
        s_lower = search_str.lower()
        filtered = [
            p for p in filtered
            if s_lower in str(p.get("name", "")).lower()
            or s_lower in str(p.get("category", "")).lower()
            or s_lower in str(p.get("description", "")).lower()
        ]

    # Filter by min rating
    if min_rating_flt is not None:
        filtered = [
            p for p in filtered
            if p.get("rating") is not None and float(p.get("rating", 0)) >= min_rating_flt
        ]

    # Filter by min growth score
    if min_growth_flt is not None:
        filtered = [
            p for p in filtered
            if p.get("growth_score") is not None and float(p.get("growth_score", 0)) >= min_growth_flt
        ]

    total_count = len(filtered)
    paginated_results = filtered[offset_int : offset_int + limit_int]

    return {
        "total": total_count,
        "limit": limit_int,
        "offset": offset_int,
        "products": paginated_results
    }


@app.get("/api/products/{product_id}", tags=["Products Intelligence"])
def get_product_by_id(
    product_id: str = APIPath(..., description="Unique product ID (e.g., prod-101)")
) -> Dict[str, Any]:
    """Retrieves detailed record for a specific product by ID."""
    products = output_manager.load_enriched_products()
    if not products:
        products = output_manager.load_processed_products()
    if not products:
        products = output_manager.load_raw_products()

    for item in products:
        if str(item.get("id")) == str(product_id):
            return item

    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Product with ID '{product_id}' not found."
    )


@app.get("/api/startups", tags=["Startups Intelligence"])
def list_startups(
    industry: Optional[str] = Query(None, description="Filter startups by industry vertical"),
    search: Optional[str] = Query(None, description="Search term across name, industry, description, tech stack"),
    min_funding: Optional[float] = Query(None, description="Filter by minimum total funding in USD"),
    min_growth_score: Optional[float] = Query(None, description="Filter by minimum AI growth score (0.0 - 100.0)"),
    limit: int = Query(50, ge=1, le=500, description="Max number of items to return"),
    offset: int = Query(0, ge=0, description="Offset index for pagination")
) -> Dict[str, Any]:
    """
    Queries startups intelligence dataset with filtering and pagination.
    Pulls enriched startup records if available, otherwise falls back to cleaned or raw records.
    """
    industry_str = _str_val(industry)
    search_str = _str_val(search)
    min_funding_flt = _float_val(min_funding)
    min_growth_flt = _float_val(min_growth_score)
    limit_int = _int_val(limit, 50)
    offset_int = _int_val(offset, 0)

    startups = output_manager.load_enriched_startups()
    if not startups:
        startups = output_manager.load_processed_startups()
    if not startups:
        startups = output_manager.load_raw_startups()

    filtered = startups

    # Filter by industry
    if industry_str:
        ind_lower = industry_str.lower()
        filtered = [
            s for s in filtered
            if ind_lower in str(s.get("industry_vertical", "")).lower()
        ]

    # Filter by search term
    if search_str:
        s_lower = search_str.lower()
        filtered = [
            s for s in filtered
            if s_lower in str(s.get("name", "")).lower()
            or s_lower in str(s.get("industry_vertical", "")).lower()
            or s_lower in str(s.get("description", "")).lower()
            or any(s_lower in str(t).lower() for t in s.get("tech_stack", []))
            or any(s_lower in str(d).lower() for d in s.get("domain_tags", []))
        ]

    # Filter by min funding
    if min_funding_flt is not None:
        filtered = [
            s for s in filtered
            if s.get("total_funding_usd") is not None and float(s.get("total_funding_usd", 0)) >= min_funding_flt
        ]

    # Filter by min growth score
    if min_growth_flt is not None:
        filtered = [
            s for s in filtered
            if s.get("growth_score") is not None and float(s.get("growth_score", 0)) >= min_growth_flt
        ]

    total_count = len(filtered)
    paginated_results = filtered[offset_int : offset_int + limit_int]

    return {
        "total": total_count,
        "limit": limit_int,
        "offset": offset_int,
        "startups": paginated_results
    }


@app.get("/api/startups/{startup_id}", tags=["Startups Intelligence"])
def get_startup_by_id(
    startup_id: str = APIPath(..., description="Unique startup ID (e.g., start-201)")
) -> Dict[str, Any]:
    """Retrieves detailed record for a specific startup by ID."""
    startups = output_manager.load_enriched_startups()
    if not startups:
        startups = output_manager.load_processed_startups()
    if not startups:
        startups = output_manager.load_raw_startups()

    for item in startups:
        if str(item.get("id")) == str(startup_id):
            return item

    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Startup with ID '{startup_id}' not found."
    )


@app.get("/api/feature-matrix", tags=["AI & ML Features"])
@app.get("/api/features", tags=["AI & ML Features"])
def get_feature_matrix() -> Dict[str, Any]:
    """Returns AI/ML feature matrix containing vector embeddings and extracted entities."""
    matrix = output_manager.load_feature_matrix()
    if not matrix:
        return {
            "status": "empty",
            "message": "No feature matrix generated yet. Run pipeline orchestration to extract features.",
            "total_items": 0,
            "feature_matrix": {}
        }
    return matrix


@app.post("/api/pipeline/run", tags=["Pipeline Orchestration"], status_code=status.HTTP_200_OK)
@app.post("/api/trigger", tags=["Pipeline Orchestration"], status_code=status.HTTP_200_OK)
def trigger_pipeline_execution(
    payload: PipelineRunRequest,
    background_tasks: BackgroundTasks
) -> Dict[str, Any]:
    """
    Triggers end-to-end data collection, cleaning, feature extraction, and storage pipeline.
    Supports synchronous execution or background async execution.
    """
    logger.info(f"Received API trigger for pipeline execution (async_run={payload.async_run})")

    if payload.async_run:
        background_tasks.add_task(
            run_pipeline,
            products_source=payload.products_file,
            startups_source=payload.startups_file,
            export_csv=payload.export_csv,
            export_sqlite=payload.export_sqlite,
            verbose=False
        )
        return {
            "status": "accepted",
            "message": "Pipeline execution scheduled in background task.",
            "timestamp": datetime.now().isoformat()
        }

    try:
        result = run_pipeline(
            products_source=payload.products_file,
            startups_source=payload.startups_file,
            export_csv=payload.export_csv,
            export_sqlite=payload.export_sqlite,
            verbose=False
        )
        return {
            "status": "completed",
            "timestamp": datetime.now().isoformat(),
            "records_processed": result.get("unified_summary", {}).get("total_records_processed", 0),
            "summary": result.get("unified_summary", {}),
            "saved_paths": {k: str(v) for k, v in result.get("saved_paths", {}).items()}
        }
    except Exception as e:
        logger.error(f"Pipeline execution error via REST API: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Pipeline execution failed: {str(e)}"
        )


# -----------------------------------------------------------------------------
# CLI Entry Point
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parses command-line options for running the REST server directly."""
    parser = argparse.ArgumentParser(
        description="Cognitive Data Processing Pipeline REST API Server"
    )
    parser.add_argument(
        "--host",
        type=str,
        default=config.HOST,
        help=f"Host address to bind server (default: {config.HOST})"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=config.PORT,
        help=f"Port number to bind server (default: {config.PORT})"
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload on code changes (development mode)"
    )
    return parser.parse_args()


def main() -> None:
    """Launches the Uvicorn ASGI server with parsed arguments."""
    args = parse_args()
    logger.info(f"Starting REST API server on http://{args.host}:{args.port}")
    uvicorn.run(
        "src.server:app",
        host=args.host,
        port=args.port,
        reload=args.reload
    )


if __name__ == "__main__":
    main()

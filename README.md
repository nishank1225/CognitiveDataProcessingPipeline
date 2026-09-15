# Cognitive Data Processing Pipeline

![Python Version](https://img.shields.io/badge/python-3.9%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![Pipeline Architecture](https://img.shields.io/badge/pipeline-5--Stage-green?style=for-the-badge)
![Status](https://img.shields.io/badge/status-active-brightgreen?style=for-the-badge)
![License](https://img.shields.io/badge/license-MIT-orange?style=for-the-badge)

An end-to-end framework built to ingest, sanitize, analyze (AI/ML feature extraction), store, and visualize market intelligence for **Products** and **Startups**.

This pipeline automates the entire market data lifecycle: from multi-source collection and cleaning, through AI feature extraction and vector similarity indexing, all the way to structured storage and interactive visual dashboards.

---

##  Domain Data Collection Targets

| Domain | Collected Metrics & Features | Primary Collector File |
| :--- | :--- | :--- |
| **Products Intelligence** | Tech Specs, Customer Reviews, Rating Polarity, Usage Telemetry | [`src/products/collector.py`](src/products/collector.py) |
| **Startups Intelligence** | Funding Rounds, Valuation Metrics, Team Size, Tech Stack, Domain Tags | [`src/startups/collector.py`](src/startups/collector.py) |

---

##  System Architecture

Data flows through a structured **5-stage processing pipeline**:

```text
                      ┌─────────────────────────┐
                      │      DATA SOURCES       │
                      │  (Products & Startups)  │
                      └────────────┬────────────┘
                                   │  Raw payloads
                                   ▼
                      ┌─────────────────────────┐
                      │      1. Collectors      │  --> products/collector.py & startups/collector.py
                      │  Products / Startups    │
                      └────────────┬────────────┘
                                   │  Uncleaned JSON/CSV
                                   ▼
                      ┌─────────────────────────┐
                      │     2. Preprocessing    │  --> processing/cleaner.py (Sanitization & Validation)
                      │     Clean & Validate    │
                      └────────────┬────────────┘
                                   │  Sanitized records
                                   ▼
                      ┌─────────────────────────┐
                      │   3. Data Processing    │  --> processing/feature_extractor.py & processor.py
                      │  Feature Extraction &   │      (NER, Sentiment, Growth Index, Vectors)
                      │     AI/ML Analysis      │
                      └────────────┬────────────┘
                                   │  Enriched feature matrix
                                   ▼
                      ┌─────────────────────────┐
                      │       4. Storage        │  --> storage/output_manager.py
                      │    JSON / CSV / DB      │      (Persists to raw/, processed/, output/)
                      └────────────┬────────────┘
                                   │  Persisted datasets
                                   ▼
                      ┌─────────────────────────┐
                      │      5. Dashboard       │  --> dashboard/app.py & src/server.py
                      │   Visualization & REST  │      (Interactive UI & REST API Endpoints)
                      └─────────────────────────┘
```

---

##  Main Pipeline Flow

```text
Data Sources (Products & Startups)
   ↓
Collectors (src/products/ & src/startups/)
   ↓
Preprocessing (src/processing/cleaner.py - HTML Stripping, Validation, Deduplication)
   ↓
Data Processing (src/processing/feature_extractor.py - Sentiment, NER, Vector Embeddings)
   ↓
Storage Layer (src/storage/output_manager.py -> data/raw/ | data/processed/ | data/output/)
   ↓
Dashboard (dashboard/app.py) & REST API Server (src/server.py)
```

---

##  Directory & Module Structure

```text
CognitiveDataProcessingPipeline/
├── README.md                          # Main project guide & setup documentation
├── requirements.txt                   # Project dependencies (FastAPI, Pandas, scikit-learn, etc.)
├── .gitignore                         # Keeps caches and temporary data out of Git
├── .env.example                       # Template for local environment variables
│
├── src/                               # Core Python source package
│   ├── __init__.py
│   ├── main.py                        # CLI runner to trigger collection & processing pipeline
│   ├── server.py                      # REST API server exposing pipeline data & trigger endpoints
│   │
│   ├── products/                      # Product Intelligence Collectors
│   │   ├── __init__.py
│   │   └── collector.py               # Ingests tech specs, user reviews & product metrics
│   │
│   ├── startups/                      # Startup Intelligence Collectors
│   │   ├── __init__.py
│   │   └── collector.py               # Ingests funding rounds, market tags & team metrics
│   │
│   ├── processing/                    # Preprocessing & AI/ML Cognitive Engine
│   │   ├── __init__.py
│   │   ├── cleaner.py                 # Data validation, null handling, HTML stripping & deduplication
│   │   ├── processor.py               # Main processing orchestrator combining clean + feature extraction
│   │   └── feature_extractor.py       # Sentiment analysis, entity extraction, TF-IDF vector embeddings
│   │
│   ├── storage/                       # Storage & Data Management
│   │   ├── __init__.py
│   │   └── output_manager.py          # Writes data to data/raw/, data/processed/, data/output/
│   │
│   └── utils/                         # Helper Utilities
│       ├── __init__.py
│       ├── config.py                  # Global configurations & thresholds
│       └── logger.py                  # Structured console logging
│
├── dashboard/                         # Analytics Dashboard UI
│   ├── app.py                         # Web dashboard application entrypoint
│   └── components/                    # UI component cards & visual chart generators
│
├── data/                              # Data Lifecycle Directories
│   ├── raw/                           # Raw uncleaned payloads from collectors
│   ├── processed/                     # Cleaned & standardized records
│   └── output/                        # AI-enriched final datasets
│
├── tests/                             # Unit Test Suite
│   ├── test_products.py
│   ├── test_startups.py
│   ├── test_processing.py
│   └── test_storage.py
│
└── docs/
    └── architecture.md                # Technical architecture & pipeline documentation
```

---

##  Component Specifications

### 1. Collectors (`src/products/` & `src/startups/`)
- **Product Collector** ([`src/products/collector.py`](src/products/collector.py)): Pulls product specifications, user reviews, ratings, and technical usage metadata.
- **Startup Collector** ([`src/startups/collector.py`](src/startups/collector.py)): Pulls startup profiles, funding stages, valuation histories, team sizes, and industry tags.

### 2. Preprocessing & Cleaning (`src/processing/cleaner.py`)
- **HTML Sanitization**: Strips markup tags and unescapes text.
- **Missing Value Handling**: Imputes default schemas for missing fields.
- **Deduplication**: Removes duplicate entity records based on checksum hashes.
- **Validation**: Enforces strict Pydantic type validation.

### 3. Data Processing & AI Engine (`src/processing/feature_extractor.py` & `processor.py`)
- **Sentiment Scoring**: Computes stance polarity ($[-1.0, +1.0]$) and review tone.
- **Entity Extraction (NER)**: Extracts key entities (Tech Stack, Investors, Companies).
- **Vector Embeddings**: Computes TF-IDF feature vectors and Cosine Similarity matrices.
- **Growth Indexing**: Calculates market momentum & innovation scores.

### 4. Storage Layer (`src/storage/output_manager.py`)
- Manages routing and saving data through a structured 3-tier lifecycle across `data/raw/`, `data/processed/`, and `data/output/` in JSON and CSV formats.

### 5. Dashboard UI & REST Server (`dashboard/` & `src/server.py`)
- **REST API Server** ([`src/server.py`](src/server.py)): Serves FastAPI endpoints (`/api/products`, `/api/startups`, `/api/metrics`).
- **Visual Dashboard** ([`dashboard/app.py`](dashboard/app.py)): Renders interactive charts, metric summary cards, and search filters.

---

##  Key Highlights

- **Decoupled Architecture**: 5 independent pipeline stages operating with clear data contracts.
- **Automated Data Cleaning**: Eliminates missing values, sanitizes text, and removes duplicate entries.
- **Cognitive ML Features**: Computes sentiment scores, entity tags, growth metrics, and dense vector embeddings.
- **Multi-Tier Persistence**: Automatic data archiving across `data/raw/` $\rightarrow$ `data/processed/` $\rightarrow$ `data/output/`.
- **API & UI Layer**: FastAPI endpoints and visual dashboard for real-time data inspection.
- **Automated Testing**: Complete test coverage across modules in `tests/`.

---

##  Tech Stack

**Python 3.9+** · `Pandas` · `NumPy` · `scikit-learn` · `FastAPI` · `Uvicorn` · `Pydantic` · `python-dotenv` · `pytest`

---

##  Installation & Setup

### 1. Clone & Navigate
```bash
git clone https://github.com/your-username/CognitiveDataProcessingPipeline.git
cd CognitiveDataProcessingPipeline
```

### 2. Set Up Virtual Environment
```bash
# On Windows
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# On macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment File
```bash
cp .env.example .env
```

---

##  Run Instructions

###  1. CLI Pipeline Mode
To trigger end-to-end data ingestion, cleaning, feature extraction, and storage:
```bash
python -m src.main
```

###  2. REST API Server Mode
To launch the FastAPI server:
```bash
python -m src.server
```
*Access interactive API documentation at `http://localhost:8000/docs`.*

###  3. Web Dashboard Mode
To launch the interactive visual analytics dashboard:
```bash
python dashboard/app.py
```

###  4. Run Test Suite
To run unit tests across all modules:
```bash
python -m unittest discover -s tests
```


##  Output Lifecycle

Results are automatically persisted into structured data folders:

```text
data/
├── raw/                               # Raw uncleaned collector dumps
│   ├── products_raw.json
│   └── startups_raw.json
│
├── processed/                         # Sanitized & schema-validated data
│   ├── products_cleaned.json
│   └── startups_cleaned.json
│
└── output/                            # AI/ML feature-enriched final outputs
    ├── products.json
    ├── startups.json
    ├── feature_matrix.json
    └── unified_summary.json
```

---

##  What I Learned

- **Resilient Pipeline Design**: Structuring an end-to-end 5-stage pipeline where data ingestion, preprocessing, AI analysis, storage, and visualization operate independently.
- **Data Quality Engineering**: Handling noisy external payloads through HTML stripping, missing value imputation, and Pydantic schema validation.
- **Cognitive Feature Modeling**: Extracting Sentiment Scores, Entity Tags (NER), and building TF-IDF Vector Embeddings.
- **Clean Decoupling**: Exposing data via REST API endpoints (`src/server.py`) and Web Dashboard (`dashboard/app.py`) without tightly coupling visual components to data processing scripts.

---

##  Future Roadmap

- [ ] PostgreSQL / SQLite Database integration
- [ ] Scheduled background cron jobs for recurring data ingestion
- [ ] Enhanced vector similarity search engine (Cosine & K-NN clustering)
- [ ] Expanded data source collectors for Research Papers and Market News
- [ ] System telemetry, health checks, and logging dashboards

---

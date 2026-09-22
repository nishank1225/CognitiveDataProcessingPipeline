"""
AI/ML Feature Extractor module for Cognitive Data Processing Pipeline.
Handles sentiment scoring, entity/keyword extraction (NER), TF-IDF vector embeddings,
cosine similarity computation, and startup/product growth index calculation.
"""

import re
from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.utils.config import config
from src.utils.logger import logger


class FeatureExtractor:
    """
    AI/ML Feature Extraction Engine responsible for analyzing textual payloads,
    extracting key domain entities, computing sentiment polarities, generating
    TF-IDF embeddings, and evaluating growth/innovation scores.
    """

    # Domain sentiment lexicon for rule-based polarity scoring
    POSITIVE_WORDS = {
        "outstanding", "excellent", "seamless", "top", "fast", "great", "solid",
        "magic", "blazing", "robust", "impressively", "cut", "save", "saves",
        "pro", "advanced", "autonomous", "predictive", "optimized", "high",
        "innovative", "accelerates", "scalable", "secure", "leader", "best",
        "growth", "efficient", "breakthrough", "superior", "positive"
    }

    NEGATIVE_WORDS = {
        "slow", "high memory", "high latency", "false positive", "overhead",
        "complex", "high cost", "issue", "bug", "flaw", "poor", "bad",
        "difficult", "legacy", "bottleneck", "vulnerable", "missing", "hard",
        "fail", "failure", "degraded", "negative"
    }

    # Pre-defined tech stack vocabulary for entity extraction
    TECH_VOCABULARY = {
        "python", "pytorch", "tensorflow", "jax", "c++", "cuda", "fastapi",
        "docker", "kubernetes", "react", "vue.js", "next.js", "postgresql",
        "kafka", "redis", "spark", "influxdb", "aws", "gRPC", "rest", "grpc",
        "microservices", "rust", "apache arrow", "snowflake", "bigquery",
        "mongo", "verilog", "embedded linux", "scikit-learn", "pandas", "numpy"
    }

    def __init__(self, max_tfidf_features: Optional[int] = None):
        """
        Initialize FeatureExtractor.

        Args:
            max_tfidf_features: Maximum vocabulary size for TF-IDF vectorizer.
        """
        self.max_tfidf_features = max_tfidf_features or config.TFIDF_MAX_FEATURES
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_tfidf_features,
            stop_words="english",
            token_pattern=r"(?u)\b\w+\b"
        )
        self.is_vectorizer_fitted = False

    def analyze_sentiment(self, text: Union[str, List[str]]) -> Dict[str, Any]:
        """
        Calculates sentiment polarity score [-1.0, +1.0] and tone label
        ('positive', 'negative', 'neutral') for a string or list of text strings.

        Args:
            text: Input string or list of review/description strings.

        Returns:
            Dict[str, Any]: Contains 'score' (float) and 'label' (str).
        """
        if isinstance(text, list):
            text_str = " ".join([t for t in text if isinstance(t, str)])
        elif isinstance(text, str):
            text_str = text
        else:
            text_str = ""

        if not text_str.strip():
            return {"score": 0.0, "label": "neutral"}

        words = re.findall(r"\b\w+\b", text_str.lower())
        if not words:
            return {"score": 0.0, "label": "neutral"}

        pos_count = sum(1 for word in words if word in self.POSITIVE_WORDS)
        neg_count = sum(1 for word in words if word in self.NEGATIVE_WORDS)

        # Check multi-word negative phrases
        lower_text = text_str.lower()
        for phrase in ["high memory", "high latency", "false positive", "high cost"]:
            if phrase in lower_text:
                neg_count += 1

        total_matches = pos_count + neg_count
        if total_matches == 0:
            polarity = 0.0
        else:
            polarity = (pos_count - neg_count) / float(total_matches)

        polarity = round(max(-1.0, min(1.0, polarity)), 3)

        if polarity >= config.SENTIMENT_POSITIVE_THRESHOLD:
            label = "positive"
        elif polarity <= config.SENTIMENT_NEGATIVE_THRESHOLD:
            label = "negative"
        else:
            label = "neutral"

        return {"score": polarity, "label": label}

    def extract_entities(
        self,
        text: str,
        existing_tags: Optional[List[str]] = None
    ) -> Dict[str, List[str]]:
        """
        Extracts tech stack entities, proper noun organizations, and keyword tags from text.

        Args:
            text: Raw text content (e.g. description, summary).
            existing_tags: Optional list of known tags to integrate.

        Returns:
            Dict[str, List[str]]: Categorized extracted entities.
        """
        if not isinstance(text, str):
            text = ""

        words = re.findall(r"\b[A-Za-z0-9+#\.]+\b", text)

        # 1. Match tech stack terms
        tech_entities = set()
        lower_text = text.lower()
        for tech in self.TECH_VOCABULARY:
            if tech in lower_text:
                tech_entities.add(tech)

        # 2. Match proper noun capitalized phrases (e.g. "Sequoia Capital", "AWS")
        proper_entities = set(re.findall(r"\b[A-Z][a-zA-Z0-9]+(?:\s+[A-Z][a-zA-Z0-9]+)*\b", text))
        # Filter out common sentence start words
        stop_words = {"An", "The", "Building", "Autonomous", "Zero", "Rock", "Needs", "Great", "Caught", "Querying"}
        proper_entities = {e for e in proper_entities if e not in stop_words and len(e) > 2}

        # 3. Keyword extraction (non-stop words > 4 chars)
        keywords = set()
        for word in words:
            word_clean = word.strip().lower()
            if len(word_clean) > 4 and word_clean not in {"about", "which", "their", "there", "using", "platform"}:
                keywords.add(word_clean)

        if existing_tags:
            for tag in existing_tags:
                if isinstance(tag, str) and tag.strip():
                    keywords.add(tag.strip())

        return {
            "tech_entities": sorted(list(tech_entities)),
            "proper_entities": sorted(list(proper_entities)),
            "keywords": sorted(list(keywords))[:15]
        }

    def fit_transform_tfidf(self, corpus: List[str]) -> np.ndarray:
        """
        Fits TF-IDF Vectorizer on corpus and returns dense feature matrix array.

        Args:
            corpus: List of document strings.

        Returns:
            np.ndarray: Dense 2D TF-IDF feature embeddings matrix.
        """
        if not corpus or all(not doc.strip() for doc in corpus):
            logger.warning("Empty corpus provided for TF-IDF feature extraction.")
            return np.zeros((len(corpus), self.max_tfidf_features))

        matrix = self.vectorizer.fit_transform(corpus)
        self.is_vectorizer_fitted = True
        return matrix.toarray()

    def compute_similarity_matrix(self, corpus: List[str]) -> np.ndarray:
        """
        Computes pairwise Cosine Similarity matrix for a list of document strings.

        Args:
            corpus: List of text documents.

        Returns:
            np.ndarray: Square 2D Cosine Similarity matrix.
        """
        if not corpus:
            return np.array([[]])

        tfidf_matrix = self.fit_transform_tfidf(corpus)
        sim_matrix = cosine_similarity(tfidf_matrix)
        return np.round(sim_matrix, 4)

    def calculate_startup_growth_score(self, record: Dict[str, Any]) -> float:
        """
        Calculates normalized Growth/Innovation Score (0.0 to 100.0) for a startup record.

        Args:
            record: Startup profile dictionary.

        Returns:
            float: Composite growth index score.
        """
        weights = config.GROWTH_WEIGHTS

        # 1. Funding score (log-scaled up to $100M)
        funding = float(record.get("total_funding_usd") or 0.0)
        funding_score = min(1.0, np.log10(max(1.0, funding)) / 8.0)

        # 2. Team growth score (normalized up to 100% growth)
        growth_pct = float(record.get("team_growth_pct") or 0.0)
        team_score = min(1.0, max(0.0, growth_pct / 100.0))

        # 3. Sentiment score
        summary_text = record.get("summary_text") or record.get("description", "")
        sentiment_res = self.analyze_sentiment(summary_text)
        sentiment_score = (sentiment_res["score"] + 1.0) / 2.0  # map [-1,1] to [0,1]

        # 4. Rating score
        rating = float(record.get("customer_rating") or 4.0)
        rating_score = min(1.0, max(0.0, rating / 5.0))

        composite = (
            weights.get("funding", 0.4) * funding_score +
            weights.get("team_growth", 0.3) * team_score +
            weights.get("sentiment", 0.2) * sentiment_score +
            weights.get("rating", 0.1) * rating_score
        )

        return round(float(composite * 100.0), 2)

    def calculate_product_growth_score(self, record: Dict[str, Any]) -> float:
        """
        Calculates normalized Growth/Innovation Score (0.0 to 100.0) for a product record.

        Args:
            record: Product dictionary.

        Returns:
            float: Composite innovation index score.
        """
        # 1. Active users score (log-scaled up to 100k MAU)
        mau = float(record.get("monthly_active_users") or record.get("active_users") or 0.0)
        users_score = min(1.0, np.log10(max(1.0, mau)) / 5.0)

        # 2. Rating score
        rating = float(record.get("rating") or 4.0)
        rating_score = min(1.0, max(0.0, rating / 5.0))

        # 3. Sentiment of user reviews
        reviews = record.get("reviews", [])
        sentiment_res = self.analyze_sentiment(reviews)
        sentiment_score = (sentiment_res["score"] + 1.0) / 2.0

        # Composite score weighting
        composite = (0.4 * users_score) + (0.35 * rating_score) + (0.25 * sentiment_score)
        return round(float(composite * 100.0), 2)

    def extract_product_features(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enriches a single product record with ML feature vectors, sentiment analysis,
        entity tags, and innovation scores.

        Args:
            record: Cleaned product record dictionary.

        Returns:
            Dict[str, Any]: Enriched product record dictionary.
        """
        enriched = dict(record)

        # Sentiment analysis on reviews + description
        reviews = enriched.get("reviews", [])
        desc = enriched.get("description", "")
        combined_text = f"{desc} " + " ".join(reviews)
        
        sentiment = self.analyze_sentiment(reviews if reviews else desc)
        enriched["sentiment_score"] = sentiment["score"]
        enriched["sentiment_label"] = sentiment["label"]

        # Entity and keyword extraction
        entities = self.extract_entities(combined_text)
        enriched["extracted_entities"] = entities["tech_entities"]
        enriched["proper_entities"] = entities["proper_entities"]
        enriched["keywords"] = entities["keywords"]

        # Innovation / Growth Score
        enriched["growth_score"] = self.calculate_product_growth_score(enriched)

        return enriched

    def extract_startup_features(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enriches a single startup profile record with ML feature vectors, sentiment analysis,
        entity tags, and growth scores.

        Args:
            record: Cleaned startup record dictionary.

        Returns:
            Dict[str, Any]: Enriched startup record dictionary.
        """
        enriched = dict(record)

        # Sentiment analysis on summary + description
        summary = enriched.get("summary_text", "")
        desc = enriched.get("description", "")
        combined_text = f"{summary} {desc}"

        sentiment = self.analyze_sentiment(combined_text)
        enriched["sentiment_score"] = sentiment["score"]
        enriched["sentiment_label"] = sentiment["label"]

        # Entity extraction integrating existing tech_stack & domain_tags
        existing_tags = enriched.get("tech_stack", []) + enriched.get("domain_tags", [])
        entities = self.extract_entities(combined_text, existing_tags=existing_tags)

        enriched["extracted_entities"] = sorted(list(set(entities["tech_entities"] + enriched.get("tech_stack", []))))
        enriched["proper_entities"] = entities["proper_entities"]
        enriched["keywords"] = entities["keywords"]

        # Startup Growth Index
        enriched["growth_score"] = self.calculate_startup_growth_score(enriched)

        return enriched

    def process_product_batch(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Processes a batch of product records, extracting features and computing
        TF-IDF similarity embeddings.

        Args:
            records: List of product dictionaries.

        Returns:
            List[Dict[str, Any]]: List of feature-enriched product records.
        """
        logger.info(f"Extracting AI/ML features for batch of {len(records)} product records...")
        enriched_records = [self.extract_product_features(r) for r in records]

        # Compute corpus TF-IDF embeddings
        corpus = [
            f"{r.get('name', '')} {r.get('category', '')} {r.get('description', '')}"
            for r in enriched_records
        ]
        sim_matrix = self.compute_similarity_matrix(corpus)

        # Attach top similar product IDs to each record
        for idx, rec in enumerate(enriched_records):
            if len(records) > 1:
                sim_scores = sim_matrix[idx]
                top_indices = np.argsort(sim_scores)[::-1]
                # Filter out self-index
                top_indices = [i for i in top_indices if i != idx][:3]
                rec["similar_item_ids"] = [records[i].get("id") for i in top_indices if i < len(records)]
            else:
                rec["similar_item_ids"] = []

        logger.info(f"Product feature extraction complete for {len(enriched_records)} records.")
        return enriched_records

    def process_startup_batch(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Processes a batch of startup records, extracting features and computing
        TF-IDF similarity embeddings.

        Args:
            records: List of startup dictionaries.

        Returns:
            List[Dict[str, Any]]: List of feature-enriched startup records.
        """
        logger.info(f"Extracting AI/ML features for batch of {len(records)} startup records...")
        enriched_records = [self.extract_startup_features(r) for r in records]

        # Compute corpus TF-IDF embeddings
        corpus = [
            f"{r.get('name', '')} {r.get('industry_vertical', '')} {r.get('description', '')} {r.get('summary_text', '')}"
            for r in enriched_records
        ]
        sim_matrix = self.compute_similarity_matrix(corpus)

        # Attach top similar startup IDs to each record
        for idx, rec in enumerate(enriched_records):
            if len(records) > 1:
                sim_scores = sim_matrix[idx]
                top_indices = np.argsort(sim_scores)[::-1]
                top_indices = [i for i in top_indices if i != idx][:3]
                rec["similar_item_ids"] = [records[i].get("id") for i in top_indices if i < len(records)]
            else:
                rec["similar_item_ids"] = []

        logger.info(f"Startup feature extraction complete for {len(enriched_records)} records.")
        return enriched_records


if __name__ == "__main__":
    from src.products.collector import ProductCollector
    from src.startups.collector import StartupCollector
    from src.processing.cleaner import DataCleaner

    cleaner = DataCleaner()
    extractor = FeatureExtractor()

    # Test products
    prod_coll = ProductCollector()
    raw_prods = prod_coll.collect_sample_data()
    cleaned_prods = cleaner.clean_products(raw_prods)
    enriched_prods = extractor.process_product_batch(cleaned_prods)
    print(f"Enriched {len(enriched_prods)} products. Example feature output:")
    if enriched_prods:
        print(f"Product: {enriched_prods[0]['name']}")
        print(f"  Sentiment: {enriched_prods[0]['sentiment_score']} ({enriched_prods[0]['sentiment_label']})")
        print(f"  Growth Score: {enriched_prods[0]['growth_score']}")
        print(f"  Entities: {enriched_prods[0]['extracted_entities']}")
        print(f"  Similar items: {enriched_prods[0]['similar_item_ids']}")

    # Test startups
    start_coll = StartupCollector()
    raw_starts = start_coll.collect_sample_data()
    cleaned_starts = cleaner.clean_startups(raw_starts)
    enriched_starts = extractor.process_startup_batch(cleaned_starts)
    print(f"\nEnriched {len(enriched_starts)} startups. Example feature output:")
    if enriched_starts:
        print(f"Startup: {enriched_starts[0]['name']}")
        print(f"  Sentiment: {enriched_starts[0]['sentiment_score']} ({enriched_starts[0]['sentiment_label']})")
        print(f"  Growth Score: {enriched_starts[0]['growth_score']}")
        print(f"  Entities: {enriched_starts[0]['extracted_entities']}")
        print(f"  Similar items: {enriched_starts[0]['similar_item_ids']}")

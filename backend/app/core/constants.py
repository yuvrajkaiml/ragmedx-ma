# Owner: Pranav Sukale
"""Shared application constants for RAGMedX-MA.

These are stable, non-secret values that are referenced in multiple modules.
Centralising them here prevents magic numbers from being scattered across the
codebase and makes tuning for the research paper straightforward.

Secrets and environment-specific values belong in ``app.config.Settings``,
not here.
"""

# ============================================================
# Application identity
# ============================================================

APP_NAME = "RAGMedX-MA"
APP_VERSION = "0.1.0"
APP_DESCRIPTION = (
    "A Multi-Agent Iterative Retrieval Framework for "
    "High-Confidence Medical Question Answering"
)

# ============================================================
# API
# ============================================================

API_V1_PREFIX = "/api/v1"

# Maximum question length accepted by the query endpoint (characters).
MAX_QUESTION_LENGTH = 5_000
# Minimum question length (characters).
MIN_QUESTION_LENGTH = 5

# ============================================================
# Document upload
# ============================================================

MAX_UPLOAD_SIZE_MB = 50
ALLOWED_UPLOAD_CONTENT_TYPES = frozenset({
    "application/pdf",
    "application/x-pdf",
})

# ============================================================
# Chunking
# ============================================================

# Default chunker settings — also declared in Settings but mirrored here
# for components that don't have Settings access (e.g. scripts).
DEFAULT_CHUNK_SIZE = 1_000          # characters per chunk
DEFAULT_CHUNK_OVERLAP = 150         # overlap carry-over characters
DEFAULT_CHUNK_MIN_CHARS = 50        # chunks shorter than this are discarded

# Legacy config.py values (kept for backward compatibility).
CHUNK_SIZE = 2_000
CHUNK_OVERLAP = 400

# ============================================================
# Retrieval
# ============================================================

# Embedding dimension for BGE-M3 (BAAI/bge-m3).
# Must match the FAISS index dimension — do NOT change without rebuilding the index.
BGE_M3_EMBEDDING_DIM = 1_024

# Number of candidates retrieved before cross-encoder reranking.
TOP_K_INITIAL = 50
# Final number of evidence chunks returned to the agents after reranking.
TOP_K_FINAL = 5

# Default embedding model identifier.
BGE_M3_MODEL_NAME = "BAAI/bge-m3"
# Default cross-encoder model identifier.
BGE_RERANKER_MODEL_NAME = "BAAI/bge-reranker-v2-m3"

# ============================================================
# Multi-agent pipeline
# ============================================================

# Maximum number of iterative retrieval passes the orchestrator will perform.
# Increasing this improves recall but raises latency linearly.
MAX_RETRIEVAL_ITERATIONS = 3

# Maximum critic–revision rounds before the orchestrator forces abstention.
MAX_CRITIC_ROUNDS = 2

# ============================================================
# Confidence calibration
# ============================================================

# Minimum calibrated confidence required to serve an answer.
# Below this threshold the Abstention Agent returns an abstention response.
CONFIDENCE_THRESHOLD = 0.65

# Confidence bands (for display and reporting).
CONFIDENCE_HIGH_THRESHOLD = 0.80    # score >= 0.80 → "high"
CONFIDENCE_MODERATE_THRESHOLD = 0.65  # score >= 0.65 → "moderate"; else "low"

# Weighted-sum component weights for ConfidenceCalibrationAgent.
# Must sum to 1.0.
CONFIDENCE_RETRIEVAL_WEIGHT = 0.25
CONFIDENCE_EVIDENCE_QUALITY_WEIGHT = 0.30
CONFIDENCE_ANSWER_CONSISTENCY_WEIGHT = 0.25
CONFIDENCE_INTER_SOURCE_AGREEMENT_WEIGHT = 0.20

assert (
    abs(
        CONFIDENCE_RETRIEVAL_WEIGHT
        + CONFIDENCE_EVIDENCE_QUALITY_WEIGHT
        + CONFIDENCE_ANSWER_CONSISTENCY_WEIGHT
        + CONFIDENCE_INTER_SOURCE_AGREEMENT_WEIGHT
        - 1.0
    ) < 1e-9
), "Confidence calibration weights must sum to 1.0"

# ============================================================
# Abstention safety thresholds
# ============================================================

# Individual component thresholds — below any of these the Abstention Agent
# will ABSTAIN regardless of the composite confidence score.
ABSTENTION_MIN_EVIDENCE_SCORE = 0.40
ABSTENTION_MIN_FACTUAL_CONSISTENCY = 0.50
ABSTENTION_MIN_CITATION_SUPPORT = 0.50
ABSTENTION_MIN_HALLUCINATION_SAFETY = 0.50  # 1 - hallucination_risk

# ============================================================
# Token / context limits
# ============================================================

# Maximum context window for the LLM (tokens).
LLM_MAX_CONTEXT_TOKENS = 8_192
# Default max new tokens generated per LLM call.
LLM_DEFAULT_MAX_NEW_TOKENS = 512

# ============================================================
# Evaluation
# ============================================================

# Default number of equal-width bins for ECE computation.
ECE_DEFAULT_N_BINS = 10

# Default nDCG cutoff rank.
NDCG_DEFAULT_K = 5

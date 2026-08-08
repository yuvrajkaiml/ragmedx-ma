from pathlib import Path

# Project Root
BASE_DIR = Path(__file__).resolve().parent.parent

# Data Directories

DATA_DIR = BASE_DIR / "data"

DATASETS_DIR = DATA_DIR / "datasets"
DOCUMENTS_DIR = DATA_DIR / "documents"

EMBEDDINGS_DIR = DATA_DIR / "embeddings"
FAISS_INDEX_DIR = DATA_DIR / "faiss_index"

CACHE_DIR = DATA_DIR / "cache"

RESULTS_DIR = DATA_DIR / "results"
EXPERIMENTS_DIR = DATA_DIR / "experiments"

# Retrieval Parameters
CHUNK_SIZE = 2000
CHUNK_OVERLAP = 400

TOP_K_INITIAL = 50
TOP_K_FINAL = 5

MAX_ITERATIONS = 3
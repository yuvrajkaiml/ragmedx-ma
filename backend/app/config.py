# Owner: Pranav Sukale — not yet implemented, see Task_Assignment.md

from pathlib import Path

# Project Root
BASE_DIR = Path(__file__).resolve().parent.parent

# Data Directories
DATA_DIR = BASE_DIR / "data"
DOCUMENTS_DIR = DATA_DIR / "documents"
FAISS_INDEX_DIR = DATA_DIR / "faiss_index"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
CACHE_DIR = DATA_DIR / "cache"

# Retrieval Parameters
CHUNK_SIZE = 2000
CHUNK_OVERLAP = 400

TOP_K_INITIAL = 50
TOP_K_FINAL = 5

MAX_ITERATIONS = 3
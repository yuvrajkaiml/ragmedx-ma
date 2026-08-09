from pathlib import Path

from app.retrieval.document_loader import DocumentLoader
from app.config import DATASETS_DIR

loader = DocumentLoader(DATASETS_DIR)

files = loader.load_medqa()

print(files)
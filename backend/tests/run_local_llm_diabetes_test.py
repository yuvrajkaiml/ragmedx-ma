"""Small manual test for the local Qwen3-8B wrapper."""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.llm.local_llm import LocalLLMWrapper


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    answer = LocalLLMWrapper().generate("What is diabetes?")
    print(answer)

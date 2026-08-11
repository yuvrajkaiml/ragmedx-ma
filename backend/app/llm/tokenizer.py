import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


def count_tokens(text: str, tokenizer: Optional[Any] = None) -> int:
    """
    Count number of tokens in a text string.

    Args:
        text: Input string.
        tokenizer: HuggingFace PreTrainedTokenizer Base instance or None.

    Returns:
        Integer token count.
    """
    if not text:
        return 0

    if tokenizer is not None:
        try:
            tokens = tokenizer.encode(text, add_special_tokens=False)
            return len(tokens)
        except Exception as e:
            logger.warning(f"Error using model tokenizer for token count: {e}. Falling back to character estimate.")

    # Fallback heuristic: ~4 characters per token for English/medical text
    return max(1, len(text) // 4)


def truncate_to_max_tokens(text: str, max_tokens: int, tokenizer: Optional[Any] = None) -> str:
    """
    Truncate text to fit within a maximum token count.

    Args:
        text: Input string.
        max_tokens: Maximum allowed tokens.
        tokenizer: HuggingFace PreTrainedTokenizer Base instance or None.

    Returns:
        Truncated string.
    """
    if not text or max_tokens <= 0:
        return ""

    if tokenizer is not None:
        try:
            tokens = tokenizer.encode(text, add_special_tokens=False)
            if len(tokens) <= max_tokens:
                return text
            truncated_tokens = tokens[:max_tokens]
            return tokenizer.decode(truncated_tokens, skip_special_tokens=True)
        except Exception as e:
            logger.warning(f"Error truncating with model tokenizer: {e}. Falling back to character estimate.")

    # Fallback: ~4 chars per token
    max_chars = max_tokens * 4
    return text[:max_chars]

import json
import re
from typing import Any, Dict, Tuple, Union


class LLMError(Exception):
    """Base exception for LLM-related operations."""
    pass


class ModelLoadError(LLMError):
    """Raised when the local LLM cannot be loaded."""
    pass


class LLMGenerationError(LLMError):
    """Raised when generation fails."""
    pass


class LLMParseError(LLMError):
    """Raised when parsing LLM response into structured JSON fails."""
    pass


def extract_raw_text(raw_response: Union[str, Tuple[str, Any], Dict[str, Any]]) -> str:
    """Extract string text from various LLM response signatures."""
    if isinstance(raw_response, str):
        return raw_response
    if isinstance(raw_response, tuple) and len(raw_response) > 0:
        return str(raw_response[0])
    if isinstance(raw_response, dict):
        if "text" in raw_response:
            return str(raw_response["text"])
        if "response" in raw_response:
            return str(raw_response["response"])
        if "answer" in raw_response:
            return str(raw_response["answer"])
    return str(raw_response)


def parse_json_from_llm(raw_response: Union[str, Tuple[str, Any], Dict[str, Any]]) -> Dict[str, Any]:
    """
    Robustly extract and parse JSON object from LLM output.

    Handles:
    - Accidental markdown code fences (```json ... ``` or ``` ... ```)
    - Conversational preambles or postscripts around JSON
    - Trailing commas before closing braces
    - Whitespace trimming
    """
    text = extract_raw_text(raw_response).strip()

    if not text:
        raise LLMParseError("LLM response is empty.")

    # 1. Strip markdown code fences if wrapped
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
    if fence_match:
        text = fence_match.group(1).strip()

    # 2. Try direct JSON parsing
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass

    # 3. Extract outermost JSON object { ... } if text contains surrounding preambles
    start_idx = text.find("{")
    end_idx = text.rfind("}")
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        json_candidate = text[start_idx : end_idx + 1].strip()
        try:
            data = json.loads(json_candidate)
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            # Try removing trailing commas before } or ]
            cleaned = re.sub(r",\s*([}\]])", r"\1", json_candidate)
            try:
                data = json.loads(cleaned)
                if isinstance(data, dict):
                    return data
            except json.JSONDecodeError:
                pass

    raise LLMParseError(f"Failed to parse valid JSON object from LLM response: {text[:200]}")

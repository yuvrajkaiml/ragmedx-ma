import logging
import os
import time
from enum import Enum
from typing import Any, Dict, Optional, Tuple, Union

from app.llm.parser import LLMError, LLMGenerationError, ModelLoadError
from app.llm.tokenizer import count_tokens, truncate_to_max_tokens

logger = logging.getLogger(__name__)


class LLMRuntimeMode(str, Enum):
    MOCK = "mock"
    LOCAL = "local"
    AUTO = "auto"


class LocalLLMWrapper:
    """
    Local LLM Wrapper for Qwen3 / Qwen2.5 / Llama 3.1 supporting 4-bit quantization
    via bitsandbytes (NF4), memory optimization, token counting, context truncation,
    and explicit mock/local runtime modes.
    """

    def __init__(
        self,
        model_name_or_path: str = "Qwen/Qwen3-8B",
        quantization_bits: int = 4,
        device_map: str = "cuda:0",
        max_context_length: int = 8192,
        mode: Union[LLMRuntimeMode, str] = LLMRuntimeMode.AUTO,
        allow_fallback: bool = False,
        use_mock_fallback: Optional[bool] = None,
    ):
        self.model_name_or_path = model_name_or_path
        self.quantization_bits = quantization_bits
        self.device_map = device_map
        self.max_context_length = max_context_length

        # Handle backward compatibility parameter use_mock_fallback
        if use_mock_fallback is True:
            self.mode = LLMRuntimeMode.MOCK
            self.allow_fallback = True
        elif use_mock_fallback is False:
            self.mode = LLMRuntimeMode.LOCAL
            self.allow_fallback = False
        else:
            self.mode = LLMRuntimeMode(mode) if isinstance(mode, str) else mode
            self.allow_fallback = allow_fallback

        self.model = None
        self.tokenizer = None
        self.is_loaded = False
        self.active_runtime_mode = self.mode

    def load_model(self) -> bool:
        """
        Initialize and load model and tokenizer according to the configured runtime mode.

        - MOCK mode: Initializes mock generator immediately.
        - LOCAL mode: Strictly attempts local load; raises ModelLoadError on failure (no silent fallback).
        - AUTO mode: Attempts local load; falls back to mock only if allow_fallback=True, otherwise raises ModelLoadError.
        """
        if self.mode == LLMRuntimeMode.MOCK:
            logger.info("LocalLLMWrapper initialized explicitly in MOCK mode.")
            self.active_runtime_mode = LLMRuntimeMode.MOCK
            self.is_loaded = True
            return True

        # LOCAL or AUTO mode: Attempt real model loading
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

            if not torch.cuda.is_available():
                raise ModelLoadError(
                    f"CUDA is not available on this host. Cannot load {self.model_name_or_path} for GPU inference."
                )

            logger.info(f"Loading tokenizer for {self.model_name_or_path}...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name_or_path,
                trust_remote_code=True,
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            quantization_config = None
            if self.quantization_bits == 4:
                try:
                    # Force the native extension import before downloading/loading
                    # model weights. Windows Application Control can block these
                    # DLLs; a blocked backend must be repaired by policy approval,
                    # not hidden by a memory-unsafe full-precision retry.
                    import bitsandbytes  # noqa: F401
                    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_compute_dtype=compute_dtype,
                    )
                    logger.info("Configured 4-bit bitsandbytes NF4 quantization.")
                except Exception as bnb_err:
                    raise ModelLoadError(
                        "bitsandbytes 4-bit NF4 initialization failed. Do not retry "
                        "this 8 GB deployment in unquantized mode; verify the approved "
                        f"Windows bitsandbytes DLL installation. Original error: {bnb_err}"
                    ) from bnb_err

            logger.info(f"Loading model {self.model_name_or_path}...")
            torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name_or_path,
                quantization_config=quantization_config,
                device_map=self.device_map,
                torch_dtype=torch_dtype,
                trust_remote_code=True,
            )

            self.is_loaded = True
            self.active_runtime_mode = LLMRuntimeMode.LOCAL
            logger.info("Local LLM model loaded successfully into VRAM.")
            return True

        except Exception as e:
            logger.error(f"Failed to load local model '{self.model_name_or_path}': {e}")
            if self.mode == LLMRuntimeMode.LOCAL or not self.allow_fallback:
                self.is_loaded = False
                raise ModelLoadError(
                    f"Failed to load local model '{self.model_name_or_path}': {e}. "
                    "Silent fallback is disabled. Set mode='mock' or allow_fallback=True for mock execution."
                ) from e

            # AUTO mode with allow_fallback=True
            logger.warning(
                f"Falling back to MOCK mode due to model loading error: {e}. "
                "Active runtime is explicitly marked as MOCK."
            )
            self.active_runtime_mode = LLMRuntimeMode.MOCK
            self.is_loaded = True
            return False

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.2,
        top_p: float = 0.95,
        system_prompt: Optional[str] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Generate text response from LLM.

        Args:
            prompt: User/Agent prompt string.
            max_new_tokens: Max new tokens to generate.
            temperature: Sampling temperature.
            top_p: Nucleus sampling top_p.
            system_prompt: Optional system instruction.

        Returns:
            Tuple of (generated_text_string, token_usage_dict).
        """
        if not self.is_loaded:
            self.load_model()

        full_prompt = prompt
        if system_prompt:
            full_prompt = f"System: {system_prompt}\nUser: {prompt}\nAssistant:"

        input_tokens_count = self.count_tokens(full_prompt)

        # Handle MOCK runtime mode
        if self.active_runtime_mode == LLMRuntimeMode.MOCK or self.model is None:
            mock_response = self._generate_mock_response(prompt)
            output_tokens_count = self.count_tokens(mock_response)
            token_usage = {
                "prompt_tokens": input_tokens_count,
                "completion_tokens": output_tokens_count,
                "total_tokens": input_tokens_count + output_tokens_count,
                "mode": LLMRuntimeMode.MOCK.value,
                "is_mock": True,
                "model_name": self.model_name_or_path,
            }
            return mock_response, token_usage

        # Handle LOCAL real inference
        try:
            import torch

            max_input_tokens = self.max_context_length - max_new_tokens
            truncated_prompt = truncate_to_max_tokens(full_prompt, max_input_tokens, self.tokenizer)

            # Qwen and other instruction-tuned chat models expect their native
            # chat template. Raw completion prompts make structured control-plane
            # agents prepend prose before JSON, which can exhaust their bounded
            # response budget. Keep the raw path only for tokenizers without a
            # usable template.
            chat_template = getattr(self.tokenizer, "chat_template", None)
            if chat_template:
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                    # ``full_prompt`` already embeds system text for legacy raw
                    # prompts; give chat models the clean user content instead.
                    truncated_prompt = truncate_to_max_tokens(prompt, max_input_tokens, self.tokenizer)
                messages.append({"role": "user", "content": truncated_prompt})
                model_prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            else:
                model_prompt = truncated_prompt

            inputs = self.tokenizer(model_prompt, return_tensors="pt").to(self.model.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature if temperature > 0 else 1.0,
                    do_sample=temperature > 0,
                    top_p=top_p,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

            generated_ids = outputs[0][inputs["input_ids"].shape[1] :]
            generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

            output_tokens_count = len(generated_ids)
            token_usage = {
                "prompt_tokens": len(inputs["input_ids"][0]),
                "completion_tokens": output_tokens_count,
                "total_tokens": len(inputs["input_ids"][0]) + output_tokens_count,
                "mode": LLMRuntimeMode.LOCAL.value,
                "is_mock": False,
                "model_name": self.model_name_or_path,
            }

            return generated_text, token_usage

        except Exception as e:
            logger.error(f"Error during local LLM generation: {e}")
            raise LLMGenerationError(f"Local LLM generation failed: {e}") from e

    def _generate_mock_response(self, prompt: str) -> str:
        """Deterministic mock responses tailored to standard agent prompt types."""
        prompt_lower = prompt.lower()

        # Query Understanding prompt
        if "query understanding" in prompt_lower or "medical specialty" in prompt_lower:
            return (
                '{\n'
                '    "medical_specialty": "general_medicine",\n'
                '    "intent": "treatment_inquiry",\n'
                '    "expanded_query": "first-line medical management and evidence",\n'
                '    "abbreviations": {},\n'
                '    "confidence": 0.90\n'
                '}'
            )

        # Critic prompt
        if "medical answer critic" in prompt_lower or "factual consistency" in prompt_lower:
            return (
                '{\n'
                '    "passed": true,\n'
                '    "factual_consistency": 0.90,\n'
                '    "citation_support": 0.88,\n'
                '    "hallucination_risk": 0.05,\n'
                '    "issues": [],\n'
                '    "revision_instructions": null,\n'
                '    "confidence": 0.92\n'
                '}'
            )

        # Gap analysis / Iterative retrieval prompt
        if "gap analysis" in prompt_lower or "sufficient to safely answer" in prompt_lower:
            return (
                '{\n'
                '    "sufficient": true,\n'
                '    "reasoning": "The supplied evidence is adequate to answer the clinical inquiry.",\n'
                '    "missing_information": [],\n'
                '    "sub_query": null,\n'
                '    "confidence": 0.88\n'
                '}'
            )

        # Evidence Re-Ranking Agent prompt
        if "evidence re-ranking agent" in prompt_lower:
            return (
                '{\n'
                '    "contradictory": false,\n'
                '    "selected_markers": [],\n'
                '    "reasoning": "No material contradiction identified in the supplied evidence."\n'
                '}'
            )

        # Generator or general answer
        return f"[MOCK_GENERATED_ANSWER for: {prompt[:60]}...]"

    def count_tokens(self, text: str) -> int:
        """Token counting helper."""
        return count_tokens(text, self.tokenizer)

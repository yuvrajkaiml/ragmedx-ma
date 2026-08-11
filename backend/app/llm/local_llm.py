import logging
import os
import time
from typing import Any, Dict, Optional, Tuple

from app.llm.tokenizer import count_tokens, truncate_to_max_tokens

logger = logging.getLogger(__name__)


class LocalLLMWrapper:
    """
    Local LLM Wrapper for Qwen3 / Qwen2.5 / Llama 3.1 supporting 4-bit quantization
    via bitsandbytes (NF4), memory optimization, token counting, and context truncation.
    """

    def __init__(
        self,
        model_name_or_path: str = "Qwen/Qwen3-8B",
        quantization_bits: int = 4,
        device_map: str = "auto",
        max_context_length: int = 8192,
        load_in_8bit: bool = False,
        use_mock_fallback: bool = False
    ):
        self.model_name_or_path = model_name_or_path
        self.quantization_bits = quantization_bits
        self.device_map = device_map
        self.max_context_length = max_context_length
        self.use_mock_fallback = use_mock_fallback

        self.model = None
        self.tokenizer = None
        self.is_loaded = False

    def load_model(self) -> bool:
        """
        Initialize and load HuggingFace Transformers model and tokenizer.
        Supports 4-bit bitsandbytes quantization with fallback to FP16/BF16 on Windows App Control policy limits.
        """
        if self.use_mock_fallback:
            logger.info("LocalLLMWrapper initialized in mock mode.")
            self.is_loaded = True
            return True

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

            logger.info(f"Loading tokenizer for {self.model_name_or_path}...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name_or_path,
                trust_remote_code=True
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            quantization_config = None
            if self.quantization_bits == 4:
                try:
                    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_compute_dtype=compute_dtype
                    )
                    logger.info("Configured 4-bit bitsandbytes NF4 quantization.")
                except Exception as bnb_err:
                    logger.warning(f"bitsandbytes configuration error: {bnb_err}. Falling back to native PyTorch float16/bfloat16.")
                    quantization_config = None

            logger.info(f"Loading model {self.model_name_or_path}...")
            torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name_or_path,
                    quantization_config=quantization_config,
                    device_map=self.device_map,
                    torch_dtype=torch_dtype,
                    trust_remote_code=True
                )
            except Exception as load_err:
                if quantization_config is not None:
                    logger.warning(f"Model load with bitsandbytes failed ({load_err}). Retrying with native PyTorch {torch_dtype}...")
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_name_or_path,
                        quantization_config=None,
                        device_map=self.device_map,
                        torch_dtype=torch_dtype,
                        trust_remote_code=True
                    )
                else:
                    raise load_err

            self.is_loaded = True
            logger.info("Local LLM model loaded successfully.")
            return True

        except Exception as e:
            logger.warning(
                f"Failed to load model '{self.model_name_or_path}': {e}. "
                "Falling back to mock generation mode for testing/scaffolding."
            )
            self.use_mock_fallback = True
            self.is_loaded = True
            return False

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.2,
        top_p: float = 0.95,
        system_prompt: Optional[str] = None
    ) -> Tuple[str, Dict[str, int]]:
        """
        Generate text response from LLM.

        Args:
            prompt: User/Agent prompt string.
            max_new_tokens: Max new tokens to generate.
            temperature: Sampling temperature (0.0 to 1.0).
            top_p: Nucleus sampling top_p.
            system_prompt: Optional system instruction.

        Returns:
            Tuple of (generated_text_string, token_usage_dict).
        """
        if not self.is_loaded:
            self.load_model()

        # Combine system prompt if provided
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"System: {system_prompt}\nUser: {prompt}\nAssistant:"

        input_tokens_count = count_tokens(full_prompt, self.tokenizer)

        if self.use_mock_fallback or self.model is None:
            # Mock generation response
            mock_response = f"[Mock Generated Response for prompt: {prompt[:60]}...]"
            output_tokens_count = count_tokens(mock_response, self.tokenizer)
            token_usage = {
                "prompt_tokens": input_tokens_count,
                "completion_tokens": output_tokens_count,
                "total_tokens": input_tokens_count + output_tokens_count
            }
            return mock_response, token_usage

        try:
            import torch

            # Truncate prompt if exceeding context limit
            max_input_tokens = self.max_context_length - max_new_tokens
            truncated_prompt = truncate_to_max_tokens(full_prompt, max_input_tokens, self.tokenizer)

            inputs = self.tokenizer(truncated_prompt, return_tensors="pt").to(self.model.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature if temperature > 0 else 1.0,
                    do_sample=temperature > 0,
                    top_p=top_p,
                    pad_token_id=self.tokenizer.pad_token_id
                )

            # Decode output excluding prompt tokens
            generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
            generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

            output_tokens_count = len(generated_ids)
            token_usage = {
                "prompt_tokens": len(inputs["input_ids"][0]),
                "completion_tokens": output_tokens_count,
                "total_tokens": len(inputs["input_ids"][0]) + output_tokens_count
            }

            return generated_text, token_usage

        except Exception as e:
            logger.error(f"Error during LLM generation: {e}")
            fallback_msg = f"[Generation Error: {e}]"
            return fallback_msg, {"prompt_tokens": input_tokens_count, "completion_tokens": 10, "total_tokens": input_tokens_count + 10}

    def count_tokens(self, text: str) -> int:
        """
        Token counting helper.
        """
        return count_tokens(text, self.tokenizer)

"""
Local Hugging Face LLM wrapper for RAGMedX-MA.

Supports:
- CUDA/NVIDIA GPU when available
- CPU fallback when CUDA is unavailable

Default model:
    Qwen/Qwen2.5-1.5B-Instruct
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class LocalLLMWrapper:
    """Load a local Hugging Face model and generate responses."""

    def __init__(
        self,
        model_name_or_path: str = "Qwen/Qwen2.5-1.5B-Instruct",
    ) -> None:
        self.model_name_or_path = model_name_or_path
        self.model = None
        self.tokenizer = None
        self.device = None

    def load_model(self) -> None:
        """Load the tokenizer and model on CUDA when available, otherwise CPU."""

        if self.model is not None and self.tokenizer is not None:
            return

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            # ---------------------------------------------------------
            # Select device
            # ---------------------------------------------------------
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
                logger.info("CUDA detected. Loading local LLM on GPU.")

                if torch.cuda.is_bf16_supported():
                    dtype = torch.bfloat16
                else:
                    dtype = torch.float16

            else:
                self.device = torch.device("cpu")
                dtype = torch.float32
                logger.info(
                    "CUDA not available. Loading local LLM on CPU."
                )

            logger.info(
                "Loading tokenizer for %s",
                self.model_name_or_path,
            )

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name_or_path
            )

            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            logger.info(
                "Loading model with device=%s, dtype=%s",
                self.device,
                dtype,
            )

            # ---------------------------------------------------------
            # Load model
            # ---------------------------------------------------------
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name_or_path,
                torch_dtype=dtype,
            )

            self.model.to(self.device)
            self.model.eval()

            logger.info(
                "Local model loaded successfully on %s.",
                self.device,
            )

        except Exception as exc:
            self.model = None
            self.tokenizer = None
            self.device = None

            logger.exception("Unable to load model.")

            raise RuntimeError(
                f"Unable to load model '{self.model_name_or_path}'."
            ) from exc

    def generate(
        self,
        prompt: str,
        temperature: float = 0.2,
        max_new_tokens: int = 256,
    ) -> str:
        """Generate a response from the local LLM."""

        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Prompt must be a non-empty string.")

        if temperature < 0:
            raise ValueError("Temperature cannot be negative.")

        if max_new_tokens <= 0:
            raise ValueError(
                "max_new_tokens must be greater than zero."
            )

        # Load model lazily.
        if self.model is None or self.tokenizer is None:
            self.load_model()

        try:
            import torch

            # ---------------------------------------------------------
            # Build chat prompt
            # ---------------------------------------------------------
            messages = [
                {
                    "role": "user",
                    "content": prompt,
                }
            ]

            logger.info("Applying chat template...")

            if hasattr(self.tokenizer, "apply_chat_template"):
                rendered_prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            else:
                # Fallback for tokenizers without a chat template.
                rendered_prompt = prompt

            # ---------------------------------------------------------
            # Tokenize
            # ---------------------------------------------------------
            logger.info("Tokenizing prompt...")

            inputs = self.tokenizer(
                rendered_prompt,
                return_tensors="pt",
            )

            # Move tensors to the same device as the model.
            inputs = {
                key: value.to(self.device)
                for key, value in inputs.items()
            }

            logger.info(
                "Starting generation on %s...",
                self.device,
            )

            # ---------------------------------------------------------
            # Generate
            # ---------------------------------------------------------
            with torch.inference_mode():

                generation_kwargs = {
                    "max_new_tokens": max_new_tokens,
                    "pad_token_id": self.tokenizer.pad_token_id,
                    "eos_token_id": self.tokenizer.eos_token_id,
                }

                if temperature > 0:
                    generation_kwargs.update(
                        {
                            "do_sample": True,
                            "temperature": temperature,
                        }
                    )
                else:
                    generation_kwargs["do_sample"] = False

                output_ids = self.model.generate(
                    **inputs,
                    **generation_kwargs,
                )

            logger.info("Generation finished.")

            # ---------------------------------------------------------
            # Remove original prompt tokens
            # ---------------------------------------------------------
            input_length = inputs["input_ids"].shape[1]

            response_ids = output_ids[0][input_length:]

            response = self.tokenizer.decode(
                response_ids,
                skip_special_tokens=True,
            ).strip()

            logger.info("Generated response successfully.")

            return response

        except Exception as exc:
            logger.exception("Generation failed.")

            raise RuntimeError(
                "Failed to generate response."
            ) from exc
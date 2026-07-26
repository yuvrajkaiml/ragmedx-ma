"""Local Hugging Face wrapper for the Qwen2.5-1.5B-Instruct language model."""

import logging

logger = logging.getLogger(__name__)


class LocalLLMWrapper:
    """Load a local Hugging Face model and generate responses."""

    def __init__(self, model_name_or_path: str = "Qwen/Qwen2.5-1.5B-Instruct") -> None:
        self.model_name_or_path = model_name_or_path
        self.model = None
        self.tokenizer = None

    def load_model(self) -> None:
        """Load the tokenizer and model."""
        if self.model is not None and self.tokenizer is not None:
            return

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            if not torch.cuda.is_available():
                raise RuntimeError("CUDA-capable GPU not found.")

            logger.info("Loading tokenizer for %s", self.model_name_or_path)
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name_or_path
            )

            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            dtype = (
                torch.bfloat16
                if torch.cuda.is_bf16_supported()
                else torch.float16
            )

            logger.info("Loading model...")

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name_or_path,
                device_map="auto",
                dtype=dtype,
            )

            self.model.eval()

            logger.info("Local model loaded successfully.")

        except Exception as exc:
            self.model = None
            self.tokenizer = None
            logger.exception("Unable to load model.")
            raise RuntimeError(
                f"Unable to load model '{self.model_name_or_path}'"
            ) from exc

    def generate(
    self,
    prompt: str,
    temperature: float = 0.2,
    max_new_tokens: int = 256,
) -> str:
        """Generate a response."""

        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Prompt must be a non-empty string.")

        if self.model is None or self.tokenizer is None:
            self.load_model()

        try:
            import torch

            messages = [
                {
                    "role": "user",
                    "content": prompt,
                }
            ]

            logger.info("Applying chat template...")

            rendered_prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            logger.info("Tokenizing...")

            inputs = self.tokenizer(
                rendered_prompt,
                return_tensors="pt",
            )

            device = next(self.model.parameters()).device

            logger.info("Moving inputs to %s", device)

            inputs = {
                k: v.to(device)
                for k, v in inputs.items()
            }

            logger.info("Starting generation...")

            with torch.inference_mode():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=temperature > 0,
                    temperature=temperature,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

            logger.info("Generation finished.")

            response_ids = output_ids[0][inputs["input_ids"].shape[1]:]

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
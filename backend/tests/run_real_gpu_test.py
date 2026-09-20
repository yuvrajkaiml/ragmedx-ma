import sys
import os
import time

# Ensure backend root is on sys.path
backend_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, backend_path)

import torch
from app.llm.local_llm import LLMRuntimeMode, LocalLLMWrapper

def run_test():
    print("=== REAL GPU EXECUTION TEST ===")
    print(f"PyTorch Version: {torch.__version__}")
    cuda_avail = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_avail}")
    if cuda_avail:
        device_name = torch.cuda.get_device_name(0)
        print(f"GPU Device Name: {device_name}")
    else:
        print("ERROR: CUDA is not available.")
        sys.exit(1)

    model_id = "Qwen/Qwen3-8B"
    print(f"\nTarget Model: {model_id}")
    print("Starting model download and initialization (use_mock_fallback=False)...")
    
    start_time = time.time()
    
    llm = LocalLLMWrapper(
        model_name_or_path=model_id,
        quantization_bits=4,
        device_map="auto",
        use_mock_fallback=False
    )
    
    loaded_ok = llm.load_model()
    if not loaded_ok or llm.active_runtime_mode != LLMRuntimeMode.LOCAL:
        print("Model load failed or did not remain in explicit local mode.")
        sys.exit(1)

    print("\nModel loaded successfully into VRAM. Executing inference generation...")
    prompt = "What is the first-line medical treatment for uncomplicated essential hypertension?"
    
    response, token_usage = llm.generate(
        prompt=prompt,
        max_new_tokens=128,
        temperature=0.2
    )

    mem_allocated_gb = torch.cuda.memory_allocated(0) / (1024 ** 3)
    mem_reserved_gb = torch.cuda.memory_reserved(0) / (1024 ** 3)
    
    end_time = time.time()
    total_wall_clock_time = end_time - start_time

    print("\n=== GENERATION METRICS & RESULTS ===")
    print(f"1. PyTorch Version: {torch.__version__} (CUDA: {torch.cuda.is_available()})")
    print(f"2. GPU Device: {torch.cuda.get_device_name(0)}")
    print(f"3. Generated Response:\n{response}\n")
    print(f"4. Token Usage: {token_usage}")
    print(f"5. VRAM Memory Allocated: {mem_allocated_gb:.3f} GB")
    print(f"6. VRAM Memory Reserved:  {mem_reserved_gb:.3f} GB")
    print(f"7. Total Wall-Clock Time: {total_wall_clock_time:.2f} seconds ({total_wall_clock_time / 60:.2f} minutes)")
    print("=== TEST COMPLETED SUCCESSFULLY WITH ZERO OOM ERRORS ===")

if __name__ == "__main__":
    run_test()

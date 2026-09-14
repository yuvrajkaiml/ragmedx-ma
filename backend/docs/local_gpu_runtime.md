# Local GPU runtime

The shared baseline in `requirements.txt` targets Windows CUDA 12.4 and remains the
default for existing developer environments. It is not compatible with an sm_120 RTX
5060: PyTorch 2.6.0+cu124 ships kernels only through sm_90. This is a versioned-wheel
limitation, not a general Windows or PyTorch limitation.

For a local RTX 50-series development environment, use a separate Python 3.11 virtual
environment and the committed CUDA 12.8 overlay. It deliberately does not replace the
shared baseline for teammates without a compatible GPU.

```powershell
Set-Location D:\Ragmed_2\backend
py -3.11 -m venv ..\.venv-cu128
..\.venv-cu128\Scripts\python.exe -m pip install --upgrade pip
..\.venv-cu128\Scripts\python.exe -m pip install -r requirements.txt
..\.venv-cu128\Scripts\python.exe -m pip install --upgrade -r requirements-local-gpu-cu128.txt
```

Verify the runtime before loading a model:

```powershell
..\.venv-cu128\Scripts\python.exe -c "import torch, bitsandbytes as bnb; print(torch.__version__, torch.version.cuda, torch.cuda.get_arch_list(), torch.cuda.is_available()); print(bnb.__version__)"
```

`DEBUG=false` is the normal local/test setting. `DEBUG=true` enables SQLAlchemy
engine echo, which logs every database statement and is useful only while diagnosing
database access.

## Windows Application Control

`bitsandbytes` loads native CUDA DLLs. If Application Control blocks one of those
DLLs, do **not** fall back silently to an unquantized model: that can exceed the
available VRAM. Capture the blocked DLL path and policy event, then ask the machine
administrator to allow the signed/vendor-approved wheel and its DLLs by publisher or
hash. After approval, reinstall the version in the selected runtime overlay and rerun the verification
command above followed by:

```powershell
..\.venv-cu128\Scripts\python.exe tests\run_real_gpu_test.py
```

## Verified installation result (2026-09-15)

The original `D:\Ragmed_2\.venv-cu124` environment (Python 3.11, `torch
2.6.0+cu124`, `bitsandbytes 0.45.5`) detected the RTX 5060 but failed real NF4
quantization with `CUDA error: no kernel image is available for execution on the
device`. Its PyTorch wheel ended at sm_90.

The local `D:\Ragmed_2\.venv-cu128` environment was verified on 2026-09-15 with
Python 3.11, `torch 2.11.0+cu128`, and `bitsandbytes 0.50.2`. PyTorch reported
`sm_120` in its architecture list; `tests\\run_real_gpu_test.py` loaded
`Qwen/Qwen3-8B` using NF4, generated a non-mock response, and completed without an
OOM or kernel-image error. Peak allocated VRAM was 5.667 GB (5.873 GB reserved) on
the 8 GB RTX 5060 Laptop GPU. WSL2 is therefore unnecessary for this host. The gap
was specifically the older CUDA 12.4/bitsandbytes wheel set, not Windows PyTorch
support for sm_120.

## Model-fit decision

Qwen3-14B at NF4 is not supported on the 8 GB RTX 5060 deployment target: weights
alone are roughly 7–8 GB after quantization and runtime/KV-cache overhead leaves no
safe headroom. Until a larger GPU is provisioned, use `Qwen/Qwen3-8B` with a small
context/window and retain the model choice in experiment metadata. This is a capacity
constraint, not an unrecorded model downgrade.

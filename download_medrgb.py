
from huggingface_hub import snapshot_download
from pathlib import Path

# Save MedRGB alongside your existing project datasets
dataset_path = Path("backend/data/datasets/medrgb")
dataset_path.mkdir(parents=True, exist_ok=True)

snapshot_download(
    repo_id="ngotrnghia1811/MedRGB",
    repo_type="dataset",
    local_dir=str(dataset_path),
)

print(f"MedRGB downloaded to: {dataset_path.resolve()}")

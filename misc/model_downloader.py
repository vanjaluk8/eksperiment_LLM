from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="mistralai/Mistral-7B-Instruct-v0.3",
    local_dir="/raid/models/hgf_models/Mistral-7B-Instruct-v0.3",
    local_dir_use_symlinks=False  # ensures actual files, not symlinks
)
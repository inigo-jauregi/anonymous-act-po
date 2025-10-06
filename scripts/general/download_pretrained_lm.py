import os
from huggingface_hub import snapshot_download

model_name = "Qwen/Qwen3-8B"

# Set the destination folder on your desktop
# Replace 'your_username' with your actual username
local_path = f"./pretrained_lms/{model_name.replace('/', '-')}"

# Create the directory if it doesn't exist
os.makedirs(local_path, exist_ok=True)

print(f"Starting download of pretrained model to {local_path}")

# Download the model files
snapshot_download(
    repo_id=model_name,
    cache_dir=None,
    local_dir=local_path,
    local_dir_use_symlinks=False  # Set to True if you want to use symlinks instead of copying files
)

print("\nDownload complete! The model is now saved at:", local_path)

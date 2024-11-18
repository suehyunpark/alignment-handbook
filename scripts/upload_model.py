from huggingface_hub import HfApi
import os

# Define paths
local_dir = "suehyunpark/potpourri-8b-inst-fft-induction-bc-optimal-action-max1-per-task"
repo_id = "suehyunpark/potpourri-8b-inst-fft-induction-bc-optimal-action-max1-per-task"

# Initialize API
api = HfApi()

# Upload all files (excluding checkpoints directory and hidden files)
files_to_upload = [
    f for f in os.listdir(local_dir)
    if not f.startswith('.') and not f.startswith('checkpoint-') 
    and os.path.isfile(os.path.join(local_dir, f))
]

for file in files_to_upload:
    file_path = os.path.join(local_dir, file)
    print(f"Uploading {file}")
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file,
        repo_id=repo_id,
        repo_type="model",
        commit_message=f"Upload after 3 epochs",
    )

print("Upload complete!")
#!/usr/bin/env python3
"""
Push the monolithic kernel code to the Hugging Face Model Hub as source code in a model repo.
"""
import os
from huggingface_hub import HfApi, HfFolder, upload_folder

def main():
    repo_id = input("Enter your Hugging Face model repo (e.g. username/model-name): ").strip()
    kernel_dir = input("Enter path to kernel code directory (e.g. kernels/monolithic): ").strip()
    token = os.environ.get("HF_TOKEN") or input("Enter your Hugging Face token: ").strip()

    api = HfApi()
    HfFolder.save_token(token)
    print(f"Pushing {kernel_dir} to {repo_id}/kernels ...")
    upload_folder(
        repo_id=repo_id,
        folder_path=kernel_dir,
        repo_type="model",
        token=token,
        path_in_repo="kernels/monolithic",
        commit_message="Upload monolithic MI300X kernel code"
    )
    print("✅ Kernel code pushed to Hugging Face Model Hub!")

if __name__ == "__main__":
    main() 
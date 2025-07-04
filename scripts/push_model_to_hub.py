#!/usr/bin/env python3
"""
Push the exported Kernel Tonic model to the Hugging Face Model Hub.
"""
from huggingface_hub import HfApi, HfFolder, upload_folder
import os

def main():
    repo_id = input("Enter your Hugging Face model repo (e.g. username/model-name): ").strip()
    model_dir = input("Enter path to exported model directory (e.g. ./models/kernel-tonic): ").strip()
    token = os.environ.get("HF_TOKEN") or input("Enter your Hugging Face token: ").strip()

    api = HfApi()
    HfFolder.save_token(token)
    print(f"Pushing {model_dir} to {repo_id} ...")
    upload_folder(
        repo_id=repo_id,
        folder_path=model_dir,
        repo_type="model",
        token=token,
        commit_message="Upload Kernel Tonic model"
    )
    print("✅ Model pushed to Hugging Face Hub!")

if __name__ == "__main__":
    main() 
"""
Data loader for Colossal OSCAR 1.0 (all languages) using Hugging Face token from environment.
"""

import os
from datasets import load_dataset

def get_oscar_dataset(split='train', streaming=True):
    """
    Load the Colossal OSCAR 1.0 dataset (all languages) with authentication.
    Args:
        split: Dataset split (default: 'train')
        streaming: Whether to stream the dataset (recommended for large datasets)
    Returns:
        Hugging Face datasets iterable
    """
    hf_token = os.environ.get('HF_TOKEN')
    if not hf_token:
        raise RuntimeError("Please set the HF_TOKEN environment variable with your Hugging Face token.")
    dataset = load_dataset(
        'oscar-corpus/colossal-oscar-1.0',
        split=split,
        use_auth_token=hf_token,
        streaming=streaming
    )
    return dataset

if __name__ == "__main__":
    ds = get_oscar_dataset()
    for i, sample in enumerate(ds):
        print(sample)
        if i >= 4:
            break 
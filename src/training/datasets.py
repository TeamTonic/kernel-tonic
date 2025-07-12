"""
Data loader for CohereLabs/aya_collection_language_split (all languages, all splits) using Hugging Face token from environment.
"""

import os
from datasets import load_dataset, interleave_datasets, concatenate_datasets

AYA_LANG_CODES = [
    'ace', 'acm', 'acq', 'aeb', 'afr', 'ajp', 'als', 'amh', 'apc', 'arb', 'ars', 'ary', 'arz', 'azb', 'azj', 'bel', 'ben', 'bjn', 'bul', 'cat', 'ceb', 'ces', 'ckb', 'cym', 'dan', 'deu', 'ell', 'eng', 'epo', 'est', 'eus', 'fin', 'fra', 'gla', 'gle', 'glg', 'guj', 'hat', 'hau', 'heb', 'hin', 'hun', 'hye', 'ibo', 'ind', 'isl', 'ita', 'jav', 'jpn', 'kan', 'kas', 'kat', 'kaz', 'khk', 'khm', 'kir', 'kmr', 'knc', 'kor', 'lao', 'lit', 'ltz', 'lvs', 'mal', 'mar', 'min', 'mkd', 'mlt', 'mni', 'mri', 'mya', 'nld', 'nno', 'nob', 'npi', 'nso', 'pbt', 'pes', 'plt', 'pol', 'por', 'ron', 'rus', 'sin', 'slk', 'slv', 'smo', 'sna', 'snd', 'som', 'sot', 'spa', 'srp', 'sun', 'swe', 'swh', 'tam', 'taq', 'tel', 'tgk', 'tha', 'tur', 'ukr', 'urd', 'uzn', 'vie', 'xho', 'ydd', 'yor', 'yue', 'zho-Hans', 'zho-Hant', 'zsm', 'zul', 'arq', 'ban', 'bbc', 'bem', 'fil', 'fon', 'hrv', 'kin', 'lij', 'mad', 'nij', 'nor', 'pan', 'twi', 'wol', 'zho'
]
AYA_SPLITS = ['train', 'validation', 'test']


def get_aya_dataset(streaming=True):
    """
    Load the CohereLabs/aya_collection_language_split dataset (all languages, all splits) with authentication.
    Returns a streaming interleaved dataset over all language codes and splits.
    """
    hf_token = os.environ.get('HF_TOKEN')
    if not hf_token:
        raise RuntimeError("Please set the HF_TOKEN environment variable with your Hugging Face token.")

    datasets = []
    for lang in AYA_LANG_CODES:
        for split in AYA_SPLITS:
            try:
                ds = load_dataset(
                    'CohereLabs/aya_collection_language_split',
                    lang,
                    split=split,
                    use_auth_token=hf_token,
                    streaming=streaming
                )
                datasets.append(ds)
            except Exception as e:
                # Some splits may not exist for all languages
                continue
    if not datasets:
        raise RuntimeError("No datasets loaded from CohereLabs/aya_collection_language_split.")
    # Interleave for streaming, concatenate for non-streaming
    if streaming:
        from datasets import interleave_datasets
        return interleave_datasets(datasets)
    else:
        from datasets import concatenate_datasets
        return concatenate_datasets(datasets)

if __name__ == "__main__":
    ds = get_aya_dataset()
    for i, sample in enumerate(ds):
        print(sample)
        if i >= 4:
            break 
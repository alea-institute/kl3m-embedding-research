"""
Load tokenizers and upload to HF Hub with relevant metadata.
"""

# ignore order for HF token path
# pylint: disable=wrong-import-position

# imports
import os
from pathlib import Path

# set hf token path to ../.hftoken
os.environ["HF_TOKEN_PATH"] = (
    (Path(__file__).parent.parent / ".hftoken").resolve().as_posix()
)

# packages
from transformers import AutoTokenizer

TOKENIZERS = (
    "alea-institute/kl3m-001-32k",
    "alea-institute/kl3m-003-64k",
)

if __name__ == "__main__":
    for tokenizer_name in TOKENIZERS:
        # load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        vocab = tokenizer.get_vocab()
        special_tokens = tokenizer.special_tokens_map
        print(f"Tokenizer: {tokenizer_name}")
        print(f"Vocab size: {len(vocab)}")
        print(f"Special tokens: {len(special_tokens)}")

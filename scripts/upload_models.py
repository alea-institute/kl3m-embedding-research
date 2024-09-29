"""
Load models and upload to HF Hub with relevant metadata.
"""

# imports
from pathlib import Path

# packages
from transformers import AutoModel, AutoTokenizer

# get paths
PATH_LIST: list[str] = [
    # this is just getting the absolute path for `../kl3m-embedding-00*` as a string
    (Path(__file__).parent.parent / p).resolve().as_posix()
    for p in (
        "./kl3m-embedding-001",
        "./kl3m-embedding-002",
        "./kl3m-embedding-003",
        "./kl3m-embedding-004",
    )
]


if __name__ == "__main__":
    # iterate through paths, load the model, and print the basic model info
    # - name
    # - tokenizer name
    # - tokenizer config
    # - model type
    # - model config

    # track model metadata
    model_metadata = {}

    for path in PATH_LIST:
        # load them both
        model = AutoModel.from_pretrained(path)
        tokenizer = AutoTokenizer.from_pretrained(path)

        # populate metadata
        model_metadata[path] = {
            "model": model,
            "tokenizer": tokenizer,
        }

        print(f"Model Name: {model.name_or_path}")
        print(f"Tokenizer Name: {tokenizer.name_or_path}")
        print(f"Model Type: {model.__class__.__name__}")
        print(f"Model Config: {model.config}")
        print("\n")

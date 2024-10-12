"""
Load models and upload to HF Hub with relevant metadata.

TODO: fix after resolving version issues
"""

# imports
import os
from pathlib import Path
from typing import Any

# set hf token path to ../.hftoken
os.environ["HF_TOKEN_PATH"] = (
    (Path(__file__).parent.parent / ".hftoken").resolve().as_posix()
)


# packages
from huggingface_hub import HfApi

# get paths
PATH_LIST: list[str] = [
    # this is just getting the absolute path for `../kl3m-embedding-00*` as a string
    (Path(__file__).parent.parent / p).resolve().as_posix()
    for p in (
        "/data0/checkpoints/kl3m-embedding-001",
        "/data0/checkpoints/kl3m-embedding-002",
        "/data0/checkpoints/kl3m-embedding-003",
        "/data0/checkpoints/kl3m-embedding-004",
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
    model_metadata: dict[str, Any] = {}

    for path in PATH_LIST:
        # get the model display name
        model_name = str(path).rsplit("/", maxsplit=1)[-1]
        repo_id = f"alea-institute/{model_name}"

        """
        NOTE: this is a disaster full of silent changes to tensors and config.

        # model = AutoModel.from_pretrained(path)
        # tokenizer = AutoTokenizer.from_pretrained(path)

        # push privately to HF Hub
        model.push_to_hub(
            repo_id=f"alea-institute/{model_name}", revision="main", private=True
        )
        tokenizer.push_to_hub(
            repo_id=f"alea-institute/{model_name}", revision="main", private=True
        )
        """

        # use HfApi to upload all files from that folder
        hf_api = HfApi()
        try:
            repo_url = hf_api.create_repo(
                repo_id=repo_id,
                repo_type="model",
                private=True,
            )
        except Exception as e:
            print(f"Error creating repo {repo_id}: {e}")

        # now upload all files to main
        hf_api.upload_folder(
            repo_id=repo_id,
            folder_path=path,
        )

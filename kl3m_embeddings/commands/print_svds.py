"""
Print the singular value distributions of the hidden layers.
"""

# imports
import argparse

# packages
from transformers import AutoModel

# project
from kl3m_embeddings.stats.svds import get_model_svd_stats

if __name__ == "__main__":
    # parse args to input path
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name", type=str, help="Name of the model to load")
    args = parser.parse_args()

    # load model with huggingface to get everything loaded as expected
    model = AutoModel.from_pretrained(args.model_name)

    # get the state dict
    state_dict = model.state_dict()

    # get the svds
    svd_metrics = get_model_svd_stats(state_dict)

    # print the svds
    import json

    print(json.dumps(svd_metrics, indent=2, default=str))

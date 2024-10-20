"""
Change the tokenizer for a model by swapping the embedding layers and updating the model configuration.

Usage:
$ PYTHONPATH=. poetry run python3 kl3m_embeddings/embeddings/change_tokenizer.py \
    --input_path model1 \
    --output_path model2 \
    --tokenizer tokenizer_name_or_path
"""

# imports
import argparse
from pathlib import Path

# packages
from transformers import AutoModel, AutoTokenizer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Change the tokenizer for a model by swapping the embedding layers and updating the model configuration."
    )
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Path to the model to change the tokenizer.",
    )
    parser.add_argument(
        "--output_path", type=str, required=True, help="Path to save the new model."
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        required=True,
        help="Name or path of the tokenizer to use.",
    )
    args = parser.parse_args()

    # load the tokenizer
    try:
        new_tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    except Exception as e:  # pylint: disable=broad-except
        raise RuntimeError("Failed to load tokenizer.") from e

    # load the model
    try:
        input_model = AutoModel.from_pretrained(args.input_path)
    except Exception as e:  # pylint: disable=broad-except
        raise RuntimeError("Failed to load model.") from e

    # confirm that we have a resize_token_embeddings method
    if not hasattr(input_model, "resize_token_embeddings"):
        raise RuntimeError(
            "Model does not have a resize_token_embeddings method; please manually splice layers."
        )

    # resize the model embeddings
    new_vocab = new_tokenizer.get_vocab()
    print(f"Resizing to vocab size: {len(new_vocab)}")
    input_model.resize_token_embeddings(len(new_vocab))

    # now set all the special token IDs into the model config
    input_model.config.bos_token_id = new_tokenizer.bos_token_id
    input_model.config.eos_token_id = new_tokenizer.eos_token_id
    input_model.config.pad_token_id = new_tokenizer.pad_token_id
    input_model.config.sep_token_id = new_tokenizer.sep_token_id
    input_model.config.cls_token_id = new_tokenizer.cls_token_id
    input_model.config.mask_token_id = new_tokenizer.mask_token_id

    # save the model with the new tokenizer in the folder
    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    print("Saving model to output path.")
    input_model.save_pretrained(output_path)

    print("Saving new tokenizer to output path.")
    new_tokenizer.save_pretrained(output_path)

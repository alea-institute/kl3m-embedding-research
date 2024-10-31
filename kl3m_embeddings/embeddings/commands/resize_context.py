"""
Resize the position embeddings of a model to extend the context window
by:
 1. creating a copy of the model config
 2. extending the max_position_embeddings in the config
 3. initializing a new model from the config
 4. copying the weights from the old model to the new model
 5. copying the buffers from the old model to the new model into the upper position of the rotary embeddings
6. saving the new model
"""

# imports
import argparse
import copy
from pathlib import Path

# packages
import torch
from transformers import AutoModel, AutoTokenizer


def resize_model(
    input_path: Path, output_path: Path, new_position_embedding_size: int = 1024
):
    # load the input model
    input_tokenizer = AutoTokenizer.from_pretrained(input_path)
    input_model = AutoModel.from_pretrained(input_path)
    input_config = copy.deepcopy(input_model.config)  # type: ignore
    output_config = copy.deepcopy(input_model.config)  # type: ignore

    # extend the max position embeddings
    old_position_embedding_size = input_config.max_position_embeddings
    output_config.max_position_embeddings = new_position_embedding_size
    print(f"old position embedding size: {old_position_embedding_size}")
    print(f"new position embedding size: {new_position_embedding_size}")

    # set dynamic rope scaling
    # output_config.rope_scaling = {"type": "linear",  "factor": 8.0}

    # create the new model
    output_model = AutoModel.from_config(output_config)

    # update the output model state dict from the input model
    print("updating output model state dict")
    output_model.load_state_dict(input_model.state_dict())

    # update the output model buffers from the input model
    print("updating output model buffers")
    input_buffers = list(input_model.named_buffers())
    output_buffers = list(output_model.named_buffers())
    for i in range(len(input_buffers)):
        input_name, input_buffer = input_buffers[i]
        output_name, output_buffer = output_buffers[i]
        if input_name == output_name:
            if input_buffer.shape == output_buffer.shape:
                output_buffer.copy_(input_buffer)
            else:
                if len(input_buffer.shape) == 2:
                    print("updating 2D buffer @ layer ", input_name)
                    output_buffer[:, :old_position_embedding_size] = input_buffer
                elif len(input_buffer.shape) == 4:
                    print("updating 4D buffer @ layer ", input_name)
                    output_buffer[
                        :, :, :old_position_embedding_size, :old_position_embedding_size
                    ] = input_buffer.view(1, -1)
                else:
                    raise ValueError(
                        f"Unexpected input buffer shape: {input_buffer.shape}"
                    )

    # now save it to the output path
    print("saving output model")
    output_model.save_pretrained(output_path)
    input_tokenizer.model_max_length = new_position_embedding_size
    input_tokenizer.save_pretrained(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input",
        type=Path,
        help="Path to input model",
    )
    parser.add_argument(
        "output",
        type=Path,
        help="Path to output model",
    )
    parser.add_argument(
        "new_position_embedding_size",
        type=int,
        help="new position embedding size",
    )
    args = parser.parse_args()
    resize_model(args.input, args.output, args.new_position_embedding_size)

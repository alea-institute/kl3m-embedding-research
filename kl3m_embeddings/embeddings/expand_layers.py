"""
Expand the size of a model by duplicating layers that match a specific
name pattern.

Usage:
$ PYTHONPATH=. poetry run python3 kl3m_embeddings/embeddings/compare_layers.py <model path>
"""

# imports
import argparse
from typing import Literal

# packages
from transformers import AutoModel, AutoTokenizer, PreTrainedModel


def expand_layers(
    model: PreTrainedModel,
    layer_pattern: str = "encoder.layer.",
    layer_config_field: str = "num_hidden_layers",
    strategy: Literal["stack", "interleave"] = "stack",
):
    """
    Expand a model by duplicating layers that match a particular name pattern.

    For example, if we have:
        Layer 5: encoder.layer.0.attention.self.query.weight, torch.Size([256, 256]), torch.float32, torch.Size([256, 256])
        ...
        Layer 20: encoder.layer.0.output.LayerNorm.bias, torch.Size([256]), torch.float32, torch.Size([256])
        Layer 21: encoder.layer.1.attention.self.query.weight, torch.Size([256, 256]), torch.float32, torch.Size([256, 256])
        ...
        Layer 36: encoder.layer.1.output.LayerNorm.bias, torch.Size([256]), torch.float32, torch.Size([256])
        ...
        Layer 52: encoder.layer.2.output.LayerNorm.bias, torch.Size([256]), torch.float32, torch.Size([256])
        ...
        Layer 68: encoder.layer.3.output.LayerNorm.bias, torch.Size([256]), torch.float32, torch.Size([256])

    We want to duplicate layers 5-20, 21-36, 37-52, and 53-68.

    We also need to rename the layers to avoid conflicts.

    Args:
        model (PreTrainedModel): The model.
        layer_pattern (str): The pattern to match the layers.
        layer_config_field (str): The field in the model config that specifies the number of layers.
        strategy (Literal["stack", "interleave"]): The expansion strategy:
         - stack: stack the model layers on top of each other, eg.., 00-4 -> 0,1,2,3,0,1,2,3
         - interleave: interleave the model layers, eg.., 0-4 -> 0,0,1,1,2,2,3,3,
    """
    # create a copy of the model with the config
    model_config = model.config

    if not hasattr(model_config, layer_config_field):
        raise ValueError(f"Model config does not have field {layer_config_field}")

    # otherwise, double it
    setattr(
        model_config, layer_config_field, 2 * getattr(model_config, layer_config_field)
    )

    # create new model from new config
    new_model = model.__class__(model_config)

    # print the layers
    state_dict = model.state_dict()
    new_state_dict = {}

    # get the list of layers by getting the first level of the layer pattern
    # example:
    # - pattern = encoder.layer.
    # - inputs = ["encoder.layer.0.output.LayerNorm.bias", ..., "encoder.layer.3.attention.self.query.weight"]
    # - outputs: ["0", "1", "2", "3"]
    all_layer_names = list(state_dict.keys())

    match_layer_names = [
        name for name in all_layer_names if name.startswith(layer_pattern)
    ]

    # assume integer-valued layer ids
    original_layer_id_list = {
        int(name[len(layer_pattern) :].split(".", 1)[0]) for name in match_layer_names
    }

    # build a dictionary to map them
    new_layer_id_map: dict[int, int] = {}
    if strategy == "stack":
        # stack the layers
        for i, layer_id in enumerate(original_layer_id_list):
            new_layer_id_map[i] = layer_id
            new_layer_id_map[i + len(original_layer_id_list)] = layer_id
    elif strategy == "interleave":
        # interleave the layers
        for i, layer_id in enumerate(original_layer_id_list):
            new_layer_id_map[i * 2] = layer_id
            new_layer_id_map[i * 2 + 1] = layer_id

    # now build the new state dict
    for layer_name in all_layer_names:
        # get the layer id
        if layer_name not in match_layer_names:
            print("copying", layer_name)
            new_state_dict[layer_name] = state_dict[layer_name]

    # next, copy the layers that match the pattern
    for layer_name in match_layer_names:
        # iterate through layer maps
        for new_layer_id, original_layer_id in new_layer_id_map.items():
            new_layer_name = layer_name.replace(
                f".{original_layer_id}.",
                f".{new_layer_id}.",
            )
            print("copying", layer_name, "to", new_layer_name)
            new_state_dict[new_layer_name] = state_dict[layer_name]

    # set the new state dict into the model
    new_model.load_state_dict(new_state_dict)

    # return the model
    return new_model


def main():
    """
    Main method
    """
    # parse the arguments
    parser = argparse.ArgumentParser(
        description="Double the layers of a model that match a certain pattern."
    )
    parser.add_argument("input_path", type=str, help="The path to the model.")
    parser.add_argument(
        "output_path", type=str, help="The output path for the new model."
    )
    args = parser.parse_args()

    # print the layers
    model = AutoModel.from_pretrained(args.input_path)
    tokenizer = AutoTokenizer.from_pretrained(args.input_path)
    new_model = expand_layers(model)
    new_model.save_pretrained(args.output_path)
    tokenizer.save_pretrained(args.output_path)


if __name__ == "__main__":
    main()

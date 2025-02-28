"""
Print out all layers of a model.

Usage:
$ PYTHONPATH=. poetry run python3 kl3m_embeddings/embeddings/compare_layers.py <model path>
"""

# imports
import argparse

# packages
from transformers import AutoModel, PreTrainedModel


def print_layers(model: PreTrainedModel):
    """
    Print out all layers of a model.

    Args:
        model (PreTrainedModel): The model.
    """
    # print the layers
    state_dict = model.state_dict()
    for i, (name, param) in enumerate(model.named_parameters()):
        # layer type from state dict
        if name in state_dict:
            state_dict_shape = state_dict[name].shape
            print(
                f"Layer {i}: {name}, {param.shape}, {param.dtype}, {state_dict_shape}"
            )
        else:
            print(f"Layer {i}: {name}, {param.shape}, {param.dtype}")


def main():
    """
    Main method
    """
    # parse the arguments
    parser = argparse.ArgumentParser(description="Print out all layers of a model.")
    parser.add_argument("model_path", type=str, help="The path to the model.")
    args = parser.parse_args()

    # print the layers
    model = AutoModel.from_pretrained(args.model_path)
    print_layers(model)


if __name__ == "__main__":
    main()

"""
Print out all layers of a model.

Usage:
$ PYTHONPATH=. poetry run python3 kl3m_embeddings/embeddings/compare_layers.py <model path>
"""

# imports
import argparse

# packages
from transformers import AutoModel, PreTrainedModel


def compare_layers(model1: PreTrainedModel, model2: PreTrainedModel):
    """
    Compare the layers of two models.

    Args:
        model1 (PreTrainedModel): The first model.
        model2 (PreTrainedModel): The second model.
    """
    # get all names, shapes, dtypes
    names1 = [name for name, _ in model1.named_parameters()]
    names2 = [name for name, _ in model2.named_parameters()]

    # compare the layers by name first
    if set(names1) != set(names2):
        print("Layer names are different.")

        # print only in 1
        for name in set(names1) - set(names2):
            print(f"Only in model 1: {name}")

        # print only in 2
        for name in set(names2) - set(names1):
            print(f"Only in model 2: {name}")

        return

    # compare the layers by shape and dtype
    for name in names1:
        shape1 = model1.state_dict()[name].shape
        shape2 = model2.state_dict()[name].shape
        dtype1 = model1.state_dict()[name].dtype
        dtype2 = model2.state_dict()[name].dtype

        if shape1 != shape2:
            print(f"Shape mismatch for layer {name}: {shape1} vs {shape2}")

        if dtype1 != dtype2:
            print(f"Dtype mismatch for layer {name}: {dtype1} vs {dtype2}")


def main():
    """
    Main method
    """
    # parse the arguments
    parser = argparse.ArgumentParser(description="Print out all layers of a model.")
    parser.add_argument("model_path_1", type=str, help="The path to the model.")
    parser.add_argument("model_path_2", type=str, help="The path to the model.")
    args = parser.parse_args()

    # print the layers
    model1 = AutoModel.from_pretrained(args.model_path_1)
    model2 = AutoModel.from_pretrained(args.model_path_2)
    compare_layers(model1, model2)


if __name__ == "__main__":
    main()

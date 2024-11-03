"""
Average two models by adding their layers together and dividing by the number of models.

Usage;
$ python3 kl3m_embeddings/embeddings/commands/average_models.py \
    <model1 path> <model2 path> ... <output path>

Alternatively, model paths can have a :## suffix to indicate the **percentage** weight to give to the model.

For example:
$ python3 kl3m_embeddings/embeddings/commands/average_models.py \
    model1:10 model2:80 model3:10 output_model
"""


# imports
import argparse
from pathlib import Path
from typing import List, Dict

# packages
from transformers import PreTrainedModel, AutoModel, AutoTokenizer


def average_models(model_path_weights: Dict[Path, float], output_path: Path) -> Path:
    """
    Average the models by adding their layers together and dividing by the number of models.

    Args:
        model_path_weights (Dict[Path, float]): The paths to the models and their weights.
        output_path (Path): The path to save the averaged model.

    Returns:
        Path to the averaged model.
    """

    # load the models one at a time
    current_average_model = None
    tokenizer = None

    # add the layers together
    for model_path, model_weight in model_path_weights.items():
        print(f"Loading model from {model_path}")
        model = AutoModel.from_pretrained(model_path)

        if current_average_model is None:
            # set
            current_average_model = model
            tokenizer = AutoTokenizer.from_pretrained(model_path)

            # reweight
            for name, param in current_average_model.named_parameters():
                current_average_model.state_dict()[name] *= model_weight
        else:
            # iterate through state dict
            for name, param in model.named_parameters():
                if name in current_average_model.state_dict():
                    # add the parameters
                    current_average_model.state_dict()[name] += model_weight * param.data
                else:
                    print(f"Layer {name} not found in current model.")

                # output layer shape and current mean value
                print(f"Layer {name}: {param.data.shape}, {param.data.mean()}")

    # save the model with the tokenizer
    output_path.mkdir(parents=True, exist_ok=True)
    current_average_model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)

    return output_path


def main():
    """
    Main method for the script.
    """
    # parse the arguments
    parser = argparse.ArgumentParser(description="Average two models by adding their layers together and dividing by the number of models.")
    parser.add_argument("model_paths", type=str, nargs="+", help="The paths to the models.")
    parser.add_argument("output_path", type=Path, help="The output path for the averaged model.")
    args = parser.parse_args()

    # model paths and weights
    num_models = len(args.model_paths)
    model_path_weights = {}
    for model_path in args.model_paths:
        # split the path and weight
        if ":" in model_path:
            path, weight = model_path.split(":")
            model_path_weights[Path(path)] = float(weight) / 100
        else:
            model_path_weights[Path(model_path)] = 1.0 / num_models

    print(f"Averaging models:")
    for model_path, weight in model_path_weights.items():
        print(f"  {model_path}: {weight}")

    # NB: weights don't really have to add to one, but this will obviously result in
    # a change in all magnitude-related properties

    # average the models
    average_models(model_path_weights, args.output_path)


if __name__ == "__main__":
    main()
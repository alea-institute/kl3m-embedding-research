# kl3m-embedding-research

### Versions
**There have been a number of breaking changes in the `transformers`, `tokenizers`, and `cuda` libraries over the last 12 months.**

In order to replicate the training process or run `mteb` benchmarks, you may need to:
 * Use the `transformers` and `tokenizers` versions specified in the `pyproject.toml` file.
 * Remove cached files under `~/.cache/huggingface/hub/`
 * Check that you are on H100s with CUDA version between 535-560.

## Description

This ALEA project contains the research pipeline for the KL3M embedding models.

(The KL3M tokenizers have been moved to the [kl3m-tokenizers](https://github.com/alea-institute/kl3m-tokenizers) repository.)

## Model Cards

TODO

## Training a Model

You can replicate or train your own model like this:

1. Pick a model configuration under the `models/` directory.
2. Review the `config.json` and `training.json` files for details related to the model architecture and training parameters.
3. Run the training script for the model you want to train using the commands below.
4. Monitor progress with the `describe.py` script using the commands below.

Model training can be resumed as long as the `log.jsonl` file is present in the model configuration or checkpoint path.

### DeBERTa-based Models

```bash
$ PYTHONPATH=. poetry run python3 kl3m_embeddings/embeddings/deberta/train_deberta_single.py models/kl3m-embedding-005/
```

### Monitoring Progress

```bash
$ PYTHONPATH=. poetry run python3 kl3m_embeddings/embeddings/describe.py models/kl3m-embedding-005/log.jsonl
```

## Example Outputs

**Progress Example**
```
Training:   0%|          | 1004/2000000 [00:52<80:15:01,  6.92it/s, loss=8.25, loss_100=8.275, loss_1000=9.945, lr=5.5e-05, step_time=0.08]
```

![loss_by_step.png](loss_by_step.png)

![step_time_components.png](step_time_components.png)

![learning_rate_loss.png](learning_rate_loss.png)


## License

This ALEA project is released under the MIT License. See the [LICENSE](LICENSE) file for details.

## Support

If you encounter any issues or have questions about using this ALEA project, please [open an issue](https://github.com/alea-institute/kl3m-embedding-research/issues) on GitHub.

## Learn More

To learn more about ALEA and its software and research projects like KL3M, visit the [ALEA website](https://aleainstitute.ai/).

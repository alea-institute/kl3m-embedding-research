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


#### torch only (single GPU)
```bash
$ PYTHONPATH=. poetry run python3 kl3m_embeddings/embeddings/deberta/train_deberta_single.py models/kl3m-embedding-005/
```

#### torch + deepspeed (multi-node, multi-GPU)
```bash
$ DS_SKIP_CUDA_CHECK=1 PYTHONPATH=. poetry run deepspeed kl3m_embeddings/embeddings/deberta/train_deberta_deepspeed.py models/kl3m-embedding-005-deepspeed-2/
```


### Monitoring Progress

```bash
$ PYTHONPATH=. poetry run python3 kl3m_embeddings/embeddings/describe.py models/kl3m-embedding-005/log.jsonl
```

## Example Outputs

**Progress Example**
```
Training:   1%|▌         | 1207/200000 [01:21<2:35:35, 21.29it/s, loss=5.54, loss_100=5.098, loss_1000=6.985, last_eval=7.50, lr=2.0e-04, step_time=0.05]
```

**Sample Log Line** (`log.jsonl`)
```json
{"step": 424, "epoch": 1, "lr": 9.037e-05, "reduced_dim": 8, "task": "mlm", "num_samples": 64, "num_identifiers": 4, "num_tokens": 8192, "samples_by_dataset": {"reg_docs": 16, "govinfo": 16, "dockets": 16, "fdlp": 16}, "tokens_by_dataset": {"reg_docs": 2048, "govinfo": 2048, "dockets": 2048, "fdlp": 2048}, "sample_time": 0.02530217170715332, "loss": 7.4093098640441895, "forward_time": 0.014125823974609375, "backward_time": 0.004723310470581055, "optimizer_time": 0.0007536411285400391, "step_time": 0.046399831771850586, "time": "2024-10-20T12:24:45.957532"}
```

**Sample Eval Line** (`eval.jsonl`)
```json
{"step": 1000, "mean": 7.500783353328705, "median": 7.503348350524902, "std": 1.1186098803630797, "min": 1.8519301414489746, "p5": 5.78562373161316, "p95": 9.532702102661133, "max": 11.14339828491211, "num_samples": 1000}
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

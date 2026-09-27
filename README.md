# GPT-2 (124M) From Scratch

A from-scratch PyTorch implementation and training pipeline for GPT-2 (124M parameters), built as a learning project to understand the core mechanics behind transformer language models — architecture, tokenization, data sharding, gradient accumulation, and the training loop.

Based on the approach in Andrej Karpathy's ["Let's reproduce GPT-2 (124M)"](https://www.youtube.com/watch?v=l8pRSuU81PU) video and the [build-nanogpt](https://github.com/karpathy/build-nanogpt) repo, adapted and scaled down to run on a single consumer GPU.

## What's in here

| File | Purpose |
|---|---|
| `train_gpt2.py` | GPT-2 model definition (attention, MLP, transformer blocks) + full training loop |
| `fineweb_small.py` | Downloads a small streamed slice of the FineWeb-Edu dataset, tokenizes it with `tiktoken`, and writes it to disk as `.npy` shards |
| `plot_results.py` | Reads the training log and generates a train/val loss chart |
| `generate.py` | Loads a saved checkpoint and generates sample text — no retraining needed |

## Setup

```powershell
pip install torch numpy tiktoken datasets tqdm matplotlib
```

Requires an NVIDIA GPU with CUDA for reasonable training speed (works on CPU too, just much slower).

## Usage

Run these in order:

```powershell
# 1. Prepare a small slice of training data (writes shards to edu_fineweb_small/)
python fineweb_small.py

# 2. Train the model (saves checkpoint + loss log to log/)
python train_gpt2.py

# 3. Plot the training/validation loss curve (saves loss_plot.png)
python plot_results.py

# 4. Generate text from the trained checkpoint (fast, no retraining)
python generate.py
```

## Model

Standard GPT-2 (124M) config:
- 12 transformer layers
- 12 attention heads
- 768 embedding dimension
- 1024 context length (block size)
- 50,257 vocabulary (GPT-2 BPE tokenizer via `tiktoken`)

## Notes on scale

This is a **small-scale learning run**, not a full reproduction:
- Karpathy's original run trains on the full FineWeb-Edu 10B-token sample for ~19,000 steps (~1 hour on an 8×A100 node)
- This version streams a few thousand documents (~5M tokens) and trains for 200 steps on a single consumer GPU, as a way to verify the entire pipeline works correctly end-to-end

With this small-scale setup: training loss drops from ~11.0 (near-random, close to `ln(50257) ≈ 10.8`) down to ~6.5 over 200 steps, with train and validation loss tracking closely (no overfitting). Generated text at this stage shows locally plausible English phrasing but no long-range coherence — expected given the limited data and steps.

## Scaling up

To move toward a more complete reproduction:
- Increase `num_docs_to_take` in `fineweb_small.py` (or run the original `fineweb.py` for the full 10B-token sample)
- Increase `max_steps` and `warmup_steps` in `train_gpt2.py` accordingly
- Ideally run on multiple GPUs via `torchrun` (the code is DDP-ready) or a cloud instance with more VRAM

## Acknowledgments

Built by following Andrej Karpathy's Zero to Hero series, specifically ["Let's reproduce GPT-2 (124M)"](https://www.youtube.com/watch?v=l8pRSuU81PU).

# Wan 2.1 video generation: single-GPU vs multi-GPU benchmark

Containerized text-to-video inference with **Wan 2.1 T2V-1.3B** ([diffusers](https://github.com/huggingface/diffusers)), measured three ways:

| Mode | What it does | Optimizes |
|---|---|---|
| `single` | One GPU renders one video | baseline |
| `dp` (data parallel) | Each GPU renders its own video | throughput (videos/hour) |
| `cp` (context parallel, Ulysses) | All GPUs render one video, splitting the token sequence | latency (time to one video) |

This is a small-scale check of the multi-GPU approach I proposed for an 8-GPU video generation server: a queue of data-parallel workers for throughput, plus sequence parallelism for long high-resolution clips.

## Results

> Filled in from a real run. Hardware: Kaggle, 2× NVIDIA T4 (16 GB).

_Results table goes here (output of `python src/summarize.py`)._

**Findings:**

- _to be added after the run_

## How it works

- **Memory layout for a 16 GB GPU.** The UMT5-XXL text encoder (~11 GB) runs once on the CPU in bf16 and is then released; only the 1.3B transformer and the VAE live on the GPU. Ranks load one after another so two copies of the text encoder never sit in RAM at once.
- **Precision.** `fp16` on T4 (no fast bf16 on Turing), `bf16` on Ampere and newer. The VAE runs in fp32. The script flags NaN output, the typical sign of fp16 overflow.
- **Timing.** One warmup run, then a measured run with CUDA synchronization; per-step timestamps come from the diffusers step callback. Peak VRAM via `torch.cuda.max_memory_allocated`.
- **Multi-GPU.** Launched with `torchrun` (NCCL). Context parallelism uses diffusers `ContextParallelConfig(ulysses_degree=N)`. Wan 1.3B has 12 attention heads, so Ulysses degree 2, 3, 4 or 6 divides them evenly.

## Run on Kaggle (free, 2× T4)

1. Kaggle → **Create → New Notebook → File → Import Notebook** → `notebooks/kaggle_wan_benchmark.ipynb`.
2. Settings: **Accelerator: GPU T4 x2**, **Internet: On**.
3. **Run All**, then download `results.zip` from the Output panel.

## Run with Docker on a Linux GPU host

Requires the NVIDIA driver and [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

```bash
docker build -t wan-video-gpu-benchmark .

# one GPU
docker run --rm --gpus all --shm-size 8g \
  -v hf-cache:/cache -v "$PWD/results:/app/results" \
  wan-video-gpu-benchmark --mode single

# two GPUs, data parallel
docker run --rm --gpus all --shm-size 8g \
  -v hf-cache:/cache -v "$PWD/results:/app/results" \
  --entrypoint torchrun wan-video-gpu-benchmark \
  --nproc_per_node=2 src/bench.py --mode dp

python src/summarize.py results
```

On Ampere or newer GPUs add `--dtype bf16`.

## Build on macOS

macOS has no NVIDIA GPU support in Docker, so the image can be built and inspected but not run:

```bash
docker build --platform linux/amd64 -t wan-video-gpu-benchmark .
docker image ls wan-video-gpu-benchmark
```

## Options

```
--mode single|dp|cp   --dtype fp16|bf16   --height 480 --width 832
--frames 33 (4k+1)    --steps 20          --guidance 5.0
--seed 42             --warmup 1          --prompt "..."
```

## Repository layout

```
src/bench.py                     benchmark (single / dp / cp)
src/summarize.py                 results/*.json -> Markdown table + speedups
notebooks/kaggle_wan_benchmark.ipynb
Dockerfile, docker-compose.yml   container setup
results/                         JSON metrics and sample videos
```

## Limitations

- T4 is a 2018-generation GPU: no fast bf16, no FlashAttention 2, PCIe interconnect. Absolute numbers are far below a modern server; the relative scaling between modes is the point.
- Small model and short clips (33 frames, 480p). Sequence parallelism pays off more on longer, higher-resolution videos, where attention dominates compute.

"""
Benchmark text-to-video generation with Wan 2.1 T2V-1.3B (diffusers).

Modes:
  single  - one GPU, one video (baseline)
  dp      - data parallel: every GPU renders its own video (throughput)
  cp      - context (sequence) parallel, Ulysses: all GPUs render ONE video (latency)

Launch:
  python src/bench.py --mode single
  torchrun --nproc_per_node=2 src/bench.py --mode dp
  torchrun --nproc_per_node=2 src/bench.py --mode cp

Each run writes results/<tag>.json (+ an .mp4 from rank 0).
"""

import argparse
import gc
import json
import os
import platform
import time
from pathlib import Path

import torch
import torch.distributed as dist

MODEL_ID = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
DEFAULT_PROMPT = (
    "A cinematic shot of a modern glass office building at sunset, "
    "camera slowly flying towards the entrance, warm light, high detail"
)
NEGATIVE_PROMPT = (
    "blurry, low quality, distorted, static, overexposed, watermark, text, "
    "deformed, ugly, jpeg artifacts"
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["single", "dp", "cp"], default="single")
    p.add_argument("--dtype", choices=["fp16", "bf16"], default="fp16",
                   help="fp16 for T4/older GPUs; bf16 for Ampere+ (A100, L40S, RTX PRO 6000)")
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--width", type=int, default=832)
    p.add_argument("--frames", type=int, default=33, help="must be 4k+1 (e.g. 17, 33, 49, 81)")
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--guidance", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--prompt", default=DEFAULT_PROMPT)
    p.add_argument("--warmup", type=int, default=1,
                   help="warmup runs (2 steps each) before the measured run")
    p.add_argument("--out", default="results")
    p.add_argument("--tag", default=None, help="result file name; default is derived from settings")
    return p.parse_args()


def setup_distributed():
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
    return world, rank, torch.device(f"cuda:{local_rank}")


def log(rank, *msg):
    if rank == 0:
        print("[bench]", *msg, flush=True)


def load_pipeline(dtype, device, rank):
    """Load Wan so it fits on a 16 GB GPU.

    - text encoder (UMT5-XXL, ~11 GB) stays on CPU in bf16: fp16 T5 is prone to overflow,
      and it is only needed once to encode the prompt;
    - VAE in fp32 for clean decoding (small model);
    - transformer (1.3B) on GPU in the chosen dtype.
    """
    from diffusers import AutoencoderKLWan, WanPipeline
    from transformers import UMT5EncoderModel

    t0 = time.perf_counter()
    text_encoder = UMT5EncoderModel.from_pretrained(
        MODEL_ID, subfolder="text_encoder", torch_dtype=torch.bfloat16)
    vae = AutoencoderKLWan.from_pretrained(MODEL_ID, subfolder="vae", torch_dtype=torch.float32)
    pipe = WanPipeline.from_pretrained(
        MODEL_ID, vae=vae, text_encoder=text_encoder, torch_dtype=dtype)
    pipe.transformer.to(device)
    pipe.vae.to(device)
    try:
        pipe.vae.enable_tiling()  # lowers VAE decode memory if supported
    except Exception:
        pass
    log(rank, f"pipeline loaded in {time.perf_counter() - t0:.1f}s")
    return pipe


def encode_prompt(pipe, prompt, dtype, device):
    """Encode on CPU, move embeddings to GPU, then drop the text encoder to free RAM."""
    with torch.no_grad():
        pe, npe = pipe.encode_prompt(
            prompt=prompt,
            negative_prompt=NEGATIVE_PROMPT,
            do_classifier_free_guidance=True,
            num_videos_per_prompt=1,
            device=torch.device("cpu"),
        )
    pe, npe = pe.to(device, dtype), npe.to(device, dtype)
    pipe.text_encoder = None
    gc.collect()
    return pe, npe


def enable_context_parallel(pipe, world, rank):
    from diffusers import ContextParallelConfig

    cfg = ContextParallelConfig(ulysses_degree=world)
    tr = pipe.transformer
    # PyTorch SDPA ("native") supports context parallelism and works on T4,
    # unlike FlashAttention backends that need Ampere or newer.
    tr.set_attention_backend("native")
    if hasattr(tr, "enable_parallelism"):
        tr.enable_parallelism(config=cfg)
    elif hasattr(tr, "parallelize"):
        tr.parallelize(config=cfg)
    else:
        raise RuntimeError("this diffusers version has no context parallelism; pip install -U diffusers")
    log(rank, f"context parallelism enabled: Ulysses degree {world}")


def run_once(pipe, args, pe, npe, device, seed, steps, step_times=None):
    def on_step_end(_pipe, _i, _t, kwargs):
        if step_times is not None:
            torch.cuda.synchronize(device)
            step_times.append(time.perf_counter())
        return kwargs

    gen = torch.Generator(device=device).manual_seed(seed)
    out = pipe(
        prompt_embeds=pe,
        negative_prompt_embeds=npe,
        height=args.height,
        width=args.width,
        num_frames=args.frames,
        num_inference_steps=steps,
        guidance_scale=args.guidance,
        generator=gen,
        output_type="np",
        callback_on_step_end=on_step_end,
    )
    return out.frames[0]


def main():
    args = parse_args()
    world, rank, device = setup_distributed()
    if args.mode != "single" and world < 2:
        raise SystemExit(f"--mode {args.mode} needs torchrun with 2+ GPUs")
    if args.mode == "single" and world > 1:
        raise SystemExit("--mode single runs without torchrun")

    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    gpu_name = torch.cuda.get_device_name(device)
    log(rank, f"mode={args.mode} gpus={world}x {gpu_name} dtype={args.dtype} "
              f"{args.width}x{args.height} frames={args.frames} steps={args.steps}")

    # Load ranks one after another: each rank briefly holds the ~11 GB text encoder in RAM,
    # and Kaggle has ~30 GB, so loading in parallel could run out of memory.
    for turn in range(world):
        if rank == turn:
            pipe = load_pipeline(dtype, device, rank)
            pe, npe = encode_prompt(pipe, args.prompt, dtype, device)
        if world > 1:
            dist.barrier()

    if args.mode == "cp":
        enable_context_parallel(pipe, world, rank)

    # dp: each rank renders a different video; cp/single: same seed everywhere
    seed = args.seed + (rank if args.mode == "dp" else 0)

    for _ in range(args.warmup):
        run_once(pipe, args, pe, npe, device, seed, steps=2)
    if world > 1:
        dist.barrier()

    torch.cuda.reset_peak_memory_stats(device)
    step_times = []
    torch.cuda.synchronize(device)
    t_start = time.perf_counter()
    frames = run_once(pipe, args, pe, npe, device, seed, args.steps, step_times)
    torch.cuda.synchronize(device)
    t_end = time.perf_counter()

    total_s = t_end - t_start
    denoise_s = (step_times[-1] - t_start) if step_times else total_s
    peak_gb = torch.cuda.max_memory_allocated(device) / 1024**3
    has_nan = bool(torch.isnan(torch.from_numpy(frames)).any())

    # gather per-rank numbers on rank 0
    local = torch.tensor([total_s, denoise_s, peak_gb], device=device)
    if world > 1:
        gathered = [torch.zeros_like(local) for _ in range(world)]
        dist.all_gather(gathered, local)
        per_rank = [g.tolist() for g in gathered]
    else:
        per_rank = [local.tolist()]

    if rank == 0:
        import diffusers
        from diffusers.utils import export_to_video

        total_wall = max(r[0] for r in per_rank)
        videos = world if args.mode == "dp" else 1
        result = {
            "mode": args.mode,
            "gpus": world,
            "gpu_name": gpu_name,
            "dtype": args.dtype,
            "resolution": f"{args.width}x{args.height}",
            "frames": args.frames,
            "steps": args.steps,
            "total_s": round(total_wall, 1),
            "denoise_s": round(max(r[1] for r in per_rank), 1),
            "s_per_step": round(max(r[1] for r in per_rank) / args.steps, 2),
            "peak_vram_gb": round(max(r[2] for r in per_rank), 2),
            "videos": videos,
            "videos_per_hour": round(videos * 3600 / total_wall, 1),
            "nan_in_output": has_nan,
            "torch": torch.__version__,
            "diffusers": diffusers.__version__,
            "python": platform.python_version(),
        }
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        tag = args.tag or f"{args.mode}_{world}gpu_{args.dtype}_{args.frames}f_{args.steps}s"
        (out / f"{tag}.json").write_text(json.dumps(result, indent=2))
        export_to_video(list(frames), str(out / f"{tag}.mp4"), fps=16)
        print(json.dumps(result, indent=2), flush=True)
        if has_nan:
            print("[bench] WARNING: NaN in output (black video). Try --dtype bf16.", flush=True)

    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

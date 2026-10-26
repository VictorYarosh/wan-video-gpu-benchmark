"""Collect results/*.json into a Markdown table and print speedups vs the single-GPU run."""

import json
import sys
from pathlib import Path

COLUMNS = [
    ("mode", "Mode"),
    ("gpus", "GPUs"),
    ("gpu_name", "GPU"),
    ("dtype", "dtype"),
    ("resolution", "Resolution"),
    ("frames", "Frames"),
    ("steps", "Steps"),
    ("total_s", "Total, s"),
    ("s_per_step", "s/step"),
    ("peak_vram_gb", "Peak VRAM, GB"),
    ("videos_per_hour", "Videos/hour"),
]


def main(results_dir="results"):
    rows = [json.loads(p.read_text()) for p in sorted(Path(results_dir).glob("*.json"))]
    if not rows:
        print(f"no results in {results_dir}/")
        return

    lines = ["| " + " | ".join(h for _, h in COLUMNS) + " |",
             "|" + "---|" * len(COLUMNS)]
    for r in rows:
        lines.append("| " + " | ".join(str(r.get(k, "")) for k, _ in COLUMNS) + " |")
    table = "\n".join(lines)
    print(table)

    base = next((r for r in rows if r["mode"] == "single"), None)
    if base:
        print()
        for r in rows:
            if r is base:
                continue
            latency = base["total_s"] / r["total_s"]
            throughput = r["videos_per_hour"] / base["videos_per_hour"]
            eff = throughput / r["gpus"] * 100
            print(f"{r['mode']} x{r['gpus']}: latency speedup {latency:.2f}x, "
                  f"throughput {throughput:.2f}x, scaling efficiency {eff:.0f}%")

    bad = [r for r in rows if r.get("nan_in_output")]
    if bad:
        print("\nWARNING: NaN outputs in:", ", ".join(r["mode"] for r in bad))

    Path(results_dir, "summary.md").write_text(table + "\n")


if __name__ == "__main__":
    main(*sys.argv[1:])

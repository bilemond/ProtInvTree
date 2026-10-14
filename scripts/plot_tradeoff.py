#!/usr/bin/env python3
"""Plot measured benchmark summaries; one point per method and configuration."""
import argparse
from collections import defaultdict
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summaries", type=Path, nargs="+", required=True)
    parser.add_argument("--metric", choices=("novelty", "diversity"), default="novelty")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    groups = defaultdict(list)
    comparable = None
    for path in args.summaries:
        with path.open() as handle:
            result = json.load(handle)
        metadata = result["metadata"]
        cfg = metadata["configuration"]
        if not metadata.get("complete", False):
            parser.error(f"Refusing to plot an incomplete evaluation: {path}")
        benchmark = (cfg["dataset"], cfg["subset"], cfg["min_length"], cfg["max_length"],
                     cfg["start_index"], cfg["end_index"], cfg["input_sha256"], cfg["samples_per_target"])
        if comparable is not None and benchmark != comparable:
            parser.error("Summaries must evaluate the same dataset, split, range and sampling budget")
        comparable = benchmark
        value = result["aggregate"].get(args.metric)
        if value is None:
            parser.error(f"No {args.metric} available in {path}; diversity requires at least two samples")
        search = cfg["models_and_search"]
        label = cfg["method"]
        if label == "mcts":
            label += f" ({search['rollout_mode']}, {search['reward_reference']})"
        groups[label].append((result["aggregate"]["sc_tm_score"], value))
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for label, points in groups.items():
        points.sort()
        ax.plot([p[0] for p in points], [p[1] for p in points], marker="o", label=label)
    ax.set_xlabel("sc-TMscore")
    ax.set_ylabel(args.metric.capitalize())
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=300, bbox_inches="tight")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()

"""Atomic result persistence and per-backbone benchmark summaries."""

from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
from statistics import mean

from .metrics import diversity

METRICS = ("target_tm_score", "target_rmsd", "sc_tm_score", "sc_rmsd", "recovery",
           "novelty", "reward", "solve_time_seconds", "simulation_steps", "node_count")


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def sample_seed(base_seed, name, sample_index):
    text = f"{base_seed}:{name}:{sample_index}".encode()
    return int.from_bytes(hashlib.sha256(text).digest()[:4], "big") % (2**31)


def summarize(records):
    groups = defaultdict(list)
    for record in records:
        groups[record["name"]].append(record)
    chains = []
    for name, samples in groups.items():
        row = {"name": name, "samples": len(samples), "length": len(samples[0]["sequence"]),
               "diversity": diversity([sample["sequence"] for sample in samples])}
        for metric in METRICS:
            row[metric] = mean(sample[metric] for sample in samples)
        chains.append(row)
    aggregate = {"backbones": len(chains), "samples": len(records)}
    if chains:
        for metric in METRICS:
            aggregate[metric] = mean(chain[metric] for chain in chains)
        diversities = [chain["diversity"] for chain in chains if chain["diversity"] is not None]
        aggregate["diversity"] = mean(diversities) if diversities else None
        aggregate["backbones_with_diversity"] = len(diversities)
    return {"aggregate": aggregate, "per_backbone": chains}


def write_tables(output, records, metadata):
    output = Path(output)
    atomic_json(output / "results.json", {"metadata": metadata, "records": records})
    atomic_json(output / "summary.json", {"metadata": metadata, **summarize(records)})
    fields = ["name", "sample_index", "seed", "method", "sequence", *METRICS,
              "search_value", "search_complete"]
    temp = output / "results.csv.tmp"
    with temp.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    temp.replace(output / "results.csv")

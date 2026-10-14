#!/usr/bin/env python3
"""Recompute metrics from one exported results.json."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from protinvtree.results import atomic_json, summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--subset-file", type=Path, help="Optional JSON with a test-name list")
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve():
        parser.error("Output must differ from input")
    with args.input.open() as handle:
        exported = json.load(handle)
    records = exported["records"]
    if args.subset_file:
        with args.subset_file.open() as handle:
            names = set(json.load(handle)["test"])
        records = [record for record in records if record["name"] in names]
    result = {"metadata": exported["metadata"], **summarize(records)}
    atomic_json(args.output, result)
    print(json.dumps(result["aggregate"], indent=2))


if __name__ == "__main__":
    main()

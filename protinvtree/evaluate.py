"""CATH 4.2/4.3 evaluation with resumable, independent samples per backbone."""

import argparse
from datetime import datetime, timezone
import json
import logging
import math
from pathlib import Path
import platform

from .data import AMINO_ACIDS, read_cath
from .options import add_model_options, parse_options
from .results import METRICS, atomic_json, sample_seed, write_tables


def validate_saved_result(result, backbone, sample_index, seed, method):
    if not isinstance(result, dict):
        raise ValueError("Saved sample must be a JSON object")
    if (result.get("name"), result.get("sample_index"), result.get("seed"), result.get("method")) != (
        backbone.name, sample_index, seed, method
    ):
        raise ValueError("Saved sample identity does not match the evaluation")
    sequence = result.get("sequence")
    if (not isinstance(sequence, str) or len(sequence) != len(backbone.sequence)
            or not set(sequence) <= AMINO_ACIDS
            or result.get("reference_sequence") != backbone.sequence
            or result.get("original_length") != backbone.original_length
            or result.get("residue_indices") != backbone.residue_indices):
        raise ValueError("Saved sample sequence or backbone metadata is invalid")
    for field in (*METRICS, "search_value"):
        value = result.get(field)
        if field == "search_value" and value is None and field in result:
            continue
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError(f"Saved sample has missing or nonfinite metric: {field}")
    if type(result.get("search_complete")) is not bool:
        raise ValueError("Saved sample search_complete must be a boolean")


def create_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("CATH4.2", "CATH4.3"), default="CATH4.2")
    parser.add_argument("--data-dir", type=Path, required=True, help="Directory containing chain_set.jsonl and split JSON")
    parser.add_argument("--subset", choices=("all", "short", "single-chain"), default="all")
    parser.add_argument("--min-length", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=500)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, help="Exclusive upper bound in the filtered test set")
    parser.add_argument("--samples-per-target", type=int, default=1)
    parser.add_argument("--method", choices=("mcts", "esm3"), default="mcts")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--save-pdb", action="store_true")
    parser.add_argument("--save-tree", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and write evaluation plan without model imports")
    add_model_options(parser)
    return parser


def main(argv=None):
    parser = create_parser()
    args, cfg = parse_options(parser, argv)
    if args.min_length < 1 or args.max_length < args.min_length or args.samples_per_target < 1:
        parser.error("Invalid sequence lengths or samples-per-target")
    if args.start_index < 0 or (args.end_index is not None and args.end_index <= args.start_index):
        parser.error("Expected 0 <= start-index < end-index")
    records, filtering, hashes = read_cath(args.data_dir, args.subset, args.min_length, args.max_length)
    selected = records[args.start_index:args.end_index]
    if not selected:
        parser.error("No eligible chains in the selected range")
    plan = [{"index": i + args.start_index, "name": record.name,
             "original_length": record.original_length, "length": len(record.sequence),
             "residue_indices": record.residue_indices} for i, record in enumerate(selected)]
    configuration = {"models_and_search": cfg, "dataset": args.dataset, "subset": args.subset,
                     "generation_alphabet": "".join(sorted(AMINO_ACIDS)),
                     "sampling_seed_protocol": "after-prompt-encoding-v1",
                     "min_length": args.min_length, "max_length": args.max_length,
                     "start_index": args.start_index, "end_index": args.end_index,
                     "samples_per_target": args.samples_per_target, "method": args.method,
                     "input_sha256": hashes, "save_pdb": args.save_pdb, "save_tree": args.save_tree}
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    if args.dry_run:
        if manifest_path.exists():
            parser.error("Use a separate output directory for a dry run of an existing evaluation")
        atomic_json(output / "dry_run.json", {"configuration": configuration, "filtering": filtering,
                                               "plan": plan, "expected_samples": len(plan) * args.samples_per_target})
        print(f"Validated {len(plan)} backbones; {len(plan) * args.samples_per_target} independent designs planned")
        print(json.dumps(filtering, indent=2))
        return 0
    if manifest_path.exists():
        if not args.resume:
            parser.error("Output already contains an evaluation; use --resume or another output directory")
        with manifest_path.open() as handle:
            manifest = json.load(handle)
        if manifest["configuration"] != configuration:
            parser.error("Resume configuration or dataset content differs from the saved run")
    else:
        if (output / "samples").exists():
            parser.error("Samples exist without a manifest; choose another output directory")
        manifest = {"configuration": configuration, "filtering": filtering, "plan": plan,
                    "created_at": datetime.now(timezone.utc).isoformat(), "python": platform.python_version()}
        atomic_json(manifest_path, manifest)
    logging.basicConfig(filename=output / "evaluation.log", level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    from .runner import DesignRunner
    from importlib.metadata import version
    manifest["dependency_versions"] = {package: version(package) for package in
                                       ("torch", "transformers", "biotite", "tmtools", "numpy", "huggingface-hub")}
    atomic_json(manifest_path, manifest)
    runner, results, failures = None, [], []
    expected = len(plan) * args.samples_per_target
    for plan_item, backbone in zip(plan, selected):
        protein, prepared = None, False
        for sample_index in range(args.samples_per_target):
            sample_id = f"{plan_item['index']:06d}_{sample_index:03d}"
            sample_dir = output / "samples" / sample_id
            result_path = sample_dir / "result.json"
            seed = sample_seed(cfg["seed"], backbone.name, sample_index)
            try:
                if args.resume and result_path.exists():
                    with result_path.open() as handle:
                        result = json.load(handle)
                    validate_saved_result(result, backbone, sample_index, seed, args.method)
                    required = [sample_dir / "sequence.fasta"]
                    if args.save_pdb:
                        required.append(sample_dir / "folded.pdb")
                    if args.save_tree:
                        required.append(sample_dir / "tree.json")
                    if all(path.is_file() and path.stat().st_size > 0 for path in required):
                        results.append(result)
                        continue
                if runner is None:
                    runner = DesignRunner(cfg)
                if protein is None:
                    protein = backbone.to_protein()
                if not prepared:
                    runner.prepare(protein, backbone.sequence)
                    prepared = True
                result, pdb, tree = runner.design(protein, backbone.sequence, seed, args.method)
                result.update({"name": backbone.name, "sample_index": sample_index,
                               "original_length": backbone.original_length,
                               "residue_indices": backbone.residue_indices})
                sample_dir.mkdir(parents=True, exist_ok=True)
                (sample_dir / "sequence.fasta").write_text(f">{backbone.name}_sample_{sample_index}\n{result['sequence']}\n")
                if args.save_pdb:
                    (sample_dir / "folded.pdb").write_text(pdb)
                if args.save_tree:
                    atomic_json(sample_dir / "tree.json", tree)
                # Commit last so a failed write never looks like a completed sample.
                atomic_json(result_path, result)
                results.append(result)
                print(f"[{len(results)}/{expected}] {backbone.name} sample={sample_index} scTM={result['sc_tm_score']:.4f}", flush=True)
            except Exception as error:
                failure = {"name": backbone.name, "sample_index": sample_index,
                           "seed": seed, "error": f"{type(error).__name__}: {error}"}
                failures.append(failure)
                logging.exception("Failed sample %s", sample_id)
                atomic_json(output / "failures.json", failures)
                write_tables(output, results, {**manifest, "expected_samples": expected,
                                               "failed_samples": failures, "complete": False})
                if not args.continue_on_error:
                    raise
    atomic_json(output / "failures.json", failures)
    metadata = {**manifest, "expected_samples": expected, "failed_samples": failures,
                "complete": len(results) == expected and not failures}
    write_tables(output, results, metadata)
    print(f"Saved {len(results)}/{expected} designs to {output}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

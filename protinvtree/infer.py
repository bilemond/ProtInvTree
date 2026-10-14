"""Design a sequence for one PDB chain."""

import argparse
from pathlib import Path

from .options import add_model_options, parse_options
from .results import atomic_json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdb", type=Path, required=True)
    parser.add_argument("--chain-id", default="detect", help="PDB chain ID; detect selects the first chain")
    parser.add_argument("--output-dir", type=Path, required=True)
    add_model_options(parser)
    args, config = parse_options(parser, argv)
    if not args.pdb.is_file():
        parser.error(f"PDB does not exist: {args.pdb}")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("Output directory must be empty")
    import torch
    from esm.sdk.api import ESMProtein
    from .runner import DesignRunner
    protein = ESMProtein.from_pdb(args.pdb, chain_id=args.chain_id)
    reference = protein.sequence
    if not reference or any(x not in "ACDEFGHIKLMNPQRSTVWY" for x in reference):
        parser.error("PDB chain must have a nonempty canonical amino acid sequence")
    if not torch.isfinite(protein.coordinates[:, [0, 1, 2, 4]]).all():
        parser.error("PDB chain must have complete N, CA, C and O backbone atoms")
    runner = DesignRunner(config)
    runner.prepare(protein, reference)
    result, pdb, tree = runner.design(protein, reference, config["seed"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "sequence.fasta").write_text(f">{args.pdb.stem}_design\n{result['sequence']}\n")
    (args.output_dir / "folded.pdb").write_text(pdb)
    atomic_json(args.output_dir / "tree.json", tree)
    atomic_json(args.output_dir / "result.json", {"configuration": config, **result})
    print(f"Saved design to {args.output_dir}; scTM={result['sc_tm_score']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Read the CATH chain_set JSONL format without loading models or tokenizers."""

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

AMINO_ACIDS = frozenset("ACDEFGHIKLMNPQRSTVWY")
ATOMS = ("N", "CA", "C", "O")


@dataclass
class Backbone:
    name: str
    sequence: str
    coordinates: list
    original_length: int
    residue_indices: list

    def to_protein(self):
        import numpy as np
        from esm.sdk.api import ESMProtein
        from esm.utils.structure.protein_chain import ProteinChain
        chain = ProteinChain.from_backbone_atom_coordinates_N_CA_C_O(
            # The original JSONL loader uses NumPy float64. Preserve the target
            # coordinates for reward computation; ESM encoding casts to float32.
            np.asarray(self.coordinates, dtype=np.float64)
        )
        protein = ESMProtein.from_protein_chain(chain)
        protein.sequence = None
        return protein


def file_sha256(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def read_cath(data_dir, subset="all", min_length=1, max_length=500):
    """Retain file order and drop residues missing any of N/CA/C/O, as in the source."""
    data_dir = Path(data_dir)
    split_path = data_dir / "chain_set_splits.json"
    with split_path.open() as handle:
        split = json.load(handle)
    if "test" not in split or not isinstance(split["test"], list):
        raise ValueError(f"{split_path} must contain a test list")
    names = set(split["test"])
    files = [data_dir / "chain_set.jsonl", split_path]
    if subset != "all":
        subset_file = data_dir / {"short": "test_split_L100.json", "single-chain": "test_split_sc.json"}[subset]
        with subset_file.open() as handle:
            subset_names = set(json.load(handle)["test"])
        if not subset_names <= names:
            raise ValueError("Subset contains names outside the official test split")
        names = subset_names
        files.append(subset_file)
    records, seen = [], set()
    stats = {"selected_split_names": len(names), "noncanonical": 0,
             "too_long": 0, "too_short_after_filter": 0, "removed_residues": 0}
    with files[0].open() as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            entry = json.loads(line)
            name = entry["name"]
            if name not in names:
                continue
            if name in seen:
                raise ValueError(f"Duplicate CATH chain name at line {line_no}: {name}")
            seen.add(name)
            sequence = entry["seq"]
            if not set(sequence) <= AMINO_ACIDS:
                stats["noncanonical"] += 1
                continue
            if len(sequence) > max_length:
                stats["too_long"] += 1
                continue
            coords = entry["coords"]
            if any(len(coords[atom]) != len(sequence) for atom in ATOMS):
                raise ValueError(f"Coordinate length does not match sequence for {name}")
            coordinates, indices = [], []
            for i in range(len(sequence)):
                atoms = [coords[atom][i] for atom in ATOMS]
                if any(x is not None and len(x) != 3 for x in atoms):
                    raise ValueError(f"Expected xyz triplets for {name}, residue {i}")
                if any(x is None or any(v is None or not math.isfinite(float(v)) for v in x) for x in atoms):
                    continue
                coordinates.append([[float(v) for v in x] for x in atoms])
                indices.append(i)
            stats["removed_residues"] += len(sequence) - len(indices)
            if len(indices) < min_length:
                stats["too_short_after_filter"] += 1
                continue
            records.append(Backbone(name, "".join(sequence[i] for i in indices),
                                    coordinates, len(sequence), indices))
    missing = sorted(names - seen)
    if missing:
        raise ValueError(f"Test split refers to {len(missing)} missing chains: {missing[:5]}")
    stats["eligible_chains"] = len(records)
    return records, stats, {path.name: file_sha256(path) for path in files}

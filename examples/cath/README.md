# Synthetic CATH-format fixture

These two short sequences and artificial coordinates exercise data parsing and
missing-coordinate filtering. They are not real CATH records, designed proteins,
or benchmark results. Do not use them for biological validation.

The JSONL order deliberately differs from the split-list order. `synthetic_b`
has a missing oxygen coordinate at residue index 2, so its clean sequence is
`HILMN`, with five retained residues. All residue indices in exported manifests
are zero-based indices into the original JSONL sequence.

# Software validation

On 2026-10-14, 51 regression tests passed using Python 3.12, PyTorch 2.5.0,
Transformers 4.46.3, NumPy 1.26.4, Biotite 0.41.2, and TM-tools 0.3.0.
Tests use real ESM tokenizers/sampling helpers and PDB parsing/TM-align with
synthetic coordinates and tiny fake models. Pretrained weights are not required.

Coverage includes sequence masking and focus sampling, iterative/jumpy rollouts,
mixed-length batch padding, average-reward backpropagation, budget fallback,
threshold semantics, reward validation and native-label dependence, CATH subsets,
sequence metrics, result aggregation, independent seeds, and evaluation resume.
Additional regression cases cover noncanonical logits, baseline alphabet
restrictions, corrupt/nonfinite saved results, missing requested artifacts,
sampling-protocol changes, and interrupted downloads/extraction with safe retry.

```bash
python -m unittest discover -s tests -v
python scripts/evaluate_cath.py --data-dir examples/cath \
  --output-dir results/format-check --dry-run
```

The synthetic examples verify input format and program behavior, not biological
design quality. The GitHub CI matrix passed for the earlier `07df17c` revision;
the current fixes were checked locally. Real CATH GPU inference and paper
scores/timings have not been validated. See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for
protocol choices that affect comparisons with the paper.

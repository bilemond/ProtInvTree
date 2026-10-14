# Software validation

On 2026-10-14, 53 regression tests passed using Python 3.12, PyTorch 2.5.0,
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
The baseline exercises the actual ESM token sampling API without invoking
unneeded decoders. A stochastic lazy-initialization fixture checks identical
sequences and search trees for cold/warm model state with the same input/seed.

```bash
python -m unittest discover -s tests -v
python scripts/evaluate_cath.py --data-dir examples/cath \
  --output-dir results/format-check --dry-run
```

The synthetic examples verify input format and program behavior, not biological
design quality. The GitHub CI matrix passed for the earlier `20f3a6b` revision;
the current fixes were checked locally.

## Real GPU examples

On 2026-10-14, the actual CLI ran successfully using two RTX 3090 GPUs, the same
Python/dependency environment above, and these fixed model revisions:

- ESM3: `47f0545b2b6daf26a93439a3cd610f4f7f3d5478`.
- ESMFold: `75a3841ee059df2bf4d56688166c8fb459ddd97a`.

The main weights were checked for exact size and SHA-256 against published LFS
metadata. Settings were depth 4, expansion width 2, one rollout, pre-expansion
depth 1, at most 20 simulations, temperature 0.3, random focus, cosine schedule,
target-backbone reward with alpha 1, and folding chunk size 128.

| Input / method | Residues | Target TM-score | sc-TMscore | Sequence recovery |
| --- | --- | --- | --- | --- |
| 1UBQ chain A / iterative MCTS | 76 | 0.957905 | 1.000000 | 1.000000 |
| CATH 4.3, 1ahl.A / iterative MCTS | 49 | 0.514661 | 0.538220 | 0.367347 |
| CATH 4.3, 1ahl.A / jumpy MCTS | 49 | 0.346262 | 0.442640 | 0.346939 |
| CATH 4.3, 1ahl.A / ESM3 baseline | 49 | 0.404191 | 0.454999 | 0.346939 |

FASTA sequence, predicted PDB C-alpha count, result length, finite scores,
recovery/novelty consistency, tree node count and completed evaluation metadata
were checked. Both CATH MCTS runs resumed without regeneration. Removing a
requested PDB and resuming regenerated an identical sequence, PDB and scores.

Real ESM3 generation was repeated with cold/warm encoder state. Changing the
1UBQ native sequence label to all A, or removing sidechain coordinates, preserved
the generated sequence at the same seed in this check. These finite examples
do not prove invariance for all inputs or absence of foundation-model training
overlap. The 1UBQ example reconstructs the native sequence and has zero novelty.

The CATH files were found locally; their sizes match the public release, and
local SHA-256 values were recorded. That release provides no published digest,
so identity with the public download or the authors' private dataset is not
established by these checks.

These are single-backbone smoke tests. They establish that these execution paths
work with real pretrained models, not the paper's full benchmark scores, timing
claims, or general method rankings. See [REPRODUCIBILITY.md](REPRODUCIBILITY.md)
for protocol choices affecting paper comparisons.

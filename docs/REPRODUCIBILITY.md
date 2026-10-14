# Reproducibility and method mapping

This is a refactoring of the supplied research working tree, not a claim that
its defaults reproduce every NeurIPS table. Preserve the manifest and report
which reward reference and rollout mode were used for any published run.

## Method and source mapping

| Component | Supplied source | Prepared module |
| --- | --- | --- |
| Focus-and-grounding / sequence sampling | `code/models/policy_model.py` | `protinvtree/policy.py` |
| Expansion and rollout rewards | `code/models/agent.py` | `protinvtree/agent.py` |
| UCT traversal / backpropagation | `code/models/searching/` | `protinvtree/searching/` |
| Folding reward | `code/models/reward_model.py` | `protinvtree/reward.py` |
| CATH preprocessing | `code/src/datasets/cath_dataset.py`, `featurizer.py` | `protinvtree/data.py` |
| Evaluation driver | `code/train/inference_MCTS.py`, `model_interface.py` | `protinvtree/evaluate.py`, `runner.py` |

The policy and agent retain their main sampling bodies. The MCTS implementation
preserves the source's recursive UCB-ordered selection of nonterminal leaves and
mean-reward backpropagation. JSON search trees contain sequences, parent IDs,
depths, visits, and node values; no tensors or pickle objects are required.

## Explicit differences between paper and source

- The paper describes **single-step jumpy denoising**. The supplied rollout
  method instead loops through all remaining denoising steps. `iterative` retains
  that source behavior; `jumpy` uses the final scheduled step to fill all remaining
  masks in one model forward pass. `configs/jumpy.json` enables the latter.
- The supplied reward compares ESMFold(candidate) with the input target backbone,
  corresponding to paper equation 11. The appendix's sc-TMscore and Algorithm 1
  compare folded candidate and folded native reference. `--reward-reference`
  selects the optimization target; both structural metrics are always reported.
- Appendix Algorithm 1 collects terminal sequences with reward >= threshold from
  one tree. Here the threshold controls early exit using >, and budget fallback
  can return a lower-scoring sequence. Each sample is an independent search.
  This is a different output protocol, not a threshold acceptance filter.
- Novelty-weighted and folded-native rewards let the native sequence influence
  search. The default target-only reward passed finite same-length label-change
  checks, but its interface still stores and passes native sequence to TM-align.
  These finite software checks do not prove complete isolation from native labels.
- The source's `value` output is a historical MCTS node value and can be a mean
  over rollout candidates. For unfinished search, the source returned a newly
  rolled-out sequence with that earlier score. This version re-folds/scores the
  actual final sequence and reports `search_value` separately.
- The source defaults include temperature 0.3, depth 4, width 2, pre-expansion
  width/depth 2, one rollout, 50 simulations, random focus, cosine schedule,
  exploration coefficient 0.01, and threshold 0.0. These are recorded in
  `configs/source_defaults.json`. They are not inferred as every table's settings.
- The old driver skipped the first 901 test chains by default. This version starts
  at zero and exposes an exclusive end index. Its seed derivation is stable by
  base seed, chain name and sample index, allowing consistent job partitioning;
  it does not recreate the old process-wide random-number stream.

## Correctness fixes and practical changes

- Sequence and auxiliary annotation tracks are removed before building policy
  prompts, avoiding native-sequence conditioning in PDB use.
- Sampling state comparisons and deduplication compare tensor contents; a tensor
  is never implicitly evaluated as a boolean or deduplicated by object identity.
- Mixed-length batch outputs are trimmed to each state's true token length;
  padding is removed before rollout decoding and further expansion.
- A supplied helper returned a bare sequence tensor when a track already existed;
  it now returns the complete `ESMProteinTensor`.
- Completed paths are counted once per current tree, rather than cumulatively
  on every MCTS iteration. Pre-expanded completed trees and zero search budgets
  have defined return behavior.
- Fully unmasked states terminate even when short chains complete before the
  nominal depth. Empty expansions stop being selected indefinitely.
- Threshold-based early exit chooses the highest-valued terminal node, as in the
  source. If the budget is exhausted without qualifying early exit, the default
  `--final-selection greedy` follows the greedy highest-average-value path, as in
  the source and paper section 4.2. `best-terminal` is an explicit alternative
  and may change the returned sequence. The initial prepared version always
  preferred a terminal node; the subsequent source audit restored greedy as the default.
- Sequence decoding during MCTS does not instantiate unused ESM3 structure and
  function decoders. The ESM3 baseline retains the supplied generation API.
- The reference sequence is folded once per backbone, and an LRU cache avoids
  repeated candidate folds. Evaluation metrics and timing are saved separately.
- Plotting reads actual exported summaries. Unrelated training, LLM planning,
  generic inverse-folding baselines, notebook widgets, IDE files, caches,
  credentials, internal audit records, and generated artifacts are excluded.

## Metrics and aggregation

`target_tm_score` and `target_rmsd` compare predicted C-alpha atoms with the input
target; `sc_tm_score` and `sc_rmsd` compare predicted candidate and native-reference
C-alpha atoms. TM scores use reference-length normalization. RMSD comes from the
TM-align structural alignment, rather than an independently specified all-residue
Kabsch calculation; report this distinction when comparing protocols.

Recovery is positional sequence identity; novelty is one minus recovery. Diversity
is average normalized pairwise Hamming distance among samples for the same
backbone. It is undefined (`null`) for one sample. Summary metrics first average
samples within each backbone and then average across backbones. A failed or
partial run is explicitly marked incomplete; no failure is converted into a zero
score or claimed as a complete benchmark.

`solve_time_seconds` includes search, final rollout, final folding/evaluation and
tree serialization preparation. It excludes model loading and the once-per-target
reference preparation. It is not a direct reproduction of the old search-only
timing field. Model folding caches are shared across samples in one process.

## Full scientific reproduction still requires

The real benchmark files and official subsets, the intended model snapshots,
CUDA inference, confirmed per-table hyperparameters and reward references, and a
comparison with the original experiment outputs. None of those numerical claims
can be established from a source-code refactoring and synthetic tests alone.

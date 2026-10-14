# ProtInvTree

Research code for **ProtInvTree: Deliberate Protein Inverse Folding with
Reward-guided Tree Search**, NeurIPS 2025.

Mengdi Liu, Xiaoxue Cheng, Zhangyang Gao, Hong Chang, Cheng Tan, Shiguang Shan,
and Xilin Chen.

ProtInvTree uses a pretrained ESM3 policy to build sequences through
focus-and-grounding actions, with MCTS and ESMFold-based structural rewards.
This repository preparation provides single-chain PDB inference, CATH 4.2/4.3
evaluation and an ESM3 sampling baseline.
The locally modified ESM implementation is included so research-specific sampling
behavior does not depend on an unrelated installed `esm` version.

**Licensing status:** the authors have deferred the research-code license
decision. See [LICENSE.md](LICENSE.md).
GPU inference and reproduction of the paper's numerical results have not yet
been validated for this refactoring. See [validation](docs/VALIDATION.md).
Method choices and remaining differences are documented in
[reproducibility notes](docs/REPRODUCIBILITY.md).

## Installation

Use an isolated Python 3.10–3.12 environment. Linux with an NVIDIA GPU is the
intended inference environment; the paper reports A100 GPUs with 80 GB memory.

```bash
conda create -n protinvtree python=3.10 -y
conda activate protinvtree
python -m pip install -e .
```

Install a PyTorch build appropriate for your CUDA environment. `torchvision`
and `torchtext` are not required by the evaluation pipeline. If your environment
already contains them, their versions must be compatible with PyTorch: a broken
optional `torchvision` installation can prevent Transformers from importing.
Use this project in an environment without a separately installed ESM package,
since both provide the `esm` import namespace.

ESM3 weights are published through Hugging Face. The former EvolutionaryScale URL
redirects to [the Biohub model snapshot](https://huggingface.co/biohub/esm3-sm-open-v1).
See [verified public assets](docs/PUBLIC_ASSETS.md) for fixed-version downloads.
Authenticate with your usual Hugging Face credentials if needed; do not put tokens
in source files. The default ESM3 loader downloads a snapshot; for local weights set:

```bash
export ESM3_MODEL_DIR=/path/to/esm3-snapshot
```

The directory must contain `data/weights/esm3_sm_open_v1.pth` and the other
snapshot assets required by ESM3. ESMFold defaults to
`facebook/esmfold_v1`; use `--fold-model /path/to/esmfold-snapshot` for local weights.

## Validate the CATH input format

This small example uses **synthetic coordinates**, requires only the Python
standard library, and does not load model weights:

```bash
python scripts/evaluate_cath.py \
  --dataset CATH4.2 --data-dir examples/cath \
  --output-dir results/format-check --dry-run
```

It validates the test split, sequence and backbone lengths, missing-coordinate
filtering, and the selected evaluation range. Use real benchmark data for model
evaluation; the synthetic fixture is not suitable for protein design.

## CATH evaluation

Prepare the files described in [docs/DATA.md](docs/DATA.md), then run:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/evaluate_cath.py \
  --dataset CATH4.2 --data-dir /path/to/cath4.2 \
  --config configs/source_defaults.json \
  --samples-per-target 20 --output-dir results/cath42-mcts
```

For CATH 4.3, change both `--dataset CATH4.3` and the data directory. To verify a
few real targets before a full run, add `--end-index 2 --samples-per-target 1` and
choose a new output directory. `--start-index` and `--end-index` refer to the
filtered test-set order; the upper bound is exclusive.

Useful options:

| Option | Meaning |
| --- | --- |
| `--subset all/short/single-chain` | Select All, L100 or the supplied single-chain split |
| `--method esm3` | Independent ESM3 iterative sampling baseline |
| `--rollout-mode jumpy` | Complete each evaluated partial state in one ESM3 forward step |
| `--reward-reference folded-native` | Compare folded candidates with the folded native sequence |
| `--resume` | Skip completed samples after checking config and dataset SHA-256 |
| `--save-pdb --save-tree` | Save final ESMFold structures and JSON search trees |
| `--policy-device cuda:0 --reward-device cuda:1` | Put ESM3 and ESMFold on separate visible GPUs |
| `--value-threshold 1.0` | Disable early threshold exit for a pure TM-score reward |
| `--final-selection greedy/best-terminal` | Budget-exhaustion selection; greedy matches the original source |

The source defaults use iterative rollouts and a target-backbone reward. The
policy restricts sampled residues to the 20 canonical amino acids accepted by
the folding and evaluation pipeline, for both MCTS and the ESM3 baseline. The
paper describes single-step jumpy denoising; use `--config configs/jumpy.json` to
enable that path. Both settings are explicit and are not asserted to reproduce
all paper tables. For metric definitions and discrepancies in the supplied
research code, read [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).

`--value-threshold` controls early exit; it does not reject the final output when
the budget is exhausted. Each sample runs its own search, rather than collecting
all above-threshold terminal sequences from one tree as in appendix Algorithm 1.
Novelty-weighted rewards (`--reward-alpha` below 1) and `folded-native` rewards use
the native sequence during search; disclose this when defining the benchmark.

Results include:

- `manifest.json`: resolved search options, input hashes, versions, and selected chains.
- `samples/*/result.json` and `sequence.fasta`: one independent search per sample.
- `results.json` / `results.csv`: generated sequences and per-sample metrics.
- `summary.json`: per-backbone metrics and macro-averages across backbones.
- `failures.json`: explicit failures; the command exits with a nonzero status if any fail.

`sc_tm_score` compares ESMFold(candidate) to ESMFold(native); `target_tm_score`
compares ESMFold(candidate) to the input backbone. Both TM scores are normalized
by reference length. They are reported separately from the MCTS node's historical
`search_value`. Diversity is the mean pairwise sequence difference within each
backbone and is `null` when only one sample is available.

Resume validates saved sample identity, sequence metadata and finite metrics.
Missing requested FASTA/PDB/tree files cause that sample to be regenerated;
corrupt result JSON is recorded as a failure. The generation alphabet is recorded
in the manifest; runs created before this restriction require a new output
directory to avoid mixing sampling protocols.

## Single PDB inference

```bash
python scripts/infer_pdb.py --pdb /path/to/target.pdb --chain-id A \
  --config configs/source_defaults.json --output-dir results/pdb-design
```

The selected PDB chain must contain canonical residues and complete N/CA/C/O
atoms. Native sequence information is removed from policy prompts; it is used
for reference folding and sequence evaluation. The output contains a FASTA,
ESMFold structure, metrics, and a JSON search tree.

`scripts/run_cath_sweep.sh` provides temperature and width sweeps.

## Tests

```bash
python -m unittest discover -s tests -v
```

Tests use synthetic structures and small fake models; they do not download
pretrained weights or claim biological validation.

## Citation

```bibtex
@inproceedings{liu2025protinvtree,
  title={ProtInvTree: Deliberate Protein Inverse Folding with Reward-guided Tree Search},
  author={Liu, Mengdi and Cheng, Xiaoxue and Gao, Zhangyang and Chang, Hong and
          Tan, Cheng and Shan, Shiguang and Chen, Xilin},
  booktitle={Advances in Neural Information Processing Systems},
  year={2025}
}
```

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for ESM attribution and model
terms. A research-code license remains to be selected by the authors.

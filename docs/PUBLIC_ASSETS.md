# Verified public data and model sources

Endpoints and API metadata were checked on 2026-10-13. Their identity with the
authors' unpublished local snapshots has not been established.

## CATH data

The [ProteinInvBench dataset release](https://github.com/A4Bio/ProteinInvBench/releases/tag/dataset_release)
links to [Zenodo record 10778307](https://zenodo.org/records/10778307). Both metadata
sources list `data.tar.gz` as 422,815,658 bytes. Zenodo reports MD5
`d30b9cd6ffbd7554cd00fc32b03f0b67`.

The [CATH 4.3 release](https://github.com/A4Bio/ProteinInvBench/releases/tag/kw_design_cath4.3)
provides `chain_set.jsonl` (536,944,512 bytes) and `chain_set_splits.json`
(201,137 bytes). These assets do not include L100/single-chain subset files.
Supply the intended subset memberships separately; do not fabricate them.

```bash
python scripts/download_cath.py --dataset CATH4.2 \
  --output-dir ../reproduction-assets/cath4.2 --cache-dir ../reproduction-assets/downloads
python scripts/download_cath.py --dataset CATH4.3 \
  --output-dir ../reproduction-assets/cath4.3 --cache-dir ../reproduction-assets/downloads
python scripts/evaluate_cath.py --dataset CATH4.2 \
  --data-dir ../reproduction-assets/cath4.2 --output-dir results/cath42-input-check --dry-run
```

The helper verifies the CATH 4.2 archive against the published size and MD5,
supports `cath`/`cath4.2` directory conventions, records extracted names and file
hashes, and refuses to overwrite datasets. Archive member names and extraction
have not been validated against a full downloaded archive.
Files are prepared in a temporary staging directory and published only after
the required files and checksums succeed. Interrupted downloads can be retried
with the same output directory; keep the cache files outside that directory.

## Model snapshots

The former EvolutionaryScale ESM3 model page redirects to
[biohub/esm3-sm-open-v1](https://huggingface.co/biohub/esm3-sm-open-v1).
Verified revision: `47f0545b2b6daf26a93439a3cd610f4f7f3d5478`.
Its `data/weights/` contains the `.pth` files required by the supplied ESM code.
The full snapshot is approximately 5.5 GB.

[facebook/esmfold_v1](https://huggingface.co/facebook/esmfold_v1) has verified revision
`75a3841ee059df2bf4d56688166c8fb459ddd97a`; its `pytorch_model.bin` is approximately
8.44 GB. Both model APIs reported ungated public repositories when checked.

Download fixed snapshots with the installed huggingface_hub package:

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="biohub/esm3-sm-open-v1",
    revision="47f0545b2b6daf26a93439a3cd610f4f7f3d5478",
    local_dir="../reproduction-assets/esm3",
)
snapshot_download(
    repo_id="facebook/esmfold_v1",
    revision="75a3841ee059df2bf4d56688166c8fb459ddd97a",
    local_dir="../reproduction-assets/esmfold",
)
```

Then run a small GPU check before a full evaluation:

```bash
export ESM3_MODEL_DIR="$(realpath ../reproduction-assets/esm3)"
python scripts/evaluate_cath.py --dataset CATH4.2 \
  --data-dir ../reproduction-assets/cath4.2 --fold-model ../reproduction-assets/esmfold \
  --config configs/source_defaults.json --end-index 2 --samples-per-target 1 \
  --output-dir results/cath42-model-check
```

These model checkpoints were not loaded for the software checks. Consult
their asset terms. The fixed commits are verified public versions; their equality
with the paper's local weight files has not been proved.

## Public reproduction settings

The [official NeurIPS paper page](https://proceedings.nips.cc/paper_files/paper/2025/hash/291a1c80e9025f0ac2273fcf017fbd63-Abstract-Conference.html)
points to ProteinInvBench. The GitHub API listed `release`, `lightning`, and
`BeforeSunrise710-patch-1` branches. The visible README describes ProteinInvBench;
a complete per-table ProtInvTree configuration was not verified. The supplied
local working tree remains the source for this audit. No exhaustive claim about
every public commit is made.

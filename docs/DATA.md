# CATH benchmark data

Supply the exact preprocessed CATH 4.2 or CATH 4.3 benchmark files used for your
experiment. The paper points to `https://github.com/A4Bio/ProteinInvBench/` for
code. This repository does not bundle the benchmark, rebuild the official split,
or claim a particular download is identical to the authors' local data.

Verified download locations and the download helper are documented in
[PUBLIC_ASSETS.md](PUBLIC_ASSETS.md). Data are kept outside the code release.

```text
cath4.2/                   # or cath4.3/
  chain_set.jsonl
  chain_set_splits.json
  test_split_L100.json      # needed for --subset short
  test_split_sc.json        # needed for --subset single-chain
```

Each JSONL line has the form:

```json
{"name":"chain_name","seq":"ACD","coords":{"N":[[0,0,0],[1,0,0],[2,0,0]],"CA":[[0,1,0],[1,1,0],[2,1,0]],"C":[[0,2,0],[1,2,0],[2,2,0]],"O":[[0,3,0],[1,3,0],[2,3,0]]}}
```

The numbers above illustrate the schema only. Each atom array must have exactly
the same residue count as `seq`. Use canonical amino acid letters. Missing
coordinates may be `null` or nonfinite values in a locally produced JSONL file;
the loader drops the affected residue from both the sequence and coordinates.
The backbone atom order passed to ESM3 is N, CA, C, O. Conversion to ESM's atom37
representation places oxygen at atom37 index 4.

The main split JSON must include a `test` list of chain names; training and
validation lists are allowed but are not loaded for evaluation. Optional subset
files use the same `{"test": [...]}` format and must be contained in the main
test split. Single-chain membership is read from the supplied subset file rather
than guessed from chain names. Short membership likewise uses the supplied L100
file, preserving the intended benchmark split.

Filtering follows the original source's test-data behavior:

1. Keep chains in the selected official test split, in JSONL file order.
2. Exclude chains with noncanonical residues or raw length above `--max-length`.
3. Remove residues with missing N/CA/C/O coordinates, synchronously in all arrays.
4. Exclude chains shorter than `--min-length` after filtering.
5. Apply the requested start/end range to this filtered order.

The default maximum length is 500, matching the supplied CATH loader. The default
minimum is 1: the old inference script exposed a minimum of 40 but never passed
it to that loader. Set `--min-length 40` explicitly if desired, and report this
change in the evaluation protocol. No random cropping is applied.

The manifest records skipped-chain counts, removed residues, retained indices,
selected names, and input hashes. Missing split members, duplicate selected chain
names, and mismatched coordinate lengths produce errors instead of a silently
smaller test set.

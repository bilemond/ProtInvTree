#!/usr/bin/env python3
"""Download official ProteinInvBench CATH data; leave datasets outside the code archive."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import urllib.request

DATA_ARCHIVE = "https://github.com/A4Bio/ProteinInvBench/releases/download/dataset_release/data.tar.gz"
DATA_BYTES = 422815658
DATA_MD5 = "d30b9cd6ffbd7554cd00fc32b03f0b67"
CATH43_BASE = "https://github.com/A4Bio/ProteinInvBench/releases/download/kw_design_cath4.3/"
FILES = ("chain_set.jsonl", "chain_set_splits.json", "test_split_L100.json", "test_split_sc.json")


def checksum(path, algorithm="sha256"):
    h = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def download(url, path, expected_size=None, expected_md5=None):
    if path.exists():
        if (expected_size is None or path.stat().st_size == expected_size) and (
            expected_md5 is None or checksum(path, "md5") == expected_md5
        ):
            return
        raise ValueError(f"Existing download does not match expected archive: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    total, next_report = 0, 32 * 1024 * 1024
    with urllib.request.urlopen(url, timeout=120) as response, temporary.open("wb") as handle:
        while block := response.read(1024 * 1024):
            handle.write(block)
            total += len(block)
            if total >= next_report:
                print(f"Downloading {path.name}: {total / 1024**2:.0f} MiB", flush=True)
                next_report += 32 * 1024 * 1024
    if expected_size is not None and total != expected_size:
        raise ValueError(f"Incomplete download: {total} bytes, expected {expected_size}")
    if expected_md5 and checksum(temporary, "md5") != expected_md5:
        raise ValueError("Archive MD5 differs from the official Zenodo metadata")
    temporary.replace(path)


def extract_cath(archive, output, dataset):
    aliases = {"CATH4.2": {"cath", "cath4.2"}, "CATH4.3": {"cath4.3"}}[dataset]
    extracted = []
    output.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as bundle:
        for member in bundle:
            parts = PurePosixPath(member.name).parts
            if not member.isfile() or not parts or parts[-1] not in FILES:
                continue
            if not any(part.lower() in aliases for part in parts[:-1]):
                continue
            target = output / parts[-1]
            if target.exists():
                raise FileExistsError(f"Refusing to overwrite dataset file: {target}")
            with bundle.extractfile(member) as source, target.open("wb") as destination:
                shutil.copyfileobj(source, destination)
            extracted.append(member.name)
    return extracted


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("CATH4.2", "CATH4.3", "archive"), default="CATH4.2")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("Output directory must be empty")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.dataset == "CATH4.3":
        for name, size in (("chain_set.jsonl", 536944512), ("chain_set_splits.json", 201137)):
            cached = args.cache_dir / "cath4.3" / name
            download(CATH43_BASE + name, cached, expected_size=size)
            shutil.copyfile(cached, args.output_dir / name)
        source = CATH43_BASE
        # Subsets are not part of this release asset. Do not fabricate memberships.
        extracted = list(FILES[:2])
    else:
        archive = args.cache_dir / "data.tar.gz"
        download(DATA_ARCHIVE, archive, DATA_BYTES, DATA_MD5)
        source = DATA_ARCHIVE
        if args.dataset == "archive":
            with tarfile.open(archive, "r:gz") as bundle:
                names = [member.name for member in bundle if member.isfile()]
            (args.output_dir / "archive_members.json").write_text(json.dumps(names, indent=2) + "\n")
            print(json.dumps([name for name in names if "cath" in name.lower()], indent=2))
            return 0
        extracted = extract_cath(archive, args.output_dir, args.dataset)
    for name in FILES[:2]:
        if not (args.output_dir / name).exists():
            raise FileNotFoundError(f"Official download did not provide {name}; inspect archive members")
    provenance = {"dataset": args.dataset, "source_url": source, "extracted_members": extracted,
                  "sha256": {name: checksum(args.output_dir / name) for name in FILES
                             if (args.output_dir / name).exists()}}
    (args.output_dir / "download_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"Saved {args.dataset} to {args.output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

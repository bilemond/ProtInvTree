"""Exercise download failures and publication using local fixtures, without network."""

import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location(
    "download_cath", Path(__file__).resolve().parents[1] / "scripts/download_cath.py")
download_cath = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(download_cath)


class DownloadTests(unittest.TestCase):
    def test_cache_inside_output_is_rejected_before_downloading(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "dataset"
            with patch.object(download_cath, "download") as download:
                with self.assertRaises(SystemExit):
                    download_cath.main(["--dataset", "CATH4.3", "--output-dir", str(output),
                                        "--cache-dir", str(output)])
                download.assert_not_called()
            self.assertFalse(output.exists())

    def test_interrupted_cath43_download_can_retry_without_partial_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output, cache = root / "dataset", root / "cache"
            output.mkdir()
            interrupted = True

            def download(url, path, **kwargs):
                if path.name == "chain_set_splits.json" and interrupted:
                    raise OSError("synthetic network interruption")
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("{}")

            args = ["--dataset", "CATH4.3", "--output-dir", str(output), "--cache-dir", str(cache)]
            with patch.object(download_cath, "download", side_effect=download):
                with self.assertRaisesRegex(OSError, "interruption"):
                    download_cath.main(args)
                self.assertEqual(list(output.iterdir()), [])
                self.assertEqual(list(root.glob("dataset.download-*")), [])
                interrupted = False
                self.assertEqual(download_cath.main(args), 0)
            for filename in ("chain_set.jsonl", "chain_set_splits.json", "download_provenance.json"):
                self.assertTrue((output / filename).is_file())

    def test_archive_extraction_publishes_only_after_required_files_exist(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output, cache = root / "dataset", root / "cache"
            include_splits = False

            def download(url, path, *args):
                path.parent.mkdir(parents=True, exist_ok=True)
                with tarfile.open(path, "w:gz") as bundle:
                    names = ["chain_set.jsonl"] + (["chain_set_splits.json"] if include_splits else [])
                    for name in names:
                        member = tarfile.TarInfo("data/cath/" + name)
                        member.size = 2
                        bundle.addfile(member, io.BytesIO(b"{}"))

            args = ["--dataset", "CATH4.2", "--output-dir", str(output), "--cache-dir", str(cache)]
            with patch.object(download_cath, "download", side_effect=download):
                with self.assertRaises(FileNotFoundError):
                    download_cath.main(args)
                self.assertFalse(output.exists())
                self.assertEqual(list(root.glob("dataset.download-*")), [])
                include_splits = True
                self.assertEqual(download_cath.main(args), 0)
            provenance = json.loads((output / "download_provenance.json").read_text())
            self.assertEqual(provenance["sha256"]["chain_set.jsonl"], hashlib.sha256(b"{}").hexdigest())

    def test_incomplete_cached_download_is_not_committed_and_can_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "download.json"
            with patch.object(download_cath.urllib.request, "urlopen", return_value=io.BytesIO(b"bad")):
                with self.assertRaisesRegex(ValueError, "Incomplete download"):
                    download_cath.download("https://example.invalid/fixture", path, expected_size=10)
            self.assertFalse(path.exists())
            with patch.object(download_cath.urllib.request, "urlopen", return_value=io.BytesIO(b"0123456789")):
                download_cath.download("https://example.invalid/fixture", path, expected_size=10)
            self.assertEqual(path.read_bytes(), b"0123456789")
            self.assertFalse(path.with_suffix(".json.partial").exists())

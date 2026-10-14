import json
from pathlib import Path
import tempfile
import unittest

from protinvtree.data import read_cath
from protinvtree.metrics import diversity, recovery
from protinvtree.results import sample_seed, summarize

FIXTURE = Path(__file__).resolve().parents[1] / "examples/cath"


class DataTests(unittest.TestCase):
    def test_file_order_and_missing_residue_alignment(self):
        records, stats, hashes = read_cath(FIXTURE)
        self.assertEqual([item.name for item in records], ["synthetic_a", "synthetic_b"])
        self.assertEqual(records[1].sequence, "HILMN")
        self.assertEqual(records[1].residue_indices, [0, 1, 3, 4, 5])
        self.assertEqual(stats["removed_residues"], 1)
        self.assertEqual(len(hashes["chain_set.jsonl"]), 64)

    def test_official_subsets(self):
        records, _, _ = read_cath(FIXTURE, "single-chain")
        self.assertEqual([item.name for item in records], ["synthetic_a"])

    def test_length_filter_is_explicit(self):
        records, stats, _ = read_cath(FIXTURE, min_length=6)
        self.assertEqual(len(records), 1)
        self.assertEqual(stats["too_short_after_filter"], 1)

    def test_missing_split_member_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "chain_set_splits.json").write_text(json.dumps({"test": ["absent"]}))
            (root / "chain_set.jsonl").write_text("")
            with self.assertRaisesRegex(ValueError, "missing chains"):
                read_cath(root)


class MetricTests(unittest.TestCase):
    def test_recovery_and_diversity(self):
        self.assertEqual(recovery("AAAA", "AACA"), 0.75)
        self.assertEqual(diversity(["AAAA", "CCCC", "AACC"]), 2 / 3)
        self.assertIsNone(diversity(["AAAA"]))
        with self.assertRaises(ValueError):
            recovery("AA", "A")

    def test_stable_seed_independent_of_job_partition(self):
        seed = sample_seed(111, "target", 0)
        self.assertEqual(seed, sample_seed(111, "target", 0))
        self.assertNotEqual(seed, sample_seed(111, "target", 1))
        self.assertNotEqual(seed, sample_seed(112, "target", 0))

    def test_diversity_groups_by_backbone(self):
        from protinvtree.results import METRICS
        rows = []
        for name, sequence, value in [("a", "AA", 1), ("a", "CC", 1), ("b", "AC", 0)]:
            rows.append({"name": name, "sequence": sequence, **{metric: value for metric in METRICS}})
        result = summarize(rows)
        self.assertEqual(result["aggregate"]["sc_tm_score"], 0.5)
        self.assertEqual(result["aggregate"]["diversity"], 1.0)
        self.assertEqual(result["aggregate"]["backbones_with_diversity"], 1)

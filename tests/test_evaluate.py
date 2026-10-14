import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from protinvtree.data import Backbone
from protinvtree.evaluate import main
from protinvtree.results import METRICS

FIXTURE = Path(__file__).resolve().parents[1] / "examples/cath"


class FakeRunner:
    calls = 0

    def __init__(self, cfg):
        pass

    def prepare(self, protein, reference):
        pass

    def design(self, protein, reference, seed, method):
        self.__class__.calls += 1
        result = {metric: 0.5 for metric in METRICS}
        result.update(sequence="A" * len(reference), reference_sequence=reference,
                      seed=seed, method=method, search_value=0.2, search_complete=False)
        return result, "synthetic", []


class EvaluateTests(unittest.TestCase):
    def args(self, directory):
        return ["--data-dir", str(FIXTURE), "--output-dir", directory, "--samples-per-target", "2"]

    def test_dry_run_does_not_import_models(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch("builtins.__import__", wraps=__import__) as imports:
                self.assertEqual(main(self.args(directory) + ["--dry-run"]), 0)
            names = [call.args[0] for call in imports.call_args_list]
            self.assertNotIn("runner", names)
            self.assertNotIn("torch", names)
            plan = json.loads((Path(directory) / "dry_run.json").read_text())
            self.assertEqual(plan["expected_samples"], 4)

    def test_evaluation_resume_and_configuration_check(self):
        FakeRunner.calls = 0
        with tempfile.TemporaryDirectory() as directory:
            args = self.args(directory)
            with patch("protinvtree.runner.DesignRunner", FakeRunner), patch.object(Backbone, "to_protein", return_value=object()):
                self.assertEqual(main(args), 0)
                self.assertEqual(FakeRunner.calls, 4)
                self.assertEqual(main(args + ["--resume"]), 0)
                self.assertEqual(FakeRunner.calls, 4)
                with self.assertRaises(SystemExit):
                    main(args + ["--resume", "--temperature", "0.5"])
            summary = json.loads((Path(directory) / "summary.json").read_text())
            self.assertTrue(summary["metadata"]["complete"])
            self.assertEqual(summary["aggregate"]["samples"], 4)
            self.assertEqual(len(list((Path(directory) / "samples").glob("*/sequence.fasta"))), 4)

    def test_failed_samples_are_reported_and_exit_nonzero(self):
        class FailingRunner(FakeRunner):
            def design(self, *args):
                raise ValueError("synthetic failure")
        with tempfile.TemporaryDirectory() as directory:
            with patch("protinvtree.runner.DesignRunner", FailingRunner), patch.object(Backbone, "to_protein", return_value=object()):
                self.assertEqual(main(self.args(directory) + ["--continue-on-error"]), 1)
            summary = json.loads((Path(directory) / "summary.json").read_text())
            self.assertFalse(summary["metadata"]["complete"])
            self.assertEqual(len(summary["metadata"]["failed_samples"]), 4)

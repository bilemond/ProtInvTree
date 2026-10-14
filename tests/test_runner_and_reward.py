import math
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from esm.sdk.api import ESMProtein

from protinvtree.options import DEFAULTS
from protinvtree.reward import Reward_model
from protinvtree.runner import DesignRunner
from test_sampling import FakeESM3


def synthetic_pdb(length):
    lines = []
    for i in range(length):
        x, y, z = 3.8 * i, 2.0 * math.sin(i), 2.0 * math.cos(i)
        lines.append(f"ATOM  {i+1:5d}  CA  ALA A{i+1:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00 80.00           C  ")
    return "\n".join(lines) + "\nTER\nEND\n"


class FakeFold(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0
        self.trunk = SimpleNamespace(set_chunk_size=lambda size: None)

    def infer_pdbs(self, sequences):
        self.calls += 1
        return [synthetic_pdb(len(sequence)) for sequence in sequences]


class RewardTests(unittest.TestCase):
    def setUp(self):
        self.fold = FakeFold()
        with patch("protinvtree.reward.EsmForProteinFolding.from_pretrained", return_value=self.fold):
            self.reward = Reward_model(device="cpu")
        coordinates = torch.zeros(8, 37, 3)
        coordinates[:, 1] = torch.as_tensor(self.reward._ca_from_pdb(synthetic_pdb(8)))
        self.protein = ESMProtein(sequence="A" * 8, coordinates=coordinates)
        self.reward.update_label(self.protein, self.protein.sequence)

    def test_real_pdb_parser_and_tm_align_identical_structure(self):
        metrics, pdb = self.reward.evaluate("A" * 8)
        self.assertAlmostEqual(metrics["target_tm_score"], 1.0, places=6)
        self.assertAlmostEqual(metrics["sc_tm_score"], 1.0, places=6)
        self.assertEqual(metrics["novelty"], 0.0)
        self.assertEqual(self.fold.calls, 1)

    def test_reward_selects_the_requested_reference(self):
        references = []
        def align(coords, reference, sequence, native):
            references.append(reference)
            return SimpleNamespace(tm_norm_chain2=0.7, rmsd=1.0)
        with patch("protinvtree.reward.tm_align", side_effect=align):
            self.reward.get_reward("C" * 8)
            self.assertIs(references[-1], self.reward.target_ca)
            self.reward.reference = "folded-native"
            self.reward.get_reward("C" * 8)
            self.assertIs(references[-1], self.reward.native_ca)

    def test_wrong_sequence_length_fails(self):
        calls = self.fold.calls
        with self.assertRaisesRegex(ValueError, "length differs"):
            self.reward.evaluate("AAAA")
        self.assertEqual(self.fold.calls, calls)

    def test_target_reward_does_not_fold_native_reference(self):
        self.assertEqual(self.fold.calls, 0)
        self.reward.get_reward("C" * 8)
        self.assertEqual(self.fold.calls, 1)
        self.assertIsNone(self.reward.native_ca)

    def test_folded_native_reference_is_prepared_only_once(self):
        self.reward.reference = "folded-native"
        self.reward.get_reward("C" * 8)
        self.reward.get_reward("C" * 8)
        self.assertEqual(self.fold.calls, 2)
        self.assertIsNotNone(self.reward.native_ca)

    def test_invalid_candidates_fail_before_folding(self):
        calls = self.fold.calls
        for sequence in ("", "AAAA_AAA", "AAAAXAAA", None):
            with self.assertRaises(ValueError):
                self.reward.get_reward(sequence)
        self.assertEqual(self.fold.calls, calls)

    def test_invalid_constructor_settings_fail_before_model_loading(self):
        with patch("protinvtree.reward.EsmForProteinFolding.from_pretrained") as loader:
            for options in ({"alpha": float("nan")}, {"alpha": -0.1}, {"alpha": 1.1},
                            {"reference": "typo"}, {"chunk_size": -1}):
                with self.assertRaises(ValueError):
                    Reward_model(device="cpu", **options)
            loader.assert_not_called()

    def test_invalid_alignment_score_is_not_backpropagated(self):
        with patch("protinvtree.reward.tm_align", return_value=SimpleNamespace(
            tm_norm_chain2=float("nan"), rmsd=0.0
        )):
            with self.assertRaisesRegex(ValueError, "invalid reference-normalized"):
                self.reward.get_reward("C" * 8)

    def test_target_coordinates_are_a_snapshot(self):
        coordinates = self.protein.coordinates.to(torch.float64)
        self.reward.update_label(ESMProtein(coordinates=coordinates), "A" * 8)
        expected = self.reward.target_ca.copy()
        coordinates[:, 1] += 10.0
        np.testing.assert_array_equal(self.reward.target_ca, expected)

    def test_failed_new_native_fold_cannot_reuse_old_reference(self):
        self.reward.prepare_native_reference()
        old_native = self.reward.native_ca
        self.reward.update_label(self.protein, "C" * 8)
        self.reward.reference = "folded-native"
        with patch.object(self.fold, "infer_pdbs", side_effect=RuntimeError("synthetic failure")):
            with self.assertRaisesRegex(RuntimeError, "synthetic failure"):
                self.reward.get_reward("A" * 8)
        self.assertIsNone(self.reward.native_ca)
        self.assertEqual(self.reward.gt_sequence, "C" * 8)
        self.assertIsNotNone(old_native)


class FakeReward(nn.Module):
    def update_label(self, *args):
        pass

    def prepare_native_reference(self):
        pass

    def get_reward_from_rollout(self, candidates):
        return [0.2] * len(candidates), 0.0

    def evaluate(self, sequence):
        return {"target_tm_score": 0.8, "target_rmsd": 1.0, "sc_tm_score": 0.7,
                "sc_rmsd": 1.2, "recovery": 0.3, "novelty": 0.7}, synthetic_pdb(len(sequence))


class RunnerTests(unittest.TestCase):
    def runner(self, **overrides):
        cfg = {**DEFAULTS, "policy_device": "cpu", "reward_device": "cpu", **overrides}
        model = FakeESM3()
        with patch("protinvtree.policy.ESM3.from_pretrained", return_value=model), patch("protinvtree.runner.Reward_model", return_value=FakeReward()):
            return DesignRunner(cfg)

    def test_zero_budget_fallback_scores_actual_returned_sequence(self):
        runner = self.runner(simulations=0, pre_expansion_depth=0)
        protein = ESMProtein(sequence="ACDEFGHIKL", coordinates=torch.zeros(10, 37, 3))
        runner.prepare(protein, protein.sequence)
        result, _, tree = runner.design(protein, protein.sequence, seed=111)
        self.assertFalse(result["search_complete"])
        self.assertEqual(result["search_value"], 0.0)
        self.assertEqual(result["reward"], 0.8)
        self.assertEqual(result["sequence"], "A" * 10)
        self.assertEqual(len(tree), 1)

    def test_actual_policy_agent_mcts_produces_terminal_design(self):
        runner = self.runner(total_steps=2, pre_expansion_depth=1, simulations=3)
        protein = ESMProtein(sequence="ACDEFGHIKL", coordinates=torch.zeros(10, 37, 3))
        runner.prepare(protein, protein.sequence)
        result, _, tree = runner.design(protein, protein.sequence, seed=111)
        self.assertTrue(result["search_complete"])
        self.assertEqual(result["sequence"], "A" * 10)
        self.assertAlmostEqual(result["search_value"], 0.2)
        self.assertEqual(result["reward"], 0.8)
        self.assertGreater(len(tree), 2)
        self.assertFalse(runner.policy.model.saw_native_sequence)

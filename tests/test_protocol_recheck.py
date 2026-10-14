"""Executable checks for reward-label dependence and the actual output protocol."""
import unittest
from unittest.mock import patch

import torch
from esm.sdk.api import ESMProtein

from protinvtree.reward import Reward_model
from protinvtree.searching.search import MCTS
from protinvtree.searching.tree import Node
from test_runner_and_reward import FakeFold, synthetic_pdb
from test_search import ToyAgent


class ProtocolRecheckTests(unittest.TestCase):
    def make_reward(self, alpha=1.0, reference="target", fold=None):
        fold = FakeFold() if fold is None else fold
        with patch("protinvtree.reward.EsmForProteinFolding.from_pretrained", return_value=fold):
            reward = Reward_model(alpha=alpha, reference=reference, device="cpu")
        coordinates = torch.zeros(8, 37, 3, dtype=torch.float64)
        coordinates[:, 1] = torch.tensor(reward._ca_from_pdb(synthetic_pdb(8)))
        return reward, ESMProtein(coordinates=coordinates), fold

    def test_default_target_reward_invariant_to_same_length_native_labels(self):
        reward, protein, fold = self.make_reward()
        scores = []
        for native in ("A" * 8, "C" * 8, "ACDEFGHI"):
            reward.update_label(protein, native)
            scores.append(reward.get_reward("A" * 8))
            self.assertIsNone(reward.native_ca)
        self.assertEqual(scores, [1.0, 1.0, 1.0])
        self.assertEqual(fold.calls, 1)

    def test_novelty_weight_introduces_native_label_feedback(self):
        reward, protein, _ = self.make_reward(alpha=0.5)
        reward.update_label(protein, "A" * 8)
        matching_label_score = reward.get_reward("A" * 8)
        reward.update_label(protein, "C" * 8)
        different_label_score = reward.get_reward("A" * 8)
        self.assertEqual(matching_label_score, 0.5)
        self.assertEqual(different_label_score, 1.0)

    def test_folded_native_reference_introduces_native_label_feedback(self):
        class SequenceAwareFold(FakeFold):
            def infer_pdbs(self, sequences):
                self.calls += 1
                result = []
                for sequence in sequences:
                    if sequence.startswith("C"):
                        lines = []
                        for i in range(len(sequence)):
                            x, y, z = 3.8 * i, 20.0 * (-1)**i, float(i*i)
                            lines.append(f"ATOM  {i+1:5d}  CA  ALA A{i+1:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00 80.00           C  ")
                        result.append("\n".join(lines) + "\nTER\nEND\n")
                    else:
                        result.append(synthetic_pdb(len(sequence)))
                return result
        reward, protein, _ = self.make_reward(reference="folded-native", fold=SequenceAwareFold())
        reward.update_label(protein, "A" * 8)
        score_before = reward.get_reward("A" * 8)
        reward.update_label(protein, "C" * 8)
        score_after = reward.get_reward("A" * 8)
        self.assertAlmostEqual(score_before, 1.0)
        self.assertLess(score_after, score_before)

    def test_value_threshold_is_early_exit_not_output_acceptance_filter(self):
        search = MCTS(ToyAgent(depth=1), "root", value_threshold=1.0)
        best, _, _ = search.run_search_with_pre_expansion(1, 0)
        self.assertTrue(best.is_terminated())
        self.assertLess(best.value(), search.value_threshold)

    def test_threshold_equality_does_not_trigger_early_exit(self):
        search = MCTS(ToyAgent(), "root", value_threshold=0.5)
        terminal = Node(parent=search.root, state="complete", initial_value=0.5)
        terminal.set_terminated()
        search.root.add_child(terminal)
        self.assertIsNone(search._threshold_solution())

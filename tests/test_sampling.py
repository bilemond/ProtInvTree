"""Exercise the real ESM tokenizers/samplers using a tiny deterministic fake model."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn
from esm.sdk.api import ESMProtein, ESMProteinTensor, ForwardTrackData, LogitsOutput
from esm.tokenization import (TokenizerCollection, EsmSequenceTokenizer, StructureTokenizer,
                              SecondaryStructureTokenizer, SASADiscretizingTokenizer,
                              InterProQuantizedTokenizer)
from esm.utils.constants.models import ESM3_OPEN_SMALL
from protinvtree.policy import Policy_model
from protinvtree.searching.utils import clean_outputs, same_state


class FakeESM3(nn.Module):
    def __init__(self):
        super().__init__()
        # Residue-annotation vocabulary is a model-snapshot asset. This pipeline
        # samples only sequence tokens, so use a stub for the unused track.
        self.tokenizers = TokenizerCollection(
            EsmSequenceTokenizer(), StructureTokenizer(), SecondaryStructureTokenizer("ss8"),
            SASADiscretizingTokenizer(), InterProQuantizedTokenizer(),
            SimpleNamespace(mask_token_id=0, pad_token_id=0, bos_token_id=0, eos_token_id=0),
        )
        self.calls = 0
        self.saw_native_sequence = False

    def encode(self, protein):
        self.saw_native_sequence |= protein.sequence is not None
        coordinates = torch.nn.functional.pad(protein.coordinates, (0, 0, 0, 0, 1, 1), value=float("inf"))
        return ESMProteinTensor(coordinates=coordinates)

    def logits(self, tokens, config):
        self.calls += 1
        shape = (*tokens.sequence.shape, 64)
        logits = torch.full(shape, -100.0)
        logits[..., self.tokenizers.sequence.encode("A", add_special_tokens=False)[0]] = 100.0
        return LogitsOutput(logits=ForwardTrackData(sequence=logits))


class SamplingTests(unittest.TestCase):
    def setUp(self):
        self.model = FakeESM3()
        with patch("protinvtree.policy.ESM3.from_pretrained", return_value=self.model):
            self.policy = Policy_model(device="cpu")
        self.protein = ESMProtein(sequence="ACDEFGHIKL", coordinates=torch.zeros(10, 37, 3))
        self.params = {"total_steps": 4, "temperature": 0.3, "top_p": 1.0,
                       "strategy": "random", "schedule": "cosine", "n": 1}

    def test_native_sequence_is_removed_and_tensor_type_is_preserved(self):
        initial = self.policy.prepare_initial_state(self.protein)
        self.assertFalse(self.model.saw_native_sequence)
        self.assertIsInstance(initial, ESMProteinTensor)
        self.assertEqual(int((initial.sequence == 32).sum()), 10)
        self.assertIs(self.policy._maybe_mock_input(initial, len(initial)), initial)

    def test_jumpy_rollout_is_one_forward_and_does_not_mutate_parent(self):
        initial = self.policy.prepare_initial_state(self.protein)
        original = initial.sequence.clone()
        batch = self.policy.batch_ESMTensor([initial])
        self.policy.rollout_mode = "jumpy"
        output = self.policy.roll_out_final_states_given_batched_states(batch, self.params, [12], [10], 1)
        self.assertEqual(self.model.calls, 1)
        self.assertEqual(output[0][0].sequence, "A" * 10)
        self.assertTrue(torch.equal(initial.sequence, original))

    def test_iterative_rollout_preserves_residues_and_uses_remaining_steps(self):
        initial = self.policy.prepare_initial_state(self.protein)
        initial.sequence[1] = self.model.tokenizers.sequence.encode("C", add_special_tokens=False)[0]
        batch = self.policy.batch_ESMTensor([initial])
        output = self.policy.roll_out_final_states_given_batched_states(batch, self.params, [12], [10], 1)
        self.assertEqual(self.model.calls, 3)
        self.assertEqual(output[0][0].sequence, "C" + "A" * 9)

    def test_tensor_deduplication_uses_contents(self):
        left = ESMProteinTensor(sequence=torch.tensor([0, 5, 2]))
        right = ESMProteinTensor(sequence=torch.tensor([0, 5, 2]))
        self.assertTrue(same_state(left, right))
        self.assertEqual(len(clean_outputs([[left, right]])), 1)

    def test_focus_updates_only_subset_and_keeps_parent_unchanged(self):
        initial = self.policy.prepare_initial_state(self.protein)
        expanded = self.policy.expand_nodes_given_batched_states(
            self.policy.batch_ESMTensor([initial]), self.params, [12], [10], [0])
        child = expanded[0][0]
        self.assertGreater(int((child.sequence == 32).sum()), 0)
        self.assertLess(int((child.sequence == 32).sum()), 10)
        self.assertEqual(int((initial.sequence == 32).sum()), 10)

    def test_mixed_length_rollouts_trim_padding_before_decoding(self):
        short = self.policy.prepare_initial_state(ESMProtein(coordinates=torch.zeros(5, 37, 3)))
        initial = self.policy.prepare_initial_state(self.protein)
        for mode in ("iterative", "jumpy"):
            self.policy.rollout_mode = mode
            output = self.policy.roll_out_final_states_given_batched_states(
                self.policy.batch_ESMTensor([initial, short]), self.params, [12, 7], [10, 5], 1)
            self.assertEqual([candidates[0].sequence for candidates in output], ["A" * 10, "A" * 5])
            self.assertEqual([len(candidates[0].coordinates) for candidates in output], [10, 5])

    def test_mixed_length_expansion_preserves_lengths_for_next_rollout(self):
        short = self.policy.prepare_initial_state(ESMProtein(coordinates=torch.zeros(5, 37, 3)))
        initial = self.policy.prepare_initial_state(self.protein)
        expanded = self.policy.expand_nodes_given_batched_states(
            self.policy.batch_ESMTensor([initial, short]), self.params, [12, 7], [10, 5], [0, 0])
        children = [candidates[0] for candidates in expanded]
        self.assertEqual([len(child) for child in children], [12, 7])
        self.assertEqual([len(child.coordinates) for child in children], [12, 7])
        remaining_masks = [int((child.sequence == self.model.tokenizers.sequence.mask_token_id).sum())
                           for child in children]
        self.policy.rollout_mode = "jumpy"
        output = self.policy.roll_out_final_states_given_batched_states(
            self.policy.batch_ESMTensor(children), self.params, [len(child) for child in children],
            remaining_masks, 1)
        self.assertEqual([candidates[0].sequence for candidates in output], ["A" * 10, "A" * 5])

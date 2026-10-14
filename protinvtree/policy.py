from typing import Sequence
import copy
import attr
import torch
from torch import nn
from tqdm import tqdm
from esm.models.esm3 import ESM3
from esm.utils.sampling import _BatchedESMProteinTensor
from esm.utils.generation import _batch_forward, _sample_per_prompt, _slice_tensor_dataclass, _trim_sequence_tensor_dataclass, _get_iterative_sampling_mask_for_prompt_and_step, _stack_protein_tensors, _make_masked_inputs
from esm.sdk.api import ESM3InferenceClient, ESMProtein, ESMProteinError, ESMProteinTensor, GenerationConfig, LogitsOutput, SamplingConfig, SamplingTrackConfig
from esm.tokenization import TokenizerCollectionProtocol
from esm.utils.decoding import decode_sequence
from .data import AMINO_ACIDS


def canonical_invalid_ids(tokenizer):
    """Restrict generated residues to the alphabet accepted by folding/evaluation."""
    valid_ids = set(tokenizer.encode("".join(sorted(AMINO_ACIDS)), add_special_tokens=False))
    return sorted(set(tokenizer.all_token_ids) - valid_ids)

class Policy_model(nn.Module):

    def __init__(self, model_name="esm3-open", device="cuda", rollout_mode="iterative"):
        super(Policy_model, self).__init__()
        self.model: ESM3InferenceClient = ESM3.from_pretrained(model_name, device=torch.device(device))
        self.tokenizers: TokenizerCollectionProtocol = self.model.tokenizers
        self.rollout_mode = rollout_mode

    def decode_state(self, state):
        """Decode just the sequence, without loading structure/function decoder weights."""
        return ESMProtein(
            sequence=decode_sequence(state.sequence, self.tokenizers.sequence),
            coordinates=None if state.coordinates is None else state.coordinates[1:-1],
        )

    def prepare_initial_state(self, structure_input: ESMProtein, with_structure: bool=False, with_coordinates: bool=True) -> ESMProteinTensor:
        structure_input = attr.evolve(structure_input, sequence=None, secondary_structure=None, sasa=None, function_annotations=None)
        structure_tensor = self.ESMProtein_to_ESMTensor(structure_input, with_structure, with_coordinates)
        initial_state = self._maybe_mock_input(structure_tensor, len(structure_tensor), fn='sequence')
        return initial_state

    def ESMProtein_to_ESMTensor(self, inputs: ESMProtein, with_structure: bool=False, with_coordinates: bool=True) -> ESMProteinTensor:
        """
        Convert ESMProtein to ESMProteinTensor.
        """
        if isinstance(inputs, ESMProtein):
            inputs = [inputs]
        tokens_tensor = [self.model.encode(protein) for protein in inputs]
        if with_structure == False:
            for tokens in tokens_tensor:
                tokens.structure = None
        if with_coordinates == False:
            for tokens in tokens_tensor:
                tokens.coordinates = None
        return tokens_tensor[0] if len(tokens_tensor) == 1 else tokens_tensor

    def _maybe_mock_input(self, input_tokens: ESMProteinTensor, sequence_length: int, fn: str='sequence') -> ESMProteinTensor:
        tensors = getattr(input_tokens, fn)
        if tensors is not None:
            return input_tokens
        tensors = _make_masked_inputs(fn, sequence_length, self.tokenizers)
        if tensors is not None:
            tensors = tensors.to(input_tokens.device)
        return attr.evolve(input_tokens, **{fn: tensors})

    def batch_ESMTensor(self, sampled_tokens: list[ESMProteinTensor]) -> _BatchedESMProteinTensor:
        """
        Convert a list of ESMProteinTensor to _BatchedESMProteinTensor.
        """
        sequence_lengths = [len(tokens) for tokens in sampled_tokens]
        devices = set([t.device for t in sampled_tokens])
        batched_tokens = _stack_protein_tensors(sampled_tokens, sequence_lengths, self.tokenizers, devices.pop())
        return batched_tokens

    def roll_out_final_states_given_batched_states(self, batched_tokens: _BatchedESMProteinTensor, sampling_params: dict, sequence_lengths: Sequence[int], total_to_sample: Sequence[int], depth: int) -> list[ESMProteinTensor]:
        total_steps = sampling_params['total_steps']
        generating_config = GenerationConfig(track='sequence', num_steps=total_steps, temperature=sampling_params['temperature'], top_p=sampling_params['top_p'], strategy=sampling_params['strategy'], schedule=sampling_params['schedule'], invalid_ids=canonical_invalid_ids(self.tokenizers.sequence))
        track_sample_config = SamplingTrackConfig(temperature=sampling_params['temperature'], top_p=sampling_params['top_p'], invalid_ids=generating_config.invalid_ids)
        sampling_config = SamplingConfig(**{'sequence': track_sample_config})
        expand_N = sampling_params['n']
        final_states = []
        errors: dict[int, ESMProteinError] = {}
        for num in tqdm(range(expand_N), desc='Rolling out'):
            batched_tokens_per_num = copy.deepcopy(batched_tokens)
            batch_state = []
            steps = [total_steps - 1] if self.rollout_mode == "jumpy" else range(depth, total_steps)
            for t in steps:
                forward_out = _batch_forward(self.model, batched_tokens_per_num)
                for i in range(batched_tokens_per_num.sequence.shape[0]):
                    per_prompt_cur_sampled = _BatchedESMProteinTensor.from_protein_tensor(batched_tokens_per_num.slice(i))
                    per_prompt_forward_out: LogitsOutput = _slice_tensor_dataclass(forward_out, i, keep_dim=True)
                    per_prompt_forward_out = _trim_sequence_tensor_dataclass(per_prompt_forward_out, len(per_prompt_cur_sampled))
                    per_prompt_forward_and_sample_output = _sample_per_prompt(per_prompt_cur_sampled, per_prompt_forward_out, sampling_config, self.tokenizers, decode_sasa_tokens=False)
                    per_prompt_new_sampled = per_prompt_forward_and_sample_output.protein_tensor
                    try:
                        where_to_sample = _get_iterative_sampling_mask_for_prompt_and_step(per_prompt_cur_sampled, torch.tensor(sequence_lengths[i]), torch.tensor(total_to_sample[i]), t, per_prompt_forward_and_sample_output.entropy, generating_config, self.tokenizers)
                    except ValueError:
                        if not (per_prompt_cur_sampled.sequence == self.tokenizers.sequence.mask_token_id).any():
                            continue
                        raise
                    where_to_sample = where_to_sample.to(batched_tokens_per_num.device)
                    old_track_samples = getattr(per_prompt_cur_sampled, generating_config.track)
                    new_track_samples = getattr(per_prompt_new_sampled, generating_config.track)
                    new_track_samples = torch.where(where_to_sample, new_track_samples, old_track_samples)
                    getattr(batched_tokens_per_num, generating_config.track)[i, ...] = new_track_samples[0]
            for i in range(batched_tokens_per_num.sequence.shape[0]):
                per_prompt_cur_sampled = _BatchedESMProteinTensor.from_protein_tensor(batched_tokens_per_num.slice(i))
                per_prompt_cur_original = _BatchedESMProteinTensor.from_protein_tensor(batched_tokens.slice(i))
                length = sequence_lengths[i]
                final_tokens = ESMProteinTensor(
                    sequence=per_prompt_cur_sampled.sequence[0, :length],
                    structure=None if per_prompt_cur_original.structure is None else per_prompt_cur_original.structure[0, :length],
                    coordinates=None if per_prompt_cur_original.coordinates is None else per_prompt_cur_original.coordinates[0, :length],
                )
                final_raw_state = self.decode_state(final_tokens)
                batch_state.append(final_raw_state)
            final_states.append(batch_state)
        final_states_transposed = list(map(list, zip(*final_states)))
        return final_states_transposed

    def expand_nodes_given_batched_states(self, batched_tokens: _BatchedESMProteinTensor, expansion_sampling_params: dict, sequence_lengths: Sequence[int], total_to_sample: Sequence[int], depth: Sequence[int]) -> list[list[ESMProteinTensor]]:
        """
        Expand the next layer of n leaf nodes from a given node (batched states).
        Args:
            batched_tokens: The current state of the MCTS tree. batched_tokens
            expansion_sampling_params: The configuration for sampling.
            layer: The timestep & tree layer to sample. layer=0表示从树的根节点(layer0)向下扩展节点, 也就是batched_tokens属于第几层
        returns:
            expand_states: A list of expanded states. 每一行表示第一个token的n个采样结果
        """
        total_steps = expansion_sampling_params['total_steps']
        generating_config = GenerationConfig(track='sequence', num_steps=total_steps, temperature=expansion_sampling_params['temperature'], schedule=expansion_sampling_params['schedule'], top_p=expansion_sampling_params['top_p'], strategy=expansion_sampling_params['strategy'], invalid_ids=canonical_invalid_ids(self.tokenizers.sequence))
        track_sample_config = SamplingTrackConfig(temperature=expansion_sampling_params['temperature'], top_p=expansion_sampling_params['top_p'], invalid_ids=generating_config.invalid_ids)
        sampling_config = SamplingConfig(**{'sequence': track_sample_config})
        expand_N = expansion_sampling_params['n']
        expand_states = []
        errors: dict[int, ESMProteinError] = {}
        for num in tqdm(range(expand_N), desc='Expanding'):
            forward_out = _batch_forward(self.model, batched_tokens)
            expand_batch_state = []
            for i in range(batched_tokens.sequence.shape[0]):
                per_prompt_cur_sampled = _BatchedESMProteinTensor.from_protein_tensor(batched_tokens.slice(i))
                per_prompt_forward_out: LogitsOutput = _slice_tensor_dataclass(forward_out, i, keep_dim=True)
                per_prompt_forward_out = _trim_sequence_tensor_dataclass(per_prompt_forward_out, len(per_prompt_cur_sampled))
                per_prompt_forward_and_sample_output = _sample_per_prompt(per_prompt_cur_sampled, per_prompt_forward_out, sampling_config, self.tokenizers, decode_sasa_tokens=False)
                per_prompt_new_sampled = per_prompt_forward_and_sample_output.protein_tensor
                try:
                    where_to_sample = _get_iterative_sampling_mask_for_prompt_and_step(per_prompt_cur_sampled, torch.tensor(sequence_lengths[i]), torch.tensor(total_to_sample[i]), depth[i], per_prompt_forward_and_sample_output.entropy, generating_config, self.tokenizers)
                except ValueError:
                    raise
                where_to_sample = where_to_sample.to(batched_tokens.device)
                old_track_samples = getattr(per_prompt_cur_sampled, generating_config.track)
                new_track_samples = getattr(per_prompt_new_sampled, generating_config.track)
                new_track_samples = torch.where(where_to_sample, new_track_samples, old_track_samples)
                length = sequence_lengths[i]
                next_state = ESMProteinTensor(
                    sequence=new_track_samples[0, :length],
                    structure=None if per_prompt_cur_sampled.structure is None else per_prompt_cur_sampled.structure[0, :length],
                    coordinates=None if per_prompt_cur_sampled.coordinates is None else per_prompt_cur_sampled.coordinates[0, :length],
                )
                expand_batch_state.append(next_state)
            expand_states.append(expand_batch_state)
        expand_states_transposed = list(map(list, zip(*expand_states)))
        return expand_states_transposed

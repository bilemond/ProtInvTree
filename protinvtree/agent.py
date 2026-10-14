import time
import torch
import torch.nn as nn
from esm.utils.generation import _get_non_special_tokens
from .searching.tree import Node
from .searching.utils import clean_outputs, get_unique_value, same_state

class Agent(nn.Module):

    def __init__(self, reward_model, policy_model, total_steps=10, temperature=0.85, pre_expand_N=1, expand_N=1, rollout_N=1, strategy='random', schedule='cosine', do_value=False, mode='inference'):
        """
        Args:
            reward_model: Reward model to evaluate the generated sequences.
            policy_model: Policy model to generate sequences.
            total_steps: Total steps for sampling.
            temperature: Temperature for sampling.
            pre_expand_N: Number of pre-expansion samples.
            expand_N: Number of expansion samples.
            rollout_N: Number of rollout samples.
            do_value: Whether to compute value or not.
            mode: Mode of operation ('inference' or 'gen_data').
        """
        super(Agent, self).__init__()
        self.reward_model = reward_model
        self.policy_model = policy_model
        self.pre_expansion_sampling_params = {'total_steps': total_steps, 'temperature': temperature, 'top_p': 1, 'n': pre_expand_N, 'strategy': strategy, 'schedule': schedule}
        self.reasoning_sampling_params = {'total_steps': total_steps, 'temperature': temperature, 'top_p': 1, 'n': expand_N, 'strategy': strategy, 'schedule': schedule}
        self.rollout_sampling_params = {'total_steps': total_steps, 'temperature': temperature, 'top_p': 1, 'n': rollout_N, 'strategy': strategy, 'schedule': schedule}
        self.value_model = None
        self.do_value = do_value
        assert mode in ['inference', 'gen_data']
        self.mode = mode
        self.rollout_res = []

    def perform_planning(self, leaf_node, logger, speedup=None):
        stats = {}
        if leaf_node.is_terminated():
            return []
        current_state = [leaf_node.state]
        current_state_batch = self.policy_model.batch_ESMTensor(current_state)
        depth = [leaf_node.get_depth()]
        sequence_lengths = [len(tokens) for tokens in current_state]
        total_to_sample = [_get_non_special_tokens(protein, self.policy_model.tokenizers) for protein in current_state]
        t0 = time.time()
        outputs = self.policy_model.expand_nodes_given_batched_states(current_state_batch, self.reasoning_sampling_params, sequence_lengths, total_to_sample, depth)
        print('expand the {}th layer, node id is {}'.format(leaf_node.get_depth(), leaf_node.get_index_in_layer()))
        t1 = time.time()
        stats['expansion/reasoning_time'] = t1 - t0
        zipped_responses = clean_outputs(outputs)
        responses = [zipped_response[1] for zipped_response in zipped_responses]
        new_nodes = []
        for new_state in responses:
            if new_state is None:
                continue
            repeat = 0
            if not leaf_node.is_leaf():
                for node in leaf_node.children():
                    if same_state(new_state, node.state):
                        repeat = 1
                        break
            if repeat:
                continue
            new_node = Node(parent=leaf_node, state=new_state)
            if new_node.get_depth() >= self.reasoning_sampling_params['total_steps'] or not (new_state.sequence == self.policy_model.tokenizers.sequence.mask_token_id).any():
                new_node.state = self.policy_model.decode_state(new_node.state)
                new_node.set_terminated()
            new_nodes.append(new_node)
        t2 = time.time()
        print('after cut:', len(new_nodes))
        final_rollout_solutions = self.perform_rollout(new_nodes)
        t3 = time.time()
        stats['simulation/rollout_compute_time'] = t3 - t2
        final_rewards, final_rm_times = self.reward_model.get_reward_from_rollout(final_rollout_solutions)
        planning_results = []
        for j in range(len(new_nodes)):
            new_nodes[j]._initial_value = final_rewards[j]
            planning_results.append((new_nodes[j], final_rewards[j]))
        t4 = time.time()
        simulation_reward_time = t4 - t3
        stats['simulation/reward_compute_time'] = simulation_reward_time
        logger.info(stats)
        return planning_results

    def perform_layer_reasoning(self, parents_nodes, logger, stage='pre_expand'):
        stats = {}
        layer_states = [node.state for node in parents_nodes]
        layer_states_batch = self.policy_model.batch_ESMTensor(layer_states)
        depth = [node.get_depth() for node in parents_nodes]
        sequence_lengths = [len(tokens) for tokens in layer_states]
        total_to_sample = [_get_non_special_tokens(protein, self.policy_model.tokenizers) for protein in layer_states]
        t0 = time.time()
        if stage == 'pre_expand':
            print('=====Pre-Expansion=====')
            expansion_sampling_params = self.pre_expansion_sampling_params
        else:
            expansion_sampling_params = self.reasoning_sampling_params
        outputs = self.policy_model.expand_nodes_given_batched_states(layer_states_batch, expansion_sampling_params, sequence_lengths, total_to_sample, depth)
        t1 = time.time()
        stats['expansion/pre_reasoning_time'] = t1 - t0
        planning_results = []
        new_nodes = []
        zipped_responses = clean_outputs(outputs)
        for zipped_response in zipped_responses:
            parent_idx = zipped_response[0]
            parent_node = parents_nodes[parent_idx]
            new_state = zipped_response[1]
            if new_state is None:
                continue
            repeat = 0
            if not parent_node.is_leaf():
                for node in parent_node.children():
                    if same_state(new_state, node.state):
                        repeat = 1
                        break
            if repeat:
                continue
            new_node = Node(parent=parent_node, state=new_state)
            if new_node.get_depth() >= expansion_sampling_params['total_steps'] or not (new_state.sequence == self.policy_model.tokenizers.sequence.mask_token_id).any():
                new_node.state = self.policy_model.decode_state(new_node.state)
                new_node.set_terminated()
            new_nodes.append(new_node)
        t2 = time.time()
        final_rollout_solutions = self.perform_rollout(new_nodes)
        t3 = time.time()
        stats['simulation/rollout_compute_time'] = t3 - t2
        final_rewards, final_rm_times = self.reward_model.get_reward_from_rollout(final_rollout_solutions)
        for j in range(len(new_nodes)):
            new_nodes[j]._initial_value = final_rewards[j]
            planning_results.append((new_nodes[j], final_rewards[j]))
            new_nodes[j].parent().add_child(new_nodes[j])
        t4 = time.time()
        simulation_reward_time = t4 - t3
        stats['simulation/reward_compute_time'] = simulation_reward_time
        logger.info(stats)
        return planning_results

    def perform_rollout(self, nodes_to_rollout):
        rollout_states = []
        terminate_idx = []
        for i, node in enumerate(nodes_to_rollout):
            if not node.is_terminated():
                rollout_states.append(node.state)
            else:
                terminate_idx.append(i)
        if len(rollout_states) != 0:
            batched_rollout_states = self.policy_model.batch_ESMTensor(rollout_states)
            depth_list = [node.get_depth() for node in nodes_to_rollout if not node.is_terminated()]
            depth = get_unique_value(depth_list)
            sequence_lengths = [len(tokens) for tokens in rollout_states]
            total_to_sample = [_get_non_special_tokens(protein, self.policy_model.tokenizers) for protein in rollout_states]
            outputs = self.policy_model.roll_out_final_states_given_batched_states(batched_rollout_states, self.rollout_sampling_params, sequence_lengths, total_to_sample, depth)
        all_rollout_solutions = []
        terminate_count = 0
        for i in range(len(nodes_to_rollout)):
            node = nodes_to_rollout[i]
            if i not in terminate_idx:
                output = outputs[i - terminate_count]
                all_rollout_solutions.append(output)
                if self.mode == 'gen_data':
                    node.rollout_res.extend(output)
            else:
                terminate_count += 1
                all_rollout_solutions.append([node.state])
                if self.mode == 'gen_data':
                    node.rollout_res.extend([node.state])
        return all_rollout_solutions

"""Reusable inference engine: load the two pretrained models once per process."""

import logging
import random
import time

import numpy as np
import torch

from .agent import Agent
from .policy import Policy_model, canonical_invalid_ids
from .reward import Reward_model
from .searching.search import MCTS
from .searching.utils import node_count, expand_count
from esm.utils.generation import _get_non_special_tokens
from esm.sdk.api import ESMProteinError, GenerationConfig


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class DesignRunner:
    def __init__(self, config):
        self.config = config
        seed_everything(config["seed"])
        self.policy = Policy_model(config["policy_model"], config["policy_device"], config["rollout_mode"]).eval()
        self.reward = Reward_model(config["reward_alpha"], config["fold_model"], config["reward_device"],
                                   config["reward_reference"], config["fold_chunk_size"]).eval()
        self.agent = Agent(self.reward, self.policy, total_steps=config["total_steps"],
                           temperature=config["temperature"], pre_expand_N=config["pre_expand_n"],
                           expand_N=config["expand_n"], rollout_N=config["rollout_n"],
                           strategy=config["selection"], schedule=config["schedule"])
        self.logger = logging.getLogger("protinvtree")

    def prepare(self, protein, reference_sequence):
        self.reward.update_label(protein, reference_sequence)
        # The benchmark reports sc-TMscore for both reward modes. Prepare this
        # reference outside the timed search, retaining the established protocol.
        self.reward.prepare_native_reference()

    def _rollout(self, node):
        state = node.state
        return self.policy.roll_out_final_states_given_batched_states(
            self.policy.batch_ESMTensor([state]), self.agent.rollout_sampling_params,
            [len(state)], [_get_non_special_tokens(state, self.policy.tokenizers)], node.get_depth()
        )[0][0]

    @torch.inference_mode()
    def design(self, protein, reference_sequence, seed, method="mcts"):
        seed_everything(seed)
        started = time.perf_counter()
        cfg = self.config
        tree, simulations, search_value, search_complete = [], 0, None, True
        if method == "esm3":
            # Remove all native sequence and annotation tracks from the prompt.
            import attr
            prompt = attr.evolve(protein, sequence=None, secondary_structure=None, sasa=None,
                                 function_annotations=None)
            generated = self.policy.model.generate(prompt, GenerationConfig(
                track="sequence", num_steps=cfg["total_steps"], temperature=cfg["temperature"],
                strategy=cfg["selection"], schedule=cfg["schedule"],
                invalid_ids=canonical_invalid_ids(self.policy.tokenizers.sequence)))
            if isinstance(generated, ESMProteinError):
                raise RuntimeError(generated.error_msg)
        else:
            initial = self.policy.prepare_initial_state(protein)
            node_count.set_count(0)
            expand_count.set_count(0)
            search = MCTS(self.agent, initial, self.logger, cfg["c_param"], cfg["value_threshold"],
                          cfg["complete_path_n"], cfg["final_selection"])
            best, nodes, simulations = search.run_search_with_pre_expansion(
                cfg["simulations"], cfg["pre_expansion_depth"])
            search_value = float(best.value())
            search_complete = best.is_terminated()
            generated = best.state if search_complete else self._rollout(best)
            for node in nodes:
                seq = node.state.sequence
                if not isinstance(seq, str):
                    seq = self.policy.decode_state(node.state).sequence
                tree.append({"id": node.idx, "parent": node.parent().idx if node.parent() else None,
                             "depth": node.get_depth(), "visits": node.visit_count(),
                             "value": float(node.value()), "terminal": node.is_terminated(),
                             "sequence": seq})
        sequence = generated.sequence
        if sequence is None or any(residue not in "ACDEFGHIKLMNPQRSTVWY" for residue in sequence):
            raise ValueError("Generated sequence is incomplete or contains noncanonical residues")
        metrics, pdb = self.reward.evaluate(sequence)
        actual_reward = cfg["reward_alpha"] * metrics[
            "target_tm_score" if cfg["reward_reference"] == "target" else "sc_tm_score"
        ] + (1.0 - cfg["reward_alpha"]) * metrics["novelty"]
        result = {"sequence": sequence, "reference_sequence": reference_sequence, "seed": seed,
                  "method": method, **metrics, "reward": actual_reward,
                  "search_value": search_value, "search_complete": search_complete,
                  "simulation_steps": simulations, "node_count": len(tree),
                  "solve_time_seconds": time.perf_counter() - started}
        return result, pdb, tree

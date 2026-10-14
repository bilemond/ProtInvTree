"""ESMFold reward and explicitly named structural evaluation metrics."""

from collections import OrderedDict
import io
import math
import time

import numpy as np
import torch
from torch import nn
from biotite.structure.io.pdb import PDBFile
from tmtools import tm_align
from transformers import EsmForProteinFolding

from .metrics import novelty
from .data import AMINO_ACIDS


class Reward_model(nn.Module):
    def __init__(self, alpha=1.0, model_name="facebook/esmfold_v1", device="cuda",
                 reference="target", chunk_size=128):
        super().__init__()
        alpha = float(alpha)
        if not math.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
            raise ValueError("Reward alpha must be finite and lie in [0, 1]")
        if reference not in ("target", "folded-native"):
            raise ValueError("Reward reference must be target or folded-native")
        if chunk_size < 0:
            raise ValueError("Fold chunk size cannot be negative")
        self.esmfold = EsmForProteinFolding.from_pretrained(model_name).to(device).eval()
        if chunk_size:
            self.esmfold.trunk.set_chunk_size(chunk_size)
        self.alpha = alpha
        self.reference = reference
        self._cache = OrderedDict()
        self.gt_sequence = None
        self.target_ca = None
        self.native_ca = None

    @staticmethod
    def _validate_sequence(sequence):
        if not isinstance(sequence, str) or not sequence or not set(sequence) <= AMINO_ACIDS:
            raise ValueError("Folding requires a nonempty canonical amino acid sequence")

    def _validate_candidate(self, sequence):
        if self.gt_sequence is None:
            raise RuntimeError("Call update_label before evaluating a candidate")
        self._validate_sequence(sequence)
        if len(sequence) != len(self.gt_sequence):
            raise ValueError("Generated sequence length differs from target length")

    @staticmethod
    def _ca_from_pdb(pdb):
        atoms = PDBFile.read(io.StringIO(pdb)).get_structure(model=1)
        return np.asarray(atoms.coord[atoms.atom_name == "CA"], dtype=np.float64)

    @torch.inference_mode()
    def fold(self, sequence):
        self._validate_sequence(sequence)
        if sequence in self._cache:
            self._cache.move_to_end(sequence)
            return self._cache[sequence]
        pdb = self.esmfold.infer_pdbs([sequence])[0]
        coords = self._ca_from_pdb(pdb)
        if len(coords) != len(sequence) or not np.isfinite(coords).all():
            raise ValueError("ESMFold returned invalid C-alpha coordinates")
        result = (pdb, coords)
        self._cache[sequence] = result
        if len(self._cache) > 128:
            self._cache.popitem(last=False)
        return result

    def update_label(self, structure_inputs, gt_sequence, structure=None):
        self._validate_sequence(gt_sequence)
        protein = structure_inputs if structure is None else structure
        coordinates = protein.coordinates
        if coordinates is None or coordinates.ndim != 3 or coordinates.shape[1] < 2 or coordinates.shape[2] != 3:
            raise ValueError("Target coordinates must have shape (length, atoms, 3) and contain C-alpha")
        target_ca = coordinates[:, 1].detach().to(device="cpu", dtype=torch.float64).numpy().copy()
        if len(target_ca) != len(gt_sequence) or not np.isfinite(target_ca).all():
            raise ValueError("Target coordinates must match the reference sequence and be finite")
        # Commit a valid new target together; never leave an old native reference
        # attached to the new target after a failed reference folding operation.
        self.gt_sequence = gt_sequence
        self.target_ca = target_ca
        self.native_ca = None

    def prepare_native_reference(self):
        """Prepare the folded native only when its reward or metric is needed."""
        if self.gt_sequence is None:
            raise RuntimeError("Call update_label before preparing the native reference")
        if self.native_ca is None:
            self.native_ca = self.fold(self.gt_sequence)[1]
        return self.native_ca

    def _alignment(self, coordinates, sequence, reference_ca):
        self._validate_candidate(sequence)
        result = tm_align(coordinates, reference_ca, sequence, self.gt_sequence)
        if not math.isfinite(float(result.tm_norm_chain2)) or not 0.0 <= result.tm_norm_chain2 <= 1.0:
            raise ValueError("TM-align returned an invalid reference-normalized TM-score")
        if not math.isfinite(float(result.rmsd)) or result.rmsd < 0:
            raise ValueError("TM-align returned an invalid RMSD")
        return result

    def get_reward(self, sequence):
        self._validate_candidate(sequence)
        if self.reference == "target":
            reference_ca = self.target_ca
        elif self.reference == "folded-native":
            reference_ca = self.prepare_native_reference()
        else:
            raise ValueError("Reward reference must be target or folded-native")
        coords = self.fold(sequence)[1]
        score = self._alignment(coords, sequence, reference_ca).tm_norm_chain2
        return self.alpha * score + (1.0 - self.alpha) * novelty(sequence, self.gt_sequence)

    def get_reward_from_rollout(self, all_rollout):
        values, started = [], time.perf_counter()
        for candidates in all_rollout:
            if not candidates:
                raise ValueError("Cannot score an empty rollout")
            rewards = [self.get_reward(candidate if isinstance(candidate, str) else candidate.sequence)
                       for candidate in candidates]
            values.append(sum(rewards) / len(rewards))
        return values, time.perf_counter() - started

    def evaluate(self, sequence):
        self._validate_candidate(sequence)
        native_ca = self.prepare_native_reference()
        pdb, coords = self.fold(sequence)
        target = self._alignment(coords, sequence, self.target_ca)
        native = self._alignment(coords, sequence, native_ca)
        return {"target_tm_score": float(target.tm_norm_chain2),
                "target_rmsd": float(target.rmsd),
                "sc_tm_score": float(native.tm_norm_chain2),
                "sc_rmsd": float(native.rmsd),
                "recovery": 1.0 - novelty(sequence, self.gt_sequence),
                "novelty": novelty(sequence, self.gt_sequence)}, pdb

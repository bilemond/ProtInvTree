# Third-party notices

## EvolutionaryScale ESM

The `esm/` directory was copied from the supplied working tree, whose original
package metadata identifies ESM version 3.1.7. It includes local research changes;
the exact upstream Git commit was unavailable in that working tree.

Upstream: <https://github.com/evolutionaryscale/esm>

The original license and contribution information are preserved in
`third_party/ESM_LICENSE.md` and `third_party/ESM_CONTRIBUTIONS.md`.
Unused notebook widgets, experimental SDK helpers and SageMaker clients are
omitted. The license information distinguishes the code license
from model-weight and service terms; consult the linked upstream agreements.
No model weights are bundled.

This preparation additionally changes `esm/utils/constants/esm3.py` to accept
`ESM3_MODEL_DIR` or download the model snapshot, replacing a server-specific path.
The working tree already contained support for `ar` position selection and
`ProteinChain.from_backbone_atom_coordinates_N_CA_C_O`.

## ProtInvTree research sources

`protinvtree/policy.py` and `protinvtree/agent.py` are extracted and cleaned from
`code/models/policy_model.py` and `code/models/agent.py`. Search nodes and MCTS are
refactored from `code/models/searching/`. The CATH data format and filtering are
based on `code/src/datasets/cath_dataset.py` and `code/src/datasets/featurizer.py`.
The paper identifies <https://github.com/A4Bio/ProteinInvBench/> as its code site.
This preparation does not assert that the working tree matches that public repo.

The authors have deferred the research-code license decision. This preparation
does not apply an MIT or other new license to any copied research code.

## Pretrained weights and benchmark data

ESM3 and ESMFold weights, and CATH benchmark data, must be obtained separately:

- ESM3: <https://huggingface.co/biohub/esm3-sm-open-v1>
- ESMFold: <https://huggingface.co/facebook/esmfold_v1>
- CATH benchmark preprocessing and splits: see the paper's code site above.

Use the terms associated with each asset. The repository contains only a tiny
synthetic CATH-format fixture for software validation, clearly labeled as such.

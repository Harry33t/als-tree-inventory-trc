# Reliable Individual-Tree Inventory From ALS

Code release for the manuscript *Reliable Individual-Tree Inventory From Airborne Laser
Scanning With Competition-Aware Segmentation and Trait-Specific Selective Delivery*
(submitted to IEEE Transactions on Geoscience and Remote Sensing).

The repository contains the method and evaluation code only. It contains no point-cloud
data, no model weights, no figure-generation scripts, and no manuscript tooling.

## Contents

| Path | What it is |
|---|---|
| `epco/epco.py` | EPCO, the equivalence-preserving contact-ownership objective. Group evidence by size-normalized log-mean-exp, admissible-rival selection, Gaussian contact support under stop-gradient, contact-strength-modulated temperature, and the contact-weighted ownership loss. |
| `epco/test_epco_formal.py` | Unit tests for the EPCO objective, including the query-duplication invariance of the group evidence. |
| `configs/` | The four training configurations of the paper. They differ only in `acpe_enabled`, `acpe_loss_weight`, `epco_enabled`, and `epco_loss_weight`, and inherit everything else from the base configuration of the segmenter. |
| `training/run_official47_benchmark_train.py` | Training runner for the four configurations, including the scope guards that keep the BlueCat extension source out of the training split. |
| `trc/run_source_heldout_trc.py` | Trait-specific reliability control: descriptor assembly, the three risk heads, nested leave-one-physical-source-out validation, probability calibration, risk--coverage curves, budgeted review, and the hierarchical bootstrap. |
| `trc/finalize_trc.py`, `trc/score_trc_risk_bundle.py`, `trc/extract_sealed_instance_confidence.py` | Freezing of the selected risk models, scoring of new predictions with the frozen bundle, and extraction of the sealed inference-confidence descriptors. |
| `evaluation/evaluate_external_baselines_official28.py` | Evaluation of the external methods under the same matcher and the same inventory. |
| `evaluation/evaluate_l1a_external_source.py` | External-source evaluation: re-measurement of sealed predictions with the frozen trait routine and rescoring with the frozen risk models. |

## Not included

- **The ACPE prototype objective and the base training configurations.** The configurations
  in `configs/` switch ACPE on through `acpe_enabled`, but the module that implements the
  P2 prototype loss, and the `_base_` configurations they inherit from, are held in the
  training environment and are not part of this snapshot.
- **All data.** The FOR-instanceV2 benchmark, the published ForestFormer3D checkpoint and the released models of the external methods are obtained from their own distributions and remain under their own licences. The external field-measured source L1A is not part of a public benchmark; requests for access should be addressed to the corresponding author.
- Figure-generation scripts, manuscript sources, and internal packaging or audit tooling.

## Relation to the upstream segmenter

`epco/epco.py` is written for ForestFormer3D, which builds on OneFormer3D and MMDetection3D.
It is a standalone module: it is imported by the instance criterion of the segmenter and
called once per training sample. Neither ForestFormer3D nor OneFormer3D source files are
redistributed here. Obtain them from their own releases and call `epco_ownership_loss` once per training sample
from the instance criterion, after the final decoder layer has produced its mask logits.
Frozen settings: contact kernel `sigma_xy = 0.60 m`, support threshold `0.05`, at most `4096`
contact voxels per target tree, suppressed-rival logit `-20`, temperature `tau = 1.50 - 0.75 w`.
The support term is computed under a stop-gradient, so gradients reach the network only
through the evidence terms of the ownership competition.

## Citation

Please cite the article once it appears. Until then, cite this repository and the
FOR-instanceV2 benchmark.

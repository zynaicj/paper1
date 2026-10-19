# BSCNet — Structure-Aware Network for Underwater Image Semantic Segmentation

Research code accompanying **Structure-Aware Network for Underwater Image Semantic
Segmentation** by Yinuo Zheng and Jiaqi Ma. The implementation is based on
MMSegmentation 1.2.2 and SegFormer / MiT-B0.

This repository contains the local model implementation, training and evaluation
entry points, and experiment configurations. Dataset images, annotations, split
lists, trained checkpoints, manuscripts, training logs, and experiment outputs
are not distributed in this source-code repository.

## Paper terminology and implementation

The source retains the implementation names used during the experiments:

| Paper module | Implementation | Source |
|---|---|---|
| SMBAS: shallow multiscale boundary auxiliary supervision | `BoundaryHead` | `mmseg/models/decode_heads/boundary_head.py` |
| CSDA: cross-scale decoder feature alignment | `SemanticGuidedFuse`, assembled by `SegformerResidualASPPSGPCFHead` | `mmseg/models/decode_heads/segformer_residual_aspp_sgpcf_head.py` |
| MCAR: multisource context adaptive refinement | `ResidualASPP` and `ASPPContextBlock` | `mmseg/models/decode_heads/segformer_residual_aspp_head.py` |

The full model uses `SegformerResidualASPPSGPCFHead` with a `BoundaryHead`
auxiliary branch. The `BPSF` classes and other optional heads retained in the
local framework are alternative implementations, not the full model evaluated
in the submitted manuscript. Use the configurations listed below.

## Environment

The recorded training environment used Linux, Python 3.10.20, PyTorch
2.1.2+cu118, CUDA 11.8, MMCV 2.1.0, MMEngine 0.10.7, and one NVIDIA RTX 3090.
Use a CUDA-capable Linux environment for the recorded NCCL configurations.

Example installation in a fresh Python 3.10 environment:

```bash
python -m pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118
python -m pip install "numpy<2" openmim
mim install "mmcv==2.1.0"
python -m pip install -r requirements-paper.txt
python -m pip install -e .
```

The repository includes the modified `mmseg` source; do not replace it with an
unmodified MMSegmentation package. The optional models retained from the local
framework may have additional dependencies when explicitly selected.

## Datasets and preprocessing

Obtain the original datasets from their providers:

- SUIM: <https://irvlab.cs.umn.edu/resources/suim-dataset>
- DUT-USEG: <https://github.com/baxiyi/DUT-USEG>

The local name `USEG` denotes DUT-USEG. Prepare the following layout:

```text
data/
  SUIM/
    JPEGImages/<sample_id>.jpg
    Segmentation/<sample_id>.png
    boundary/<sample_id>.png
    splits/train.txt
    splits/val.txt
  USEG/
    JPEGImages/<sample_id>.jpg
    Segmentation/<sample_id>.png
    boundary/<sample_id>.png
    ImageSets/Segmentation/train.txt
    ImageSets/Segmentation/val.txt
```

Semantic masks must be single-channel class-index masks, not RGB visualization
masks. SUIM class IDs 0–7 are background, human diver, aquatic plants, wrecks and
ruins, robots, reefs, fish, and sea floor. USEG class IDs 0–4 are background,
holothurian, echinus, scallop, and starfish. The palettes in the dataset classes
are display colors; they are not a converter for the original RGB masks.

The original experiment splits contain 1,220/305 training/validation samples for
SUIM and 1,338/149 for USEG. Split files are not included in this code-only
release; supply the corresponding lists locally in the paths above. Different
splits will change the evaluation results. These counts do not imply evaluation
on the independent SUIM official test set.

For the existing class-index masks, boundary targets are the 3×3 morphological
gradient of the foreground mask (`mask > 0`), saved as 0/255 PNGs. Generate them
and validate the data layout using:

```bash
python tools/prepare_paper_data.py data/SUIM --dataset SUIM --write-boundaries
python tools/prepare_paper_data.py data/USEG --dataset USEG --write-boundaries
```

This helper does not download data, infer a class mapping, regenerate splits,
or overwrite existing boundary files. It checks existing boundaries against
the foreground-gradient rule. Convert the original semantic masks to the
class IDs above before using it.

## Pretrained backbone

Place the ImageNet-pretrained MiT-B0 checkpoint at
`checkpoints/mit_b0_20220624-7e0fe6dd.pth`. Its upstream download URL is:

<https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b0_20220624-7e0fe6dd.pth>

The pretrained backbone and trained experiment weights are not included here.

## Configurations and training

`configs/segformer/` contains the editable source configurations and their base
dependencies. `configs/recorded/` contains the fully expanded configurations
saved with the corresponding experiments. In the recorded snapshots, only the
original machine-specific project path prefix has been removed.

| Dataset / stage | Recorded configuration |
|---|---|
| SUIM baseline | `configs/recorded/suim_baseline.py` |
| SUIM SMBAS | `configs/recorded/suim_smbas.py` |
| SUIM SMBAS + MCAR | `configs/recorded/suim_smbas_mcar.py` |
| SUIM full BSCNet | `configs/recorded/suim_bscnet.py` |
| USEG baseline | `configs/recorded/useg_baseline.py` |
| USEG full BSCNet | `configs/recorded/useg_bscnet.py` |

**Checkpoint initialization matters.** The recorded SUIM sequence trains the
baseline, then SMBAS from the baseline checkpoint, then SMBAS+MCAR from the
SMBAS checkpoint, then the full model from SMBAS+MCAR. The recorded USEG full
model starts from the final baseline checkpoint, with a lower learning rate.
These are staged experiments; they must not be described as independent
from-scratch runs with equal total training budgets.

Train a baseline:

```bash
python tools/train.py configs/recorded/suim_baseline.py --work-dir work_dirs/paper1/suim_baseline
python tools/train.py configs/recorded/useg_baseline.py --work-dir work_dirs/paper1/useg_baseline
```

Train each subsequent SUIM stage with its actual predecessor checkpoint:

```bash
python tools/train.py configs/recorded/suim_smbas.py --work-dir work_dirs/paper1/suim_smbas --cfg-options load_from=PATH_TO_SUIM_BASELINE_CHECKPOINT
python tools/train.py configs/recorded/suim_smbas_mcar.py --work-dir work_dirs/paper1/suim_smbas_mcar --cfg-options load_from=PATH_TO_SUIM_SMBAS_CHECKPOINT
python tools/train.py configs/recorded/suim_bscnet.py --work-dir work_dirs/paper1/suim_bscnet --cfg-options load_from=PATH_TO_SUIM_SMBAS_MCAR_CHECKPOINT
```

Train the USEG full model after training the baseline:

```bash
python tools/train.py configs/recorded/useg_bscnet.py --work-dir work_dirs/paper1/useg_bscnet --cfg-options load_from=work_dirs/paper1/useg_baseline/iter_160000.pth
```

Replace `PATH_TO_...` with real files produced by the preceding stage; the
original paths in `load_from` refer to local experiment artifacts not included
in this repository. If starting the full model directly from the pretrained
backbone, explicitly override `load_from=None`; this is a different experiment
and is not expected to reproduce the recorded staged results.

## Evaluation

```bash
python tools/test.py configs/recorded/suim_bscnet.py PATH_TO_TRAINED_SUIM_CHECKPOINT --cfg-options model.backbone.init_cfg=None
python tools/test.py configs/recorded/useg_bscnet.py PATH_TO_TRAINED_USEG_CHECKPOINT --cfg-options model.backbone.init_cfg=None
```

The test dataloaders in these configurations use the fixed validation splits.
The manuscript reports validation mIoU of 67.95% for SUIM and 69.39% for USEG.
These are the original experiment results, not fresh reruns of this release.

The recorded SUIM baseline validates at the end of 160k iterations; the improved
SUIM configurations validate every 10k iterations. Most recorded SUIM configs
do not pin their randomly generated seed. The USEG full-model snapshot pins
seed 0. The saved configurations preserve these details rather than silently
equalizing them. This source release does not include trained weights, so
retraining is required before evaluation.

## Attribution and license

The original MMSegmentation Apache-2.0 license and source copyright notices are
retained in `LICENSE` and the source files. This code builds on OpenMMLab
MMSegmentation and SegFormer. The semantic-guided fusion implementation notes
its relationship to PIDNet's PagFM, and the balanced boundary loss retains its
upstream attribution. Dataset rights remain with their original providers.

## Release validation

The source release was checked for Python syntax, local configuration
dependencies, exclusion of experimental data, and accidental inclusion of
credentials or large binary artifacts. Model training and inference were not
rerun in the release-preparation environment, which lacks PyTorch/MMCV and a
configured training runtime.

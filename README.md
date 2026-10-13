# ESPI-Net

Official PyTorch implementation of **ESPI-Net: Encoder-Stage Phase Interaction
Network for Multiphasic Breast DCE-MRI Segmentation**.

ESPI-Net combines three phase-aware components with a compact 3D HCMA-UNet
backbone:

- **Input PAF** reweights the ordered DCE-MRI phases before spatial encoding.
- **MSPF** injects phase-specific features at encoder stages E0-E4.
- **TC loss** aligns deep phase-specific and fused representations during
  training and adds no inference-time branch.

The paper has been accepted for publication. This repository contains the
paper-aligned implementation, data conversion utilities, controlled ablation
trainers, and evaluation scripts.

## Results

| Dataset | Dice (%) | Precision (%) | IoU (%) | ASD (mm) |
|---|---:|---:|---:|---:|
| MAMA-MIA (5-fold CV) | 79.23 ± 2.15 | 80.71 ± 2.33 | 68.92 ± 2.38 | 4.49 ± 0.95 |
| Yunnan (external test) | 78.08 ± 3.19 | 74.29 ± 3.79 | 66.62 ± 3.67 | 6.47 ± 1.89 |

Values are patient-wise means ± 95% confidence-interval half-widths reported
in the accepted manuscript. MAMA-MIA was used for five-fold training and
validation; Yunnan was evaluated without target-dataset fine-tuning.

## Repository layout

```text
ESPI-Net/
├── nnUNet/
│   ├── nnunetv2/
│   │   ├── dataset_conversion/Dataset003_DCE_MRI_4phase.py
│   │   └── training/
│   │       ├── loss/compound_losses.py
│   │       └── nnUNetTrainer/
│   │           ├── nnUNetTrainerESPINet.py
│   │           └── variants/network_architecture/HCMA.py
│   └── pyproject.toml
└── scripts/                     # ablation and statistical-analysis tools
```

The main network class is `ESPINet`; `HCMA4D` remains as a compatibility alias
for checkpoints produced during development.

## Requirements

- Linux
- Python 3.9 or newer
- CUDA-capable NVIDIA GPU (the paper used one RTX 3090)
- PyTorch 2.1.2 or newer
- CUDA-compatible `mamba-ssm`

Create an environment and install PyTorch for your CUDA version first. Then
install Mamba and this nnU-Net fork:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip packaging ninja

# Install the correct PyTorch build from https://pytorch.org/get-started/locally/
python -m pip install torch torchvision
python -m pip install "mamba-ssm>=2.2.2" --no-build-isolation
python -m pip install -e ./nnUNet
```

The repository does not track locally compiled CUDA extensions. They must be
built for the CUDA and PyTorch versions of the target machine.

## Data

The experiments use four registered volumes per patient in temporal order:

| nnU-Net channel | DCE-MRI phase |
|---:|---|
| `0000` | V0, pre-contrast |
| `0001` | V1, first post-contrast |
| `0002` | V2, second post-contrast |
| `0003` | V3, third post-contrast |

Public datasets used in the paper:

- [MAMA-MIA](https://www.synapse.org/Synapse:syn60868042/wiki/628716)
- [Yunnan breast DCE-MRI](https://zenodo.org/records/8068383)

Dataset access and use remain subject to the terms set by their respective
providers. No patient data or trained checkpoints are included here.

For a manifest-based conversion example, geometry checks, and missing-phase
handling, see
[`nnUNet/documentation/dce_mri_4phase.md`](nnUNet/documentation/dce_mri_4phase.md).

## Preprocessing

Set the standard nnU-Net paths:

```bash
export nnUNet_raw=/path/to/nnUNet_raw
export nnUNet_preprocessed=/path/to/nnUNet_preprocessed
export nnUNet_results=/path/to/nnUNet_results
```

Place each case in the nnU-Net raw layout and run:

```bash
nnUNetv2_plan_and_preprocess -d 3 -c 3d_fullres --verify_dataset_integrity
```

All four phases of one patient must have identical geometry and must stay in
the same cross-validation fold.

## Training

Train the complete model for one fold:

```bash
nnUNetv2_train 3 3d_fullres 0 -tr nnUNetTrainerESPINet
```

Repeat with folds `0 1 2 3 4` for the paper's five-fold protocol. The trainer
uses 128³ patches, batch size 2, foreground oversampling 0.33, AdamW,
OneCycleLR, 500 epochs, mixed precision, gradient clipping at 12, and TC weight
0.20.

## Inference

```bash
nnUNetv2_predict \
  -i /path/to/imagesTs \
  -o /path/to/predictions \
  -d 3 -c 3d_fullres -f 0 1 2 3 4 \
  -tr nnUNetTrainerESPINet
```

TC and the regional feature loss are training-only objectives. Inference uses
the spatial backbone, Input PAF, and MSPF.

## Controlled ablations

The following trainers reproduce the component ablations while keeping the
backbone and training protocol fixed:

| Trainer | Spatial input | Input PAF | MSPF | TC |
|---|---|:---:|:---:|:---:|
| `nnUNetTrainerESPINet_SingleV1` | V1 | - | - | - |
| `nnUNetTrainerESPINet_Concat` | concatenated V0-V3 | - | - | - |
| `nnUNetTrainerESPINet_InputPAF` | fused V0-V3 | yes | - | - |
| `nnUNetTrainerESPINet_MSPF` | concatenated V0-V3 | - | E0-E4 | - |
| `nnUNetTrainerESPINet_NoTC` | fused V0-V3 | yes | E0-E4 | - |
| `nnUNetTrainerESPINet` | fused V0-V3 | yes | E0-E4 | yes |

Fusion-depth variants are `nnUNetTrainerESPINet_E4`, `_E3E4`, `_E2E4`, and
`_E1E4`; the complete trainer uses E0-E4. Run the matrix with:

```bash
DATASET_ID=3 FOLDS="0 1 2 3 4" bash scripts/run_ablation.sh
```

## Evaluation utilities

```bash
# Aggregate validation summary files
python scripts/aggregate_ablation.py \
  --root "$nnUNet_results/Dataset003_DCE-MRI" \
  --out ablation_summary.json

# Paired statistical comparison
python scripts/paired_stats.py \
  --summary_a /path/to/run_a/summary.json \
  --summary_b /path/to/run_b/summary.json \
  --metric Dice

# Check macro and micro metric calculations
python scripts/check_metrics_consistency.py --summary /path/to/summary.json

# Recalculate case-wise overlap metrics and optionally export CSV
python scripts/calculate_metrics.py \
  --summary /path/to/summary.json \
  --save_csv per_case_metrics.csv
```

## Citation

If this code is useful in your research, please cite:

```text
Jiajin Xu, Jun Li, Mingxin Liu, Jiahe Liu, Weiquan Shen, and Jun Xu.
"ESPI-Net: Encoder-Stage Phase Interaction Network for Multiphasic Breast
DCE-MRI Segmentation." Accepted conference paper, 2026.
```

A final BibTeX entry will be added when the proceedings metadata and DOI are
available.

## Acknowledgements and licensing

This implementation is built on [nnU-Net](https://github.com/MIC-DKFZ/nnUNet)
and the HCMA-UNet spatial backbone. The bundled nnU-Net-derived code retains
its Apache-2.0 license in [`nnUNet/LICENSE`](nnUNet/LICENSE). `mamba-ssm` is an
external dependency and is not vendored. Dataset licenses are independent of
the code. No license is granted for third-party datasets or model weights.

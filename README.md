# ESPI-Net

**ESPI-Net: Encoder-Stage Phase Interaction Network for Multiphasic Breast DCE-MRI Segmentation**.

This paper has been accepted by the BIBM 2026.

## Requirements

- Linux
- Python 3.12
- CUDA-capable NVIDIA GPU (the paper used one RTX 3090)
- PyTorch 2.5.1
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

## Acknowledgements and licensing

This implementation is built on [nnU-Net](https://github.com/MIC-DKFZ/nnUNet)
and the HCMA-UNet spatial backbone. The bundled nnU-Net-derived code retains
its Apache-2.0 license in [`nnUNet/LICENSE`](nnUNet/LICENSE). `mamba-ssm` is an
external dependency and is not vendored. Dataset licenses are independent of
the code. No license is granted for third-party datasets or model weights.

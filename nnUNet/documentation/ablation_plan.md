# ESPI-Net controlled ablations

Use the same dataset split, preprocessing plans, patch sampling, optimizer,
checkpoint selection, and evaluation code for every configuration.

## Component ablations

| Trainer | V0-V3 spatial input | Input PAF | MSPF | TC |
|---|---|:---:|:---:|:---:|
| `nnUNetTrainerESPINet_SingleV1` | V1 only | - | - | - |
| `nnUNetTrainerESPINet_Concat` | channel concatenation | - | - | - |
| `nnUNetTrainerESPINet_InputPAF` | attention fusion | yes | - | - |
| `nnUNetTrainerESPINet_MSPF` | channel concatenation | - | E0-E4 | - |
| `nnUNetTrainerESPINet_NoTC` | attention fusion | yes | E0-E4 | - |
| `nnUNetTrainerESPINet` | attention fusion | yes | E0-E4 | yes |

The concatenation and MSPF-only variants both feed the same four-channel input
to the spatial backbone. This isolates the contribution of encoder-stage phase
fusion.

## Fusion-depth ablations

The `_E4`, `_E3E4`, `_E2E4`, and `_E1E4` trainers compute the same phase
pyramid and scale-specific weights but inject them only at the named encoder
stages. The complete trainer injects at E0-E4.

## Run

```bash
DATASET_ID=3 FOLDS="0 1 2 3 4" bash scripts/run_ablation.sh
```

Each fold is an independent training run. Keep all phases from a patient in a
single fold, and determine checkpoint-selection rules using only the training
dataset.


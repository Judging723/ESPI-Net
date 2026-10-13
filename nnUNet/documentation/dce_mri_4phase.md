# Four-phase breast DCE-MRI setup

ESPI-Net expects one registered 3D image per DCE-MRI phase. nnU-Net represents
the temporal phase axis as four input channels:

```text
nnUNet_raw/Dataset003_DCE-MRI/
├── imagesTr/
│   ├── Case001_0000.nii.gz  # V0: pre-contrast
│   ├── Case001_0001.nii.gz  # V1: first post-contrast
│   ├── Case001_0002.nii.gz  # V2: second post-contrast
│   └── Case001_0003.nii.gz  # V3: third post-contrast
├── labelsTr/
│   └── Case001.nii.gz
└── imagesTs/
    ├── Case101_0000.nii.gz
    ├── Case101_0001.nii.gz
    ├── Case101_0002.nii.gz
    └── Case101_0003.nii.gz
```

All phases belonging to a patient must have the same size, spacing, origin,
direction, and anatomical registration. Split at the patient level so that no
phase from a patient appears in a different fold.

## Manifest-based conversion

The included converter copies registered images into nnU-Net format and checks
their geometry. Create a training CSV:

```csv
case_id,phase0,phase1,phase2,phase3,label
Case001,/data/Case001_v0.nii.gz,/data/Case001_v1.nii.gz,/data/Case001_v2.nii.gz,/data/Case001_v3.nii.gz,/data/Case001_label.nii.gz
```

An optional test CSV uses the same columns without `label`. Then run:

```bash
python nnUNet/nnunetv2/dataset_conversion/Dataset003_DCE_MRI_4phase.py \
  --dataset_name Dataset003_DCE-MRI \
  --train_csv /path/to/train.csv \
  --test_csv /path/to/test.csv \
  --fill nearest
```

If a phase is unavailable, leave its CSV field empty. `--fill nearest`
duplicates the closest available temporal phase; `--fill zero` creates a
zero-valued volume with the reference geometry. Record any imputation policy
in the experiment metadata, because it is not part of the paper's complete
four-phase protocol.

The converter creates a `dataset.json` with these channel names:

```json
{
  "channel_names": {
    "0": "V0_pre_contrast",
    "1": "V1_post_contrast",
    "2": "V2_post_contrast",
    "3": "V3_post_contrast"
  },
  "labels": {
    "background": 0,
    "lesion": 1
  },
  "file_ending": ".nii.gz"
}
```

## Preprocess and train

```bash
export nnUNet_raw=/path/to/nnUNet_raw
export nnUNet_preprocessed=/path/to/nnUNet_preprocessed
export nnUNet_results=/path/to/nnUNet_results

nnUNetv2_plan_and_preprocess -d 3 -c 3d_fullres --verify_dataset_integrity
nnUNetv2_train 3 3d_fullres 0 -tr nnUNetTrainerESPINet
```

The complete trainer enables Input PAF, MSPF at E0-E4, and TC loss with weight
0.20. See the repository README for controlled component and fusion-depth
ablation trainer names.


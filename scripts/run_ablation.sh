#!/usr/bin/env bash
set -euo pipefail

DATASET_ID="${DATASET_ID:-3}"
CONFIGURATION="${CONFIGURATION:-3d_fullres}"
FOLDS="${FOLDS:-0}"
EXPERIMENT_GROUP="${EXPERIMENT_GROUP:-espi_component_ablation}"

trainers=(
  nnUNetTrainerESPINet_SingleV1
  nnUNetTrainerESPINet_Concat
  nnUNetTrainerESPINet_InputPAF
  nnUNetTrainerESPINet_MSPF
  nnUNetTrainerESPINet_NoTC
  nnUNetTrainerESPINet
)

for trainer in "${trainers[@]}"; do
  for fold in ${FOLDS}; do
    nnUNetv2_train \
      "${DATASET_ID}" "${CONFIGURATION}" "${fold}" \
      -tr "${trainer}" --exp_name "${EXPERIMENT_GROUP}"
  done
done


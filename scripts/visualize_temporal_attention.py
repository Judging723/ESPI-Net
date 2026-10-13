"""Visualize Input PAF and stage-wise MSPF phase responses for one case."""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from nnunetv2.run.run_training import get_trainer_from_args
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDataset
from nnunetv2.training.dataloading.utils import get_case_identifiers


def _pick_case(preprocessed_folder: str, case_id: str) -> str:
    if case_id:
        return case_id
    cases = get_case_identifiers(preprocessed_folder)
    if not cases:
        raise RuntimeError(f"No cases found in {preprocessed_folder}")
    return cases[0]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot spatially averaged Input PAF and MSPF phase weights."
    )
    parser.add_argument("--dataset_id", required=True)
    parser.add_argument("--configuration", default="3d_fullres")
    parser.add_argument("--fold", default=0, type=int)
    parser.add_argument("--trainer", default="nnUNetTrainerESPINet")
    parser.add_argument("--exp_name", default="")
    parser.add_argument("--case_id", default="")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--out", default="phase_responses.png")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = get_trainer_from_args(
        dataset_name_or_id=args.dataset_id,
        configuration=args.configuration,
        fold=args.fold,
        trainer_name=args.trainer,
        exp_name=args.exp_name,
        device=device,
    )
    trainer.initialize()
    checkpoint = args.checkpoint or os.path.join(
        trainer.output_folder, "checkpoint_best.pth"
    )
    trainer.load_checkpoint(checkpoint)
    trainer.network.eval()

    network = getattr(trainer.network, "module", trainer.network)
    modules = []
    labels = []
    if getattr(network, "input_fusion", None) is not None:
        modules.append(network.input_fusion)
        labels.append("Input PAF")
    for index, module in enumerate(getattr(network, "phase_fusion", [])):
        modules.append(module)
        labels.append(f"E{index}")
    if not modules:
        raise RuntimeError("The selected trainer has no phase-attention modules.")

    captured = []
    handles = []

    def capture_weights(_module, _inputs, output):
        weights = output[2].detach().float().cpu()
        reduce_dims = tuple(index for index in range(weights.ndim) if index != 1)
        captured.append(weights.mean(dim=reduce_dims).numpy())

    for module in modules:
        handles.append(module.register_forward_hook(capture_weights))

    dataset = nnUNetDataset(trainer.preprocessed_dataset_folder)
    case_id = _pick_case(trainer.preprocessed_dataset_folder, args.case_id)
    data, _, _ = dataset.load_case(case_id)
    x = torch.from_numpy(data).float().unsqueeze(0).to(device)
    try:
        with torch.no_grad():
            network(x)
    finally:
        for handle in handles:
            handle.remove()

    if len(captured) != len(labels):
        raise RuntimeError(
            f"Captured {len(captured)} responses from {len(labels)} modules."
        )

    matrix = np.stack(captured)
    fig, ax = plt.subplots(figsize=(6.4, 0.75 * len(labels) + 2.0))
    image = ax.imshow(matrix, vmin=0.0, vmax=1.0, cmap="viridis", aspect="auto")
    ax.set_xticks(np.arange(matrix.shape[1]), [f"V{i}" for i in range(matrix.shape[1])])
    ax.set_yticks(np.arange(len(labels)), labels)
    ax.set_xlabel("DCE-MRI phase")
    ax.set_title(f"ESPI-Net phase responses: {case_id}")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            ax.text(column, row, f"{matrix[row, column]:.3f}", ha="center", va="center", color="white")
    fig.colorbar(image, ax=ax, label="Mean softmax weight")
    fig.tight_layout()
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved phase-response heatmap to {args.out}")


if __name__ == "__main__":
    main()


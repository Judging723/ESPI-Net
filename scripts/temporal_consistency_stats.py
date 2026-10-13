"""Measure deep phase-feature cosine similarity for an ESPI-Net checkpoint."""

import argparse
import csv
import itertools
import os

import numpy as np
import torch
import torch.nn.functional as F

from nnunetv2.run.run_training import get_trainer_from_args
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDataset
from nnunetv2.training.dataloading.utils import get_case_identifiers


def _pairwise_similarity(phase_features: torch.Tensor) -> dict[str, float]:
    if phase_features is None or phase_features.ndim != 6:
        raise ValueError("Expected deep phase features with shape (B,T,C,D,H,W)")
    result = {}
    for left, right in itertools.combinations(range(phase_features.shape[1]), 2):
        similarity = F.cosine_similarity(
            phase_features[:, left], phase_features[:, right], dim=1, eps=1e-8
        ).mean()
        result[f"V{left}_V{right}"] = float(similarity.item())
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_id", required=True)
    parser.add_argument("--configuration", default="3d_fullres")
    parser.add_argument("--fold", default=0, type=int)
    parser.add_argument("--trainer", default="nnUNetTrainerESPINet")
    parser.add_argument("--exp_name", default="")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--out_csv", default="phase_feature_similarity.csv")
    parser.add_argument("--limit", type=int, default=0)
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

    dataset = nnUNetDataset(trainer.preprocessed_dataset_folder)
    case_ids = get_case_identifiers(trainer.preprocessed_dataset_folder)
    if args.limit > 0:
        case_ids = case_ids[: args.limit]

    rows = []
    with torch.no_grad():
        for case_id in case_ids:
            data, _, _ = dataset.load_case(case_id)
            x = torch.from_numpy(data).float().unsqueeze(0).to(device)
            features, _ = network(x, return_features=True)
            phase_features = features[1]
            rows.append({"case_id": case_id, **_pairwise_similarity(phase_features)})

    if not rows:
        raise RuntimeError("No cases were evaluated.")
    fieldnames = list(rows[0])
    with open(args.out_csv, "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    values = np.array([[row[key] for key in fieldnames[1:]] for row in rows])
    print(f"Evaluated {len(rows)} cases")
    for index, key in enumerate(fieldnames[1:]):
        print(f"{key}: {values[:, index].mean():.6f} ± {values[:, index].std():.6f}")
    print(f"Saved per-case similarities to {args.out_csv}")


if __name__ == "__main__":
    main()


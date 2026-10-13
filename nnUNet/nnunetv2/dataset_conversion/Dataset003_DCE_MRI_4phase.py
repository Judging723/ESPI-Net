import argparse
import csv
import os
import shutil
from typing import List, Optional

import SimpleITK as sitk
import numpy as np

from batchgenerators.utilities.file_and_folder_operations import maybe_mkdir_p, save_json
from nnunetv2.paths import nnUNet_raw


def _read_sitk(path: str) -> sitk.Image:
    return sitk.ReadImage(path)


def _make_zero_like(ref: sitk.Image) -> sitk.Image:
    arr = sitk.GetArrayFromImage(ref)
    zeros = np.zeros_like(arr)
    out = sitk.GetImageFromArray(zeros)
    out.CopyInformation(ref)
    return out


def _choose_nearest_available(missing_idx: int, available: List[int]) -> int:
    return min(available, key=lambda x: abs(x - missing_idx))


def _load_or_fill_phase(phase_path: Optional[str], ref: sitk.Image, fill: str) -> sitk.Image:
    if phase_path is None or phase_path == "":
        if fill == "zero":
            return _make_zero_like(ref)
        raise ValueError("Missing phase but fill != zero. Use fill=zero or fill=nearest.")
    return _read_sitk(phase_path)


def _validate_geometry(ref: sitk.Image, img: sitk.Image, path: str) -> None:
    if ref.GetSize() != img.GetSize():
        raise ValueError(f"Geometry mismatch size for {path}")
    if ref.GetSpacing() != img.GetSpacing():
        raise ValueError(f"Geometry mismatch spacing for {path}")
    if ref.GetOrigin() != img.GetOrigin():
        raise ValueError(f"Geometry mismatch origin for {path}")
    if ref.GetDirection() != img.GetDirection():
        raise ValueError(f"Geometry mismatch direction for {path}")


def _write_phase(img: sitk.Image, out_path: str) -> None:
    sitk.WriteImage(img, out_path, True)


def _resolve_phases(paths: List[Optional[str]], fill: str) -> List[sitk.Image]:
    available = [i for i, p in enumerate(paths) if p not in (None, "")]
    if len(available) == 0:
        raise ValueError("All phases missing for a case.")
    ref = _read_sitk(paths[available[0]])
    imgs: List[sitk.Image] = [None] * 4
    # load available phases
    for i in available:
        img = _read_sitk(paths[i])
        _validate_geometry(ref, img, paths[i])
        imgs[i] = img
    # fill missing
    for i in range(4):
        if imgs[i] is None:
            if fill == "nearest":
                nearest = _choose_nearest_available(i, available)
                imgs[i] = imgs[nearest]
            elif fill == "zero":
                imgs[i] = _make_zero_like(ref)
            else:
                raise ValueError("fill must be one of: nearest, zero")
    return imgs


def _write_case_phases(case_id: str, phases: List[sitk.Image], out_dir: str) -> None:
    for i, img in enumerate(phases):
        out_path = os.path.join(out_dir, f"{case_id}_{i:04d}.nii.gz")
        _write_phase(img, out_path)


def _read_csv_rows(path: str) -> List[dict]:
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        return [row for row in reader]


def convert_dataset(
    dataset_name: str,
    train_csv: str,
    test_csv: Optional[str],
    fill: str,
) -> None:
    target_folder = os.path.join(nnUNet_raw, dataset_name)
    imagesTr = os.path.join(target_folder, "imagesTr")
    imagesTs = os.path.join(target_folder, "imagesTs")
    labelsTr = os.path.join(target_folder, "labelsTr")
    maybe_mkdir_p(imagesTr)
    maybe_mkdir_p(imagesTs)
    maybe_mkdir_p(labelsTr)

    train_rows = _read_csv_rows(train_csv)
    for row in train_rows:
        case_id = row["case_id"]
        phase_paths = [row.get(f"phase{i}", "") for i in range(4)]
        phases = _resolve_phases(phase_paths, fill)
        _write_case_phases(case_id, phases, imagesTr)
        label_path = row.get("label", "")
        if label_path == "":
            raise ValueError(f"Missing label for case {case_id}")
        shutil.copy(label_path, os.path.join(labelsTr, f"{case_id}.nii.gz"))

    test_rows: List[dict] = []
    if test_csv:
        test_rows = _read_csv_rows(test_csv)
        for row in test_rows:
            case_id = row["case_id"]
            phase_paths = [row.get(f"phase{i}", "") for i in range(4)]
            phases = _resolve_phases(phase_paths, fill)
            _write_case_phases(case_id, phases, imagesTs)

    # dataset.json
    dataset_json = {
        "name": "DCE4D",
        "description": "4-phase DCE-MRI breast lesion segmentation",
        "tensorImageSize": "3D",
        "reference": "",
        "licence": "",
        "release": "1.0",
        "channel_names": {
            "0": "V0_pre_contrast",
            "1": "V1_post_contrast",
            "2": "V2_post_contrast",
            "3": "V3_post_contrast",
        },
        "labels": {
            "background": 0,
            "lesion": 1,
        },
        "numTraining": len(train_rows),
        "numTest": len(test_rows),
        "file_ending": ".nii.gz",
        "training": [],
        "test": [],
    }
    save_json(dataset_json, os.path.join(target_folder, "dataset.json"), sort_keys=False)


def entry_point() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="Dataset003_DCE-MRI",
        help="Target dataset folder name in nnUNet_raw",
    )
    parser.add_argument("--train_csv", type=str, required=True)
    parser.add_argument("--test_csv", type=str, required=False, default="")
    parser.add_argument(
        "--fill",
        type=str,
        choices=["nearest", "zero"],
        default="nearest",
        help="How to fill missing phases",
    )
    args = parser.parse_args()
    convert_dataset(
        dataset_name=args.dataset_name,
        train_csv=args.train_csv,
        test_csv=args.test_csv if args.test_csv else None,
        fill=args.fill,
    )


if __name__ == "__main__":
    entry_point()

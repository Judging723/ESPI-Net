import argparse
import json
from pathlib import Path

import numpy as np


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den > 0 else float("nan")


def _pick_label_key(summary: dict, label_key: str | None) -> str:
    if label_key:
        return label_key
    metric_per_case = summary.get("metric_per_case", [])
    if not metric_per_case:
        raise ValueError("summary.json has no metric_per_case")
    keys = list(metric_per_case[0]["metrics"].keys())
    if "1" in keys:
        return "1"
    if "foreground" in keys:
        return "foreground"
    return keys[0]


def _calc_macro(per_case_metrics: list[dict]) -> dict:
    dice = []
    iou = []
    prec = []
    rec = []
    vs = []
    tp_list = []
    fp_list = []
    fn_list = []
    tn_list = []

    for m in per_case_metrics:
        tp = float(m["TP"])
        fp = float(m["FP"])
        fn = float(m["FN"])
        tn = float(m.get("TN", 0.0))
        tp_list.append(tp)
        fp_list.append(fp)
        fn_list.append(fn)
        tn_list.append(tn)

        dice.append(_safe_div(2 * tp, 2 * tp + fp + fn))
        iou.append(_safe_div(tp, tp + fp + fn))
        prec.append(_safe_div(tp, tp + fp))
        rec.append(_safe_div(tp, tp + fn))
        vs.append(1.0 - _safe_div(abs(fp - fn), (2 * tp + fp + fn)))

    return {
        "Dice": float(np.nanmean(dice)),
        "IoU": float(np.nanmean(iou)),
        "Precision": float(np.nanmean(prec)),
        "Recall": float(np.nanmean(rec)),
        "VS": float(np.nanmean(vs)),
        "TP_mean": float(np.nanmean(tp_list)),
        "FP_mean": float(np.nanmean(fp_list)),
        "FN_mean": float(np.nanmean(fn_list)),
        "TN_mean": float(np.nanmean(tn_list)),
    }


def _calc_micro(per_case_metrics: list[dict]) -> dict:
    tp = float(np.nansum([m["TP"] for m in per_case_metrics]))
    fp = float(np.nansum([m["FP"] for m in per_case_metrics]))
    fn = float(np.nansum([m["FN"] for m in per_case_metrics]))
    tn = float(np.nansum([m.get("TN", 0.0) for m in per_case_metrics]))

    return {
        "Dice": _safe_div(2 * tp, 2 * tp + fp + fn),
        "IoU": _safe_div(tp, tp + fp + fn),
        "Precision": _safe_div(tp, tp + fp),
        "Recall": _safe_div(tp, tp + fn),
        "VS": 1.0 - _safe_div(abs(fp - fn), (2 * tp + fp + fn)),
        "TP_sum": tp,
        "FP_sum": fp,
        "FN_sum": fn,
        "TN_sum": tn,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check metric consistency (macro vs micro) from nnUNet summary.json"
    )
    parser.add_argument("--summary", required=True, help="Path to summary.json")
    parser.add_argument("--label_key", default="", help="Label key, e.g. 1")
    args = parser.parse_args()

    summary_path = Path(args.summary)
    summary = json.loads(summary_path.read_text())
    label_key = _pick_label_key(summary, args.label_key or None)

    per_case = summary["metric_per_case"]
    per_case_metrics = [item["metrics"][label_key] for item in per_case]

    macro = _calc_macro(per_case_metrics)
    micro = _calc_micro(per_case_metrics)
    summary_mean = summary.get("mean", {}).get(label_key, {})

    print(f"Summary file: {summary_path}")
    print(f"Label key: {label_key}")
    print()
    print("=== Macro (mean over cases) ===")
    for k in ["Dice", "IoU", "Precision", "Recall", "VS"]:
        print(f"{k:>9}: {macro[k]:.6f}")
    print()
    print("=== Micro (global TP/FP/FN aggregation) ===")
    for k in ["Dice", "IoU", "Precision", "Recall", "VS"]:
        print(f"{k:>9}: {micro[k]:.6f}")
    print()
    print("=== summary.json mean[label] (nnUNet default) ===")
    for k in ["Dice", "IoU", "TP", "FP", "FN", "TN", "n_pred", "n_ref"]:
        if k in summary_mean:
            print(f"{k:>9}: {float(summary_mean[k]):.6f}")
    print()
    print("Note: In nnUNet summary.json, TP/FP/FN are case-wise means, not global sums.")
    print("So recomputing IoU/Precision/Recall from summary.mean TP/FP/FN will generally mismatch.")


if __name__ == "__main__":
    main()

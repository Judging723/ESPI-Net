"""Recalculate patient-wise overlap metrics from an nnU-Net summary file."""

import argparse
import csv
import json
from pathlib import Path
from statistics import mean


def _case_items(summary: dict) -> list[dict]:
    for key in ("metric_per_case", "per_case", "cases"):
        value = summary.get(key)
        if isinstance(value, list):
            return value
    raise ValueError("No per-case metric list found in the summary file.")


def _label_metrics(item: dict, label_key: str) -> dict:
    metrics = item.get("metrics", item)
    if not isinstance(metrics, dict):
        raise ValueError("Invalid per-case metrics entry.")
    if label_key in metrics and isinstance(metrics[label_key], dict):
        return metrics[label_key]
    candidates = [
        value
        for key, value in metrics.items()
        if key not in {"mean", "foreground_mean"} and isinstance(value, dict)
    ]
    return candidates[0] if candidates else metrics


def _counts(metrics: dict) -> tuple[float, float, float]:
    tp = metrics.get("TP", metrics.get("tp"))
    fp = metrics.get("FP", metrics.get("fp"))
    fn = metrics.get("FN", metrics.get("fn"))
    if tp is None or fp is None or fn is None:
        raise ValueError("Each case must contain TP, FP, and FN.")
    return float(tp), float(fp), float(fn)


def overlap_metrics(tp: float, fp: float, fn: float) -> dict[str, float]:
    def divide(numerator: float, denominator: float) -> float:
        return numerator / denominator if denominator > 0 else 0.0

    return {
        "Dice": divide(2 * tp, 2 * tp + fp + fn),
        "IoU": divide(tp, tp + fp + fn),
        "Recall": divide(tp, tp + fn),
        "Precision": divide(tp, tp + fp),
        "VS": 1.0 - divide(abs(fn - fp), 2 * tp + fp + fn),
    }


def _case_name(item: dict, index: int) -> str:
    for key in ("case_id", "name", "id", "prediction_file", "reference_file"):
        if key in item:
            return Path(str(item[key])).name
    return f"case_{index:04d}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True)
    parser.add_argument("--label", default="1")
    parser.add_argument("--save_csv", default="")
    args = parser.parse_args()

    summary = json.loads(Path(args.summary).read_text(encoding="utf-8"))
    rows = []
    for index, item in enumerate(_case_items(summary)):
        tp, fp, fn = _counts(_label_metrics(item, args.label))
        rows.append(
            {
                "case": _case_name(item, index),
                "TP": tp,
                "FP": fp,
                "FN": fn,
                **overlap_metrics(tp, fp, fn),
            }
        )

    if not rows:
        raise ValueError("No valid cases found.")
    print(f"Valid cases: {len(rows)}")
    for metric in ("Dice", "IoU", "Recall", "Precision", "VS"):
        print(f"{metric:>9}: {mean(row[metric] for row in rows):.6f}")

    if args.save_csv:
        with open(args.save_csv, "w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved per-case metrics to {args.save_csv}")


if __name__ == "__main__":
    main()


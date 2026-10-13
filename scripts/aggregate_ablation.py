"""Aggregate patient-wise metrics from ablation summary files."""

import argparse
import json
from pathlib import Path

import numpy as np


def _label_key(summary: dict) -> str:
    cases = summary.get("metric_per_case", [])
    if not cases:
        raise ValueError("summary.json has no metric_per_case entries")
    metrics = cases[0].get("metrics", {})
    for key in ("1", "foreground", "foreground_mean"):
        if key in metrics:
            return key
    if not metrics:
        raise ValueError("First case has no metrics")
    return next(iter(metrics))


def _derived_metric(metrics: dict, name: str) -> float:
    if name in metrics:
        return float(metrics[name])
    aliases = {
        "ASD": ("Average Surface Distance", "average_surface_distance"),
        "HD95": ("Hausdorff Distance 95", "hausdorff_distance_95"),
    }
    for alias in aliases.get(name, ()):
        if alias in metrics:
            return float(metrics[alias])
    tp = float(metrics.get("TP", np.nan))
    fp = float(metrics.get("FP", np.nan))
    fn = float(metrics.get("FN", np.nan))
    denominator = {
        "Dice": 2 * tp + fp + fn,
        "IoU": tp + fp + fn,
        "Precision": tp + fp,
        "Recall": tp + fn,
    }.get(name)
    numerator = {
        "Dice": 2 * tp,
        "IoU": tp,
        "Precision": tp,
        "Recall": tp,
    }.get(name)
    if denominator is None or not np.isfinite(denominator) or denominator <= 0:
        return float("nan")
    return float(numerator / denominator)


def summarize(summary: dict, metric_names: list[str]) -> dict:
    label = _label_key(summary)
    per_case = [item["metrics"][label] for item in summary["metric_per_case"]]
    result = {}
    for name in metric_names:
        values = np.asarray([_derived_metric(item, name) for item in per_case])
        valid = values[np.isfinite(values)]
        result[name] = {
            "mean": float(valid.mean()) if valid.size else None,
            "std": float(valid.std()) if valid.size else None,
            "n": int(valid.size),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--out", default="ablation_summary.json")
    parser.add_argument(
        "--metrics", nargs="+", default=["Dice", "Precision", "IoU", "ASD"]
    )
    args = parser.parse_args()

    root = Path(args.root)
    files = sorted(root.rglob("summary.json"))
    if not files:
        raise SystemExit(f"No summary.json found under {root}")
    result = {
        str(path.relative_to(root)): summarize(
            json.loads(path.read_text(encoding="utf-8")), args.metrics
        )
        for path in files
    }
    Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Wrote {len(result)} summaries to {args.out}")


if __name__ == "__main__":
    main()


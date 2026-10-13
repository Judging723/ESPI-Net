"""Paired comparison of matching cases from two nnU-Net summary files."""

import argparse
import json
from pathlib import Path

import numpy as np


def _label_key(summary: dict) -> str:
    metrics = summary["metric_per_case"][0]["metrics"]
    for key in ("1", "foreground", "foreground_mean"):
        if key in metrics:
            return key
    return next(iter(metrics))


def _case_id(item: dict, index: int) -> str:
    for key in ("case_id", "prediction_file", "reference_file", "name", "id"):
        if key in item:
            return Path(str(item[key])).name
    return f"case_{index:04d}"


def metric_by_case(summary: dict, metric: str, label: str) -> dict[str, float]:
    values = {}
    for index, item in enumerate(summary.get("metric_per_case", [])):
        metrics = item.get("metrics", {}).get(label, {})
        if metric in metrics:
            values[_case_id(item, index)] = float(metrics[metric])
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary_a", required=True)
    parser.add_argument("--summary_b", required=True)
    parser.add_argument("--metric", default="Dice")
    parser.add_argument("--label_key", default="")
    args = parser.parse_args()

    a = json.loads(Path(args.summary_a).read_text(encoding="utf-8"))
    b = json.loads(Path(args.summary_b).read_text(encoding="utf-8"))
    label = args.label_key or _label_key(a)
    a_by_case = metric_by_case(a, args.metric, label)
    b_by_case = metric_by_case(b, args.metric, label)
    case_ids = sorted(a_by_case.keys() & b_by_case.keys())
    if not case_ids:
        raise ValueError("No matching cases with the requested metric.")

    a_values = np.asarray([a_by_case[case] for case in case_ids])
    b_values = np.asarray([b_by_case[case] for case in case_ids])
    difference = a_values - b_values
    print(f"N={len(case_ids)}")
    print(f"{args.metric} A mean±std: {a_values.mean():.4f}±{a_values.std():.4f}")
    print(f"{args.metric} B mean±std: {b_values.mean():.4f}±{b_values.std():.4f}")
    print(f"Difference mean±std: {difference.mean():.4f}±{difference.std():.4f}")

    from scipy.stats import ttest_rel, wilcoxon

    print(f"Paired t-test p={ttest_rel(a_values, b_values).pvalue:.4e}")
    try:
        pvalue = wilcoxon(a_values, b_values).pvalue
        print(f"Wilcoxon p={pvalue:.4e}")
    except ValueError as error:
        print(f"Wilcoxon unavailable: {error}")


if __name__ == "__main__":
    main()


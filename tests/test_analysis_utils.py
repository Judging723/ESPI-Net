import sys
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from aggregate_ablation import summarize
from calculate_metrics import overlap_metrics
from paired_stats import metric_by_case


def _summary(cases):
    return {
        "metric_per_case": [
            {
                "case_id": case_id,
                "metrics": {"1": {"TP": tp, "FP": fp, "FN": fn, "Dice": dice}},
            }
            for case_id, tp, fp, fn, dice in cases
        ]
    }


def test_overlap_metrics():
    metrics = overlap_metrics(tp=8, fp=2, fn=2)
    assert metrics["Dice"] == 0.8
    assert metrics["IoU"] == 2 / 3
    assert metrics["Precision"] == 0.8
    assert metrics["Recall"] == 0.8
    assert metrics["VS"] == 1.0


def test_ablation_summary_derives_metrics_from_counts():
    result = summarize(
        _summary([("case_b", 8, 2, 2, 0.8), ("case_a", 3, 1, 2, 2 / 3)]),
        ["Dice", "Precision", "IoU", "ASD"],
    )
    assert result["Dice"]["n"] == 2
    assert result["Precision"]["n"] == 2
    assert result["IoU"]["n"] == 2
    assert result["ASD"] == {"mean": None, "std": None, "n": 0}


def test_paired_values_are_keyed_by_case_not_file_order():
    summary = _summary([("case_b", 1, 0, 0, 0.9), ("case_a", 1, 0, 0, 0.7)])
    assert metric_by_case(summary, "Dice", "1") == {"case_b": 0.9, "case_a": 0.7}


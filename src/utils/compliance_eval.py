"""
compliance_eval.py
-------------------
Ordinal evaluation metrics and a reusable eval runner for the compliance
checker's 4-level verdict scale (0=fully compliant ... 3=not compliant).

Binary precision/recall/F1 don't fit an ordinal 4-level scale — being off by
one level (e.g. predicting 1 when truth is 0) is a much smaller error than
being off by three (predicting 0 when truth is 3), and exact-match accuracy
alone can't distinguish those. This module scores both.
"""

from __future__ import annotations

import time

from src.compliance.checker import check_compliance


def compute_ordinal_metrics(
    results: list[dict],
    label_key: str = "ground_truth_verdict",
    pred_key: str = "predicted_verdict",
) -> dict:
    """
    Ordinal metrics for a 0-3 verdict scale:
      - exact_match_accuracy: predicted verdict == ground truth exactly
      - off_by_one_accuracy: |predicted - truth| <= 1 (a tolerant accuracy —
        e.g. predicting 1 when truth is 0 is a minor miss, not a failure)
      - severe_miss_rate: |predicted - truth| >= 2 (e.g. calling a real
        violation "fully compliant" or vice versa — the failure mode that
        actually matters for a compliance tool)
      - mean_absolute_error: average |predicted - truth| across all examples
    """
    n = len(results)
    if n == 0:
        return {
            "n": 0, "exact_match_accuracy": 0.0, "off_by_one_accuracy": 0.0,
            "severe_miss_rate": 0.0, "mean_absolute_error": 0.0,
        }

    diffs = [abs(r[pred_key] - r[label_key]) for r in results]
    exact = sum(1 for d in diffs if d == 0)
    off_by_one = sum(1 for d in diffs if d <= 1)
    severe = sum(1 for d in diffs if d >= 2)

    return {
        "n": n,
        "exact_match_accuracy": round(exact / n, 3),
        "off_by_one_accuracy": round(off_by_one / n, 3),
        "severe_miss_rate": round(severe / n, 3),
        "mean_absolute_error": round(sum(diffs) / n, 3),
    }


def run_compliance_eval(examples: list[dict], model: str | None = None) -> list[dict]:
    """
    Run check_compliance() over a labeled example set, optionally overriding
    the model (for comparing gpt-4o-mini vs. another model on the same set).

    Each example must have: id, claim, category, ground_truth_verdict.
    """
    kwargs = {"model": model} if model else {}
    results = []

    for ex in examples:
        start = time.time()
        result = check_compliance(ex["claim"], **kwargs)
        latency_ms = round((time.time() - start) * 1000)

        results.append({
            "id": ex["id"],
            "claim": ex["claim"],
            "category": ex["category"],
            "ground_truth_verdict": ex["ground_truth_verdict"],
            "predicted_verdict": result["verdict"],
            "correct": result["verdict"] == ex["ground_truth_verdict"],
            "source": result["source"],
            "cited_sections": result.get("cited_sections", []),
            "reason": result.get("reason", ""),
            "notes": result.get("notes", ""),
            "latency_ms": latency_ms,
        })

    return results


def breakdown_by_category(results: list[dict]) -> dict:
    categories = sorted(set(r["category"] for r in results))
    breakdown = {}
    for cat in categories:
        subset = [r for r in results if r["category"] == cat]
        metrics = compute_ordinal_metrics(subset)
        breakdown[cat] = metrics
    return breakdown

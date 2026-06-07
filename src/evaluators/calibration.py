"""Calibration utilities: meta-evaluate judges against human-labelled samples.

Usage:
    samples = load_calibration("data/calibration")
    metrics = compare_to_human(samples, judge_scores)

Each calibration sample is a JSON file with the structure::

    {
        "case_id": "1__cooperative",
        "human_labels": {
            "gsr": {"passed": true, "rationale": "..."},
            "kar": {"passed": false},
            ...
        }
    }

We compute simple agreement and Cohen's kappa per dimension.
"""
from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Optional


logger = logging.getLogger(__name__)


def load_calibration(folder: str | Path) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    base = Path(folder)
    if not base.exists():
        return samples
    for f in sorted(base.glob("*.json")):
        try:
            samples.append(json.loads(f.read_text(encoding="utf-8")))
        except Exception:
            logger.exception("Failed to read calibration sample %s", f)
    return samples


def cohens_kappa(pairs: Iterable[tuple[bool, bool]]) -> Optional[float]:
    items = list(pairs)
    if not items:
        return None
    n = len(items)
    agree = sum(1 for a, b in items if a == b) / n
    pa = sum(1 for a, _ in items if a) / n
    pb = sum(1 for _, b in items if b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    if pe == 1:
        return 1.0 if agree == 1.0 else 0.0
    return round((agree - pe) / (1 - pe), 4)


def compare_to_human(
    samples: list[dict[str, Any]],
    judge_results: dict[str, dict[str, bool]],
) -> dict[str, dict[str, Any]]:
    """Return per-dimension accuracy and kappa.

    judge_results[case_id][dim_id] should be a bool indicating whether the judge
    considered the dimension "passed" (e.g. score >= 0.5).
    """
    grouped: dict[str, list[tuple[bool, bool]]] = defaultdict(list)
    for sample in samples:
        case_id = sample.get("case_id")
        labels = sample.get("human_labels") or {}
        judge = judge_results.get(case_id, {})
        for dim_id, info in labels.items():
            human_pass = bool(info.get("passed"))
            if dim_id not in judge:
                continue
            judge_pass = bool(judge[dim_id])
            grouped[dim_id].append((human_pass, judge_pass))
    metrics: dict[str, dict[str, Any]] = {}
    for dim_id, items in grouped.items():
        if not items:
            continue
        agree = sum(1 for a, b in items if a == b) / len(items)
        metrics[dim_id] = {
            "n": len(items),
            "accuracy": round(agree, 4),
            "kappa": cohens_kappa(items),
        }
    return metrics

"""Meta-evaluation: compare LLM judge results to human-labelled samples.

Run after you have at least one ``run_report.json`` and a few human
labelled samples in ``data/calibration``::

    python -m src.cli.meta_eval --run reports/run_xxx
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from rich.console import Console
from rich.table import Table

from ..evaluators.calibration import compare_to_human, load_calibration


PASS_THRESHOLDS = {
    "gsr": 0.85,
    "pcr": 0.8,
    "bca": 0.8,
    "kar": 0.8,
    "hcr": 0.8,
    "scr": 0.8,
    "ter": 0.8,
    "rob": 0.7,
}


def _index_run_report(path: Path) -> dict[str, dict[str, bool]]:
    """Convert a run_report.json into ``{case_id: {dim_id: passed}}``."""
    data = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, dict[str, bool]] = {}
    for case in data.get("cases", []):
        case_id = case.get("trace", {}).get("case_id")
        if not case_id:
            continue
        passed: dict[str, bool] = {}
        for dim in case.get("dimensions", []):
            dim_id = dim.get("id")
            score = float(dim.get("score", 0.0))
            passed[dim_id] = score >= PASS_THRESHOLDS.get(dim_id, 0.8)
        out[case_id] = passed
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Meta-evaluate judge vs human labels.")
    parser.add_argument("--run", type=Path, required=True, help="Run directory or run_report.json path")
    parser.add_argument("--calibration", type=Path, default=Path("data/calibration"))
    args = parser.parse_args()

    console = Console()
    run_path = args.run
    if run_path.is_dir():
        run_path = run_path / "run_report.json"
    if not run_path.exists():
        raise SystemExit(f"run_report.json not found at {run_path}")

    judge_results = _index_run_report(run_path)
    samples = load_calibration(args.calibration)
    samples = [s for s in samples if not str(s.get("case_id", "")).endswith(".example")]
    if not samples:
        console.print("[yellow]No calibration samples found.[/]")
        return
    metrics = compare_to_human(samples, judge_results)
    if not metrics:
        console.print(
            "[yellow]No overlapping case ids between calibration set and run report.[/]"
        )
        return
    table = Table(title=f"Meta-evaluation @ {run_path.parent.name}")
    table.add_column("Dim")
    table.add_column("N", justify="right")
    table.add_column("Accuracy", justify="right")
    table.add_column("Cohen's kappa", justify="right")
    for dim_id, info in sorted(metrics.items()):
        kappa = info.get("kappa")
        kappa_str = f"{kappa:.3f}" if kappa is not None else "n/a"
        table.add_row(dim_id, str(info["n"]), f"{info['accuracy']:.3f}", kappa_str)
    console.print(table)


if __name__ == "__main__":
    main()

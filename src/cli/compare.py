"""Compare two evaluation runs side-by-side."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from rich.console import Console
from rich.table import Table


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two run reports (JSON).")
    parser.add_argument("--run-a", type=Path, required=True)
    parser.add_argument("--run-b", type=Path, required=True)
    args = parser.parse_args()

    a = _load(args.run_a / "run_report.json" if args.run_a.is_dir() else args.run_a)
    b = _load(args.run_b / "run_report.json" if args.run_b.is_dir() else args.run_b)

    console = Console()
    summary = Table(title="Overall comparison")
    summary.add_column("Metric")
    summary.add_column("Run A")
    summary.add_column("Run B")
    summary.add_column("Δ (B - A)")
    summary.add_row(
        "overall_mean",
        f"{a['aggregate']['overall_mean']:.3f}",
        f"{b['aggregate']['overall_mean']:.3f}",
        f"{b['aggregate']['overall_mean'] - a['aggregate']['overall_mean']:+.3f}",
    )
    summary.add_row("n_cases", str(a["aggregate"]["n_cases"]), str(b["aggregate"]["n_cases"]), "")
    console.print(summary)

    dim_table = Table(title="Per-dimension means")
    dim_table.add_column("Dim")
    dim_table.add_column("Run A")
    dim_table.add_column("Run B")
    dim_table.add_column("Δ")
    a_dims = a["aggregate"].get("dimensions", {})
    b_dims = b["aggregate"].get("dimensions", {})
    for dim_id in sorted(set(a_dims) | set(b_dims)):
        ma = a_dims.get(dim_id, {}).get("mean")
        mb = b_dims.get(dim_id, {}).get("mean")
        delta = (mb or 0) - (ma or 0) if (ma is not None and mb is not None) else None
        dim_table.add_row(
            f"{dim_id} ({a_dims.get(dim_id, b_dims.get(dim_id, {})).get('name', '')})",
            f"{ma:.3f}" if ma is not None else "-",
            f"{mb:.3f}" if mb is not None else "-",
            f"{delta:+.3f}" if delta is not None else "-",
        )
    console.print(dim_table)


if __name__ == "__main__":
    main()

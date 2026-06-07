"""Create human-labelling templates from an evaluation run."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from rich.console import Console


DIMENSIONS = ["gsr", "pcr", "bca", "kar", "hcr", "scr", "ter"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate calibration JSON templates.")
    parser.add_argument("--run", type=Path, required=True, help="Run directory or run_report.json")
    parser.add_argument("--out", type=Path, default=Path("data/calibration"))
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()

    run_path = args.run / "run_report.json" if args.run.is_dir() else args.run
    data = json.loads(run_path.read_text(encoding="utf-8"))
    args.out.mkdir(parents=True, exist_ok=True)
    created = 0
    for case in data.get("cases", [])[: args.limit]:
        trace = case.get("trace", {})
        case_id = trace.get("case_id")
        if not case_id:
            continue
        target = args.out / f"{data.get('run_id', run_path.parent.name)}__{case_id}.json"
        if target.exists():
            continue
        payload = {
            "case_id": case_id,
            "trace_path": f"traces/{trace.get('run_id')}/{case_id}.json",
            "labeller": "",
            "labelled_at": datetime.now(tz=timezone.utc).date().isoformat(),
            "human_labels": {
                dim: {"passed": None, "rationale": ""} for dim in DIMENSIONS
            },
            "notes": "请人工阅读trace和data/instructions中的指令后填写passed与rationale。",
        }
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        created += 1
    Console().print(f"Created {created} calibration templates in {args.out}")


if __name__ == "__main__":
    main()

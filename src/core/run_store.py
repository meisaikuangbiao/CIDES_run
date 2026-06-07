from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Optional


SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    config_json TEXT NOT NULL,
    sut_model TEXT,
    judge_model TEXT,
    user_sim_model TEXT
);

CREATE TABLE IF NOT EXISTS cases (
    case_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    instruction_id TEXT NOT NULL,
    scenario_id TEXT NOT NULL,
    trace_path TEXT NOT NULL,
    report_path TEXT,
    weighted_total REAL,
    passed INTEGER,
    created_at TEXT NOT NULL,
    FOREIGN KEY (run_id) REFERENCES runs(run_id)
);

CREATE TABLE IF NOT EXISTS reports (
    run_id TEXT PRIMARY KEY,
    aggregate_json TEXT NOT NULL,
    html_path TEXT,
    md_path TEXT,
    created_at TEXT NOT NULL
);
"""


class RunStore:
    """SQLite-backed bookkeeping for evaluation runs."""

    def __init__(self, root: str | Path = "reports") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.db_path = self.root / "runs.sqlite3"
        with self._conn() as conn:
            conn.executescript(SCHEMA)
            self._ensure_columns(conn)

    @staticmethod
    def _ensure_columns(conn: sqlite3.Connection) -> None:
        cols = {
            row[1]
            for row in conn.execute("PRAGMA table_info(cases)").fetchall()
        }
        if "passed" not in cols:
            conn.execute("ALTER TABLE cases ADD COLUMN passed INTEGER")

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        try:
            conn.row_factory = sqlite3.Row
            yield conn
            conn.commit()
        finally:
            conn.close()

    @staticmethod
    def _now() -> str:
        return datetime.now(tz=timezone.utc).isoformat()

    def register_run(
        self,
        run_id: str,
        config: dict[str, Any],
        sut_model: Optional[str] = None,
        judge_model: Optional[str] = None,
        user_sim_model: Optional[str] = None,
    ) -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO runs(run_id, created_at, config_json, sut_model, judge_model, user_sim_model) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    run_id,
                    self._now(),
                    json.dumps(config, ensure_ascii=False),
                    sut_model,
                    judge_model,
                    user_sim_model,
                ),
            )

    def add_case(
        self,
        case_id: str,
        run_id: str,
        instruction_id: str,
        scenario_id: str,
        trace_path: str,
        weighted_total: Optional[float] = None,
        passed: Optional[bool] = None,
        report_path: Optional[str] = None,
    ) -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO cases(case_id, run_id, instruction_id, scenario_id, trace_path, report_path, weighted_total, passed, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    case_id,
                    run_id,
                    instruction_id,
                    scenario_id,
                    trace_path,
                    report_path,
                    weighted_total,
                    int(passed) if passed is not None else None,
                    self._now(),
                ),
            )

    def save_report(
        self,
        run_id: str,
        aggregate: dict[str, Any],
        html_path: Optional[str] = None,
        md_path: Optional[str] = None,
    ) -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO reports(run_id, aggregate_json, html_path, md_path, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    run_id,
                    json.dumps(aggregate, ensure_ascii=False),
                    html_path,
                    md_path,
                    self._now(),
                ),
            )

    def list_runs(self) -> list[dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT run_id, created_at, sut_model, judge_model, user_sim_model FROM runs ORDER BY created_at DESC"
            ).fetchall()
            return [dict(r) for r in rows]

    def list_cases(self, run_id: str) -> list[dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM cases WHERE run_id = ? ORDER BY created_at",
                (run_id,),
            ).fetchall()
            return [dict(r) for r in rows]

    def bootstrap_from_disk(self, project_root: str | Path) -> int:
        """当 sqlite 为空时，从 reports/*/run_report.json 重建索引（便于云端部署）。"""
        if self.list_runs():
            return 0
        root = Path(project_root)
        imported = 0
        for report_file in sorted(self.root.glob("*/run_report.json")):
            run_id = report_file.parent.name
            try:
                data = json.loads(report_file.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            config = data.get("config") or {}
            models = config.get("models") or {}
            self.register_run(
                run_id=run_id,
                config=config,
                sut_model=models.get("sut"),
                judge_model=models.get("judge"),
                user_sim_model=models.get("user_sim"),
            )
            agg = data.get("aggregate") or {}
            html = report_file.parent / "report.html"
            md = report_file.parent / "report.md"
            self.save_report(
                run_id,
                agg,
                str(html) if html.exists() else None,
                str(md) if md.exists() else None,
            )
            for case in data.get("cases") or []:
                trace = case.get("trace") or {}
                case_id = trace.get("case_id")
                if not case_id:
                    continue
                trace_file = root / "traces" / run_id / f"{case_id}.json"
                if not trace_file.exists():
                    continue
                trace_path = trace_file.relative_to(root).as_posix()
                self.add_case(
                    case_id=case_id,
                    run_id=run_id,
                    instruction_id=trace.get("instruction_id") or "",
                    scenario_id=trace.get("scenario_id") or "",
                    trace_path=trace_path,
                    weighted_total=case.get("weighted_total"),
                    passed=case.get("passed"),
                )
            imported += 1
        return imported

    def get_report_meta(self, run_id: str) -> Optional[dict[str, Any]]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM reports WHERE run_id = ?", (run_id,)
            ).fetchone()
            return dict(row) if row else None

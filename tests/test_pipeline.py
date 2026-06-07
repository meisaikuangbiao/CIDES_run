from pathlib import Path
import json
import subprocess
import sys

from src.core.instruction_parser import load_specs
from src.core.orchestrator import run_dialogue
from src.core.scenario_matrix import build_scenarios
from src.core.sut_client import build_sut_client
from src.core.user_simulator import build_user_simulator
from src.evaluators.aggregator import aggregate_run, evaluate_case
from src.report.renderer import write_run_artifacts


def test_stub_pipeline_runs_end_to_end(tmp_path: Path) -> None:
    specs = load_specs("data/instructions")
    assert specs, "Expecting parsed instruction fixtures from previous step"
    spec = specs[0]
    scenarios, vmap = build_scenarios(spec, seed=42)
    chosen = scenarios[:2]
    sut = build_sut_client(use_stub=True, seed=42)
    user_sim = build_user_simulator(use_stub=True, seed=42)
    cases = []
    for sc in chosen:
        trace = run_dialogue(
            spec,
            sc,
            sut,
            user_sim,
            variables=vmap[sc.id],
            run_id="pytest",
            max_turns=8,
            trace_dir=tmp_path / "traces",
        )
        cases.append(evaluate_case(spec, sc, trace, client=None))

    report = aggregate_run("pytest", cases, config={"models": {"sut": "stub-sut"}})
    paths = write_run_artifacts(report, tmp_path / "out")
    assert paths["html"].exists()
    assert paths["md"].exists()
    assert paths["json"].exists()
    assert 0.0 <= report.aggregate["overall_mean"] <= 1.0
    assert report.aggregate["n_cases"] == len(chosen)
    assert "case_pass_rate" in report.aggregate
    assert hasattr(report.cases[0], "passed")


def test_cli_evaluates_uploaded_real_dialogues(tmp_path: Path) -> None:
    upload = [
        {
            "id": "001",
            "任务编号": 1,
            "多轮对话": [
                {"index": 0, "role": "assistant", "content": "你好，请问是陈师傅吗？我是站长。"},
                {"index": 1, "role": "user", "content": "我是陈师傅，你说。"},
            ],
        }
    ]
    upload_path = tmp_path / "real_dialogues.json"
    out_dir = tmp_path / "reports" / "real_pytest"
    upload_path.write_text(json.dumps(upload, ensure_ascii=False), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.cli.eval_run",
            "--real-dialogues",
            str(upload_path),
            "--run-id",
            "real_pytest",
            "--out",
            str(out_dir),
            "--judge",
            "offline",
            "--workers",
            "1",
        ],
        check=False,
        cwd=Path.cwd(),
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    report_data = json.loads((out_dir / "run_report.json").read_text(encoding="utf-8"))
    assert report_data["config"]["run_mode"] == "real_dialogue"
    assert report_data["cases"][0]["trace"]["case_id"] == "001"
    assert report_data["cases"][0]["trace"]["scenario_id"] == "real"
    assert Path("traces/real_pytest/001.json").exists()

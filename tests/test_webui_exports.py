import json

from src.core.schemas import InstructionSpec
from src.core.real_dialogue_importer import load_real_dialogues
from src.webui.app import (
    EVAL_ENV_GENERATED,
    EVAL_ENV_REAL,
    _build_raw_dialogue_export,
    _compact_report_html,
    _filter_runs_for_env,
    _format_case_display,
    _latest_log_excerpt,
    _nav_keys_for_env,
    _page_for_env,
    _page_map_for_env,
    _should_apply_query_env,
    _should_apply_query_page,
    _summarize_active_run,
    _trace_selector_mode_for_env,
    _is_real_dialogue_run,
)


def test_build_raw_dialogue_export_keeps_only_id_task_number_and_turns() -> None:
    report_data = {
        "cases": [
            {
                "trace": {
                    "case_id": "1__cooperative",
                    "instruction_id": "1",
                    "scenario_id": "cooperative",
                    "turns": [
                        {"index": 0, "role": "assistant", "content": "您好"},
                        {"index": 1, "role": "user", "content": "你好"},
                    ],
                },
                "weighted_total": 0.9,
            },
            {
                "trace": {
                    "case_id": "2__interrupt",
                    "instruction_id": "1",
                    "scenario_id": "interrupt",
                    "turns": [
                        {"index": 0, "role": "assistant", "content": "请问方便吗？"},
                    ],
                },
            },
        ],
    }

    exported = _build_raw_dialogue_export(report_data)

    assert exported == [
        {
            "id": "001",
            "任务编号": 1,
            "多轮对话": [
                {"index": 0, "role": "assistant", "content": "您好"},
                {"index": 1, "role": "user", "content": "你好"},
            ],
        },
        {
            "id": "002",
            "任务编号": 1,
            "多轮对话": [
                {"index": 0, "role": "assistant", "content": "请问方便吗？"},
            ],
        },
    ]


def test_build_raw_dialogue_export_numbers_only_exported_dialogues() -> None:
    report_data = {
        "cases": [
            {"trace": {"turns": []}},
            {
                "trace": {
                    "instruction_id": "3",
                    "turns": [
                        {"index": 0, "role": "assistant", "content": "第一条有效对话"},
                    ],
                },
            },
        ],
    }

    exported = _build_raw_dialogue_export(report_data)

    assert exported == [
        {
            "id": "001",
            "任务编号": 3,
            "多轮对话": [
                {"index": 0, "role": "assistant", "content": "第一条有效对话"},
            ],
        },
    ]


def test_build_raw_dialogue_export_strips_turn_runtime_metadata() -> None:
    report_data = {
        "cases": [
            {
                "trace": {
                    "turns": [
                        {
                            "index": 0,
                            "role": "assistant",
                            "content": "您好，请问方便吗？",
                            "chars": 27,
                            "latency_ms": 4697,
                            "tokens_in": 920,
                            "tokens_out": 234,
                            "model": "deepseek-v4-flash",
                            "temperature": None,
                            "error": None,
                        },
                    ],
                },
            },
        ],
    }

    exported = _build_raw_dialogue_export(report_data)

    assert exported == [
        {
            "id": "001",
            "任务编号": 1,
            "多轮对话": [
                {"index": 0, "role": "assistant", "content": "您好，请问方便吗？"},
            ],
        },
    ]


def test_compact_report_html_overrides_old_turn_whitespace() -> None:
    html = """
    <html><head><style>
    .turn .body { flex: 1; white-space: pre-wrap; }
    </style></head><body></body></html>
    """

    compacted = _compact_report_html(html)

    assert ".turn .body { flex: 1; white-space: pre-wrap; }" not in compacted
    assert ".turn .body { white-space: normal !important;" in compacted
    assert ".turn .turn-text { white-space: pre-wrap !important;" in compacted


def test_real_dialogue_run_helpers_detect_and_label_cases() -> None:
    assert _is_real_dialogue_run({"config": {"run_mode": "real_dialogue"}}) is True
    assert _is_real_dialogue_run({"config": {"run_mode": "full"}}) is False
    assert _format_case_display("001", "1", real_mode=True) == "001 · 任务1"
    assert _format_case_display("1__hesitant", "1", real_mode=False) == "任务1 · 犹豫型"


def test_raw_dialogue_export_can_be_imported_for_real_dialogue_eval(tmp_path) -> None:
    report_data = {
        "cases": [
            {
                "trace": {
                    "case_id": "1__cooperative",
                    "instruction_id": "1",
                    "turns": [
                        {"index": 0, "role": "assistant", "content": "您好"},
                        {"index": 1, "role": "user", "content": "你好"},
                    ],
                },
            },
            {
                "trace": {
                    "case_id": "3__interrupt",
                    "instruction_id": "3",
                    "turns": [
                        {"index": 0, "role": "assistant", "content": "请问方便吗？"},
                    ],
                },
            },
        ],
    }
    export_path = tmp_path / "raw_dialogues.json"
    export_path.write_text(
        json.dumps(_build_raw_dialogue_export(report_data), ensure_ascii=False),
        encoding="utf-8",
    )

    traces = load_real_dialogues(
        export_path,
        {"1": InstructionSpec(id="1", role="r", task="t"), "3": InstructionSpec(id="3", role="r", task="t")},
        run_id="real",
    )

    assert [trace.instruction_id for trace in traces] == ["1", "3"]


def test_navigation_is_split_by_evaluation_environment() -> None:
    generated_keys = _nav_keys_for_env(EVAL_ENV_GENERATED)
    real_keys = _nav_keys_for_env(EVAL_ENV_REAL)

    assert "发起评测" in generated_keys
    assert "对话复盘" in generated_keys
    assert "上传真实对话" not in generated_keys
    assert "真实对话复盘" not in generated_keys

    assert "上传真实对话" in real_keys
    assert "真实对话复盘" in real_keys
    assert "真实评测报告" in real_keys
    assert "对话复盘" not in real_keys
    assert "发起评测" not in real_keys

    generated_pages = _page_map_for_env(EVAL_ENV_GENERATED)
    real_pages = _page_map_for_env(EVAL_ENV_REAL)
    assert "对话复盘" in generated_pages
    assert "真实对话复盘" in real_pages
    assert _trace_selector_mode_for_env(EVAL_ENV_GENERATED) == "scenario"
    assert _trace_selector_mode_for_env(EVAL_ENV_REAL) == "dialogue_id"


def test_real_environment_dashboard_links_map_to_real_pages() -> None:
    assert _page_for_env("发起评测", EVAL_ENV_REAL) == "上传真实对话"
    assert _page_for_env("评测报告", EVAL_ENV_REAL) == "真实评测报告"
    assert _page_for_env("对话复盘", EVAL_ENV_REAL) == "真实对话复盘"
    assert _page_for_env("任务配置", EVAL_ENV_REAL) == "任务配置"
    assert _page_for_env("发起评测", EVAL_ENV_GENERATED) == "发起评测"


def test_filter_runs_for_env_separates_generated_and_real_runs() -> None:
    runs = [{"run_id": "gen"}, {"run_id": "real"}]
    reports = {
        "gen": {"config": {"run_mode": "full"}},
        "real": {"config": {"run_mode": "real_dialogue"}},
    }

    assert [r["run_id"] for r in _filter_runs_for_env(runs, EVAL_ENV_GENERATED, reports.get)] == ["gen"]
    assert [r["run_id"] for r in _filter_runs_for_env(runs, EVAL_ENV_REAL, reports.get)] == ["real"]


def test_query_page_does_not_override_user_sidebar_navigation() -> None:
    assert _should_apply_query_page(
        query_page="真实工作台",
        current_page="上传真实对话",
        last_synced_page="真实工作台",
    ) is False
    assert _should_apply_query_page(
        query_page="真实对话复盘",
        current_page="上传真实对话",
        last_synced_page="真实工作台",
    ) is True
    assert _should_apply_query_page(
        query_page="真实评测报告",
        current_page=None,
        last_synced_page=None,
    ) is True


def test_query_env_does_not_override_user_environment_switch() -> None:
    assert _should_apply_query_env(
        query_env=EVAL_ENV_GENERATED,
        current_env=EVAL_ENV_REAL,
        last_synced_env=EVAL_ENV_GENERATED,
    ) is False
    assert _should_apply_query_env(
        query_env=EVAL_ENV_REAL,
        current_env=EVAL_ENV_GENERATED,
        last_synced_env=EVAL_ENV_GENERATED,
    ) is True
    assert _should_apply_query_env(
        query_env=EVAL_ENV_REAL,
        current_env=None,
        last_synced_env=None,
    ) is True


def test_summarize_active_run_explains_running_without_progress_lines(tmp_path) -> None:
    log_path = tmp_path / "webui_eval.log"
    log_path.write_text(
        "\n".join([
            "Total cases to evaluate: 22 (workers=1, judge_workers=1, samples=3)",
            'HTTP Request: POST https://api.deepseek.com/chat/completions "HTTP/1.1 200 OK"',
            'HTTP Request: POST https://api.deepseek.com/chat/completions "HTTP/1.1 200 OK"',
        ]),
        encoding="utf-8",
    )

    summary = _summarize_active_run(
        run_id="real_x",
        alive=True,
        log_path=log_path,
        report_exists=False,
    )

    assert summary.done == 0
    assert summary.total == 22
    assert summary.http_ok_count == 2
    assert summary.label == "运行中 · 模型响应正常，正在评分中 · 0/22 项"
    assert "模型接口已有 2 次成功响应" in summary.detail


def test_summarize_active_run_counts_model_calls_cumulatively(tmp_path) -> None:
    log_path = tmp_path / "webui_eval.log"
    lines = ["Total cases to evaluate: 22 (workers=1)"]
    for idx in range(20):
        lines.append(f"2026-06-06 23:50:{idx:02d},000 DEBUG openai._base_client | Sending HTTP Request: POST https://api.deepseek.com/chat/completions")
        lines.append(f'2026-06-06 23:50:{idx:02d},100 INFO httpx | HTTP Request: POST https://api.deepseek.com/chat/completions "HTTP/1.1 200 OK"')
    log_path.write_text("\n".join(lines), encoding="utf-8")

    summary = _summarize_active_run(
        run_id="real_x",
        alive=True,
        log_path=log_path,
        report_exists=False,
    )

    assert "已发起 20 次模型请求" in summary.detail
    assert "模型接口已有 20 次成功响应" in summary.detail


def test_latest_log_excerpt_keeps_recent_signal_and_skips_long_prompt_noise(tmp_path) -> None:
    log_path = tmp_path / "webui_eval.log"
    log_path.write_text(
        "\n".join([
            "Run ID: real_x",
            "Total cases to evaluate: 22 (workers=1)",
            "2026-06-06 23:41:35,613 DEBUG openai._base_client | Request options: {'json_data': '"
            + ("很长的prompt" * 80)
            + "'}",
            "2026-06-06 23:41:35,862 INFO httpx | HTTP Request: POST https://api.deepseek.com/chat/completions \"HTTP/1.1 200 OK\"",
            "2026-06-06 23:42:04,618 DEBUG httpcore.http11 | receive_response_body.complete",
            "2026-06-06 23:42:05,190 ERROR judge | retrying transient failure",
        ]),
        encoding="utf-8",
    )

    excerpt = _latest_log_excerpt(log_path, max_lines=4)

    assert "Total cases to evaluate: 22" in excerpt
    assert "HTTP/1.1 200 OK" in excerpt
    assert "receive_response_body.complete" in excerpt
    assert "retrying transient failure" in excerpt
    assert "很长的prompt" not in excerpt

"""Streamlit Web UI for the dialogue evaluation system.

Pages:
- Instructions: upload Excel, view parsed JSON.
- Scenarios & Personas: review the scenario matrix per instruction.
- Evaluate: launch a fresh run with stub or LLM models.
- Reports: browse historical runs and view their HTML/Markdown.
- Trace replay: drill into a single case turn-by-turn.
- A/B compare: pick two runs and diff dimension means.

Run with::

    streamlit run src/webui/app.py
"""
from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


def _hydrate_env_from_streamlit_secrets() -> None:
    """将 st.secrets 中的关键配置同步到 os.environ。

    本地通过 .env 注入环境变量；部署到 Streamlit Community Cloud 时只能
    通过 st.secrets 配置，这里把它们映射回进程环境，让 os.getenv 调用
    （如 src/core/llm_client.py）保持不变即可工作。
    """
    try:
        secrets = st.secrets
    except Exception:
        return
    keys = (
        "DEEPSEEK_API_KEY",
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "DEEPSEEK_BASE_URL",
        "SUT_MODEL",
        "USER_SIM_MODEL",
        "JUDGE_MODEL",
        "LLM_REQUEST_TIMEOUT",
        "LLM_MAX_RETRIES",
        "LLM_HTTP_LOG_LEVEL",
        "DEEPSEEK_REASONING_EFFORT",
    )
    for key in keys:
        if os.environ.get(key):
            continue
        try:
            value = secrets[key]  # type: ignore[index]
        except Exception:
            continue
        if value in (None, ""):
            continue
        os.environ[key] = str(value)


_hydrate_env_from_streamlit_secrets()

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.config import load_config  # noqa: E402
from src.core.instruction_parser import load_specs, parse_workbook  # noqa: E402
from src.core.llm_client import build_default_client  # noqa: E402
from src.core.case_validator import validate_case_matrix  # noqa: E402
from src.core.orchestrator import (  # noqa: E402
    load_trace,
    run_dialogue,
    summarize_trace_for_memory,
)
from src.core.real_dialogue_importer import RealDialogueImportError, load_real_dialogues  # noqa: E402
from src.core.run_store import RunStore  # noqa: E402
from src.core.scenario_matrix import build_scenarios, coverage_matrix  # noqa: E402
from src.core.sut_client import build_sut_client  # noqa: E402
from src.core.user_simulator import build_user_simulator  # noqa: E402
from src.evaluators.aggregator import (  # noqa: E402
    DIMENSION_NAMES,
    aggregate_run,
    evaluate_case,
)
from src.report.renderer import render_html, write_run_artifacts  # noqa: E402


REPORTS_ROOT = ROOT / "reports"
TRACES_ROOT = ROOT / "traces"
INSTRUCTIONS_DIR = ROOT / "data" / "instructions"
SOURCE_XLSX_DEFAULT = ROOT / "data" / "source" / "命题二：外呼任务对话模型指令示例.xlsx"


# ----- 产品设计：数据看板风格（Fira Sans + 蓝琥珀色）-------------------------
COLOR = {
    "primary": "#1E40AF",
    "primary_soft": "#EFF6FF",
    "accent": "#F59E0B",
    "good": "#059669",
    "good_soft": "#ECFDF5",
    "warn": "#D97706",
    "warn_soft": "#FFFBEB",
    "bad": "#DC2626",
    "bad_soft": "#FEF2F2",
    "neutral": "#64748B",
    "muted": "#94A3B8",
    "bg": "#F8FAFC",
    "border": "#E2E8F0",
    "text": "#1E3A8A",
    "surface": "#FFFFFF",
    "sidebar": "#0F172A",
}

EVAL_ENV_GENERATED = "generated"
EVAL_ENV_REAL = "real"

NAV_PAGES: list[tuple[str, str]] = [
    ("工作台", "首页 · 看进展和下一步"),
    ("发起评测", "第1步 · 启动自动评测"),
    ("评测报告", "第2步 · 看分数和问题"),
    ("对话复盘", "第3步 · 查具体哪里错了"),
    ("任务配置", "可选 · 管理外呼任务"),
    ("版本对比", "可选 · 对比模型升级效果"),
]
NAV_KEYS = [n[0] for n in NAV_PAGES]

REAL_NAV_PAGES: list[tuple[str, str]] = [
    ("真实工作台", "首页 · 查看真实对话评测"),
    ("上传真实对话", "第1步 · 上传对话并评分"),
    ("真实评测报告", "第2步 · 看真实对话得分"),
    ("真实对话复盘", "第3步 · 按 ID 查具体对话"),
    ("任务配置", "可选 · 管理外呼任务"),
]
REAL_NAV_KEYS = [n[0] for n in REAL_NAV_PAGES]

WORKFLOW_STEPS: list[dict[str, str]] = [
    {
        "num": "1",
        "title": "发起评测",
        "desc": "选好外呼任务和用户场景，点「开始评测」",
        "page": "发起评测",
        "tip": "默认配置即可用，不用改高级设置",
    },
    {
        "num": "2",
        "title": "查看报告",
        "desc": "看综合得分、通过率，找到没通过的测试项",
        "page": "评测报告",
        "tip": "绿色徽章=全部通过，黄色/红色=有问题",
    },
    {
        "num": "3",
        "title": "对话复盘",
        "desc": "逐轮看用户说什么、模型怎么回、哪里被扣分",
        "page": "对话复盘",
        "tip": "重点看未通过项，定位模型短板",
    },
]

PAGE_GUIDES: dict[str, str] = {
    "工作台": "按下方三步走即可完成一次完整评测",
    "发起评测": "选好范围后点「开始评测」，等进度跑完再去「评测报告」",
    "评测报告": "先看顶部得分徽章，再展开「测试项明细」找未通过项",
    "对话复盘": "外呼任务显示为「任务N-角色」，再选用户类型查看通话与评分",
    "任务配置": "系统已内置任务，一般不用改；只有导入新 Excel 时才需要",
    "版本对比": "模型升级后，选「基准」和「对比」两次评测，看分数变化",
    "真实工作台": "上传真实对话后，查看真实对话评测结果和复盘",
    "上传真实对话": "上传 JSON 文件，系统按任务编号匹配任务配置并评分",
    "真实评测报告": "仅展示真实对话批次，按对话 ID 查看结果",
    "真实对话复盘": "按对话 ID 查看真实对话，不再选择用户类型",
}
_NAV_ALIAS = {
    "总览 Dashboard": "工作台",
    "评测运行": "发起评测",
    "报告中心": "评测报告",
    "Trace 复盘": "对话复盘",
    "场景与画像": "任务配置",
    "指令管理": "任务配置",
    "A/B 对比": "版本对比",
    "首页": "工作台",
    "评测": "发起评测",
    "结果": "评测报告",
    "真实结果": "真实评测报告",
    "真实复盘": "真实对话复盘",
}

SCENARIO_DISPLAY_NAMES: dict[str, str] = {
    "cooperative": "配合型",
    "hesitant": "犹豫型",
    "resistant": "抗拒型",
    "faq_drill": "追问细节",
    "out_of_scope": "越权提问",
    "interrupt": "频繁打断",
    "busy_driving": "忙碌/开车",
    "lure_violation": "诱导违规",
}

TERMINATION_LABELS: dict[str, str] = {
    "user_end_call": "用户结束通话",
    "sut_goodbye": "模型主动告别",
    "sut_error": "模型响应异常",
    "sut_invalid": "模型空回复过多",
    "user_sim_error": "用户模拟异常",
    "max_turns": "超过最大轮次",
}


def inject_global_css() -> None:
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+SC:wght@400;500;600;700&display=swap');
        :root {{
          --primary: {COLOR['primary']}; --accent: {COLOR['accent']};
          --good: {COLOR['good']}; --warn: {COLOR['warn']}; --bad: {COLOR['bad']};
          --muted: {COLOR['muted']}; --border: {COLOR['border']};
          --text: {COLOR['text']}; --bg: {COLOR['bg']}; --surface: {COLOR['surface']};
        }}
        #MainMenu, footer {{ visibility: hidden; }}
        /* 保留顶部栏，否则侧边栏折叠后无法再次展开 */
        [data-testid="stSidebarCollapsedControl"],
        [data-testid="collapsedControl"] {{
          visibility: visible !important;
        }}
        html, body, [class*="css"] {{ font-family: 'Noto Sans SC', system-ui, sans-serif; }}
        .stApp {{ background: #f1f5f9; }}
        section.main > div.block-container {{
          padding-top: 1.25rem; padding-bottom: 3rem; max-width: 960px;
        }}
        [data-testid="stSidebar"] {{
          background: {COLOR['sidebar']}; border-right: none;
        }}
        [data-testid="stSidebar"] > div:first-child {{
          background: {COLOR['sidebar']};
        }}
        [data-testid="stSidebar"] * {{ color: #e2e8f0 !important; }}
        [data-testid="stSidebar"] .ev-side-brand {{ color: white !important; }}
        [data-testid="stSidebar"] [data-testid="stRadio"] > div {{
          gap: 4px;
        }}
        [data-testid="stSidebar"] [data-testid="stRadio"] label {{
          padding: 10px 14px; border-radius: 10px; margin: 0;
          border: 1px solid transparent; transition: all .15s ease;
        }}
        [data-testid="stSidebar"] [data-testid="stRadio"] label:hover {{
          background: rgba(255,255,255,.07);
          border-color: rgba(255,255,255,.1);
        }}
        [data-testid="stSidebar"] [data-testid="stRadio"] label[data-checked="true"] {{
          background: rgba(30,64,175,.45) !important;
          border-color: rgba(96,165,250,.35) !important;
        }}
        h1, h2, h3 {{ letter-spacing: -0.02em; }}
        h1 {{ font-size: 1.4rem !important; color: var(--text) !important; }}
        h2 {{ font-size: 1rem !important; color: #334155 !important; }}
        .ev-side-brand {{
          padding: 8px 0 18px; border-bottom: 1px solid rgba(255,255,255,.1);
          margin-bottom: 16px;
        }}
        .ev-side-brand .t {{ font-size: 1.1rem; font-weight: 700; color: #fff; }}
        .ev-side-brand .s {{ color: #94a3b8; font-size: .76rem; margin-top: 4px; }}
        .ev-page-head {{
          margin-bottom: 24px; padding-bottom: 16px;
          border-bottom: 1px solid var(--border);
        }}
        .ev-page-head .title {{ font-size: 1.5rem; font-weight: 700; color: var(--text); }}
        .ev-page-head .sub {{ color: var(--muted); font-size: .9rem; margin-top: 6px; }}
        .ev-panel {{
          background: var(--surface); border: 1px solid var(--border);
          border-radius: 16px; padding: 20px 22px; margin-bottom: 16px;
          box-shadow: 0 4px 24px rgba(15,23,42,.04);
        }}
        .ev-panel-title {{
          font-size: .95rem; font-weight: 600; color: #334155; margin-bottom: 14px;
        }}
        div[data-testid="stHorizontalBlock"]:has(.ev-kpi) {{
          align-items: stretch !important;
          margin-bottom: 6px !important;
        }}
        div[data-testid="column"]:has(.ev-kpi) {{
          display: flex !important;
          flex-direction: column !important;
        }}
        div[data-testid="column"]:has(.ev-kpi) [data-testid="stMarkdown"] {{
          flex: 1 1 auto !important;
          height: auto !important;
          display: flex !important;
        }}
        div[data-testid="column"]:has(.ev-kpi) [data-testid="stMarkdown"] > div {{
          flex: 1 1 auto !important;
          height: auto !important;
          display: flex !important;
        }}
        div[data-testid="stExpanderDetails"]:has(.ev-kpi) {{
          padding-bottom: 28px !important;
        }}
        .ev-kpi {{
          background: var(--surface); border: 1px solid var(--border);
          border-radius: 14px; padding: 18px 20px;
          min-height: 112px; box-sizing: border-box;
          box-shadow: 0 2px 8px rgba(15,23,42,.03);
          display: flex; flex-direction: column; flex: 1 1 auto;
          width: 100%;
        }}
        .ev-kpi .label {{ color: var(--muted); font-size: .76rem; margin-bottom: 8px; text-transform: uppercase; letter-spacing: .04em; }}
        .ev-kpi .value {{ font-size: 1.65rem; font-weight: 700; color: var(--text); line-height: 1.2; }}
        .ev-kpi .delta {{
          color: var(--muted); font-size: .8rem; margin-top: auto;
          padding-top: 8px; min-height: 1.35em; line-height: 1.35;
        }}
        .ev-kpi .delta:empty::before {{
          content: "\\00a0";
        }}
        .ev-kpi.good {{ border-top: 3px solid var(--good); }}
        .ev-kpi.warn {{ border-top: 3px solid var(--warn); }}
        .ev-kpi.bad {{ border-top: 3px solid var(--bad); }}
        .ev-kpi.primary {{ border-top: 3px solid var(--primary); }}
        .ev-card {{
          background: var(--surface); border: 1px solid var(--border);
          border-radius: 12px; padding: 14px 16px; margin-bottom: 8px;
        }}
        .ev-pill {{
          display: inline-block; padding: 4px 10px; border-radius: 999px;
          font-size: .72rem; font-weight: 600;
          background: var(--primary-soft); color: var(--primary);
        }}
        .ev-pill.good {{ background: {COLOR['good_soft']}; color: var(--good); }}
        .ev-pill.warn {{ background: {COLOR['warn_soft']}; color: var(--warn); }}
        .ev-pill.bad {{ background: {COLOR['bad_soft']}; color: var(--bad); }}
        .ev-summary {{
          background: linear-gradient(135deg, #eff6ff, #fff);
          border: 1px solid #dbeafe; border-radius: 14px;
          padding: 16px 20px; margin-bottom: 16px;
        }}
        .ev-summary .line1 {{ font-weight: 600; color: #1e3a8a; font-size: 1rem; }}
        .ev-summary .line2 {{ color: #64748b; font-size: .86rem; margin-top: 4px; }}
        div[data-testid="stMetric"] {{
          background: #f8fafc; border: 1px solid var(--border);
          border-radius: 10px; padding: 10px 14px;
        }}
        /* 全局折叠区统一尺寸（与对话复盘三大折叠区一致） */
        div[data-testid="stExpander"] {{
          margin-bottom: 12px !important;
          border: none !important;
          background: transparent !important;
          box-shadow: none !important;
        }}
        div[data-testid="stExpander"] details {{
          border: 1px solid #e2e8f0 !important;
          border-radius: 14px !important;
          background: #fff !important;
          box-shadow: 0 2px 12px rgba(15,23,42,.05) !important;
          overflow: hidden;
          min-height: 64px;
          transition: border-color .18s ease, box-shadow .18s ease;
        }}
        div[data-testid="stExpander"] details:hover {{
          border-color: #bfdbfe !important;
          box-shadow: 0 4px 18px rgba(37,99,235,.10) !important;
        }}
        div[data-testid="stExpander"] details[open] {{
          border-color: #93c5fd !important;
          box-shadow: 0 6px 22px rgba(37,99,235,.12) !important;
        }}
        div[data-testid="stExpander"] summary {{
          padding: 16px 22px !important;
          min-height: 64px !important;
          font-weight: 700 !important;
          font-size: 1rem !important;
          color: #1e293b !important;
          background: linear-gradient(90deg, #f8fafc 0%, #fff 100%) !important;
          line-height: 1.5 !important;
          display: flex !important;
          align-items: center !important;
        }}
        div[data-testid="stExpander"] summary p,
        div[data-testid="stExpander"] summary span {{
          font-size: 1rem !important;
          font-weight: 700 !important;
          margin: 0 !important;
        }}
        div[data-testid="stExpander"] details[open] summary {{
          border-bottom: 1px solid #e2e8f0;
          background: linear-gradient(90deg, #eff6ff 0%, #fff 100%) !important;
        }}
        div[data-testid="stExpander"] [data-testid="stExpanderDetails"] {{
          padding: 12px 18px 18px !important;
        }}
        .stTabs [data-baseweb="tab-list"] {{
          gap: 6px; background: transparent;
        }}
        .stTabs [data-baseweb="tab"] {{
          border-radius: 8px; padding: 8px 16px;
          background: #f1f5f9; border: 1px solid transparent;
        }}
        .stTabs [aria-selected="true"] {{
          background: #fff !important; border-color: var(--border) !important;
        }}
        .ev-usj-row {{
          display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 12px; margin: 10px 0;
        }}
        .ev-usj-col {{
          background: var(--surface); border: 1px solid var(--border);
          border-radius: 10px; padding: 12px 14px;
        }}
        .ev-usj-col h4 {{ margin: 0 0 6px; font-size: .75rem; color: var(--muted); }}
        .ev-usj-col.user {{ border-top: 3px solid var(--accent); }}
        .ev-usj-col.sut {{ border-top: 3px solid var(--primary); }}
        .ev-usj-col.judge {{ border-top: 3px solid var(--bad); }}
        .ev-usj-col.judge.ok {{ border-top-color: var(--good); }}
        .ev-hero {{
          background: linear-gradient(135deg, #1e40af 0%, #3b82f6 55%, #60a5fa 100%);
          border-radius: 18px; padding: 28px 32px; margin-bottom: 20px;
          color: #fff; box-shadow: 0 8px 32px rgba(30,64,175,.18);
        }}
        .ev-hero .h1 {{ font-size: 1.55rem; font-weight: 700; margin: 0 0 8px; color: #fff !important; }}
        .ev-hero .h2 {{ font-size: .92rem; opacity: .92; line-height: 1.6; color: #e0e7ff !important; }}
        .ev-guide {{
          background: #fffbeb; border: 1px solid #fde68a; border-radius: 12px;
          padding: 12px 16px; margin-bottom: 18px;
          display: flex; gap: 10px; align-items: flex-start;
        }}
        .ev-guide .icon {{ font-size: 1.1rem; flex-shrink: 0; }}
        .ev-guide .text {{ color: #92400e; font-size: .88rem; line-height: 1.55; }}
        .ev-workflow {{
          display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px;
          margin-bottom: 20px;
        }}
        @media (max-width: 768px) {{
          .ev-workflow {{ grid-template-columns: 1fr; }}
        }}
        .ev-step {{
          background: #fff; border: 1px solid var(--border); border-radius: 14px;
          padding: 18px 16px; position: relative;
          box-shadow: 0 2px 8px rgba(15,23,42,.04);
        }}
        .ev-step .num {{
          width: 28px; height: 28px; border-radius: 50%;
          background: var(--primary); color: #fff;
          font-size: .82rem; font-weight: 700;
          display: flex; align-items: center; justify-content: center;
          margin-bottom: 10px;
        }}
        .ev-step .title {{ font-weight: 700; color: #1e293b; font-size: .95rem; margin-bottom: 6px; }}
        .ev-step .desc {{ color: #64748b; font-size: .82rem; line-height: 1.5; }}
        .ev-step .tip {{ color: #94a3b8; font-size: .76rem; margin-top: 8px; font-style: italic; }}
        .ev-cta {{
          background: linear-gradient(135deg, #fff7ed, #fff);
          border: 2px solid #fdba74; border-radius: 14px;
          padding: 18px 22px; margin-bottom: 20px;
        }}
        .ev-cta .label {{ color: #9a3412; font-size: .78rem; font-weight: 600; margin-bottom: 4px; }}
        .ev-cta .action {{ color: #1e293b; font-size: 1.05rem; font-weight: 700; }}
        .ev-cta .hint {{ color: #64748b; font-size: .84rem; margin-top: 6px; }}
        .ev-side-step {{
          font-size: .72rem; color: #64748b !important; margin: -2px 0 8px 14px;
        }}
        .ev-side-hint {{
          background: rgba(255,255,255,.06); border-radius: 10px;
          padding: 12px 14px; margin-top: 16px; font-size: .78rem;
          line-height: 1.55; color: #94a3b8 !important;
        }}
        .ev-side-hint b {{ color: #e2e8f0 !important; }}
        .stButton > button[kind="primary"] {{
          background: linear-gradient(135deg, #1e40af, #2563eb) !important;
          border: none !important; font-weight: 600 !important;
          border-radius: 10px !important; padding: .55rem 1.2rem !important;
        }}
        .stButton > button[kind="secondary"] {{
          border-radius: 10px !important;
        }}
        .stDownloadButton > button {{
          border-radius: 11px !important;
          font-weight: 700 !important;
          min-height: 42px !important;
          border: 1px solid #cbd5e1 !important;
          box-shadow: 0 2px 8px rgba(15,23,42,.04) !important;
          transition: transform .16s ease, box-shadow .16s ease, border-color .16s ease;
        }}
        .stDownloadButton > button:hover {{
          transform: translateY(-1px);
          border-color: #93c5fd !important;
          box-shadow: 0 8px 18px rgba(37,99,235,.14) !important;
        }}
        .stDownloadButton > button[kind="primary"] {{
          background: linear-gradient(135deg, #1e40af, #2563eb) !important;
          border: none !important;
          color: #fff !important;
          box-shadow: 0 8px 22px rgba(37,99,235,.24) !important;
        }}
        .ev-export-head {{
          background: linear-gradient(135deg, #eff6ff 0%, #fff 58%, #fffbeb 100%);
          border: 1px solid #dbeafe;
          border-radius: 14px;
          padding: 14px 16px;
          margin: 4px 0 14px;
          display: flex;
          justify-content: space-between;
          gap: 12px;
          align-items: flex-start;
        }}
        .ev-export-head .title {{
          font-weight: 800;
          color: #1e3a8a;
          font-size: .98rem;
          margin-bottom: 4px;
        }}
        .ev-export-head .desc {{
          color: #64748b;
          font-size: .82rem;
          line-height: 1.55;
        }}
        .ev-export-head .badge {{
          flex-shrink: 0;
          background: #fff;
          color: #1d4ed8;
          border: 1px solid #bfdbfe;
          border-radius: 999px;
          padding: 5px 10px;
          font-size: .72rem;
          font-weight: 700;
        }}
        [data-testid="stVerticalBlockBorderWrapper"]:has(.ev-export-card) {{
          border-color: #dbeafe !important;
          background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%) !important;
          box-shadow: 0 4px 16px rgba(15,23,42,.06) !important;
          min-height: 150px;
        }}
        .ev-export-card {{
          padding: 4px 4px 2px;
        }}
        .ev-export-card .kind {{
          color: #1e40af;
          font-size: .76rem;
          font-weight: 800;
          letter-spacing: .02em;
          margin-bottom: 4px;
        }}
        .ev-export-card .hint {{
          color: #64748b;
          font-size: .78rem;
          line-height: 1.5;
          min-height: 38px;
          margin-bottom: 10px;
        }}
        .ev-export-card.primary .kind {{
          color: #b45309;
        }}
        [data-testid="stVerticalBlockBorderWrapper"]:has(.ev-export-card.primary) {{
          border-color: #fcd34d !important;
          background: linear-gradient(180deg, #fffbeb 0%, #ffffff 72%) !important;
        }}
        [data-testid="stVerticalBlockBorderWrapper"] {{
          border-radius: 14px !important;
          border-color: #cbd5e1 !important;
          background: #fff !important;
          box-shadow: 0 2px 12px rgba(15,23,42,.06);
          margin-bottom: 14px;
          padding: 4px 2px;
        }}
        [data-testid="stVerticalBlockBorderWrapper"] [data-testid="stSelectbox"] {{
          margin-bottom: 6px;
        }}
        .ev-filter-head {{
          font-size: .95rem; font-weight: 700; color: #1e40af;
          margin: 0 0 10px; padding: 8px 12px;
          background: linear-gradient(90deg, #eff6ff, #fff);
          border-left: 4px solid #2563eb; border-radius: 0 8px 8px 0;
        }}
        .ev-filter-head.task {{
          color: #5b21b6;
          background: linear-gradient(90deg, #f5f3ff, #fff);
          border-left-color: #7c3aed;
        }}
        .ev-filter-select-host + div [data-testid="stSelectbox"] {{
          margin-top: 2px;
        }}
        .ev-filter-select-host.batch + div [data-testid="stSelectbox"] label p,
        [data-testid="stVerticalBlockBorderWrapper"]:has(.ev-filter-head.batch)
          [data-testid="stSelectbox"] label p {{
          font-size: .82rem !important; font-weight: 700 !important;
          color: #1e40af !important;
        }}
        .ev-filter-select-host.task + div [data-testid="stSelectbox"] label p,
        [data-testid="stVerticalBlockBorderWrapper"]:has(.ev-filter-head.task)
          [data-testid="stSelectbox"] label p {{
          font-size: .82rem !important; font-weight: 700 !important;
          color: #6d28d9 !important;
        }}
        .ev-filter-select-host.batch + div [data-testid="stSelectbox"] [data-baseweb="select"] > div,
        .ev-filter-select-host.batch + div [data-testid="stSelectbox"] [role="combobox"],
        [data-testid="stVerticalBlockBorderWrapper"]:has(.ev-filter-head.batch)
          [data-testid="stSelectbox"] [data-baseweb="select"] > div,
        [data-testid="stVerticalBlockBorderWrapper"]:has(.ev-filter-head.batch)
          [data-testid="stSelectbox"] [role="combobox"] {{
          background: linear-gradient(180deg, #eff6ff 0%, #fff 100%) !important;
          border: 2.5px solid #2563eb !important;
          border-radius: 12px !important;
          min-height: 46px !important;
          box-shadow: 0 2px 8px rgba(37,99,235,.12) !important;
          font-weight: 600 !important;
          color: #1e293b !important;
        }}
        .ev-filter-select-host.task + div [data-testid="stSelectbox"] [data-baseweb="select"] > div,
        .ev-filter-select-host.task + div [data-testid="stSelectbox"] [role="combobox"],
        [data-testid="stVerticalBlockBorderWrapper"]:has(.ev-filter-head.task)
          [data-testid="stSelectbox"] [data-baseweb="select"] > div,
        [data-testid="stVerticalBlockBorderWrapper"]:has(.ev-filter-head.task)
          [data-testid="stSelectbox"] [role="combobox"] {{
          background: linear-gradient(180deg, #f5f3ff 0%, #fff 100%) !important;
          border: 2.5px solid #7c3aed !important;
          border-radius: 12px !important;
          min-height: 46px !important;
          box-shadow: 0 2px 8px rgba(124,58,237,.12) !important;
          font-weight: 600 !important;
          color: #1e293b !important;
        }}
        .ev-filter-select-host.batch + div [data-testid="stSelectbox"] [data-baseweb="select"]:hover > div,
        [data-testid="stVerticalBlockBorderWrapper"]:has(.ev-filter-head.batch)
          [data-testid="stSelectbox"] [data-baseweb="select"]:hover > div {{
          border-color: #1d4ed8 !important;
          background: #fff !important;
        }}
        .ev-filter-select-host.task + div [data-testid="stSelectbox"] [data-baseweb="select"]:hover > div,
        [data-testid="stVerticalBlockBorderWrapper"]:has(.ev-filter-head.task)
          [data-testid="stSelectbox"] [data-baseweb="select"]:hover > div {{
          border-color: #6d28d9 !important;
          background: #fff !important;
        }}
        .ev-filter-select-host.batch + div [data-testid="stSelectbox"] [data-baseweb="select"] svg,
        [data-testid="stVerticalBlockBorderWrapper"]:has(.ev-filter-head.batch)
          [data-testid="stSelectbox"] svg {{
          color: #1e40af !important;
          fill: #1e40af !important;
        }}
        .ev-filter-select-host.task + div [data-testid="stSelectbox"] [data-baseweb="select"] svg,
        [data-testid="stVerticalBlockBorderWrapper"]:has(.ev-filter-head.task)
          [data-testid="stSelectbox"] svg {{
          color: #6d28d9 !important;
          fill: #6d28d9 !important;
        }}
        .ev-filter-task-desc {{
          margin-top: 8px; padding: 10px 12px;
          background: linear-gradient(90deg, #faf5ff, #fff);
          border: 1px solid #e9d5ff; border-radius: 10px;
          color: #6b7280; font-size: .82rem; line-height: 1.5;
        }}
        .ev-filter-hint {{
          font-size: .78rem; color: #64748b; margin: -4px 0 10px;
        }}
        .ev-scen-select-note {{
          display: flex; align-items: center; gap: 8px;
          color: #64748b; font-size: .78rem; margin: 2px 0 8px;
        }}
        .ev-scen-select-note .dot {{
          width: 8px; height: 8px; border-radius: 999px;
          background: linear-gradient(135deg, #2563eb, #60a5fa);
          box-shadow: 0 0 0 3px rgba(37,99,235,.10);
        }}
        .ev-scen-table-picker [data-testid="stDataFrame"] {{
          border: 2px solid #2563eb !important;
          border-radius: 12px !important;
          overflow: hidden;
          box-shadow: 0 2px 8px rgba(37,99,235,.10);
        }}
        .ev-trace-status {{
          display: flex; align-items: center; gap: 14px; flex-wrap: wrap;
          background: #f8fafc; border: 1px solid var(--border);
          border-radius: 12px; padding: 14px 18px; margin-bottom: 16px;
        }}
        .ev-trace-status .scenario {{
          font-size: 1.05rem; font-weight: 700; color: #1e293b;
        }}
        .ev-result-tag {{
          display: inline-block; padding: 4px 12px; border-radius: 999px;
          font-size: .82rem; font-weight: 700;
        }}
        .ev-result-tag.pass {{ background: {COLOR['good_soft']}; color: {COLOR['good']}; }}
        .ev-result-tag.fail {{ background: {COLOR['bad_soft']}; color: {COLOR['bad']}; }}
        .ev-trace-status .score {{
          font-size: 1.1rem; font-weight: 700; margin-left: auto;
        }}
        .ev-trace-status .meta {{
          width: 100%; color: #64748b; font-size: .82rem; margin-top: 2px;
        }}
        .ev-trace-detail-stack {{
          margin-top: 8px;
        }}
        .ev-trace-overview {{
          display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px;
          margin-bottom: 16px;
        }}
        @media (max-width: 900px) {{
          .ev-trace-overview {{ grid-template-columns: 1fr; }}
        }}
        .ev-trace-overview-card {{
          display: flex; gap: 14px; align-items: flex-start;
          background: #fff; border: 1px solid #e2e8f0; border-radius: 16px;
          padding: 18px 20px; box-shadow: 0 2px 12px rgba(15,23,42,.05);
          position: relative; overflow: hidden; min-height: 108px;
        }}
        .ev-trace-overview-card::before {{
          content: ""; position: absolute; left: 0; top: 0; bottom: 0; width: 5px;
        }}
        .ev-trace-overview-card.dialogue::before {{ background: #2563eb; }}
        .ev-trace-overview-card.score::before {{ background: #8b5cf6; }}
        .ev-trace-overview-card.score.pass::before {{ background: #16a34a; }}
        .ev-trace-overview-card.score.fail::before {{ background: #dc2626; }}
        .ev-trace-overview-card.perf::before {{ background: #f59e0b; }}
        .ev-trace-overview-card .ico {{
          width: 44px; height: 44px; border-radius: 12px;
          display: flex; align-items: center; justify-content: center;
          font-size: 1.25rem; flex-shrink: 0;
          background: #f8fafc;
        }}
        .ev-trace-overview-card.dialogue .ico {{ background: #eff6ff; }}
        .ev-trace-overview-card.score .ico {{ background: #f5f3ff; }}
        .ev-trace-overview-card.perf .ico {{ background: #fffbeb; }}
        .ev-trace-overview-card .label {{
          font-size: .82rem; color: #64748b; font-weight: 600; margin-bottom: 4px;
        }}
        .ev-trace-overview-card .value {{
          font-size: 1.45rem; font-weight: 800; color: #0f172a; line-height: 1.2;
        }}
        .ev-trace-overview-card .value .unit {{
          font-size: .88rem; font-weight: 600; color: #64748b;
        }}
        .ev-trace-overview-card .hint {{
          font-size: .8rem; color: #94a3b8; margin-top: 6px; line-height: 1.45;
        }}
        /* 对话复盘三大折叠区：左侧彩色标识 */
        .st-key-trace_exp_dialogue [data-testid="stExpander"] details {{
          border-left: 4px solid #2563eb !important;
        }}
        .st-key-trace_exp_score [data-testid="stExpander"] details {{
          border-left: 4px solid #8b5cf6 !important;
        }}
        .st-key-trace_exp_perf [data-testid="stExpander"] details {{
          border-left: 4px solid #f59e0b !important;
        }}
        .ev-trace-dim-grid {{
          display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px;
          margin-bottom: 12px;
        }}
        @media (max-width: 768px) {{
          .ev-trace-dim-grid {{ grid-template-columns: 1fr; }}
        }}
        .ev-trace-dim-item {{
          background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 10px;
          padding: 10px 12px;
        }}
        .ev-trace-dim-item .name {{
          font-size: .78rem; color: #64748b; margin-bottom: 4px;
        }}
        .ev-trace-dim-item .bar-wrap {{
          height: 6px; background: #e2e8f0; border-radius: 999px; overflow: hidden;
        }}
        .ev-trace-dim-item .bar {{
          height: 100%; border-radius: 999px;
        }}
        .ev-trace-dim-item .meta {{
          display: flex; justify-content: space-between; align-items: center;
          margin-top: 6px; font-size: .82rem; font-weight: 700; color: #1e293b;
        }}
        .ev-trace-perf-metrics {{
          display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px;
          margin-bottom: 12px;
        }}
        @media (max-width: 768px) {{
          .ev-trace-perf-metrics {{ grid-template-columns: 1fr; }}
        }}
        .ev-trace-perf-metric {{
          background: linear-gradient(180deg, #fffbeb, #fff);
          border: 1px solid #fde68a; border-radius: 12px; padding: 14px 16px;
        }}
        .ev-trace-perf-metric .k {{
          font-size: .8rem; color: #92400e; font-weight: 600;
        }}
        .ev-trace-perf-metric .v {{
          font-size: 1.25rem; font-weight: 800; color: #78350f; margin-top: 4px;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# ----- 任务交付目标：UI 多处复用 ---------------------------------------------
DELIVERY_GOALS: list[dict[str, str]] = [
    {
        "id": "user_sim",
        "name": "用户模拟器覆盖完整场景矩阵",
        "evidence": "8 类用户画像，覆盖配合、犹豫、抗拒等多种通话场景",
        "page": "场景与画像",
        "color": "good",
    },
    {
        "id": "explainable",
        "name": "评测过程可解释",
        "evidence": "每条扣分都附带对话原文、评分依据与说明",
        "page": "报告中心",
        "color": "good",
    },
    {
        "id": "quantifiable",
        "name": "评测结果可量化",
        "evidence": "8 个评分维度、加权总分与 95% 置信区间",
        "page": "报告中心",
        "color": "good",
    },
    {
        "id": "reliable",
        "name": "评测结果可靠",
        "evidence": "多次采样投票、置信度评估与一致性校验",
        "page": "A/B 对比",
        "color": "good",
    },
]


def _kpi_card(label: str, value: str, *, delta: str = "", tone: str = "primary") -> str:
    return (
        f"<div class='ev-kpi {tone}'>"
        f"<div class='label'>{label}</div>"
        f"<div class='value'>{value}</div>"
        f"<div class='delta'>{delta}</div>"
        "</div>"
    )


def _pill(text: str, tone: str = "") -> str:
    cls = f"ev-pill {tone}".strip()
    return f"<span class='{cls}'>{text}</span>"


def _page_header(title: str, subtitle: str = "") -> None:
    sub_html = f"<div class='sub'>{subtitle}</div>" if subtitle else ""
    st.markdown(
        f"<div class='ev-page-head'><div class='title'>{title}</div>{sub_html}</div>",
        unsafe_allow_html=True,
    )


def _panel_start(title: str) -> None:
    st.markdown(f"<div class='ev-panel'><div class='ev-panel-title'>{title}</div>", unsafe_allow_html=True)


def _panel_end() -> None:
    st.markdown("</div>", unsafe_allow_html=True)


def _summary_card(line1: str, line2: str = "") -> None:
    l2 = f"<div class='line2'>{line2}</div>" if line2 else ""
    st.markdown(
        f"<div class='ev-summary'><div class='line1'>{line1}</div>{l2}</div>",
        unsafe_allow_html=True,
    )


def _page_guide(page_key: str) -> None:
    text = PAGE_GUIDES.get(page_key)
    if text:
        st.markdown(
            f"<div class='ev-guide'><span class='icon'>💡</span>"
            f"<span class='text'><b>怎么用：</b>{text}</span></div>",
            unsafe_allow_html=True,
        )


def _workflow_cards() -> None:
    cards = "".join(
        f"<div class='ev-step'>"
        f"<div class='num'>{s['num']}</div>"
        f"<div class='title'>{s['title']}</div>"
        f"<div class='desc'>{s['desc']}</div>"
        f"<div class='tip'>{s['tip']}</div></div>"
        for s in WORKFLOW_STEPS
    )
    st.markdown(f"<div class='ev-workflow'>{cards}</div>", unsafe_allow_html=True)


def _recommend_next(*, has_specs: bool, has_runs: bool, has_failures: bool) -> tuple[str, str, str]:
    """Return (title, hint, target_page)."""
    if not has_specs:
        return "先导入外呼任务", "在「任务配置」上传 Excel，或确认内置任务已加载", "任务配置"
    if not has_runs:
        return "发起第一次评测", "选好任务和场景，点「开始评测」即可，默认配置就能跑", "发起评测"
    if has_failures:
        return "查看未通过项", "有测试项没通过，建议去「对话复盘」看具体哪里说错了", "对话复盘"
    return "查看最新报告", "评测已完成，去「评测报告」看综合得分和各维度分析", "评测报告"


def _page_intro(what: str, *, goals: Optional[list[str]] = None, hint: str = "") -> None:
    return


def _scenario_label(scenario_id: str) -> str:
    return SCENARIO_DISPLAY_NAMES.get(scenario_id, scenario_id)


def _termination_label(code: Optional[str]) -> str:
    if not code:
        return "未知"
    return TERMINATION_LABELS.get(code, code)


def _format_test_item(instruction_id: str, scenario_id: str) -> str:
    return f"任务{instruction_id} · {_scenario_label(scenario_id)}"


def _format_case_id(case_id: str) -> str:
    if "__" in case_id:
        instr, scen = case_id.split("__", 1)
        return _format_test_item(instr, scen)
    return case_id


def _is_real_dialogue_run(report_data: Optional[dict] = None, info: Optional[dict] = None) -> bool:
    config = (report_data or {}).get("config") or {}
    if config.get("run_mode") == "real_dialogue":
        return True
    if info and info.get("config_json"):
        try:
            stored_config = json.loads(info["config_json"])
        except (TypeError, json.JSONDecodeError):
            stored_config = {}
        return stored_config.get("run_mode") == "real_dialogue"
    return False


def _nav_keys_for_env(env: str) -> list[str]:
    return REAL_NAV_KEYS if env == EVAL_ENV_REAL else NAV_KEYS


def _page_for_env(page_name: str, env: str) -> str:
    page = _NAV_ALIAS.get(page_name, page_name)
    if env == EVAL_ENV_REAL:
        return {
            "工作台": "真实工作台",
            "发起评测": "上传真实对话",
            "评测报告": "真实评测报告",
            "对话复盘": "真实对话复盘",
        }.get(page, page)
    return {
        "真实工作台": "工作台",
        "上传真实对话": "发起评测",
        "真实评测报告": "评测报告",
        "真实对话复盘": "对话复盘",
    }.get(page, page)


def _filter_runs_for_env(
    runs: list[dict],
    env: str,
    report_loader,
) -> list[dict]:
    filtered: list[dict] = []
    for run in runs:
        report_data = report_loader(run["run_id"]) or {}
        is_real = _is_real_dialogue_run(report_data, run)
        if env == EVAL_ENV_REAL and is_real:
            filtered.append(run)
        elif env != EVAL_ENV_REAL and not is_real:
            filtered.append(run)
    return filtered


def _page_map_for_env(env: str) -> dict[str, Any]:
    if env == EVAL_ENV_REAL:
        return {
            "真实工作台": page_dashboard,
            "上传真实对话": page_evaluate_real,
            "真实评测报告": page_reports_real,
            "真实对话复盘": page_trace_real,
            "任务配置": page_config,
        }
    return {
        "工作台": page_dashboard,
        "发起评测": page_evaluate,
        "评测报告": page_reports,
        "对话复盘": page_trace,
        "任务配置": page_config,
        "版本对比": page_compare,
    }


def _trace_selector_mode_for_env(env: str) -> str:
    return "dialogue_id" if env == EVAL_ENV_REAL else "scenario"


def _should_apply_query_page(
    *,
    query_page: Optional[str],
    current_page: Optional[str],
    last_synced_page: Optional[str],
) -> bool:
    if not query_page:
        return False
    if current_page is None:
        return True
    return query_page != last_synced_page


def _should_apply_query_env(
    *,
    query_env: Optional[str],
    current_env: Optional[str],
    last_synced_env: Optional[str],
) -> bool:
    if query_env not in {EVAL_ENV_GENERATED, EVAL_ENV_REAL}:
        return False
    if current_env is None:
        return True
    return query_env != last_synced_env


def _format_case_display(case_id: str, instruction_id: str = "", *, real_mode: bool = False) -> str:
    if real_mode:
        return f"{case_id} · 任务{instruction_id or '—'}"
    return _format_case_id(case_id)


def _normalize_role_text(role: str) -> str:
    """清洗 role 字段，用于「任务N-角色」展示。"""
    text = (role or "").strip()
    text = re.sub(r"^(你是|您是)\s*", "", text)
    text = text.rstrip("。. ")
    if not text:
        return "未设置角色"
    return text


def _format_instruction_label(spec_id: str, role: str = "") -> str:
    """统一外呼任务展示：任务1-骑手站长、任务2-机构客服…"""
    return f"任务{spec_id}-{_normalize_role_text(role)}"


def _trace_task_label(instr_id: str) -> str:
    """对话复盘专用：任务N-角色（上传新 JSON 后自动生效）。"""
    for spec in load_specs(INSTRUCTIONS_DIR):
        if spec.id == instr_id:
            return _format_instruction_label(spec.id, spec.role or "")
    return _format_instruction_label(instr_id, "")


def _trace_task_desc(instr_id: str) -> str:
    for spec in load_specs(INSTRUCTIONS_DIR):
        if spec.id == instr_id:
            task = (spec.task or "").replace("\n", " ").strip()
            if len(task) > 48:
                return task[:48] + "…"
            return task
    return ""


def _scenario_status(
    scen_id: str,
    pick_instr: str,
    case_id_map: dict[tuple[str, str], str],
    case_reports: dict[str, dict],
) -> tuple[Optional[bool], Optional[float]]:
    case_id = case_id_map.get((pick_instr, scen_id))
    if not case_id:
        return None, None
    cr = case_reports.get(case_id)
    if not cr:
        return None, None
    return cr.get("passed"), cr.get("weighted_total")


def _scenario_row_meta(
    scen_id: str,
    pick_instr: str,
    case_id_map: dict[tuple[str, str], str],
    case_reports: dict[str, dict],
) -> tuple[str, str, str, str]:
    name = _scenario_label(scen_id)
    passed, score = _scenario_status(scen_id, pick_instr, case_id_map, case_reports)
    score_t = f"{score:.3f}" if score is not None else "—"
    if passed is True:
        return name, "通过", score_t, "ok"
    if passed is False:
        return name, "未通过", score_t, "bad"
    return name, "待评", score_t, "neutral"


def _scenario_picker_df(
    scen_ids: list[str],
    pick_instr: str,
    case_id_map: dict[tuple[str, str], str],
    case_reports: dict[str, dict],
) -> pd.DataFrame:
    rows = []
    for scen in scen_ids:
        name, status, score_t, _ = _scenario_row_meta(
            scen, pick_instr, case_id_map, case_reports
        )
        rows.append({"用户类型": name, "结果": status, "得分": score_t})
    return pd.DataFrame(rows)


def _style_scenario_picker_df(df: pd.DataFrame):
    def _color_status(val: str) -> str:
        if val == "通过":
            return f"color: {COLOR['good']}; font-weight: 700"
        if val == "未通过":
            return f"color: {COLOR['bad']}; font-weight: 700"
        return "color: #64748b; font-weight: 600"

    return df.style.map(_color_status, subset=["结果"])


def _picker_row_index(picker_key: str) -> Optional[int]:
    """读取表格组件当前选中的行号。"""
    if picker_key not in st.session_state:
        return None
    state = st.session_state[picker_key]
    rows: list[int] = []
    if hasattr(state, "selection") and state.selection is not None:
        rows = list(state.selection.rows or [])
    elif isinstance(state, dict):
        rows = (state.get("selection") or {}).get("rows") or []
    if not rows:
        return None
    return int(rows[0])


def _sync_pick_scen_from_picker(
    scen_ids: list[str],
    fallback: str,
    scen_key: str,
    picker_key: str,
) -> str:
    """以表格选中行为唯一来源，避免标题行与表格不一致。"""
    row = _picker_row_index(picker_key)
    if row is not None and 0 <= row < len(scen_ids):
        picked = scen_ids[row]
    elif scen_key in st.session_state and st.session_state[scen_key] in scen_ids:
        picked = st.session_state[scen_key]
    elif fallback in scen_ids:
        picked = fallback
    else:
        picked = scen_ids[0]
    st.session_state[scen_key] = picked
    return picked


def _init_picker_selection(
    picker_key: str,
    scen_ids: list[str],
    pick_scen: str,
) -> None:
    """首次进入或切换任务时，让表格高亮与当前用户类型对齐。"""
    idx = scen_ids.index(pick_scen) if pick_scen in scen_ids else 0
    if picker_key not in st.session_state:
        st.session_state[picker_key] = {
            "selection": {"rows": [idx], "columns": []},
        }
        return
    row = _picker_row_index(picker_key)
    if row is None or row >= len(scen_ids):
        st.session_state[picker_key] = {
            "selection": {"rows": [idx], "columns": []},
        }


def _on_scenario_table_select(
    scen_ids: list[str],
    scen_key: str,
    picker_key: str,
) -> None:
    """表格点选后同步用户类型。"""
    row = _picker_row_index(picker_key)
    if row is not None and 0 <= row < len(scen_ids):
        st.session_state[scen_key] = scen_ids[row]


def _render_scenario_picker(
    scen_ids: list[str],
    pick_scen: str,
    scen_key: str,
    pick_instr: str,
    case_id_map: dict[tuple[str, str], str],
    case_reports: dict[str, dict],
) -> str:
    """用户类型选择：可点选表格，列对齐且结果着色。"""
    picker_key = f"{scen_key}_picker"

    if pick_scen not in scen_ids:
        pick_scen = scen_ids[0]
    _init_picker_selection(picker_key, scen_ids, pick_scen)

    st.markdown(
        "<div class='ev-scen-select-note'>"
        "<span class='dot'></span>"
        "<span>点击下表选择用户类型；绿色=通过，红色=未通过</span>"
        "</div><div class='ev-scen-table-picker'></div>",
        unsafe_allow_html=True,
    )
    df = _scenario_picker_df(scen_ids, pick_instr, case_id_map, case_reports)
    styled = _style_scenario_picker_df(df)
    event = st.dataframe(
        styled,
        hide_index=True,
        width="stretch",
        on_select=lambda: _on_scenario_table_select(scen_ids, scen_key, picker_key),
        selection_mode="single-row",
        key=picker_key,
        height=min(36 * (len(scen_ids) + 1), 300),
    )
    if event.selection and event.selection.rows:
        row_idx = event.selection.rows[0]
        if 0 <= row_idx < len(scen_ids):
            pick_scen = scen_ids[row_idx]
    else:
        pick_scen = _sync_pick_scen_from_picker(
            scen_ids, pick_scen, scen_key, picker_key
        )
    st.session_state[scen_key] = pick_scen
    return pick_scen


_TRACE_SECTION_ICONS = {
    "dialogue": "💬",
    "score": "📊",
    "perf": "⚡",
}


@contextmanager
def _trace_detail_section(
    tone: str,
    title: str,
    badge: str = "",
    *,
    expanded: bool = False,
):
    icon = _TRACE_SECTION_ICONS.get(tone, "📁")
    label = f"{icon}  {title}"
    if badge:
        label += f"    ·    {badge}"
    with st.expander(label, expanded=expanded, key=f"trace_exp_{tone}"):
        yield


def _render_trace_detail_overview(
    *,
    turn_count: int,
    score: Optional[float],
    passed: Optional[bool],
    dim_count: int,
    fail_attr_count: int,
    total_latency_ms: int,
    tokens_in: int,
    tokens_out: int,
) -> None:
    score_text = f"{score:.3f}" if score is not None else "—"
    if passed is True:
        status_cls, status_text = "pass", "通过"
    elif passed is False:
        status_cls, status_text = "fail", "未通过"
    else:
        status_cls, status_text = "neutral", "待评"
    fail_hint = f" · {fail_attr_count} 项归因" if fail_attr_count else ""
    st.markdown(
        f"<div class='ev-trace-overview'>"
        f"<div class='ev-trace-overview-card dialogue'>"
        f"<div class='ico'>💬</div><div class='body'>"
        f"<div class='label'>逐轮对话</div>"
        f"<div class='value'>{turn_count}<span class='unit'> 轮</span></div>"
        f"<div class='hint'>用户 / 模型 / 评分三栏对照</div>"
        f"</div></div>"
        f"<div class='ev-trace-overview-card score {status_cls}'>"
        f"<div class='ico'>📊</div><div class='body'>"
        f"<div class='label'>评分详情</div>"
        f"<div class='value'>{score_text}</div>"
        f"<div class='hint'>{dim_count} 项维度 · {status_text}{fail_hint}</div>"
        f"</div></div>"
        f"<div class='ev-trace-overview-card perf'>"
        f"<div class='ico'>⚡</div><div class='body'>"
        f"<div class='label'>性能数据</div>"
        f"<div class='value'>{total_latency_ms}<span class='unit'> ms</span></div>"
        f"<div class='hint'>入 {tokens_in} / 出 {tokens_out} 字 · 可下载 JSON</div>"
        f"</div></div>"
        f"</div>",
        unsafe_allow_html=True,
    )


def _score_bar_color(score: float) -> str:
    if score >= 0.85:
        return COLOR["good"]
    if score >= 0.6:
        return "#ca8a04"
    return COLOR["bad"]


def _render_trace_dimension_cards(dimensions: list[dict]) -> None:
    if not dimensions:
        st.caption("暂无维度得分。")
        return
    items = []
    for dim in dimensions:
        raw = dim.get("score")
        score = float(raw) if raw is not None else 0.0
        conf = dim.get("confidence")
        conf_t = f"{conf:.0%}" if conf is not None else "—"
        color = _score_bar_color(score)
        pct = max(0.0, min(score * 100, 100.0))
        items.append(
            f"<div class='ev-trace-dim-item'>"
            f"<div class='name'>{dim.get('name', '—')}</div>"
            f"<div class='bar-wrap'><div class='bar' style='width:{pct:.0f}%;background:{color};'></div></div>"
            f"<div class='meta'><span>{score:.3f}</span>"
            f"<span style='color:#94a3b8;font-weight:600;'>置信 {conf_t}</span></div>"
            f"</div>"
        )
    st.markdown(
        f"<div class='ev-trace-dim-grid'>{''.join(items)}</div>",
        unsafe_allow_html=True,
    )


def _render_trace_perf_metrics(
    total_latency_ms: int,
    tokens_in: int,
    tokens_out: int,
) -> None:
    st.markdown(
        f"<div class='ev-trace-perf-metrics'>"
        f"<div class='ev-trace-perf-metric'><div class='k'>总响应耗时</div>"
        f"<div class='v'>{total_latency_ms} ms</div></div>"
        f"<div class='ev-trace-perf-metric'><div class='k'>输入字数（约）</div>"
        f"<div class='v'>{tokens_in}</div></div>"
        f"<div class='ev-trace-perf-metric'><div class='k'>输出字数（约）</div>"
        f"<div class='v'>{tokens_out}</div></div>"
        f"</div>",
        unsafe_allow_html=True,
    )


def _render_trace_result_bar(
    *,
    scenario_label: str,
    passed: Optional[bool],
    score: Optional[float],
    meta_line: str,
) -> None:
    if passed is True:
        tag = "<span class='ev-result-tag pass'>通过</span>"
        score_color = COLOR["good"]
    elif passed is False:
        tag = "<span class='ev-result-tag fail'>未通过</span>"
        score_color = COLOR["bad"]
    else:
        tag = "<span class='ev-result-tag' style='background:#f1f5f9;color:#64748b'>待评</span>"
        score_color = "#64748b"
    score_text = f"{score:.3f}" if score is not None else "—"
    st.markdown(
        f"<div class='ev-trace-status'>"
        f"<span class='scenario'>{scenario_label}</span>"
        f"{tag}"
        f"<span class='score' style='color:{score_color}'>得分 {score_text}</span>"
        f"<span class='meta'>{meta_line}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )


def _parse_case_id(case_id: str) -> tuple[str, str]:
    if "__" in case_id:
        instr, scen = case_id.split("__", 1)
        return instr, scen
    return case_id, ""


def _case_index(cases: list[dict]) -> tuple[dict[str, list[str]], dict[tuple[str, str], str]]:
    """按外呼任务分组用户类型，并建立 (任务, 类型) → case_id 映射。"""
    by_instr: dict[str, list[str]] = {}
    id_map: dict[tuple[str, str], str] = {}
    for row in cases:
        instr = row.get("instruction_id") or _parse_case_id(row["case_id"])[0]
        scen = row.get("scenario_id") or _parse_case_id(row["case_id"])[1]
        if not scen:
            continue
        by_instr.setdefault(instr, [])
        if scen not in by_instr[instr]:
            by_instr[instr].append(scen)
        id_map[(instr, scen)] = row["case_id"]
    for instr in by_instr:
        by_instr[instr].sort()
    return by_instr, id_map


def _format_run_label(run_id: str) -> str:
    if run_id.startswith("webui_"):
        return run_id.replace("webui_", "网页评测 ")
    return run_id


def _specs_signature() -> str:
    """Cheap signature of all instruction JSON files (mtime + size).

    Used as a cache-busting key so that re-parsing instructions
    invalidates downstream cached scenario matrices.
    """
    parts: list[str] = []
    if INSTRUCTIONS_DIR.exists():
        for p in sorted(INSTRUCTIONS_DIR.glob("*.json")):
            try:
                stat = p.stat()
                parts.append(f"{p.name}:{int(stat.st_mtime)}:{stat.st_size}")
            except OSError:
                continue
    return "|".join(parts)


@st.cache_data(show_spinner=False)
def _cached_scenarios(spec_id: str, seed: int, _sig: str) -> tuple[list, dict]:
    """Memoise build_scenarios per (spec_id, seed) within the streamlit session.

    Avoids re-computing the scenario matrix (and re-emitting log lines) on
    every Streamlit rerun. ``_sig`` is the spec-files signature used to
    invalidate the cache when underlying JSON changes.
    """
    specs = load_specs(INSTRUCTIONS_DIR)
    spec = next((s for s in specs if s.id == spec_id), None)
    if spec is None:
        return [], {}
    scenarios, vmap = build_scenarios(spec, seed=int(seed))
    return list(scenarios), dict(vmap)


SCENARIO_FIELD_ROWS: list[dict[str, str]] = [
    {"配置项": "场景名称", "说明": "如配合型、犹豫型等用户类型", "用于": "界面展示与测试分类"},
    {"配置项": "用户目标", "说明": "这通电话里用户想达成什么", "用于": "驱动用户模拟器行为"},
    {"配置项": "行为设定", "说明": "用户会怎么说话、怎么反应", "用于": "模拟真实通话，不含标准答案"},
    {"配置项": "需覆盖流程", "说明": "对话应走到哪些业务步骤", "用于": "检查流程是否走完"},
    {"配置项": "需检验规则", "说明": "字数、挂断方式等硬性要求", "用于": "合规性评分"},
    {"配置项": "知识问答点", "说明": "用户可能追问的 FAQ 主题", "用于": "检查回答是否准确"},
    {"配置项": "施压情况", "说明": "如打断、越权提问等", "用于": "测试模型抗压能力"},
    {"配置项": "必须出现的话", "说明": "用户开场必须说的话", "用于": "确保场景被正确触发"},
    {"配置项": "预期结束方式", "说明": "通话应如何收尾", "用于": "检查挂断是否合理"},
]


def _navigate_to_trace(run_id: str, case_id: str, *, env: str = EVAL_ENV_GENERATED) -> None:
    target_page = "真实对话复盘" if env == EVAL_ENV_REAL else "对话复盘"
    st.session_state["trace_run_id"] = run_id
    st.session_state["trace_case_id"] = case_id
    st.session_state["results_tab"] = "复盘"
    st.session_state["_nav_target"] = target_page
    st.session_state.pop("_last_synced_query_env", None)
    st.session_state.pop("_last_synced_query_page", None)
    st.query_params["eval_env"] = env
    st.query_params["page"] = target_page
    st.query_params["run"] = run_id
    st.query_params["case"] = case_id
    st.rerun()


def _navigate(page_name: str, *, env: Optional[str] = None) -> None:
    target_env = env or st.session_state.get("eval_env", EVAL_ENV_GENERATED)
    target_page = _page_for_env(page_name, target_env)
    st.session_state["_nav_target"] = target_page
    st.session_state.pop("_last_synced_query_env", None)
    st.session_state.pop("_last_synced_query_page", None)
    st.query_params["eval_env"] = target_env
    st.query_params["page"] = target_page
    st.rerun()


# ----- 配置预设：与 README 4.6 节保持一致 ------------------------------------
RUN_PRESETS: dict[str, dict[str, int]] = {
    "默认（推荐）": {"workers": 8, "judge_workers": 4, "samples_workers": 3, "max_in_flight": 32},
    "高并发（自有大配额）": {"workers": 16, "judge_workers": 8, "samples_workers": 3, "max_in_flight": 64},
    "低并发（共享配额）": {"workers": 2, "judge_workers": 1, "samples_workers": 2, "max_in_flight": 8},
    "调试串行": {"workers": 1, "judge_workers": 1, "samples_workers": 1, "max_in_flight": 0},
    "自定义": {},
}


# DeepSeek 公开价（元 / 1M tokens），仅作 UI 估算用
DEEPSEEK_PRICE = {
    "deepseek-v4-flash": {"in": 1.0, "out": 2.0},
    "deepseek-v4-pro": {"in": 3.0, "out": 6.0},
    "deepseek-chat": {"in": 1.0, "out": 2.0},
    "deepseek-reasoner": {"in": 4.0, "out": 16.0},
}


def _estimate_run(
    n_cases: int,
    *,
    max_turns: int,
    judge_samples: int,
    sut_model: str,
    judge_model: str,
    user_model: str,
    max_in_flight: int,
    avg_turn_secs: float = 5.0,
    avg_in_tokens: int = 600,
    avg_out_tokens: int = 200,
) -> dict[str, Any]:
    """Rough back-of-envelope estimate for the eval-run page.

    All numbers are pessimistic upper bounds for budgeting; cache hits
    will make actual runs cheaper.
    """
    dialogue_calls = n_cases * max_turns * 2
    judge_calls = n_cases * 4 * judge_samples
    total_calls = dialogue_calls + judge_calls

    def _cost(model: str, n: int) -> float:
        price = DEEPSEEK_PRICE.get(model)
        if not price:
            return 0.0
        cost_in = n * avg_in_tokens / 1_000_000 * price["in"]
        cost_out = n * avg_out_tokens / 1_000_000 * price["out"]
        return cost_in + cost_out

    cost = (
        _cost(sut_model, n_cases * max_turns)
        + _cost(user_model, n_cases * max_turns)
        + _cost(judge_model, judge_calls)
    )
    parallelism = max(1, max_in_flight) if max_in_flight > 0 else 1
    eta_secs = total_calls * avg_turn_secs / max(1, parallelism)
    return {
        "dialogue_calls": dialogue_calls,
        "judge_calls": judge_calls,
        "total_calls": total_calls,
        "cost_yuan": round(cost, 3),
        "eta_secs": int(eta_secs),
    }


def _format_secs(secs: int) -> str:
    if secs < 60:
        return f"{secs} 秒"
    if secs < 3600:
        return f"{secs // 60} 分 {secs % 60:02d} 秒"
    return f"{secs // 3600} 时 {(secs % 3600) // 60:02d} 分"


def _pid_alive(pid: int) -> bool:
    """Cross-platform alive check that doesn't require psutil."""
    if pid <= 0:
        return False
    try:
        if os.name == "nt":
            import ctypes

            handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)
            if not handle:
                return False
            exit_code = ctypes.c_ulong()
            ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code))
            ctypes.windll.kernel32.CloseHandle(handle)
            return exit_code.value == 259  # STILL_ACTIVE
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def _save_proc_meta(run_id: str, pid: int, cmd: list[str], log_path: Path) -> Path:
    out_dir = REPORTS_ROOT / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "pid": pid,
        "cmd": cmd,
        "log_path": str(log_path),
        "started_at": datetime.now().isoformat(),
        "run_id": run_id,
    }
    p = out_dir / ".proc.json"
    p.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return p


def _load_proc_meta(run_id: str) -> Optional[dict]:
    p = REPORTS_ROOT / run_id / ".proc.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _list_active_runs() -> list[dict]:
    """Walk reports/ for any .proc.json with a still-alive PID.

    报告（run_report.json）已生成的 run 视为已完成，并清理掉它的
    .proc.json，避免：
    - Streamlit Cloud 等容器环境下子进程退出后变 zombie，被 `kill(pid, 0)`
      误判成 alive，让 UI 永远把它列在「运行中」面板。
    - 容器重启后旧 PID 被新进程复用，造成误判。
    """
    active: list[dict] = []
    if not REPORTS_ROOT.exists():
        return active
    for proc_file in REPORTS_ROOT.glob("*/.proc.json"):
        try:
            meta = json.loads(proc_file.read_text(encoding="utf-8"))
        except Exception:
            continue
        run_dir = proc_file.parent
        if (run_dir / "run_report.json").exists():
            try:
                proc_file.unlink()
            except OSError:
                pass
            continue
        if _pid_alive(int(meta.get("pid", 0))):
            active.append(meta)
    return active


_PROGRESS_RE = re.compile(r"(\d+)\s*/\s*(\d+)")
_TOTAL_CASES_RE = re.compile(r"Total cases to evaluate:\s*(\d+)")
# 这些关键词出现在汇总/失败统计行里（例如「Case 通过率: 15.0% (3/20)」），
# 它们里面的 X/Y 数字不是真实进度，必须从进度解析中排除。
_PROGRESS_NOISE_KEYWORDS = (
    "通过率",
    "成功率",
    "失败率",
    "Pass rate",
    "Success rate",
    "通过：",
    "通过:",
    "失败：",
    "失败:",
)


def _parse_progress(log_text: str) -> tuple[int, int]:
    """Pull the most recent (done, total) from rich progress lines.

    日志末尾会写入「Case 通过率: 15.0% (3/20)」一类的统计行，里面也有
    X/Y，原先直接取最后一个匹配会把进度卡在通过数上。这里逐行扫描并
    跳过明显的统计噪声行，再取最后一次出现。
    """
    candidates: list[tuple[int, int]] = []
    for line in log_text.splitlines():
        if any(noise in line for noise in _PROGRESS_NOISE_KEYWORDS):
            continue
        for done, total in _PROGRESS_RE.findall(line):
            candidates.append((int(done), int(total)))
    if not candidates:
        return 0, 0
    return candidates[-1]


@dataclass(frozen=True)
class ActiveRunSummary:
    label: str
    detail: str
    done: int
    total: int
    http_ok_count: int
    log_updated_at: Optional[str]


def _summarize_active_run(
    *,
    run_id: str,
    alive: bool,
    log_path: Path,
    report_exists: bool,
) -> ActiveRunSummary:
    log_text = log_path.read_text(encoding="utf-8", errors="ignore") if log_path.exists() else ""
    done, total = _parse_progress(log_text)
    if total == 0:
        total_match = _TOTAL_CASES_RE.search(log_text)
        if total_match:
            total = int(total_match.group(1))
    http_ok_count = log_text.count("HTTP/1.1 200 OK")
    request_count = log_text.count("Sending HTTP Request")
    log_updated_at = None
    if log_path.exists():
        log_updated_at = datetime.fromtimestamp(log_path.stat().st_mtime).strftime("%H:%M:%S")

    if report_exists:
        status = "已完成，报告已生成"
    elif alive and done == 0 and total and http_ok_count:
        status = "运行中 · 模型响应正常，正在评分中"
    elif alive and done == 0 and total:
        status = "运行中 · 已启动评分，等待首个进度回写"
    elif alive:
        status = "运行中"
    else:
        status = "进程已结束，等待报告生成或检查日志"

    progress_text = f"{done}/{total} 项" if total else "进度待回写"
    details: list[str] = []
    if total:
        details.append(f"共需评测 {total} 组对话")
    if request_count:
        details.append(f"已发起 {request_count} 次模型请求")
    if http_ok_count:
        details.append(f"模型接口已有 {http_ok_count} 次成功响应")
    if log_updated_at:
        details.append(f"日志最后更新 {log_updated_at}")
    if not details:
        details.append("正在等待日志写入")

    return ActiveRunSummary(
        label=f"{status} · {progress_text}",
        detail="；".join(details),
        done=done,
        total=total,
        http_ok_count=http_ok_count,
        log_updated_at=log_updated_at,
    )


def _store() -> RunStore:
    return RunStore(root=REPORTS_ROOT)


def _load_run_report(run_id: str) -> dict:
    path = REPORTS_ROOT / run_id / "run_report.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def _build_raw_dialogue_export(report_data: dict) -> list[dict]:
    rows: list[dict] = []
    for case in report_data.get("cases", []):
        trace = case.get("trace") or {}
        turns = trace.get("turns")
        if not turns:
            continue
        dialogue = [
            {
                "index": turn.get("index"),
                "role": turn.get("role"),
                "content": turn.get("content"),
            }
            for turn in turns
        ]
        idx = len(rows) + 1
        instruction_id = trace.get("instruction_id") or idx
        rows.append({
            "id": f"{idx:03d}",
            "任务编号": int(instruction_id) if str(instruction_id or "").isdigit() else instruction_id,
            "多轮对话": dialogue,
        })
    return rows


_COMPACT_REPORT_HTML_CSS = """
<style id="compact-report-trace">
.turns { gap: 3px !important; margin-top: 4px !important; }
.turn { display: grid !important; grid-template-columns: 72px 1fr !important; gap: 6px !important; align-items: start !important; padding: 3px 6px !important; line-height: 1.35 !important; min-height: 0 !important; }
.turn .who { width: auto !important; font-size: 12px !important; line-height: 1.35 !important; }
.turn .body { white-space: normal !important; font-size: 13px !important; line-height: 1.35 !important; min-height: 0 !important; }
.turn .body > div { margin: 0 !important; }
.turn .turn-text { white-space: pre-wrap !important; }
.violations { margin-top: 2px !important; line-height: 1.35 !important; }
</style>
"""


def _compact_report_html(html: str) -> str:
    if "compact-report-trace" in html:
        return html
    compacted = re.sub(
        r"\.turn \.body \{[^}]*white-space:\s*pre-wrap;[^}]*\}",
        ".turn .body { font-size: 13px; line-height: 1.35; }",
        html,
    )
    if "</head>" in compacted:
        return compacted.replace("</head>", f"{_COMPACT_REPORT_HTML_CSS}\n</head>", 1)
    return _COMPACT_REPORT_HTML_CSS + compacted


def _tail_text(path: Path, limit: int = 6000) -> str:
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="ignore")
    return text[-limit:]


def _latest_log_excerpt(path: Path, *, max_lines: int = 8) -> str:
    raw = _tail_text(path, limit=20000)
    if not raw:
        return "暂无日志输出。"
    signal_keywords = (
        "Total cases to evaluate",
        "HTTP Request",
        "HTTP/1.1",
        "response_body.complete",
        "ERROR",
        "WARNING",
        "Traceback",
        "Exception",
        "Case 通过率",
        "报告",
        "cases",
    )
    selected: list[str] = []
    for line in raw.splitlines():
        text = line.strip()
        if not text:
            continue
        if len(text) > 600 and not any(keyword in text for keyword in signal_keywords):
            continue
        if any(keyword in text for keyword in signal_keywords):
            selected.append(text[:1000])
    if not selected:
        selected = [line.strip()[:1000] for line in raw.splitlines() if line.strip()]
    return "\n".join(selected[-max_lines:]) if selected else "暂无可展示日志。"


def page_instructions() -> None:
    cfg = load_config()
    samples = int(cfg.get("judge", {}).get("samples", 3))
    max_turns = int(cfg.get("runtime", {}).get("max_turns", 16))
    sut_model = cfg.get("models", {}).get("sut", "deepseek-v4-flash")
    judge_model = cfg.get("models", {}).get("judge", "deepseek-v4-pro")
    user_model = cfg.get("models", {}).get("user_sim", "deepseek-v4-flash")
    seed = int(cfg.get("runtime", {}).get("seed", 42))
    specs = load_specs(INSTRUCTIONS_DIR)
    if specs:
        st.caption(f"已导入 {len(specs)} 个外呼任务")
        rows = []
        sig = _specs_signature()
        for s in specs:
            try:
                scenarios, _ = _cached_scenarios(s.id, int(seed), sig)
                n_scen = len(scenarios)
            except Exception:
                n_scen = 0
            est = _estimate_run(
                n_scen,
                max_turns=max_turns,
                judge_samples=samples,
                sut_model=sut_model,
                judge_model=judge_model,
                user_model=user_model,
                max_in_flight=int(cfg.get("runtime", {}).get("max_in_flight", 32)),
            )
            hard_rules = []
            if s.constraints.hard.no_discount_promise:
                hard_rules.append("禁止承诺优惠")
            if s.constraints.hard.required_out_of_scope_reply:
                hard_rules.append("越权需标准回复")
            rows.append(
                {
                    "任务编号": s.id,
                    "角色设定": s.role[:60],
                    "外呼目标": s.task[:80],
                    "单条字数上限": s.constraints.hard.max_chars_per_reply,
                    "流程节点数": len(s.flow_nodes),
                    "知识问答数": len(s.knowledge),
                    "可测场景数": n_scen,
                    "预估 API 调用": est["total_calls"],
                    "预估费用(¥)": est["cost_yuan"],
                    "预估耗时": _format_secs(est["eta_secs"]),
                    "硬性规则": "、".join(hard_rules) or "—",
                }
            )
        with st.expander(f"任务列表与费用估算（{len(specs)} 项）", expanded=False):
            st.dataframe(rows, use_container_width=True, hide_index=True)
            st.caption(
                f"费用估算：单通最多 {max_turns} 轮 · 评分采样 {samples} 次 · 离线模拟不产生费用"
            )
        with st.expander("查看原始配置", expanded=False):
            chosen = st.selectbox("选择任务", [s.id for s in specs], format_func=lambda x: f"任务{x}")
            target = next((s for s in specs if s.id == chosen), None)
            if target:
                st.json(target.model_dump())
    else:
        st.info("尚未导入任何外呼任务。")

    with st.expander("导入新的外呼任务", expanded=not specs):
        uploaded = st.file_uploader("上传 Excel (.xlsx)", type=["xlsx"])
        use_llm = st.toggle("使用大模型解析（关闭则用规则解析）", value=True)
        if st.button("开始解析", disabled=uploaded is None and not SOURCE_XLSX_DEFAULT.exists()):
            if uploaded is not None:
                target_path = ROOT / "data" / "source" / uploaded.name
                target_path.parent.mkdir(parents=True, exist_ok=True)
                target_path.write_bytes(uploaded.getbuffer())
            else:
                target_path = SOURCE_XLSX_DEFAULT
            client = build_default_client() if use_llm else None
            with st.spinner("解析中..."):
                specs = parse_workbook(
                    target_path,
                    output_dir=INSTRUCTIONS_DIR,
                    client=client,
                    mode="llm" if use_llm else "offline",
                )
            _cached_scenarios.clear()
            st.success(f"已解析 {len(specs)} 条任务")
            st.rerun()


def page_scenarios() -> None:
    specs = load_specs(INSTRUCTIONS_DIR)
    if not specs:
        st.warning("请先在「外呼任务」标签页导入任务。")
        return
    spec_id = st.selectbox("选择外呼任务", [s.id for s in specs], format_func=lambda x: f"任务{x}")
    spec = next(s for s in specs if s.id == spec_id)
    seed = st.number_input("随机种子", min_value=0, value=42, step=1)
    scenarios, vmap = _cached_scenarios(spec.id, int(seed), _specs_signature())
    validation = validate_case_matrix(INSTRUCTIONS_DIR, seed=int(seed))
    _summary_card(
        f"任务 {spec_id} · {len(scenarios)} 种用户场景",
        f"全平台共 {validation['n_cases']} 个可测场景"
        + (f" · {len(validation['issues'])} 个配置问题" if validation["issues"] else ""),
    )
    if validation["issues"]:
        with st.expander("配置问题", expanded=False):
            st.warning("\n".join(f"- {x}" for x in validation["issues"][:8]))

    with st.expander(f"场景列表（{len(scenarios)} 项）", expanded=False):
        rows = [
            {
                "场景类型": _scenario_label(s.id),
                "用户目标": s.user_goal[:40] + ("…" if len(s.user_goal) > 40 else ""),
                "需覆盖流程": ", ".join(s.target_nodes),
                "预期结束": _termination_label(s.expected_termination) if s.expected_termination else "—",
            }
            for s in scenarios
        ]
        st.dataframe(rows, use_container_width=True, hide_index=True)
        with st.expander("字段说明"):
            st.dataframe(SCENARIO_FIELD_ROWS, use_container_width=True, hide_index=True)

    matrix = coverage_matrix(spec, scenarios)
    scen_ids = [s.id for s in scenarios]
    scen_labels = [_scenario_label(sid) for sid in scen_ids]
    with st.expander("流程节点覆盖情况", expanded=False):
        if spec.flow_nodes and scen_ids:
            z = []
            for node in spec.flow_nodes:
                coverers = set(matrix.get(node.id, []))
                z.append([1 if sid in coverers else 0 for sid in scen_ids])
            node_labels = [f"{n.id}: {n.desc[:18]}" for n in spec.flow_nodes]
            heat = go.Figure(data=go.Heatmap(
                z=z, x=scen_labels, y=node_labels,
                colorscale=[[0, "#f4f4f5"], [1, "#22c55e"]],
                showscale=False, xgap=2, ygap=2,
                hovertemplate="节点 %{y}<br>场景 %{x}<br>覆盖=%{z}<extra></extra>",
            ))
            heat.update_layout(
                height=max(240, 32 * len(node_labels)),
                margin=dict(l=10, r=10, t=20, b=10),
                xaxis=dict(side="top"), yaxis=dict(autorange="reversed"),
            )
            st.plotly_chart(heat, use_container_width=True)
        matrix_rows = [
            {
                "节点": node.id,
                "描述": node.desc[:50],
                "覆盖数": len(matrix.get(node.id, [])),
            }
            for node in spec.flow_nodes
        ]
        st.dataframe(matrix_rows, use_container_width=True, hide_index=True)

    with st.expander("用户行为设定", expanded=False):
        for s in scenarios:
            st.markdown(f"**{_scenario_label(s.id)}**")
            st.code(s.behaviour, language="text")


# streamlit fragment 装饰器：1.32 是 experimental_fragment，1.33+ 是 fragment。
# 没有时退化为 no-op（用户需手动点"刷新"按钮）。
_st_fragment = getattr(st, "fragment", None) or getattr(st, "experimental_fragment", None)


def _render_active_runs_body() -> None:
    """Inner body shared by both fragment and fallback paths."""
    active = _list_active_runs()
    session_run_id = st.session_state.get("eval_run_id")
    if session_run_id and not any(m["run_id"] == session_run_id for m in active):
        meta = _load_proc_meta(session_run_id)
        if meta:
            active.append(meta)
    if not active:
        return

    for meta in active:
        run_id = meta.get("run_id", "")
        pid = int(meta.get("pid", 0))
        log_path = Path(meta.get("log_path", ""))
        alive = _pid_alive(pid)
        report_exists = (REPORTS_ROOT / run_id / "run_report.json").exists()
        summary = _summarize_active_run(
            run_id=run_id,
            alive=alive,
            log_path=log_path,
            report_exists=report_exists,
        )
        st.caption(
            f"**{_format_run_label(run_id)}** · "
            f"{summary.label}"
        )
        st.caption(summary.detail)
        if summary.total:
            st.progress(min(1.0, summary.done / max(1, summary.total)))
        elif alive:
            st.info("任务已启动，正在等待评测进度写入日志。")
        with st.expander("最新日志", expanded=False):
            st.code(_latest_log_excerpt(log_path), language="text")
        if alive:
            if st.button(f"停止 {run_id}", key=f"kill_{run_id}"):
                try:
                    if os.name == "nt":
                        subprocess.run(["taskkill", "/PID", str(pid), "/F"], check=False)
                    else:
                        os.kill(pid, 9)
                    st.warning(f"已请求终止 {run_id}")
                except Exception as exc:
                    st.error(f"终止失败：{exc}")
        elif report_exists:
            if st.button(f"清理记录 {run_id}", key=f"clean_{run_id}"):
                proc_file = REPORTS_ROOT / run_id / ".proc.json"
                proc_file.unlink(missing_ok=True)
                st.session_state.pop("eval_run_id", None)
                st.rerun()
    st.divider()


# 用 fragment 包装内部 body，让它每 3s 自己 rerun（不影响主 page 的其他控件、
# 不丢用户的输入框状态）。
if _st_fragment is not None:
    @_st_fragment(run_every=3.0)
    def _render_active_runs_fragment() -> None:
        _render_active_runs_body()
else:
    _render_active_runs_fragment = _render_active_runs_body  # type: ignore


def _render_active_runs(cfg: dict) -> None:
    if not _list_active_runs() and not st.session_state.get("eval_run_id"):
        return
    with st.expander("运行中的任务", expanded=False):
        _render_active_runs_body()


def _apply_preset(preset_name: str, cfg: dict) -> None:
    if preset_name not in RUN_PRESETS or preset_name == "自定义":
        return
    preset = RUN_PRESETS[preset_name]
    if not preset:
        return
    st.session_state["cfg_workers"] = preset["workers"]
    st.session_state["cfg_judge_workers"] = preset["judge_workers"]
    st.session_state["cfg_samples_workers"] = preset["samples_workers"]
    st.session_state["cfg_max_in_flight"] = preset["max_in_flight"]


def _page_evaluate_real_dialogues(cfg: dict, specs: list) -> None:
    _panel_start("① 上传真实对话")
    uploaded = st.file_uploader(
        "上传原始对话 JSON",
        type=["json"],
        help="格式为数组，每条包含 id、任务编号、多轮对话。任务编号需对应当前任务配置。",
    )
    if uploaded is not None:
        try:
            preview_payload = json.loads(uploaded.getvalue().decode("utf-8"))
            preview_path = REPORTS_ROOT / "_preview_real_dialogues.json"
            preview_path.write_text(json.dumps(preview_payload, ensure_ascii=False), encoding="utf-8")
            traces = load_real_dialogues(
                preview_path,
                {spec.id: spec for spec in specs},
                run_id="preview",
            )
            preview_path.unlink(missing_ok=True)
            st.success(f"已识别 {len(traces)} 条真实对话")
            st.dataframe(
                [
                    {"对话 ID": t.case_id, "任务编号": t.instruction_id, "轮数": len(t.turns)}
                    for t in traces[:20]
                ],
                use_container_width=True,
                hide_index=True,
            )
        except (UnicodeDecodeError, json.JSONDecodeError, RealDialogueImportError) as exc:
            st.error(f"上传文件校验失败：{exc}")
    _panel_end()

    judge_mode = "llm"
    judge_model = cfg.get("models", {}).get("judge", "")
    judge_samples = int(cfg.get("judge", {}).get("samples", 3))
    judge_require_evidence = bool(cfg.get("judge", {}).get("require_evidence", False))
    seed = int(cfg.get("runtime", {}).get("seed", 42))
    with st.expander("评分引擎", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            judge_mode = st.radio(
                "评分引擎",
                ["llm", "offline"],
                horizontal=True,
                format_func=lambda x: "大模型评分" if x == "llm" else "规则评分",
                key="real_judge_mode",
            )
        with c2:
            judge_model = st.text_input(
                "评分模型名称",
                judge_model,
                disabled=judge_mode == "offline",
                key="real_judge_model",
            )
        c3, c4 = st.columns(2)
        judge_samples = c3.slider("评分采样次数", 1, 5, judge_samples, key="real_judge_samples")
        seed = c4.number_input("随机种子", min_value=0, value=seed, key="real_seed")
        judge_require_evidence = st.toggle(
            "评分需附原文证据",
            value=judge_require_evidence,
            key="real_judge_evidence",
        )

    with st.expander("并发与性能", expanded=False):
        p1, p2, p3 = st.columns(3)
        p1.number_input("并行评分数", 1, 64, key="cfg_workers")
        p2.number_input("维度并行数", 1, 8, key="cfg_judge_workers")
        p3.number_input("最大同时在途请求", 0, 256, key="cfg_max_in_flight")
    workers = int(st.session_state["cfg_workers"])
    judge_workers = int(st.session_state["cfg_judge_workers"])
    max_in_flight = int(st.session_state["cfg_max_in_flight"])

    _panel_start("② 确认并启动")
    _summary_card(
        "将对上传的真实对话进行评分",
        "不生成模拟对话；任务编号会直接匹配当前任务配置，并按完整任务规则评分。",
    )
    start = st.button("开始评测真实对话", type="primary", use_container_width=True)
    st.caption("评测在后台运行，跑完后去「评测报告」查看结果。")
    _panel_end()

    if start:
        if uploaded is None:
            st.error("请先上传真实对话 JSON 文件。")
            return
        run_id = f"real_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        out_dir = REPORTS_ROOT / run_id
        out_dir.mkdir(parents=True, exist_ok=True)
        upload_path = out_dir / "uploaded_dialogues.json"
        upload_path.write_bytes(uploaded.getvalue())
        log_path = out_dir / "webui_eval.log"
        cmd = [
            sys.executable, "-m", "src.cli.eval_run",
            "--run-id", run_id,
            "--real-dialogues", str(upload_path),
            "--judge", judge_mode,
            "--workers", str(workers),
            "--judge-workers", str(judge_workers),
            "--samples-workers", "1",
            "--max-in-flight", str(max_in_flight),
            "--judge-samples", str(judge_samples),
            "--judge-model", judge_model,
            "--seed", str(int(seed)),
        ]
        if judge_require_evidence:
            cmd.append("--verbose")
        log_file = log_path.open("w", encoding="utf-8")
        proc = subprocess.Popen(
            cmd, cwd=ROOT, stdout=log_file, stderr=subprocess.STDOUT, text=True,
        )
        _save_proc_meta(run_id, proc.pid, cmd, log_path)
        st.session_state["eval_run_id"] = run_id
        st.success(f"已启动真实对话评测：{run_id}（PID={proc.pid}）")
        st.code(" ".join(cmd), language="shell")
        st.rerun()


def page_evaluate() -> None:
    _page_header("发起评测", "第 1 步 · 让系统自动拨打并评分")
    _page_guide("发起评测")
    cfg = load_config()
    _render_active_runs(cfg)

    if "cfg_workers" not in st.session_state:
        st.session_state["cfg_workers"] = int(
            cfg.get("runtime", {}).get(
                "case_workers", cfg.get("runtime", {}).get("workers", 8)
            )
        )
    if "cfg_judge_workers" not in st.session_state:
        st.session_state["cfg_judge_workers"] = int(
            cfg.get("runtime", {}).get("judge_workers", 4)
        )
    if "cfg_samples_workers" not in st.session_state:
        st.session_state["cfg_samples_workers"] = int(
            cfg.get("runtime", {}).get("samples_workers") or 3
        )
    if "cfg_max_in_flight" not in st.session_state:
        st.session_state["cfg_max_in_flight"] = int(
            cfg.get("runtime", {}).get("max_in_flight", 32)
        )

    specs = load_specs(INSTRUCTIONS_DIR)
    if not specs:
        st.warning("请先解析指令。")
        return

    _panel_start("① 选择要测什么")
    c1, c2 = st.columns(2)
    with c1:
        instr_ids = st.multiselect(
            "外呼任务",
            [s.id for s in specs],
            default=[s.id for s in specs],
            format_func=lambda x: f"任务{x}",
        )
    with c2:
        _sig = _specs_signature()
        seed = int(cfg.get("runtime", {}).get("seed", 42))
        if instr_ids:
            sample_scenarios, _ = _cached_scenarios(instr_ids[0], seed, _sig)
            scenario_ids = st.multiselect(
                "用户场景",
                [s.id for s in sample_scenarios],
                default=[s.id for s in sample_scenarios],
                format_func=_scenario_label,
            )
        else:
            scenario_ids = []
    _panel_end()
    if not instr_ids:
        st.stop()
    chosen_specs = [s for s in specs if s.id in instr_ids]
    _sig = _specs_signature()

    run_mode = "full"
    sut_mode = "llm"
    user_mode = "llm"
    judge_mode = "llm"
    max_turns = int(cfg.get("runtime", {}).get("max_turns", 16))
    judge_samples = int(cfg.get("judge", {}).get("samples", 3))
    judge_require_evidence = bool(cfg.get("judge", {}).get("require_evidence", False))
    sut_model = cfg.get("models", {}).get("sut", "")
    user_sim_model = cfg.get("models", {}).get("user_sim", "")
    judge_model = cfg.get("models", {}).get("judge", "")
    reuse_trace_dir = ""
    use_cli_background = True

    with st.expander("模型与引擎", expanded=False):
        r1_mode, r1_name = st.columns([1, 1])
        with r1_mode:
            sut_mode = st.radio(
                "对话模型",
                ["llm", "stub"],
                horizontal=True,
                format_func=lambda x: "真实大模型" if x == "llm" else "离线模拟",
            )
        with r1_name:
            sut_model = st.text_input(
                "对话模型名称",
                cfg.get("models", {}).get("sut", ""),
                disabled=sut_mode == "stub",
            )
        r2_mode, r2_name = st.columns([1, 1])
        with r2_mode:
            user_mode = st.radio(
                "用户模拟",
                ["llm", "stub"],
                horizontal=True,
                format_func=lambda x: "真实大模型" if x == "llm" else "离线模拟",
            )
        with r2_name:
            user_sim_model = st.text_input(
                "用户模拟模型",
                cfg.get("models", {}).get("user_sim", ""),
                disabled=user_mode == "stub",
            )
        r3_mode, r3_name = st.columns([1, 1])
        with r3_mode:
            judge_mode = st.radio(
                "评分引擎",
                ["llm", "offline"],
                horizontal=True,
                format_func=lambda x: "大模型评分" if x == "llm" else "规则评分",
            )
        with r3_name:
            judge_model = st.text_input(
                "评分模型名称",
                cfg.get("models", {}).get("judge", ""),
                disabled=judge_mode == "offline",
            )

    with st.expander("运行参数", expanded=False):
        run_mode = st.selectbox(
            "评测阶段",
            ["full", "dialogue_only", "judge_only"],
            format_func={
                "full": "完整评测（对话 + 评分）",
                "dialogue_only": "仅生成对话记录",
                "judge_only": "仅重新评分（复用已有对话）",
            }.get,
        )
        c1, c2, c3 = st.columns(3)
        max_turns = c1.slider("单通最大轮次", 4, 30, int(cfg.get("runtime", {}).get("max_turns", 16)))
        judge_samples = c2.slider(
            "评分采样次数", 1, 5, int(cfg.get("judge", {}).get("samples", 3)),
        )
        seed = c3.number_input("随机种子", min_value=0, value=int(seed))
        judge_require_evidence = st.toggle(
            "评分需附原文证据", value=bool(cfg.get("judge", {}).get("require_evidence", False))
        )
        reuse_trace_dir = st.text_input("复用对话记录目录", value="")
        use_cli_background = st.toggle("后台运行（推荐）", value=True)

    with st.expander("并发与性能", expanded=False):
        preset_names = ["默认（推荐）", "高并发（自有大配额）", "低并发（共享配额）", "调试串行"]
        pc = st.columns(4)
        for col, name in zip(pc, preset_names):
            with col:
                if st.button(name, key=f"preset_{name}", use_container_width=True):
                    _apply_preset(name, cfg)
                    st.rerun()
        p1, p2, p3, p4 = st.columns(4)
        p1.number_input("并行对话数", 1, 64, key="cfg_workers")
        p2.number_input("并行评分数", 1, 8, key="cfg_judge_workers")
        p3.number_input("评分采样并行", 1, 5, key="cfg_samples_workers")
        p4.number_input("最大同时在途请求", 0, 256, key="cfg_max_in_flight")

    workers = int(st.session_state["cfg_workers"])
    judge_workers = int(st.session_state["cfg_judge_workers"])
    samples_workers = int(st.session_state["cfg_samples_workers"])
    max_in_flight = int(st.session_state["cfg_max_in_flight"])
    cache_dir = cfg.get("runtime", {}).get("cache_dir", "data/.llm_cache")
    sut_temp = cfg.get("generation", {}).get("sut_temperature", 0.4)
    user_temp = cfg.get("generation", {}).get("user_sim_temperature", 0.7)

    estimated_cases = sum(
        1
        for spec in chosen_specs
        for sc in _cached_scenarios(spec.id, int(seed), _sig)[0]
        if sc.id in scenario_ids
    )
    est = _estimate_run(
        estimated_cases,
        max_turns=max_turns,
        judge_samples=judge_samples,
        sut_model=sut_model or "deepseek-v4-flash",
        judge_model=judge_model or "deepseek-v4-pro",
        user_model=user_sim_model or "deepseek-v4-flash",
        max_in_flight=max_in_flight,
    )
    if run_mode == "dialogue_only":
        est["judge_calls"] = 0
        est["total_calls"] = est["dialogue_calls"]
    elif run_mode == "judge_only":
        est["dialogue_calls"] = 0
        est["total_calls"] = est["judge_calls"]
    if sut_mode == "stub":
        est["dialogue_calls"] = 0
    if judge_mode == "offline":
        est["judge_calls"] = 0
    est["total_calls"] = est["dialogue_calls"] + est["judge_calls"]
    est["eta_secs"] = int(
        est["total_calls"] * 5.0 / max(1, max_in_flight if max_in_flight > 0 else 1)
    )

    cost_text = f"¥{est['cost_yuan']:.2f}" if est["cost_yuan"] else "离线免费"
    st.markdown("<div class='ev-panel'><div class='ev-panel-title'>② 确认并启动</div>", unsafe_allow_html=True)
    _summary_card(
        f"将评测 {estimated_cases} 个测试项",
        f"预估费用 {cost_text} · 约 {_format_secs(est['eta_secs'])} · 完成后去「评测报告」查看结果",
    )
    start = st.button("开始评测", type="primary", use_container_width=True)
    st.caption("评测在后台运行，可切换到其他页面；跑完后去左侧「评测报告」查看。")
    st.markdown("</div>", unsafe_allow_html=True)
    if start:
        run_id = f"webui_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        out_dir = REPORTS_ROOT / run_id
        trace_dir = TRACES_ROOT / run_id
        if run_mode == "judge_only" and not reuse_trace_dir:
            st.error("「仅重新评分」模式需要填写复用对话记录目录。")
            return
        if not use_cli_background and run_mode != "full":
            st.error("页面内运行仅支持完整评测；其他模式请开启「后台运行」。")
            return
        if use_cli_background:
            out_dir.mkdir(parents=True, exist_ok=True)
            log_path = out_dir / "webui_eval.log"
            cmd = [
                sys.executable, "-m", "src.cli.eval_run",
                "--run-id", run_id,
                "--sut", sut_mode, "--user-sim", user_mode, "--judge", judge_mode,
                "--workers", str(int(workers)),
                "--judge-workers", str(int(judge_workers)),
                "--samples-workers", str(int(samples_workers)),
                "--max-in-flight", str(int(max_in_flight)),
                "--judge-samples", str(int(judge_samples)),
                "--max-turns", str(int(max_turns)),
                "--sut-model", sut_model,
                "--user-sim-model", user_sim_model,
                "--judge-model", judge_model,
                "--instructions", ",".join(instr_ids),
                "--scenarios", ",".join(scenario_ids),
                "--seed", str(int(seed)),
            ]
            if run_mode == "dialogue_only":
                cmd.append("--dialogue-only")
            if run_mode == "judge_only":
                cmd.extend(["--judge-only", "--reuse-traces", reuse_trace_dir])
            log_file = log_path.open("w", encoding="utf-8")
            proc = subprocess.Popen(
                cmd, cwd=ROOT, stdout=log_file, stderr=subprocess.STDOUT, text=True,
            )
            _save_proc_meta(run_id, proc.pid, cmd, log_path)
            st.session_state["eval_run_id"] = run_id
            st.success(f"已启动后台评测：{run_id}（PID={proc.pid}）")
            st.code(" ".join(cmd), language="shell")
            st.rerun()
        progress = st.progress(0.0)
        status = st.empty()
        sut_client = build_default_client(cache_dir=cache_dir, max_in_flight=max_in_flight or None) if sut_mode == "llm" else None
        user_client = build_default_client(cache_dir=cache_dir, max_in_flight=max_in_flight or None) if user_mode == "llm" else None
        judge_client = build_default_client(cache_dir=cache_dir, max_in_flight=max_in_flight or None) if judge_mode == "llm" else None
        sut = build_sut_client(use_stub=sut_mode == "stub", client=sut_client, model=sut_model, temperature=sut_temp, seed=int(seed))
        user_sim = build_user_simulator(use_stub=user_mode == "stub", client=user_client, model=user_sim_model, temperature=user_temp, seed=int(seed))
        cases = []
        total = 0
        tasks = []
        for spec in chosen_specs:
            scenarios, vmap = _cached_scenarios(spec.id, int(seed), _sig)
            for sc in scenarios:
                if sc.id not in scenario_ids:
                    continue
                tasks.append((spec, sc, vmap[sc.id]))
                total += 1
        run_started = time.time()
        max_sessions = int(cfg.get("orchestrator", {}).get("max_sessions", 1))
        retry_on_fail = bool(cfg.get("orchestrator", {}).get("retry_on_fail", False))
        case_gate = cfg.get("case_gate")
        sessions_cap = max_sessions if retry_on_fail else 1
        for i, (spec, sc, variables) in enumerate(tasks):
            status.write(f"运行 {spec.id}__{sc.id} ({i+1}/{total}) ...")
            case_id = f"{spec.id}__{sc.id}"
            prior_memory = None
            prior_sessions: list[dict] = []
            case = None
            trace = None
            for session_idx in range(1, sessions_cap + 1):
                trace = run_dialogue(
                    spec, sc, sut, user_sim,
                    variables=variables, run_id=run_id,
                    max_turns=max_turns,
                    seed=int(seed) + session_idx - 1,
                    trace_dir=None,
                    session_index=session_idx,
                    max_sessions=sessions_cap,
                    prior_memory=prior_memory,
                    prior_sessions=prior_sessions,
                )
                case = evaluate_case(
                    spec, sc, trace,
                    client=judge_client,
                    judge_model=judge_model,
                    judge_samples=judge_samples,
                    judge_temperature=cfg.get("judge", {}).get("temperature", 0.2),
                    judge_require_evidence=judge_require_evidence,
                    judge_workers=int(judge_workers),
                    samples_workers=int(samples_workers),
                    judge_seed=int(seed) + 17 + session_idx,
                    weights=cfg.get("metrics", {}).get("weights"),
                    case_gate=case_gate,
                    sessions_used=session_idx,
                )
                if case.passed or session_idx >= sessions_cap:
                    break
                prior_sessions.append(
                    {
                        "session": session_idx,
                        "score": case.weighted_total,
                        "passed": case.passed,
                        "summary": summarize_trace_for_memory(trace),
                    }
                )
                prior_memory = prior_sessions[-1]["summary"]
            if trace is not None:
                from src.core.orchestrator import save_trace

                trace.prior_sessions = prior_sessions
                save_trace(trace, trace_dir)
            cases.append(case)
            progress.progress((i + 1) / max(1, total))
        run_report = aggregate_run(
            run_id, cases, config={
                "models": {"sut": sut_model, "judge": judge_model, "user_sim": user_sim_model},
                "judge_backend": judge_mode,
                "sut_backend": sut_mode,
                "user_sim_backend": user_mode,
                "judge_samples": judge_samples,
                "case_workers": int(workers),
                "judge_workers": int(judge_workers),
                "samples_workers": int(samples_workers),
                "max_in_flight": int(max_in_flight),
                "judge_require_evidence": judge_require_evidence,
                "seed": int(seed),
                "max_turns": max_turns,
            },
            weights=cfg.get("metrics", {}).get("weights"),
            top_k_failures=cfg.get("report", {}).get("top_k_failures", 10),
        )
        paths = write_run_artifacts(run_report, out_dir)
        store = _store()
        store.register_run(run_id=run_id, config=run_report.config, sut_model=sut_model, judge_model=judge_model, user_sim_model=user_sim_model)
        for c in cases:
            store.add_case(
                case_id=c.trace.case_id,
                run_id=run_id,
                instruction_id=c.trace.instruction_id,
                scenario_id=c.trace.scenario_id,
                trace_path=str(trace_dir / f"{c.trace.case_id}.json"),
                weighted_total=c.weighted_total,
                passed=c.passed,
                report_path=str(paths["html"]),
            )
        store.save_report(run_id=run_id, aggregate=run_report.aggregate, html_path=str(paths["html"]), md_path=str(paths["md"]))
        elapsed = int(time.time() - run_started)
        st.success(
            f"评测完成 · 批次 {_format_run_label(run_id)} · 耗时 {_format_secs(elapsed)}"
        )
        agg = run_report.aggregate
        st.write(
            f"场景通过率 {agg.get('case_pass_rate', 0):.1%} "
            f"（{agg.get('n_passed', 0)}/{agg.get('n_cases', 0)} 项通过）· "
            f"综合得分 {agg['overall_mean']:.3f}，"
            f"置信区间 [{agg['confidence_interval'][0]:.3f}, "
            f"{agg['confidence_interval'][1]:.3f}]"
        )
        st.session_state["last_run"] = run_id


def page_evaluate_real() -> None:
    _page_header("上传真实对话", "第 1 步 · 上传已有对话并评分")
    _page_guide("上传真实对话")
    cfg = load_config()
    _render_active_runs(cfg)
    specs = load_specs(INSTRUCTIONS_DIR)
    if not specs:
        st.warning("请先解析指令。")
        return
    _page_evaluate_real_dialogues(cfg, specs)


def _quality_gate_badge(
    overall: float,
    passed: bool,
    *,
    pass_rate: Optional[float] = None,
    n_passed: int = 0,
    n_cases: int = 0,
) -> str:
    if passed:
        bg, text = "#16a34a", "white"
        label = "全部场景通过"
    elif (pass_rate or 0) >= 0.6:
        bg, text = "#ca8a04", "white"
        label = "部分场景未通过"
    else:
        bg, text = "#dc2626", "white"
        label = "多数场景未通过"
    rate_text = (
        f"通过率 {pass_rate * 100:.0f}% ({n_passed}/{n_cases})"
        if pass_rate is not None and n_cases
        else ""
    )
    return (
        f"<div style='background:{bg};color:{text};padding:14px 18px;"
        f"border-radius:8px;display:flex;align-items:center;justify-content:space-between;'>"
        f"<span style='font-size:1.1rem;font-weight:600;'>{label}"
        + (f"<br><span style='font-size:.85rem;opacity:.9;'>{rate_text}</span>" if rate_text else "")
        + f"</span>"
        f"<span style='font-size:1.8rem;font-weight:700;'>{overall:.3f}</span>"
        f"</div>"
    )


def _case_status_pill(passed: bool, score: float) -> str:
    if passed:
        return _pill(f"通过 {score:.3f}", "good")
    return _pill(f"未通过 {score:.3f}", "bad")


def _judge_hits_for_turn(case_report: Optional[dict], turn_index: int) -> list[dict]:
    if not case_report:
        return []
    hits: list[dict] = []
    for dim in case_report.get("dimensions", []):
        for detail in dim.get("details", []):
            if turn_index in detail.get("turn_ids", []):
                hits.append(
                    {
                        "dim_id": dim.get("id"),
                        "dim_name": dim.get("name"),
                        "passed": detail.get("passed", True),
                        "label": detail.get("label", ""),
                        "rationale": detail.get("rationale", ""),
                        "evidence": detail.get("evidence_quote", ""),
                        "disagreement": detail.get("disagreement"),
                    }
                )
    return hits


def _render_usj_turn(
    turn_index: int,
    user_text: str,
    sut_text: str,
    judge_hits: list[dict],
) -> None:
    failed = [h for h in judge_hits if not h.get("passed", True)]
    judge_cls = "judge ok" if not failed else "judge"
    judge_html = "<div style='color:#64748b;font-size:.9rem;'>本轮无扣分项</div>"
    if judge_hits:
        items = []
        for h in judge_hits:
            mark = "✓" if h.get("passed", True) else "✗"
            color = "#16a34a" if h.get("passed", True) else "#dc2626"
            disagree = ""
            if h.get("disagreement") is not None:
                disagree = f" · 分歧 {h['disagreement']:.0%}"
            items.append(
                f"<div style='color:{color};margin-bottom:6px;'>"
                f"<b>[{mark}] {h.get('dim_name') or h.get('dim_id')}</b> {h.get('label')}{disagree}"
                f"<div style='color:#475569;font-size:.85rem;'>{h.get('rationale', '')}</div>"
                + (
                    f"<div style='color:#334155;font-size:.82rem;font-style:italic;'>"
                    f"「{h.get('evidence', '')[:80]}」</div>"
                    if h.get("evidence")
                    else ""
                )
                + "</div>"
            )
        judge_html = "".join(items)
    st.markdown(
        f"<div style='color:#64748b;font-size:.82rem;margin:12px 0 4px;'>轮次 {turn_index:02d}</div>"
        f"<div class='ev-usj-row'>"
        f"<div class='ev-usj-col user'><h4>用户说什么</h4>"
        f"<div style='white-space:pre-wrap;color:#1e293b;'>{user_text or '—'}</div></div>"
        f"<div class='ev-usj-col sut'><h4>模型说什么</h4>"
        f"<div style='white-space:pre-wrap;color:#1e293b;'>{sut_text or '—'}</div></div>"
        f"<div class='ev-usj-col {judge_cls}'><h4>评分结果</h4>{judge_html}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )


def _render_trace_body(trace: Any, case_report: Optional[dict], case_pick: str) -> None:
    turn_count = sum(1 for t in trace.turns if t.role == "assistant")
    total_latency = sum(t.latency_ms for t in trace.turns)
    tokens_in = sum(t.tokens_in for t in trace.turns)
    tokens_out = sum(t.tokens_out for t in trace.turns)
    score_val = case_report.get("weighted_total") if case_report else None
    passed_val = case_report.get("passed") if case_report else None
    dim_count = len(case_report.get("dimensions", [])) if case_report else 0
    fail_attr_count = (
        len(case_report.get("failure_attribution", [])) if case_report else 0
    )

    st.markdown("<div class='ev-trace-detail-stack'>", unsafe_allow_html=True)
    _render_trace_detail_overview(
        turn_count=turn_count,
        score=score_val,
        passed=passed_val,
        dim_count=dim_count,
        fail_attr_count=fail_attr_count,
        total_latency_ms=total_latency,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
    )

    with _trace_detail_section("dialogue", "逐轮对话", f"共 {turn_count} 轮"):
        last_user = "（等待用户回应）"
        for turn in trace.turns:
            if turn.role == "assistant":
                hits = _judge_hits_for_turn(case_report, turn.index)
                _render_usj_turn(turn.index, last_user, turn.content, hits)
            elif turn.role == "user":
                last_user = turn.content

    if case_report:
        with _trace_detail_section("score", "评分详情", f"{dim_count} 项维度"):
            _render_trace_dimension_cards(case_report.get("dimensions", []))
            if case_report.get("failure_attribution"):
                st.markdown("**扣分归因**")
                for item in case_report.get("failure_attribution", []):
                    item_with_meta = dict(item)
                    item_with_meta.setdefault("instruction_id", trace.instruction_id)
                    item_with_meta.setdefault("scenario_id", trace.scenario_id)
                    st.markdown(_failure_card(item_with_meta), unsafe_allow_html=True)

    with _trace_detail_section("perf", "性能数据与下载", f"耗时 {total_latency} ms"):
        _render_trace_perf_metrics(total_latency, tokens_in, tokens_out)
        if any(t.latency_ms for t in trace.turns):
            sut_x, sut_y, usr_x, usr_y = [], [], [], []
            for t in trace.turns:
                if t.role == "assistant":
                    sut_x.append(t.index)
                    sut_y.append(t.latency_ms)
                elif t.role == "user":
                    usr_x.append(t.index)
                    usr_y.append(t.latency_ms)
            lat_fig = go.Figure()
            if sut_x:
                lat_fig.add_trace(go.Scatter(
                    x=sut_x, y=sut_y, mode="lines+markers", name="对话模型",
                    line=dict(color="#2563eb"),
                ))
            if usr_x:
                lat_fig.add_trace(go.Scatter(
                    x=usr_x, y=usr_y, mode="lines+markers", name="用户模拟",
                    line=dict(color="#f59e0b"),
                ))
            lat_fig.update_layout(
                xaxis_title="对话轮次", yaxis_title="响应耗时（毫秒）",
                height=220, margin=dict(l=10, r=10, t=20, b=10),
            )
            st.plotly_chart(lat_fig, use_container_width=True)
        if trace.variables:
            st.json(trace.variables)
        if trace.prior_sessions:
            st.markdown("**续拨记忆**")
            for sess in trace.prior_sessions:
                st.caption(
                    f"第 {sess.get('session')} 次 · 得分 {sess.get('score', '—')} · "
                    f"{'通过' if sess.get('passed') else '未通过'}"
                )
        st.download_button(
            "下载对话记录",
            data=trace.model_dump_json(indent=2),
            file_name=f"{case_pick}.json",
            mime="application/json",
        )
    st.markdown("</div>", unsafe_allow_html=True)


def _radar_chart(dims: dict[str, dict]) -> go.Figure:
    order = ["gsr", "pcr", "bca", "kar", "hcr", "scr", "ter", "rob"]
    labels = [DIMENSION_NAMES.get(k, k) for k in order]
    scores = [dims.get(k, {}).get("mean", 0.0) or 0.0 for k in order]
    closed_labels = labels + [labels[0]]
    closed_scores = scores + [scores[0]]
    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=closed_scores, theta=closed_labels, fill="toself",
            name="本次", line=dict(color="#2563eb"),
        )
    )
    fig.add_trace(
        go.Scatterpolar(
            r=[0.6] * len(closed_labels), theta=closed_labels,
            name="合格线 0.6", line=dict(color="#dc2626", dash="dash"),
        )
    )
    fig.add_trace(
        go.Scatterpolar(
            r=[0.85] * len(closed_labels), theta=closed_labels,
            name="优秀线 0.85", line=dict(color="#16a34a", dash="dot"),
        )
    )
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        height=400,
        margin=dict(l=40, r=40, t=20, b=20),
    )
    return fig


def _dim_bar(dims: dict[str, dict]) -> go.Figure:
    order = ["gsr", "pcr", "bca", "kar", "hcr", "scr", "ter", "rob"]
    rows = [(DIMENSION_NAMES.get(k, k), dims.get(k, {}).get("mean", 0.0) or 0.0) for k in order]
    rows.sort(key=lambda r: r[1])
    names = [r[0] for r in rows]
    values = [r[1] for r in rows]
    colors = ["#16a34a" if v >= 0.85 else ("#ca8a04" if v >= 0.6 else "#dc2626") for v in values]
    fig = go.Figure(
        go.Bar(
            x=values, y=names, orientation="h",
            marker_color=colors,
            text=[f"{v:.3f}" for v in values],
            textposition="auto",
        )
    )
    fig.add_vline(x=0.6, line=dict(color="#dc2626", dash="dash"), annotation_text="合格 0.6")
    fig.add_vline(x=0.85, line=dict(color="#16a34a", dash="dot"), annotation_text="优秀 0.85")
    fig.update_layout(
        xaxis=dict(range=[0, 1.05], title="均值"),
        height=320, margin=dict(l=10, r=10, t=20, b=20),
    )
    return fig


def _scenario_dim_heatmap(cases: list[dict]) -> Optional[go.Figure]:
    """Aggregate case-level dimension scores into a (scenario × dim) heatmap.

    Input is the list of case dicts from RunReport (run_report.json `cases`).
    Returns None when there's nothing to plot.
    """
    if not cases:
        return None
    order = ["gsr", "pcr", "bca", "kar", "hcr", "scr", "ter", "rob"]
    bucket: dict[str, dict[str, list[float]]] = {}
    for c in cases:
        sc_id = (c.get("trace") or {}).get("scenario_id")
        if not sc_id:
            continue
        for d in c.get("dimensions", []):
            did = d.get("id")
            score = d.get("score")
            if did is None or score is None:
                continue
            bucket.setdefault(sc_id, {}).setdefault(did, []).append(float(score))
    if not bucket:
        return None
    scenarios = sorted(bucket.keys())
    scen_labels = [_scenario_label(sc_id) for sc_id in scenarios]
    z = []
    text = []
    for sc_id in scenarios:
        z_row = []
        t_row = []
        for did in order:
            vals = bucket[sc_id].get(did, [])
            if vals:
                m = sum(vals) / len(vals)
                z_row.append(round(m, 3))
                t_row.append(f"{m:.3f}")
            else:
                z_row.append(None)
                t_row.append("-")
        z.append(z_row)
        text.append(t_row)
    dim_labels = [DIMENSION_NAMES.get(k, k) for k in order]
    fig = go.Figure(
        go.Heatmap(
            z=z, x=dim_labels, y=scen_labels,
            text=text, texttemplate="%{text}", textfont=dict(size=12),
            colorscale=[
                [0.0, COLOR["bad"]],
                [0.6, "#fef3c7"],
                [0.85, "#bbf7d0"],
                [1.0, COLOR["good"]],
            ],
            zmin=0, zmax=1,
            colorbar=dict(title="均值", thickness=12),
            hovertemplate="场景 %{y}<br>维度 %{x}<br>均值 %{z:.3f}<extra></extra>",
        )
    )
    fig.update_layout(
        height=max(280, 36 * len(scenarios) + 80),
        margin=dict(l=10, r=10, t=20, b=10),
        xaxis=dict(side="top"),
        yaxis=dict(autorange="reversed"),
    )
    return fig


def _failure_card(item: dict) -> str:
    dim = item.get("dim_name") or item.get("dim_id", "?")
    label = item.get("label", "")
    weighted = item.get("weighted_loss", 0.0)
    deduction = item.get("deduction", 0.0)
    evidence = (item.get("evidence_quote") or "").strip()
    rationale = (item.get("rationale") or "").strip()
    turns = item.get("turn_ids") or []
    instr = item.get("instruction_id", "")
    scen = item.get("scenario_id", "")
    badge_color = "#dc2626" if weighted >= 0.05 else "#ca8a04"
    return (
        f"<div style='border-left:4px solid {badge_color};padding:8px 12px;margin-bottom:8px;background:#f8fafc;'>"
        f"<div style='display:flex;justify-content:space-between;'>"
        f"<span><b>[{dim}]</b> {label}</span>"
        f"<span><b>−{weighted:.3f}</b> · 扣分 {deduction:.2f}</span>"
        f"</div>"
        f"<div style='color:#475569;font-size:0.85rem;margin:4px 0;'>"
        f"任务 {instr} · {_scenario_label(scen)} · 涉及轮次 {turns}</div>"
        + (f"<blockquote style='border-left:3px solid #cbd5e1;margin:4px 0;padding-left:8px;color:#334155;'>{evidence}</blockquote>" if evidence else "")
        + (f"<div style='color:#0f172a;'>{rationale}</div>" if rationale else "")
        + "</div>"
    )


def _render_reports_page(*, env: str) -> None:
    if env == EVAL_ENV_REAL:
        _page_header("真实评测报告", "第 2 步 · 看真实对话分数、找问题")
        _page_guide("真实评测报告")
    else:
        _page_header("评测报告", "第 2 步 · 看分数、找问题")
        _page_guide("评测报告")
    store = _store()
    runs = _filter_runs_for_env(store.list_runs(), env, _load_run_report)
    if not runs:
        target = "上传真实对话" if env == EVAL_ENV_REAL else "发起评测"
        st.info(f"尚无评测记录，请先在「{target}」页面启动一次评测。")
        return
    options = [r["run_id"] for r in runs]
    pick = st.selectbox(
        "选择评测批次",
        options,
        index=0,
        format_func=_format_run_label,
    )
    info = next(r for r in runs if r["run_id"] == pick)
    report_data = _load_run_report(pick)
    real_mode = _is_real_dialogue_run(report_data, info)
    agg = report_data.get("aggregate", {})
    if agg:
        qg = agg.get("quality_gates", {})
        overall = float(agg.get("overall_mean", 0.0))
        st.markdown(
            _quality_gate_badge(
                overall,
                bool(qg.get("passed")),
                pass_rate=agg.get("case_pass_rate"),
                n_passed=int(agg.get("n_passed", 0)),
                n_cases=int(agg.get("n_cases", 0)),
            ),
            unsafe_allow_html=True,
        )
        n_failed = int(agg.get("n_failed", 0))
        ci = agg.get("confidence_interval", [0, 0])
        _summary_card(
            f"共 {agg.get('n_cases', 0)} 项测试，{int(agg.get('n_passed', 0))} 项通过",
            f"置信区间 {ci[0]:.3f} ~ {ci[1]:.3f}"
            + (f" · {n_failed} 项未通过" if n_failed else ""),
        )

    cases = store.list_cases(pick)
    case_report_map = {
        c.get("trace", {}).get("case_id"): c for c in report_data.get("cases", [])
    }

    with st.expander(f"测试项明细（{len(cases)} 项）", expanded=False):
        case_filter = st.text_input("搜索任务或场景", value="", key="rpt_filter")
        show_low = st.checkbox("只看评分不确定或异常的项", value=False, key="rpt_low")
        flagged = {
            item.get("case_id")
            for item in (agg.get("low_confidence", []) + agg.get("fallback_or_warnings", []))
        }
        filtered_cases = []
        for row in cases:
            blob = f"{row.get('case_id')} {row.get('instruction_id')} {row.get('scenario_id')}"
            if case_filter and case_filter not in blob:
                continue
            if show_low and row.get("case_id") not in flagged:
                continue
            cr = case_report_map.get(row.get("case_id"), {})
            row = dict(row)
            row["passed"] = cr.get("passed", row.get("passed"))
            row["sessions_used"] = cr.get("sessions_used", 1)
            filtered_cases.append(row)
        if filtered_cases:
            display_rows = []
            for row in filtered_cases:
                cr = case_report_map.get(row["case_id"], {})
                passed = cr.get("passed", row.get("passed"))
                score = cr.get("weighted_total", row.get("weighted_total"))
                if real_mode:
                    display_rows.append({
                        "对话 ID": row["case_id"],
                        "任务编号": row.get("instruction_id", "—"),
                        "得分": f"{score:.3f}" if score is not None else "—",
                        "结果": "通过" if passed else "未通过",
                    })
                else:
                    display_rows.append({
                        "测试项": _format_case_id(row["case_id"]),
                        "得分": f"{score:.3f}" if score is not None else "—",
                        "结果": "通过" if passed else "未通过",
                    })
            st.dataframe(display_rows, use_container_width=True, hide_index=True)
            fail_rows = [r for r in filtered_cases if not case_report_map.get(r["case_id"], {}).get("passed")]
            if fail_rows:
                st.caption("未通过项可跳转复盘：")
                cols = st.columns(min(3, len(fail_rows)))
                for i, row in enumerate(fail_rows[:9]):
                    if cols[i % len(cols)].button(
                        _format_case_display(
                            row["case_id"],
                            row.get("instruction_id", ""),
                            real_mode=real_mode,
                        ),
                        key=f"replay_{pick}_{row['case_id']}",
                    ):
                        _navigate_to_trace(pick, row["case_id"], env=env)
        else:
            st.info("没有匹配的测试项。")

    meta = store.get_report_meta(pick)
    if meta:
        if not agg:
            agg = json.loads(meta["aggregate_json"])
        dims_summary = agg.get("dimensions", {})
        if dims_summary:
            with st.expander("维度分析", expanded=False):
                chart_type = st.radio(
                    "图表类型", ["雷达图", "条形图", "热力图", "数据表"],
                    horizontal=True, label_visibility="collapsed", key="rpt_chart",
                )
                if chart_type == "雷达图":
                    st.plotly_chart(_radar_chart(dims_summary), use_container_width=True)
                elif chart_type == "条形图":
                    st.plotly_chart(_dim_bar(dims_summary), use_container_width=True)
                elif chart_type == "热力图" and not real_mode:
                    heat_fig = _scenario_dim_heatmap(report_data.get("cases", []))
                    if heat_fig is None:
                        st.info("当前批次暂无维度数据。")
                    else:
                        st.plotly_chart(heat_fig, use_container_width=True)
                elif chart_type == "热力图":
                    st.info("真实对话模式不区分用户类型，已隐藏场景热力图。")
                else:
                    st.dataframe([
                        {"维度": v.get("name", k), "均值": v.get("mean"),
                         "最低": v.get("min"), "最高": v.get("max")}
                        for k, v in dims_summary.items()
                    ], use_container_width=True, hide_index=True)

        top_failures = agg.get("top_failures", [])
        if top_failures:
            with st.expander(f"主要扣分项（{len(top_failures)} 项）", expanded=False):
                for item in top_failures:
                    st.markdown(_failure_card(item), unsafe_allow_html=True)

        with st.expander("导出与高级信息", expanded=False):
            if agg.get("fallback_or_warnings") or agg.get("low_confidence"):
                st.markdown("**评分可靠性提示**")
                st.dataframe(
                    [{"类型": "评分降级/警告", **item} for item in agg.get("fallback_or_warnings", [])]
                    + [{"类型": "低置信评分", **item} for item in agg.get("low_confidence", [])],
                    use_container_width=True,
                )
            st.write({k: v for k, v in info.items() if k != "aggregate_json"})
            html_path = meta.get("html_path")
            html = None
            if html_path and Path(html_path).exists():
                html = _compact_report_html(Path(html_path).read_text(encoding="utf-8"))
            md = None
            if meta.get("md_path") and Path(meta["md_path"]).exists():
                md = Path(meta["md_path"]).read_text(encoding="utf-8")
            raw_dialogues = _build_raw_dialogue_export(report_data)

            st.markdown(
                f"""
                <div class="ev-export-head">
                  <div>
                    <div class="title">报告导出</div>
                    <div class="desc">选择适合后续查看、归档或数据分析的文件格式。</div>
                  </div>
                  <div class="badge">{len(raw_dialogues)} 组对话</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            c_html, c_md, c_raw = st.columns(3)
            with c_html:
                with st.container(border=True):
                    st.markdown(
                        "<div class='ev-export-card'><div class='kind'>HTML 报告</div>"
                        "<div class='hint'>适合浏览器查看，保留完整排版和图表预览。</div></div>",
                        unsafe_allow_html=True,
                    )
                    if html:
                        st.download_button(
                            "下载 HTML",
                            data=html,
                            file_name=f"{pick}.html",
                            mime="text/html",
                            use_container_width=True,
                        )
                    else:
                        st.caption("暂无 HTML 文件")
            with c_md:
                with st.container(border=True):
                    st.markdown(
                        "<div class='ev-export-card'><div class='kind'>Markdown</div>"
                        "<div class='hint'>适合沉淀到文档，便于复制、审阅和版本留痕。</div></div>",
                        unsafe_allow_html=True,
                    )
                    if md:
                        st.download_button(
                            "下载 Markdown",
                            data=md,
                            file_name=f"{pick}.md",
                            mime="text/markdown",
                            use_container_width=True,
                        )
                    else:
                        st.caption("暂无 Markdown 文件")
            with c_raw:
                with st.container(border=True):
                    st.markdown(
                        "<div class='ev-export-card primary'><div class='kind'>原始对话 JSON</div>"
                        "<div class='hint'>仅包含 id、任务编号、多轮对话，已去除模型运行元信息。</div></div>",
                        unsafe_allow_html=True,
                    )
                    if raw_dialogues:
                        st.download_button(
                            "导出原始对话",
                            data=json.dumps(raw_dialogues, ensure_ascii=False, indent=2),
                            file_name=f"{pick}_raw_dialogues.json",
                            mime="application/json",
                            type="primary",
                            use_container_width=True,
                        )
                    else:
                        st.caption("暂无可导出的原始对话")

            if html:
                with st.expander("内嵌 HTML 预览"):
                    st.components.v1.html(html, height=600, scrolling=True)
            with st.expander("原始 JSON 数据"):
                st.json(agg)


def page_reports() -> None:
    _render_reports_page(env=EVAL_ENV_GENERATED)


def page_reports_real() -> None:
    _render_reports_page(env=EVAL_ENV_REAL)


def _render_trace_page(*, env: str) -> None:
    if env == EVAL_ENV_REAL:
        _page_header("真实对话复盘", "第 3 步 · 按 ID 查看真实对话")
        _page_guide("真实对话复盘")
    else:
        _page_header("对话复盘", "第 3 步 · 看具体哪里说错了")
        _page_guide("对话复盘")
    store = _store()
    runs = _filter_runs_for_env(store.list_runs(), env, _load_run_report)
    if not runs:
        st.info("尚无评测记录。")
        return
    run_ids = [r["run_id"] for r in runs]
    default_run = st.session_state.get("trace_run_id", run_ids[0])
    if default_run not in run_ids:
        default_run = run_ids[0]

    with st.container(border=True):
        st.markdown(
            "<div class='ev-filter-head batch'>① 选择评测批次</div>",
            unsafe_allow_html=True,
        )
        st.markdown("<div class='ev-filter-select-host batch'></div>", unsafe_allow_html=True)
        pick = st.selectbox(
            "评测批次",
            run_ids,
            index=run_ids.index(default_run),
            format_func=_format_run_label,
        )
        cases = store.list_cases(pick)
        if not cases:
            st.info("该批次暂无测试项。")
            return

    report_data_early = _load_run_report(pick)
    real_mode = _is_real_dialogue_run(report_data_early)
    case_reports = {
        c.get("trace", {}).get("case_id"): c
        for c in report_data_early.get("cases", [])
        if c.get("trace", {}).get("case_id")
    }

    if _trace_selector_mode_for_env(env) == "dialogue_id":
        default_case = st.session_state.get("trace_case_id", cases[0]["case_id"])
        case_ids = [row["case_id"] for row in cases]
        if default_case not in case_ids:
            default_case = case_ids[0]
        with st.container(border=True):
            st.markdown(
                "<div class='ev-filter-head task'>② 选择真实对话</div>",
                unsafe_allow_html=True,
            )
            case_pick = st.selectbox(
                "对话 ID",
                case_ids,
                index=case_ids.index(default_case),
                format_func=lambda cid: _format_case_display(
                    cid,
                    next((c.get("instruction_id", "") for c in cases if c["case_id"] == cid), ""),
                    real_mode=True,
                ),
                key=f"trace_real_case_sel_{pick}",
            )
            st.session_state["trace_case_id"] = case_pick
        case_meta = next(c for c in cases if c["case_id"] == case_pick)
        trace_path = Path(case_meta["trace_path"])
        if not trace_path.exists():
            st.error(f"对话记录文件缺失：{trace_path}")
            return
        trace = load_trace(trace_path)
        case_report = case_reports.get(case_pick)
        score_val = case_report.get("weighted_total") if case_report else None
        passed_val = case_report.get("passed") if case_report else None
        _render_trace_result_bar(
            scenario_label=f"真实对话 {case_pick}",
            passed=passed_val,
            score=score_val,
            meta_line=(
                f"任务{trace.instruction_id} · {_termination_label(trace.terminated_by)} · "
                "完整任务规则评分"
            ),
        )
        if case_report and case_report.get("fail_reasons"):
            st.error("未通过原因：" + "；".join(case_report["fail_reasons"][:5]))
        _render_trace_body(trace, case_report, case_pick)
        return

    by_instr, case_id_map = _case_index(cases)
    if not by_instr:
        st.info("该批次暂无有效测试项。")
        return

    default_case = st.session_state.get("trace_case_id", cases[0]["case_id"])
    default_instr, default_scen = _parse_case_id(default_case)
    instr_ids = sorted(
        by_instr.keys(),
        key=lambda x: (not x.isdigit(), int(x) if x.isdigit() else x),
    )
    if default_instr not in instr_ids:
        default_instr = instr_ids[0]

    sync_key = f"{pick}:{default_case}"
    if st.session_state.get("_trace_sync") != sync_key:
        st.session_state["trace_pick_instr"] = default_instr
        st.session_state["trace_pick_scen"] = default_scen
        st.session_state["_trace_sync"] = sync_key

    pick_instr = st.session_state.get("trace_pick_instr", default_instr)
    if pick_instr not in instr_ids:
        pick_instr = instr_ids[0]

    with st.container(border=True):
        st.markdown(
            "<div class='ev-filter-head task'>② 选择外呼任务</div>",
            unsafe_allow_html=True,
        )
        st.markdown("<div class='ev-filter-select-host task'></div>", unsafe_allow_html=True)
        pick_instr = st.selectbox(
            "外呼任务",
            instr_ids,
            index=instr_ids.index(pick_instr),
            format_func=_trace_task_label,
            key=f"trace_task_sel_{pick}",
        )
        st.session_state["trace_pick_instr"] = pick_instr
        task_desc = _trace_task_desc(pick_instr)
        if task_desc:
            st.markdown(
                f"<div class='ev-filter-task-desc'>{task_desc}</div>",
                unsafe_allow_html=True,
            )

    scen_ids = by_instr[pick_instr]
    pick_scen = st.session_state.get("trace_pick_scen", default_scen)
    if pick_scen not in scen_ids:
        pick_scen = scen_ids[0]

    with st.container(border=True):
        st.markdown(
            "<div class='ev-filter-head'>③ 选择用户类型</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<div class='ev-filter-hint'>"
            "在下方表格中点击一行选择用户类型；绿色=通过，红色=未通过"
            "</div>",
            unsafe_allow_html=True,
        )
        scen_key = f"trace_scen_{pick}_{pick_instr}"
        if scen_key not in st.session_state or st.session_state[scen_key] not in scen_ids:
            init = pick_scen if pick_scen in scen_ids else scen_ids[0]
            st.session_state[scen_key] = init
        pick_scen = st.session_state.get(scen_key, pick_scen)
        pick_scen = _render_scenario_picker(
            scen_ids,
            pick_scen,
            scen_key,
            pick_instr,
            case_id_map,
            case_reports,
        )
        st.session_state["trace_pick_scen"] = pick_scen

    case_pick = case_id_map[(pick_instr, pick_scen)]
    st.session_state["trace_case_id"] = case_pick
    case_meta = next(c for c in cases if c["case_id"] == case_pick)
    trace_path = Path(case_meta["trace_path"])
    if not trace_path.exists():
        st.error(f"对话记录文件缺失：{trace_path}")
        return
    trace = load_trace(trace_path)
    report_data = report_data_early
    case_report = next(
        (
            c
            for c in report_data.get("cases", [])
            if c.get("trace", {}).get("case_id") == case_pick
        ),
        None,
    )
    score_val = case_report.get("weighted_total") if case_report else None
    passed_val = case_report.get("passed") if case_report else None
    task_title = _trace_task_label(pick_instr)
    _render_trace_result_bar(
        scenario_label=_scenario_label(pick_scen),
        passed=passed_val,
        score=score_val,
        meta_line=(
            f"{task_title} · {_termination_label(trace.terminated_by)} · "
            f"第 {trace.session_index}/{trace.max_sessions} 次通话"
        ),
    )
    if case_report and case_report.get("fail_reasons"):
        st.error("未通过原因：" + "；".join(case_report["fail_reasons"][:5]))

    _render_trace_body(trace, case_report, case_pick)


def page_trace() -> None:
    _render_trace_page(env=EVAL_ENV_GENERATED)


def page_trace_real() -> None:
    _render_trace_page(env=EVAL_ENV_REAL)


def page_compare() -> None:
    _page_header("版本对比", "可选 · 对比模型升级前后的效果")
    _page_guide("版本对比")
    store = _store()
    runs = store.list_runs()
    if len(runs) < 2:
        st.info("至少需要两次评测记录才能对比。")
        return
    options = [r["run_id"] for r in runs]
    c1, c2 = st.columns(2)
    with c1:
        a = st.selectbox("基准批次", options, index=0, format_func=_format_run_label)
    with c2:
        b = st.selectbox("对比批次", options, index=min(1, len(options) - 1), format_func=_format_run_label)
    meta_a = store.get_report_meta(a)
    meta_b = store.get_report_meta(b)
    if not meta_a or not meta_b:
        st.error("缺少汇总数据，无法比较。")
        return
    agg_a = json.loads(meta_a["aggregate_json"])
    agg_b = json.loads(meta_b["aggregate_json"])
    delta = agg_b["overall_mean"] - agg_a["overall_mean"]
    trend = "提升" if delta > 0.01 else ("下降" if delta < -0.01 else "持平")
    _summary_card(
        f"综合得分 {agg_b['overall_mean']:.3f}（{trend} {delta:+.3f}）",
        f"基准 {_format_run_label(a)} → 对比 {_format_run_label(b)}",
    )

    dims_a = agg_a.get("dimensions", {})
    dims_b = agg_b.get("dimensions", {})
    rows = []
    for dim_id in sorted(set(dims_a) | set(dims_b)):
        ma = dims_a.get(dim_id, {}).get("mean")
        mb = dims_b.get(dim_id, {}).get("mean")
        rows.append({
            "dim_id": dim_id,
            "维度": DIMENSION_NAMES.get(dim_id, dim_id),
            "基准": ma,
            "对比": mb,
            "变化": (mb or 0) - (ma or 0) if (ma is not None and mb is not None) else None,
        })
    diffs = [r for r in rows if r.get("变化") is not None]
    if diffs:
        with st.expander("各维度变化", expanded=False):
            diffs.sort(key=lambda r: r["变化"])
            names = [r["维度"] for r in diffs]
            deltas = [r["变化"] for r in diffs]
            colors = ["#16a34a" if d > 0 else ("#dc2626" if d < 0 else "#94a3b8") for d in deltas]
            bar = go.Figure(go.Bar(
                x=deltas, y=names, orientation="h",
                marker_color=colors,
                text=[f"{d:+.3f}" for d in deltas], textposition="auto",
            ))
            bar.add_vline(x=0, line=dict(color="#475569"))
            bar.update_layout(
                xaxis_title="得分变化", height=300,
                margin=dict(l=10, r=10, t=20, b=20),
            )
            st.plotly_chart(bar, use_container_width=True)
            st.dataframe(
                [{k: v for k, v in r.items() if k != "dim_id"} for r in rows],
                use_container_width=True, hide_index=True,
            )

    data_a = _load_run_report(a)
    data_b = _load_run_report(b)
    if data_a and data_b:
        cases_a = {c.get("trace", {}).get("case_id"): c for c in data_a.get("cases", [])}
        cases_b = {c.get("trace", {}).get("case_id"): c for c in data_b.get("cases", [])}
        case_rows = []
        for case_id in sorted(set(cases_a) & set(cases_b)):
            total_a = cases_a[case_id].get("weighted_total", 0)
            total_b = cases_b[case_id].get("weighted_total", 0)
            case_rows.append({
                "测试项": _format_case_id(case_id),
                "基准得分": total_a,
                "对比得分": total_b,
                "变化": total_b - total_a,
            })
        if case_rows:
            with st.expander(f"各测试项对比（{len(case_rows)} 项）", expanded=False):
                xs = [r["基准得分"] for r in case_rows]
                ys = [r["对比得分"] for r in case_rows]
                ids = [r["测试项"] for r in case_rows]
                colors = [
                    "#16a34a" if r["变化"] > 0.01
                    else ("#dc2626" if r["变化"] < -0.01 else "#94a3b8")
                    for r in case_rows
                ]
                scatter = go.Figure()
                scatter.add_trace(go.Scatter(
                    x=xs, y=ys, mode="markers", text=ids,
                    marker=dict(size=9, color=colors, line=dict(color="#1f2937", width=1)),
                    hovertemplate="<b>%{text}</b><br>基准=%{x:.3f}<br>对比=%{y:.3f}<extra></extra>",
                ))
                scatter.add_trace(go.Scatter(
                    x=[0, 1], y=[0, 1], mode="lines",
                    line=dict(color="#94a3b8", dash="dash"), showlegend=False,
                ))
                scatter.update_layout(
                    xaxis=dict(title="基准得分", range=[0, 1.05]),
                    yaxis=dict(title="对比得分", range=[0, 1.05]),
                    height=380, margin=dict(l=10, r=10, t=20, b=10),
                )
                st.plotly_chart(scatter, use_container_width=True)
                regressions = [
                    row for row in sorted(case_rows, key=lambda x: x["变化"]) if row["变化"] < 0
                ][:8]
                if regressions:
                    st.markdown("**退步最明显**")
                    st.dataframe(regressions, use_container_width=True, hide_index=True)
                else:
                    st.dataframe(
                        sorted(case_rows, key=lambda x: x["变化"], reverse=True),
                        use_container_width=True, hide_index=True,
                    )


def page_dashboard() -> None:
    env = st.session_state.get("eval_env", EVAL_ENV_GENERATED)
    specs = load_specs(INSTRUCTIONS_DIR)
    store = _store()
    runs = _filter_runs_for_env(store.list_runs() or [], env, _load_run_report)
    validation = validate_case_matrix(INSTRUCTIONS_DIR)
    latest_agg: dict = {}
    if runs:
        meta = store.get_report_meta(runs[0]["run_id"])
        if meta:
            try:
                latest_agg = json.loads(meta["aggregate_json"])
            except Exception:
                latest_agg = {}

    has_failures = bool(latest_agg.get("failed_case_ids"))
    cta_title, cta_hint, cta_page = _recommend_next(
        has_specs=bool(specs),
        has_runs=bool(runs),
        has_failures=has_failures,
    )
    cta_page = _page_for_env(cta_page, env)

    st.markdown(
        "<div class='ev-hero'>"
        "<div class='h1'>外呼对话评测平台</div>"
        "<div class='h2'>自动模拟用户打电话，评测你的对话模型说得对不对、好不好。"
        "三步即可完成一次完整评测。</div></div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        f"<div class='ev-cta'>"
        f"<div class='label'>推荐下一步</div>"
        f"<div class='action'>{cta_title}</div>"
        f"<div class='hint'>{cta_hint}</div></div>",
        unsafe_allow_html=True,
    )
    if st.button(f"前往：{cta_title}", type="primary", use_container_width=True):
        _navigate(cta_page, env=env)

    st.markdown("**使用流程**")
    _workflow_cards()
    wf_cols = st.columns(len(WORKFLOW_STEPS))
    for col, step in zip(wf_cols, WORKFLOW_STEPS):
        with col:
            target_page = _page_for_env(step["page"], env)
            if st.button(f"去{step['title']}", key=f"wf_{env}_{step['page']}", use_container_width=True):
                _navigate(target_page, env=env)

    overall = latest_agg.get("overall_mean")
    pass_rate = latest_agg.get("case_pass_rate")
    n_passed = int(latest_agg.get("n_passed", 0))
    n_cases = int(latest_agg.get("n_cases", 0))
    overall_text = f"{overall:.3f}" if overall is not None else "—"
    overall_tone = (
        "good" if (overall or 0) >= 0.85
        else ("warn" if (overall or 0) >= 0.6 else "bad")
    ) if overall is not None else "primary"
    pass_text = f"{pass_rate * 100:.0f}%" if pass_rate is not None else "—"
    pass_tone = (
        "good" if (pass_rate or 0) >= 1.0
        else ("warn" if (pass_rate or 0) >= 0.6 else "bad")
    ) if pass_rate is not None else "primary"

    with st.expander("数据概览", expanded=bool(runs)):
        k1, k2, k3, k4 = st.columns(4)
        k1.markdown(_kpi_card("外呼任务", str(len(specs)), tone="primary"), unsafe_allow_html=True)
        k2.markdown(_kpi_card("可测场景", str(validation["n_cases"]), tone="primary"), unsafe_allow_html=True)
        k3.markdown(
            _kpi_card("最新得分", overall_text,
                      delta=_format_run_label(runs[0]["run_id"]) if runs else "尚无记录",
                      tone=overall_tone),
            unsafe_allow_html=True,
        )
        k4.markdown(
            _kpi_card("通过率", pass_text,
                      delta=f"{n_passed}/{n_cases} 项通过" if n_cases else "",
                      tone=pass_tone),
            unsafe_allow_html=True,
        )

    if runs and latest_agg.get("failed_case_ids"):
        st.warning("最近批次未通过：" + "、".join(
            _format_test_item(*cid.split("__", 1)) if "__" in cid else cid
            for cid in latest_agg["failed_case_ids"][:4]
        ))

    with st.expander("历史评测记录", expanded=False):
        if runs:
            rows = []
            for r in runs[:10]:
                m = store.get_report_meta(r["run_id"])
                score, rate = "—", "—"
                if m:
                    try:
                        a = json.loads(m["aggregate_json"])
                        score = f"{a.get('overall_mean', 0):.3f}"
                        pr = a.get("case_pass_rate")
                        rate = f"{pr * 100:.0f}%" if pr is not None else "—"
                    except Exception:
                        pass
                rows.append({
                    "评测批次": _format_run_label(r["run_id"]),
                    "综合得分": score,
                    "通过率": rate,
                })
            st.dataframe(rows, use_container_width=True, hide_index=True)
        else:
            st.info("还没有评测记录。")

    with st.expander("得分趋势", expanded=False):
        if runs:
            xs, ys = [], []
            for r in reversed(runs[:8]):
                m = store.get_report_meta(r["run_id"])
                if not m:
                    continue
                try:
                    a = json.loads(m["aggregate_json"])
                    xs.append(_format_run_label(r["run_id"]))
                    ys.append(a.get("overall_mean", 0.0))
                except Exception:
                    continue
            if xs:
                fig = go.Figure(go.Scatter(
                    x=xs, y=ys, mode="lines+markers",
                    line=dict(color=COLOR["primary"], width=2),
                    marker=dict(size=8),
                ))
                fig.update_layout(
                    height=260, margin=dict(l=10, r=10, t=20, b=40),
                    yaxis=dict(range=[0, 1.05], title="综合得分"),
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.caption("有评测数据后将展示趋势图。")

    with st.expander("平台能力说明", expanded=False):
        for goal in DELIVERY_GOALS:
            friendly = {
                "user_sim": "覆盖多种用户类型",
                "explainable": "评分有据可查",
                "quantifiable": "结果可量化对比",
                "reliable": "评测稳定可靠",
            }.get(goal["id"], goal["name"])
            st.markdown(
                f"<div class='ev-card'>{_pill('已具备', 'good')} <b>{friendly}</b>"
                f"<div style='color:#64748b;margin-top:4px;font-size:.88rem;'>"
                f"{goal['evidence']}</div></div>",
                unsafe_allow_html=True,
            )
        if latest_agg.get("dimensions"):
            st.plotly_chart(_radar_chart(latest_agg["dimensions"]), use_container_width=True)


def _render_trace_panel(run_id: str, case_id: str) -> None:
    store = _store()
    cases = store.list_cases(run_id)
    case_meta = next((c for c in cases if c["case_id"] == case_id), None)
    if not case_meta:
        st.warning("测试项不存在。")
        return
    trace_path = Path(case_meta["trace_path"])
    if not trace_path.exists():
        st.error("对话记录文件缺失。")
        return
    trace = load_trace(trace_path)
    report_data = _load_run_report(run_id)
    case_report = next(
        (
            c
            for c in report_data.get("cases", [])
            if c.get("trace", {}).get("case_id") == case_id
        ),
        None,
    )
    passed = case_report.get("passed") if case_report else None
    score = case_report.get("weighted_total") if case_report else None
    badge = _pill("通过", "good") if passed else _pill("未通过", "bad")
    score_text = f"{score:.3f}" if score is not None else "—"
    st.markdown(
        f"{badge} **{_format_case_id(case_id)}** · 得分 {score_text} · "
        f"第 {trace.session_index}/{trace.max_sessions} 次通话",
        unsafe_allow_html=True,
    )
    if case_report and case_report.get("fail_reasons"):
        st.caption("原因：" + "；".join(case_report["fail_reasons"][:3]))

    turn_count = sum(1 for t in trace.turns if t.role == "assistant")
    total_latency = sum(t.latency_ms for t in trace.turns)
    tokens_in = sum(t.tokens_in for t in trace.turns)
    tokens_out = sum(t.tokens_out for t in trace.turns)
    dim_count = len(case_report.get("dimensions", [])) if case_report else 0
    fail_attr_count = (
        len(case_report.get("failure_attribution", [])) if case_report else 0
    )

    st.markdown("<div class='ev-trace-detail-stack'>", unsafe_allow_html=True)
    _render_trace_detail_overview(
        turn_count=turn_count,
        score=score,
        passed=passed,
        dim_count=dim_count,
        fail_attr_count=fail_attr_count,
        total_latency_ms=total_latency,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
    )

    with _trace_detail_section("dialogue", "逐轮对话", f"共 {turn_count} 轮"):
        last_user = "（接通）"
        for turn in trace.turns:
            if turn.role == "assistant":
                hits = _judge_hits_for_turn(case_report, turn.index)
                _render_usj_turn(turn.index, last_user, turn.content, hits)
            elif turn.role == "user":
                last_user = turn.content

    with _trace_detail_section("score", "维度明细", f"{dim_count} 项维度"):
        if case_report:
            _render_trace_dimension_cards(case_report.get("dimensions", []))

    with _trace_detail_section("perf", "性能图 / 下载", f"耗时 {total_latency} ms"):
        _render_trace_perf_metrics(total_latency, tokens_in, tokens_out)
        if trace.prior_sessions:
            st.json(trace.prior_sessions)
        st.download_button(
            "下载对话记录",
            data=trace.model_dump_json(indent=2),
            file_name=f"{case_id}.json",
            mime="application/json",
        )
    st.markdown("</div>", unsafe_allow_html=True)


def page_config() -> None:
    _page_header("任务配置", "可选 · 一般不用改，系统已内置任务")
    _page_guide("任务配置")
    t1, t2 = st.tabs(["外呼任务", "用户场景"])
    with t1:
        page_instructions()
    with t2:
        page_scenarios()


def main() -> None:
    st.set_page_config(
        page_title="外呼对话评测",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_global_css()
    _store().bootstrap_from_disk(ROOT)

    st.sidebar.markdown(
        "<div class='ev-side-brand'>"
        "<div class='t'>外呼对话评测</div>"
        "<div class='s'>左侧菜单切换页面</div></div>",
        unsafe_allow_html=True,
    )
    st.sidebar.caption("主流程：发起评测 → 评测报告 → 对话复盘")

    qp = st.query_params
    if _should_apply_query_env(
        query_env=qp.get("eval_env"),
        current_env=st.session_state.get("eval_env"),
        last_synced_env=st.session_state.get("_last_synced_query_env"),
    ):
        st.session_state["eval_env"] = qp["eval_env"]
    if st.session_state.get("eval_env") not in {EVAL_ENV_GENERATED, EVAL_ENV_REAL}:
        st.session_state["eval_env"] = EVAL_ENV_GENERATED

    env_labels = {
        EVAL_ENV_GENERATED: "生成对话",
        EVAL_ENV_REAL: "真实对话",
    }
    env_values = list(env_labels.keys())
    eval_env = st.sidebar.radio(
        "评测环境",
        env_values,
        key="eval_env",
        format_func=env_labels.get,
        horizontal=True,
    )
    nav_keys = _nav_keys_for_env(eval_env)

    query_page = None
    if qp.get("page") in NAV_KEYS + REAL_NAV_KEYS:
        query_page = qp["page"]
    elif qp.get("page") in _NAV_ALIAS:
        query_page = _NAV_ALIAS[qp["page"]]
    if _should_apply_query_page(
        query_page=query_page,
        current_page=st.session_state.get("nav_page"),
        last_synced_page=st.session_state.get("_last_synced_query_page"),
    ):
        st.session_state["_nav_target"] = query_page
    if qp.get("run"):
        st.session_state["trace_run_id"] = qp["run"]
    if qp.get("case"):
        st.session_state["trace_case_id"] = qp["case"]

    nav_target = st.session_state.pop("_nav_target", None)
    if nav_target and nav_target in nav_keys:
        st.session_state["nav_page"] = nav_target
    elif nav_target:
        mapped_target = "真实对话复盘" if eval_env == EVAL_ENV_REAL and nav_target == "对话复盘" else None
        mapped_target = mapped_target or ("对话复盘" if eval_env == EVAL_ENV_GENERATED and nav_target == "真实对话复盘" else None)
        if mapped_target in nav_keys:
            st.session_state["nav_page"] = mapped_target
    if "nav_page" not in st.session_state:
        st.session_state["nav_page"] = nav_keys[0]
    if st.session_state["nav_page"] not in nav_keys:
        st.session_state["nav_page"] = nav_keys[0]

    page = st.sidebar.radio(
        "功能导航",
        nav_keys,
        key="nav_page",
        label_visibility="collapsed",
    )

    active = len(_list_active_runs())
    if active:
        st.sidebar.markdown(f"<span class='ev-pill warn'>{active} 个任务进行中</span>", unsafe_allow_html=True)
    else:
        st.sidebar.markdown("<span class='ev-pill good'>系统就绪</span>", unsafe_allow_html=True)

    current = st.session_state.get("nav_page", nav_keys[0])
    guide = PAGE_GUIDES.get(current, "")
    st.sidebar.markdown(
        f"<div class='ev-side-hint'><b>当前页面</b><br>{guide}</div>",
        unsafe_allow_html=True,
    )

    pages = _page_map_for_env(eval_env)
    if st.query_params.get("eval_env") != eval_env:
        st.query_params["eval_env"] = eval_env
    st.session_state["_last_synced_query_env"] = eval_env
    if st.query_params.get("page") != page:
        st.query_params["page"] = page
    st.session_state["_last_synced_query_page"] = page
    pages[page]()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()

"""DeepSeek / OpenAI 兼容端点的一键连通性测试。

用法::

    python -m src.cli.ping                       # 用 .env 里的默认模型问一次 hello
    python -m src.cli.ping --model deepseek-v4-pro --thinking
    python -m src.cli.ping --json                # 测试 JSON Output 模式
"""
from __future__ import annotations

import argparse
import logging
import os
import sys

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ..core.config import load_config
from ..core.llm_client import ChatMessage, build_default_client


def main() -> None:
    parser = argparse.ArgumentParser(description="Ping the configured LLM endpoint.")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="覆盖 SUT_MODEL 进行一次测试调用，默认读取 .env / configs/default.yaml",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="用一句不超过20字的中文打个招呼。",
        help="发送给模型的用户消息",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.4,
    )
    parser.add_argument(
        "--max-tokens", type=int, default=128,
    )
    parser.add_argument(
        "--thinking",
        action="store_true",
        help="开启 DeepSeek v4 的 thinking 模式（仅当 endpoint 为 DeepSeek 时生效）",
    )
    parser.add_argument(
        "--reasoning-effort",
        default="high",
        choices=["low", "medium", "high"],
        help="thinking 模式下的推理强度",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="测试 JSON Output 模式（response_format=json_object）",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="禁用磁盘缓存，强制走网络。默认开启缓存以便复测",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )
    console = Console()
    try:
        cfg = load_config()
    except FileNotFoundError:
        cfg = {}

    client = build_default_client()
    model = (
        args.model
        or os.getenv("SUT_MODEL")
        or (cfg.get("models", {}) or {}).get("sut")
        or "deepseek-v4-flash"
    )

    info = Table.grid(padding=(0, 1))
    info.add_column(style="bold cyan")
    info.add_column()
    info.add_row("Base URL", client.base_url or "(default OpenAI)")
    info.add_row("API key", "(set)" if client.api_key else "(missing)")
    info.add_row("Model", model)
    info.add_row("DeepSeek mode", "yes" if client._is_deepseek else "no")
    info.add_row("Thinking", "on" if args.thinking else "off")
    info.add_row("JSON mode", "on" if args.json else "off")
    console.print(Panel(info, title="LLM endpoint", border_style="cyan"))

    if not client.api_key:
        console.print(
            "[red]缺少 API key。请在 .env 中设置 DEEPSEEK_API_KEY 或 OPENAI_API_KEY 后重试。[/]"
        )
        sys.exit(2)

    system = "You are a friendly Chinese-speaking assistant. 用简短中文回答。"
    if args.json:
        system += "  请只输出 JSON 对象，形如 {\"greeting\": \"...\"}。"

    try:
        result = client.chat(
            messages=[
                ChatMessage(role="system", content=system),
                ChatMessage(role="user", content=args.prompt),
            ],
            model=model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            response_format={"type": "json_object"} if args.json else None,
            thinking=args.thinking,
            reasoning_effort=args.reasoning_effort if args.thinking else None,
            cache=not args.no_cache,
        )
    except Exception as exc:
        console.print(f"[red]调用失败：[/] {exc}")
        sys.exit(1)

    out_table = Table.grid(padding=(0, 1))
    out_table.add_column(style="bold green")
    out_table.add_column()
    out_table.add_row("延迟", f"{result.latency_ms} ms")
    out_table.add_row("Prompt tokens", str(result.tokens_in))
    out_table.add_row("Output tokens", str(result.tokens_out))
    out_table.add_row("缓存命中", "是" if result.cached else "否")
    console.print(Panel(out_table, title="response stats", border_style="green"))
    console.print(Panel(result.content, title="response body", border_style="blue"))

    console.print("[bold green][OK] 连通性正常，可以正式跑评测了。[/]")


if __name__ == "__main__":
    main()

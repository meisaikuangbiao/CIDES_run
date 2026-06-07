from __future__ import annotations

import argparse
import logging
from pathlib import Path

from rich.console import Console
from rich.table import Table

from ..core.instruction_parser import parse_workbook
from ..core.llm_client import build_default_client


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse the instruction Excel into JSON specs.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/source/命题二：外呼任务对话模型指令示例.xlsx"),
        help="Path to the source xlsx file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/instructions"),
        help="Directory to write JSON / Markdown copies into.",
    )
    parser.add_argument(
        "--mode",
        choices=["llm", "offline"],
        default="llm",
        help="Parsing mode. 'llm' uses an LLM for flow extraction; 'offline' relies on regex heuristics only.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Override the parser model name.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("data/.llm_cache"),
        help="Where to cache LLM parser calls.",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )

    client = None
    if args.mode == "llm":
        client = build_default_client(cache_dir=args.cache_dir)
    if not args.input.exists():
        root_xlsx = Path("命题二：外呼任务对话模型指令示例.xlsx")
        if root_xlsx.exists():
            args.input = root_xlsx

    specs = parse_workbook(
        args.input,
        output_dir=args.output,
        client=client,
        model=args.model,
        mode=args.mode,
    )

    console = Console()
    table = Table(title="Parsed instructions", show_lines=True)
    table.add_column("ID", style="cyan", no_wrap=True)
    table.add_column("Role")
    table.add_column("Variables")
    table.add_column("Nodes", justify="right")
    table.add_column("Knowledge", justify="right")
    table.add_column("Hard Char Limit", justify="right")
    table.add_column("Out-of-scope Reply", overflow="fold")
    for spec in specs:
        table.add_row(
            spec.id,
            spec.role or "-",
            ", ".join(spec.variables) or "-",
            str(len(spec.flow_nodes)),
            str(len(spec.knowledge)),
            str(spec.constraints.hard.max_chars_per_reply or "-"),
            spec.constraints.hard.required_out_of_scope_reply or "-",
        )
    console.print(table)
    console.print(f"Saved {len(specs)} specs to [bold]{args.output}[/bold].")


if __name__ == "__main__":
    main()

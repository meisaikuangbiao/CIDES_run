"""Generate 参赛文档.pdf from 参赛文档.md (no page header/footer)."""
from __future__ import annotations

import markdown
from pathlib import Path
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent
MD_PATH = ROOT / "参赛文档.md"
PDF_PATH = ROOT / "参赛文档.pdf"
TEMP_PDF_PATH = ROOT / "_参赛文档_temp.pdf"
HTML_PATH = ROOT / "_参赛文档_temp.html"

md_text = MD_PATH.read_text(encoding="utf-8")
html_body = markdown.markdown(md_text, extensions=["tables", "fenced_code"])
css_text = (ROOT / "参赛文档-pdf.css").read_text(encoding="utf-8")
html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<style>
body {{ max-width: 920px; margin: 36px auto; }}
{css_text}
</style></head><body>{html_body}</body></html>"""

HTML_PATH.write_text(html, encoding="utf-8")

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page()
    page.goto(HTML_PATH.as_uri(), wait_until="networkidle")
    page.pdf(
        path=str(TEMP_PDF_PATH),
        format="A4",
        margin={"top": "18mm", "bottom": "18mm", "left": "14mm", "right": "14mm"},
        print_background=True,
        display_header_footer=False,
    )
    browser.close()

HTML_PATH.unlink(missing_ok=True)
try:
    TEMP_PDF_PATH.replace(PDF_PATH)
    print(f"Generated: {PDF_PATH}")
except PermissionError:
    alt = ROOT / "参赛文档_CIDES.pdf"
    TEMP_PDF_PATH.replace(alt)
    print(f"参赛文档.pdf locked; saved to: {alt}")

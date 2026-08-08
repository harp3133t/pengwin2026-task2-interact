#!/usr/bin/env python
"""Render a Markdown algorithm description to a Grand Challenge PDF."""

from __future__ import annotations

import argparse
import html
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import markdown


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def page_count(pdf: Path) -> int:
    result = subprocess.run(
        ["pdfinfo", str(pdf)], check=True, capture_output=True, text=True
    )
    for line in result.stdout.splitlines():
        if line.startswith("Pages:"):
            return int(line.split(":", 1)[1].strip())
    raise RuntimeError(f"could not determine PDF page count: {pdf}")


def drop_blank_leading_page(pdf: Path, tmp: Path) -> Path:
    pages = page_count(pdf)
    if pages <= 1:
        return pdf
    first_page = subprocess.run(
        ["pdftotext", "-f", "1", "-l", "1", str(pdf), "-"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if first_page.strip():
        return pdf
    trimmed = tmp / "description-trimmed.pdf"
    subprocess.run(
        [
            "gs",
            "-q",
            "-dNOPAUSE",
            "-dBATCH",
            "-sDEVICE=pdfwrite",
            "-dFirstPage=2",
            f"-dLastPage={pages}",
            f"-sOutputFile={trimmed}",
            str(pdf),
        ],
        check=True,
    )
    return trimmed


def main() -> None:
    args = parse_args()
    source_lines = args.source.read_text().splitlines()
    if not source_lines or not source_lines[0].startswith("# "):
        raise RuntimeError("description.md must start with a level-1 title")
    title = source_lines[0][2:].strip()
    body = markdown.markdown(
        "\n".join(source_lines[1:]), extensions=["tables", "fenced_code"]
    )
    document = f"""<!doctype html>
<html><head><meta charset="utf-8"><style>
body {{ font-family: 'DejaVu Sans', sans-serif; color: #172033; font-size: 9.5pt; line-height: 1.35; margin: 42px 46px; }}
.title {{ color: #123e70; font-size: 20pt; font-weight: bold; margin: 0 0 8pt; }}
h1 {{ color: #123e70; font-size: 16pt; margin: 0 0 8pt; }}
h2 {{ color: #18558f; font-size: 13pt; margin-top: 15pt; }}
p, li {{ orphans: 3; widows: 3; }}
table {{ border-collapse: collapse; width: 100%; font-size: 8.3pt; margin: 8pt 0; }}
th, td {{ border: 1px solid #aeb9c7; padding: 4px 5px; vertical-align: top; }}
th {{ background: #e9f1f8; }}
pre {{ background: #f4f6f8; border: 1px solid #d7dde4; padding: 7px; font-size: 7.8pt; white-space: pre-wrap; }}
code {{ font-family: 'DejaVu Sans Mono', monospace; font-size: 8.3pt; }}
</style></head><body><div class="title">{html.escape(title)}</div>{body}</body></html>"""
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="pengwin-task2-description-") as tmp_text:
        tmp = Path(tmp_text)
        source_html = tmp / "description.html"
        source_html.write_text(document)
        env = os.environ.copy()
        env["HOME"] = str(tmp)
        subprocess.run(
            [
                "libreoffice",
                "--headless",
                f"-env:UserInstallation=file://{tmp / 'lo-profile'}",
                "--convert-to",
                "pdf",
                "--outdir",
                str(tmp),
                str(source_html),
            ],
            check=True,
            env=env,
        )
        rendered = tmp / "description.pdf"
        if not rendered.is_file() or rendered.stat().st_size < 10_000:
            raise RuntimeError(f"invalid rendered PDF: {rendered}")
        rendered = drop_blank_leading_page(rendered, tmp)
        shutil.copy2(rendered, args.output)
    print(
        f"wrote {args.output} ({args.output.stat().st_size} bytes, "
        f"{page_count(args.output)} pages)"
    )


if __name__ == "__main__":
    main()

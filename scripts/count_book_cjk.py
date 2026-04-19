#!/usr/bin/env python3
"""
Count CJK Unified Ideographs (U+4E00–U+9FFF) in book/**/*.md.
Used to verify each subsection has at least 5000 Chinese characters (教材字数口径).
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

CJK_RE = re.compile(r"[\u4e00-\u9fff]")


def count_cjk(text: str) -> int:
    return len(CJK_RE.findall(text))


def main() -> int:
    ap = argparse.ArgumentParser(description="Count CJK characters in book markdown files.")
    ap.add_argument(
        "--book-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "book",
        help="Path to book directory",
    )
    ap.add_argument("--min", type=int, default=5000, help="Threshold for OK column")
    ap.add_argument("--write", type=Path, default=None, help="Optional output file for the report")
    args = ap.parse_args()
    root: Path = args.book_root
    if not root.is_dir():
        print(f"Not a directory: {root}", file=sys.stderr)
        return 1

    rows: list[tuple[int, str, Path]] = []
    for p in sorted(root.rglob("*.md")):
        try:
            text = p.read_text(encoding="utf-8")
        except OSError as e:
            print(f"Skip {p}: {e}", file=sys.stderr)
            continue
        n = count_cjk(text)
        rel = p.relative_to(root.parent)
        rows.append((n, str(rel).replace("\\", "/"), p))

    lines: list[str] = []
    lines.append(f"# book CJK baseline (threshold {args.min})\n")
    lines.append("| CJK chars | OK | path |\n")
    lines.append("|-----------|:--:|------|\n")
    bad = 0
    for n, rel, _ in sorted(rows, key=lambda x: x[0]):
        ok = "yes" if n >= args.min else "no"
        if n < args.min:
            bad += 1
        lines.append(f"| {n} | {ok} | `{rel}` |\n")
    lines.append(f"\nTotal files: {len(rows)}; below threshold: {bad}\n")

    report = "".join(lines)
    print(report, end="")
    if args.write:
        args.write.parent.mkdir(parents=True, exist_ok=True)
        args.write.write_text(report, encoding="utf-8")
        print(f"Wrote {args.write}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

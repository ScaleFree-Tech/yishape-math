#!/usr/bin/env python3
"""
Remove duplicated '## 教材深读补充' tail blocks from book/**/*.md.
Also strips the common bilingual promo block immediately preceding that section.
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "book"
MARKER = "## 教材深读补充"
# e.g. ---\n\n**中文** - ...\n\n**English** - ...\n\n---\n
PROMO_BEFORE = re.compile(
    r"\n---\n\n\*\*[^\n]+\*\*[^\n]*\n\n\*\*[^\n]+\*\*[^\n]*\n\n---\n\s*$",
    re.MULTILINE | re.DOTALL,
)


def strip_file(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    if MARKER not in text:
        return False
    idx = text.index(MARKER)
    prefix = text[:idx]
    prefix = PROMO_BEFORE.sub("\n", prefix)
    new_text = prefix.rstrip() + "\n"
    if new_text == text:
        return False
    path.write_text(new_text, encoding="utf-8")
    return True


def main() -> None:
    changed = 0
    for p in sorted(ROOT.rglob("*.md")):
        if strip_file(p):
            print(f"stripped: {p.relative_to(ROOT.parent)}")
            changed += 1
    print(f"Total updated: {changed}")


if __name__ == "__main__":
    main()

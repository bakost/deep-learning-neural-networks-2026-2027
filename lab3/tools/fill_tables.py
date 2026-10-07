"""Подставить таблицы из results/tables.md в REPORT.md.

Таблицы отчёта генерирует ``python -m lipo_ols report`` в файл
``results/tables.md`` (каждая — под заголовком «## Таблица N. …»). В
``REPORT.md`` место таблицы отмечено комментариями::

    <!-- table:2 -->
    ...содержимое заменяется...
    <!-- /table -->

Запуск из каталога лабораторной работы::

    python tools/fill_tables.py            # обновить REPORT.md на месте
    python tools/fill_tables.py --check    # только проверить, что таблицы актуальны
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HEADING = re.compile(r"^## Таблица ([^.\s]+)\.", re.MULTILINE)
MARKER = re.compile(r"(<!-- table:([^ ]+) -->\n)(.*?)(<!-- /table -->)", re.DOTALL)


def read_tables(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    heads = list(HEADING.finditer(text))
    out = {}
    for k, m in enumerate(heads):
        end = heads[k + 1].start() if k + 1 < len(heads) else len(text)
        body = text[m.end():end]
        lines = [line for line in body.splitlines() if line.startswith("|")]
        out[m.group(1)] = "\n".join(lines) + "\n"
    return out


def fill(report: str, tables: dict) -> str:
    def repl(m: re.Match) -> str:
        key = m.group(2)
        if key not in tables:
            raise KeyError(f"в results/tables.md нет таблицы {key}")
        return m.group(1) + tables[key] + m.group(4)

    return MARKER.sub(repl, report)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="не писать, только проверить")
    args = parser.parse_args()
    report_path = ROOT / "REPORT.md"
    report = report_path.read_text(encoding="utf-8")
    new = fill(report, read_tables(ROOT / "results" / "tables.md"))
    if args.check:
        if new != report:
            print("REPORT.md устарел: запустите python tools/fill_tables.py")
            return 1
        print("таблицы в REPORT.md актуальны")
        return 0
    report_path.write_text(new, encoding="utf-8")
    print(f"обновлено таблиц: {len(MARKER.findall(new))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

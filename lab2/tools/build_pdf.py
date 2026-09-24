"""Сборка REPORT.pdf из REPORT.md.

Формулы в отчёте записаны на LaTeX, поэтому PDF собирается в два шага:

1. pandoc переводит Markdown в самодостаточный HTML: формулы — в MathML,
   рисунки встраиваются внутрь файла;
2. Chrome в режиме headless печатает HTML в PDF (MathML он отображает сам,
   без сети и без JavaScript).

Запуск из каталога lab2::

    python tools/build_pdf.py                 # REPORT.md -> REPORT.pdf
    python tools/build_pdf.py --html out.html # сохранить и промежуточный HTML

Нужны pandoc (например, ``pip install pypandoc_binary``) и Google Chrome или
Chromium. Путь к браузеру можно задать переменной окружения ``CHROME``.
Chrome запускается с отдельным временным профилем и профиль пользователя не
трогает.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CHROME_CANDIDATES = (
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    "google-chrome",
    "google-chrome-stable",
    "chromium",
    "chromium-browser",
)

CSS = r"""
@page {
  size: A4;
  margin: 18mm 17mm 20mm 17mm;
  @bottom-center {
    content: counter(page);
    font-family: "PT Sans", sans-serif;
    font-size: 9pt;
    color: #555;
  }
}
html { font-size: 10.5pt; }
body {
  font-family: "STIX Two Text", "PT Serif", "Times New Roman", serif;
  line-height: 1.38;
  color: #111;
  margin: 0;
  hyphens: auto;
  text-align: justify;
}
math { font-family: "STIX Two Math", math; font-size: 1.05em; }
math[display="block"] { margin: 0.5em 0; }
/* Chrome оставляет после знака корня лишний промежуток. */
msqrt { margin-inline-end: -0.25em; }
h1, h2, h3 { font-family: "PT Sans", sans-serif; color: #1a1a1a; text-align: left;
             break-after: avoid; hyphens: manual; }
h1 { font-size: 17pt; margin: 0 0 0.4em; line-height: 1.2; }
h2 { font-size: 13.5pt; margin: 1.3em 0 0.4em; padding-bottom: 0.15em;
     border-bottom: 1px solid #bbb; }
h3 { font-size: 11.5pt; margin: 1.0em 0 0.3em; }
p { margin: 0.35em 0 0.5em; orphans: 3; widows: 3; }
ul, ol { margin: 0.3em 0 0.6em; padding-left: 1.4em; }
li { margin: 0.2em 0; }
hr { display: none; }
a { color: #1f4e8c; text-decoration: none; }
code { font-family: "PT Mono", Menlo, monospace; font-size: 0.88em; }
pre { background: #f5f5f5; border: 1px solid #e0e0e0; border-radius: 3px;
      padding: 0.5em 0.7em; font-size: 8.5pt; line-height: 1.35; overflow: hidden;
      text-align: left; break-inside: avoid; }
pre code { font-size: 1em; }
img { display: block; max-width: 100%; max-height: 118mm; width: auto; height: auto;
      margin: 0.6em auto 0.2em; break-inside: avoid; }
p:has(> img) { break-inside: avoid; break-after: avoid; margin-bottom: 0; }
table { border-collapse: collapse; margin: 0.5em auto 0.8em; font-size: 8.4pt;
        line-height: 1.25; break-inside: avoid; text-align: left; }
th, td { border: 1px solid #c8c8c8; padding: 0.22em 0.4em; vertical-align: top; }
thead th { background: #eef2f7; font-family: "PT Sans", sans-serif; font-weight: bold; }
tbody tr:nth-child(even) { background: #fafafa; }
td math, th math { font-size: 1em; }
/* Широкие таблицы (8 и более столбцов): мельче и без переносов внутри ячеек. */
table:has(th:nth-child(8)) { font-size: 7.6pt; }
table:has(th:nth-child(8)) td { white-space: nowrap; padding: 0.22em 0.3em; }
strong { font-weight: 700; }
"""


def find_pandoc() -> str:
    try:
        import pypandoc

        return pypandoc.get_pandoc_path()
    except (ImportError, OSError):
        pass
    path = shutil.which("pandoc")
    if path is None:
        sys.exit("Ошибка: не найден pandoc (pip install pypandoc_binary)")
    return path


def find_chrome() -> str:
    candidates = [os.environ["CHROME"]] if os.environ.get("CHROME") else list(CHROME_CANDIDATES)
    for candidate in candidates:
        path = candidate if os.path.isabs(candidate) else shutil.which(candidate)
        if path and os.path.exists(path):
            return path
    sys.exit("Ошибка: не найден Google Chrome/Chromium; задайте путь в переменной CHROME")


def build_html(source: Path, html: Path, workdir: Path) -> None:
    css = workdir / "report.css"
    css.write_text(CSS, encoding="utf-8")
    subprocess.run(
        [
            find_pandoc(), str(source),
            "--from", "markdown-implicit_figures-fancy_lists",
            "--to", "html5",
            "--standalone", "--embed-resources",
            "--mathml",
            "--css", str(css),
            "--resource-path", str(source.parent),
            "--metadata", "pagetitle=Лабораторная работа 2. Свойства случайных матриц",
            "--metadata", "lang=ru",
            "--output", str(html),
        ],
        check=True,
    )


def print_pdf(html: Path, pdf: Path, workdir: Path, timeout: float = 180.0) -> None:
    """Напечатать HTML в PDF.

    На macOS headless Chrome иногда не завершается после печати, поэтому
    процесс не ждётся до конца: как только PDF записан и его размер перестал
    меняться, браузер закрывается.
    """
    profile = workdir / "chrome-profile"
    target = workdir / "report.pdf"
    process = subprocess.Popen(
        [
            find_chrome(),
            "--headless=new", "--disable-gpu", "--no-first-run", "--no-default-browser-check",
            "--disable-extensions", f"--user-data-dir={profile}",
            "--no-pdf-header-footer",
            f"--print-to-pdf={target}",
            html.as_uri(),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    deadline = time.monotonic() + timeout
    last_size, stable = -1, 0
    try:
        while time.monotonic() < deadline:
            if process.poll() is not None and not target.exists():
                sys.exit(f"Ошибка: Chrome завершился с кодом {process.returncode}, PDF не создан")
            size = target.stat().st_size if target.exists() else -1
            stable = stable + 1 if size > 0 and size == last_size else 0
            if stable >= 3:
                break
            last_size = size
            time.sleep(0.5)
        else:
            sys.exit("Ошибка: Chrome не напечатал PDF за отведённое время")
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
    shutil.move(str(target), pdf)


def set_metadata(pdf: Path) -> None:
    """Название и тема в свойствах PDF (если установлен pypdf)."""
    try:
        from pypdf import PdfReader, PdfWriter
    except ImportError:
        return
    writer = PdfWriter(clone_from=PdfReader(pdf))
    writer.add_metadata({
        "/Title": "Лабораторная работа 2. Свойства случайных матриц",
        "/Subject": "Распределение расстояний между соседними собственными числами, догадка Вигнера",
    })
    with open(pdf, "wb") as handle:
        writer.write(handle)


def main() -> int:
    parser = argparse.ArgumentParser(description="Собрать REPORT.pdf из REPORT.md")
    parser.add_argument("source", nargs="?", default=str(ROOT / "REPORT.md"))
    parser.add_argument("-o", "--output", default=None, help="PDF (по умолчанию рядом с исходником)")
    parser.add_argument("--html", default=None, help="сохранить промежуточный HTML")
    args = parser.parse_args()

    source = Path(args.source).resolve()
    pdf = Path(args.output).resolve() if args.output else source.with_suffix(".pdf")
    with tempfile.TemporaryDirectory(prefix="report-pdf-") as tmp:
        workdir = Path(tmp)
        html = workdir / "report.html"
        build_html(source, html, workdir)
        print_pdf(html, pdf, workdir)
        if args.html:
            shutil.copy(html, args.html)
    set_metadata(pdf)
    print(f"Готово: {pdf} ({pdf.stat().st_size / 1e6:.1f} МБ)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

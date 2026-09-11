#!/usr/bin/env python3
"""Генератор справочника по API из докстрингов пакета.

Скрипт обходит модули пакета :mod:`jacobi_eigen` с помощью :mod:`inspect` и
собирает Markdown-документ со всеми публичными классами, функциями и их
сигнатурами. В отличие от Sphinx, он не требует установки дополнительных
пакетов, поэтому документацию можно пересобрать в любой момент::

    python tools/gen_api_docs.py              # записать docs/API.md
    python tools/gen_api_docs.py --stdout     # вывести в консоль

Sphinx-конфигурация в ``docs/conf.py`` тоже поддерживается — она даёт более
богатый HTML; этот скрипт нужен для быстрой пересборки и для того, чтобы
справочник всегда лежал в репозитории в читаемом виде.
"""

from __future__ import annotations

import argparse
import inspect
import re
import sys
from pathlib import Path
from typing import Any, List, Optional, Sequence

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import jacobi_eigen  # noqa: E402
from jacobi_eigen import (  # noqa: E402
    contracts,
    exceptions,
    kernels,
    result,
    solver,
    strategies,
    validation,
)

#: Порядок модулей в документе — от главного к вспомогательным.
MODULES = [
    (solver, "Решатель и публичный интерфейс"),
    (result, "Результат вычислений"),
    (kernels, "Ядра вращения"),
    (strategies, "Стратегии выбора элемента"),
    (validation, "Проверка входных данных"),
    (exceptions, "Иерархия исключений"),
    (contracts, "Контрактное программирование"),
]


#: Роли reStructuredText вида ``:class:`Имя``` — в докстрингах они нужны
#: Sphinx'у, а в Markdown превращаются в обычные код-спаны.
_RST_ROLE = re.compile(r":(?:class|func|meth|mod|attr|data|obj|exc|ref):`~?([^`]+)`")
#: Роль ``:math:`...``` — формула внутри строки; GitHub рендерит `$...$`.
_RST_MATH_ROLE = re.compile(r":math:`([^`]+)`")
#: Директива ``.. math::`` открывает блок формул; сам блок уже записан с
#: отступом, поэтому в Markdown он отрисуется как блок кода с формулой.
_RST_MATH_BLOCK = re.compile(r"^[ \t]*\.\.[ \t]+math::[ \t]*\n", re.MULTILINE)
#: Подчёркивание заголовка раздела внутри докстринга (``-----``).
_RST_UNDERLINE = re.compile(r"^([^\n]+)\n[-=~^]{3,}$", re.MULTILINE)


def rst_to_markdown(text: str) -> str:
    """Перевести разметку докстринга из reStructuredText в Markdown.

    Полноценный конвертер здесь не нужен: докстринги пакета используют лишь
    несколько конструкций RST, и для читаемого справочника достаточно
    заменить их на ближайшие аналоги Markdown.
    """
    text = _RST_ROLE.sub(r"`\1`", text)
    text = _RST_MATH_ROLE.sub(r"$\1$", text)
    text = _RST_MATH_BLOCK.sub("", text)
    text = _RST_UNDERLINE.sub(r"**\1**", text)
    # Двойные двоеточия перед блоком кода RST не несут смысла в Markdown.
    text = text.replace("::\n", ":\n")
    return text


def clean_docstring(obj: Any) -> str:
    """Получить докстринг без лишних отступов, переведя его в Markdown."""
    raw = inspect.getdoc(obj)
    if not raw:
        return "*(без описания)*"
    return rst_to_markdown(inspect.cleandoc(raw))


def format_signature(obj: Any) -> str:
    """Сигнатура вызова или пустая строка, если получить её нельзя."""
    try:
        return str(inspect.signature(obj))
    except (TypeError, ValueError):  # pragma: no cover - встроенные типы
        return "(...)"


def is_public(name: str) -> bool:
    """Публичное ли имя (не начинается с подчёркивания)."""
    return not name.startswith("_")


def document_function(func: Any, name: str, level: int = 3) -> List[str]:
    """Markdown-блок для функции или метода."""
    return [
        f"{'#' * level} `{name}{format_signature(func)}`",
        "",
        clean_docstring(func),
        "",
    ]


def document_class(cls: type, name: str) -> List[str]:
    """Markdown-блок для класса вместе с его публичными методами."""
    lines = [f"### `class {name}`", ""]

    bases = [base.__name__ for base in cls.__bases__ if base is not object]
    if bases:
        lines += [f"*Наследует:* `{'`, `'.join(bases)}`", ""]

    lines += [clean_docstring(cls), ""]

    contract_invariants = getattr(cls, "__contract_invariants__", None)
    if contract_invariants:
        lines += ["**Инварианты класса:**", ""]
        lines += [f"- {text}" for text in contract_invariants]
        lines += [""]

    methods = [
        (method_name, member)
        for method_name, member in vars(cls).items()
        if is_public(method_name) and (inspect.isfunction(member) or isinstance(member, property))
    ]
    if methods:
        lines += ["**Методы:**", ""]
        for method_name, member in sorted(methods):
            if isinstance(member, property):
                lines += [f"#### `{method_name}` *(свойство)*", "",
                          clean_docstring(member.fget), ""]
            else:
                lines += document_function(member, f"{name}.{method_name}", level=4)
    return lines


def document_module(module: Any, title: str) -> List[str]:
    """Markdown-раздел для одного модуля."""
    lines = [f"## {title}", "", f"Модуль `{module.__name__}`", "",
             clean_docstring(module), ""]

    exported: Optional[Sequence[str]] = getattr(module, "__all__", None)
    names = list(exported) if exported else sorted(
        name for name in vars(module) if is_public(name)
    )

    classes, functions = [], []
    for name in names:
        member = getattr(module, name, None)
        if member is None or getattr(member, "__module__", None) != module.__name__:
            continue
        if inspect.isclass(member):
            classes.append((name, member))
        elif inspect.isfunction(member):
            functions.append((name, member))

    if classes:
        lines += ["### Классы", ""]
        for name, cls in classes:
            lines += document_class(cls, name)
    if functions:
        lines += ["### Функции", ""]
        for name, func in functions:
            lines += document_function(func, name)
    return lines


def build_document() -> str:
    """Собрать полный Markdown-справочник."""
    lines = [
        "# Справочник API `jacobi_eigen`",
        "",
        f"Версия пакета: **{jacobi_eigen.__version__}**",
        "",
        "> Этот файл создаётся автоматически скриптом `tools/gen_api_docs.py` "
        "из докстрингов исходного кода. Не редактируйте его вручную — "
        "правьте документацию в коде и перезапустите генератор.",
        "",
        "## Содержание",
        "",
    ]
    for module, title in MODULES:
        anchor = title.lower().replace(" ", "-")
        lines.append(f"- [{title}](#{anchor})")
    lines += ["", "---", ""]

    for module, title in MODULES:
        lines += document_module(module, title)
        lines += ["---", ""]
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Точка входа скрипта."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-o", "--output", default=str(ROOT / "docs" / "API.md"),
                        help="куда записать справочник")
    parser.add_argument("--stdout", action="store_true", help="вывести в консоль")
    args = parser.parse_args(argv)

    document = build_document()
    if args.stdout:
        print(document)
    else:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(document + "\n", encoding="utf-8")
        print(f"Справочник записан в {path} ({len(document.splitlines())} строк)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

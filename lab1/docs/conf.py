"""Конфигурация Sphinx для сборки HTML-документации.

Собрать документацию::

    pip install -e ".[docs]"
    sphinx-build -b html docs docs/_build/html

Если Sphinx ставить не хочется, Markdown-справочник по тем же докстрингам
создаётся скриптом ``python tools/gen_api_docs.py`` без каких-либо
зависимостей.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

project = "jacobi-eigen"
author = "bakost"
release = "1.0.0"
language = "ru"

extensions = [
    "sphinx.ext.autodoc",      # документация из докстрингов
    "sphinx.ext.napoleon",     # стиль NumPy/Google в докстрингах
    "sphinx.ext.viewcode",     # ссылки на исходный код
    "sphinx.ext.mathjax",      # формулы
    "sphinx.ext.intersphinx",  # ссылки на документацию NumPy и Python
    "sphinx.ext.doctest",      # проверка примеров из докстрингов
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "API.md", "benchmarks.md"]

html_theme = "alabaster"
html_static_path = []

autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
    "member-order": "bysource",
}
autodoc_typehints = "description"

napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_rtype = False

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

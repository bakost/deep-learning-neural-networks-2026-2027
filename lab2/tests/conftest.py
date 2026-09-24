"""Общие настройки тестов.

Статистические тесты используют фиксированные зёрна, поэтому детерминированы:
допуски выбраны с запасом в 4–5 стандартных отклонений, но при повторном
запуске результат всегда один и тот же.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Пакет jacobi_eigen из ЛР1 нужен только для проверки бэкенда "jacobi";
# если соседний каталог на месте, делаем его доступным без установки.
_LAB1_SRC = Path(__file__).resolve().parents[2] / "lab1" / "src"
if _LAB1_SRC.is_dir() and str(_LAB1_SRC) not in sys.path:
    sys.path.append(str(_LAB1_SRC))


@pytest.fixture(params=["goe", "pm1", "pm1-sum", "normal-mirror"])
def ensemble_name(request) -> str:
    """Параметризация по всем ансамблям."""
    return request.param

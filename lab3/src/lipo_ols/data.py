r"""Выборка ``lipo.csv``: 82 вещества, коэффициент липофильности и 24 регрессора.

Файл лежит в корне репозитория (его выдали вместе с заданием). Столбцы:
название вещества, :math:`\log P` и 24 переменные из табл. 1 методички в
том же порядке. Путь к файлу ищется так:

1. аргумент ``path`` функции :func:`load_lipo`;
2. переменная окружения ``LIPO_CSV``;
3. ``lipo.csv`` в корне репозитория (на два уровня выше каталога ``lab3``).

>>> data = load_lipo()
>>> data.n, data.k
(82, 24)
>>> data.columns[:3], data.columns[-2:]
(('RNH2', 'R2NH', 'R3N'), ('PSA', 'MR'))
>>> data.design().shape          # с константой: n × (k + 1)
(82, 25)
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

__all__ = [
    "DESCRIPTORS",
    "DESCRIPTOR_NAMES",
    "LipoData",
    "default_data_path",
    "duplicate_groups",
    "load_lipo",
]

#: Регрессоры из табл. 1 методички: имя столбца → физический смысл.
DESCRIPTORS: Dict[str, str] = {
    "RNH2": "первичные аминогруппы",
    "R2NH": "вторичные аминогруппы",
    "R3N": "третичные аминогруппы",
    "ROPO3": "фосфатные группы",
    "ROH": "гидроксильные группы",
    "RCHO": "альдегидные группы",
    "RCOR": "кетоновые группы",
    "RCOOH": "кислотные (карбоксильные) группы",
    "RCOOR": "сложноэфирные группы",
    "ROR": "простые эфирные группы",
    "RSO2NR": "сульфонамидные группы",
    "RSR": "тиоэфирные группы",
    "RF": "фторалкильные группы",
    "RCl": "хлоралкильные группы",
    "RBr": "бромалкильные группы",
    "RSO2R": "сульфоновые группы",
    "C": "атомы углерода",
    "RINGS": "циклы (любые)",
    "AROMATIC": "ароматические кольца",
    "HBA1": "возможные точки образования водородной связи",
    "HBA2": "атомы-акцепторы водородной связи",
    "HBD": "атомы-доноры водородной связи",
    "PSA": "площадь полярной поверхности, Å²",
    "MR": "мольная рефракция, см³/моль",
}

DESCRIPTOR_NAMES: Tuple[str, ...] = tuple(DESCRIPTORS)

_ROOT = Path(__file__).resolve().parents[3]

PathLike = Union[str, "os.PathLike[str]"]


def default_data_path() -> Path:
    """Путь к ``lipo.csv`` по умолчанию (переменная ``LIPO_CSV`` или корень репозитория)."""
    env = os.environ.get("LIPO_CSV")
    return Path(env) if env else _ROOT / "lipo.csv"


@dataclass(frozen=True)
class LipoData:
    """Выборка: названия веществ, отклик :math:`y = \\log P` и матрица регрессоров.

    Attributes
    ----------
    names:
        названия веществ (без концевых пробелов, которые есть в файле);
    y:
        вектор :math:`\\log P`, форма ``(n,)``;
    X:
        регрессоры без константы, форма ``(n, k)``;
    columns:
        имена столбцов ``X``.
    """

    names: Tuple[str, ...]
    y: np.ndarray
    X: np.ndarray
    columns: Tuple[str, ...]

    def __post_init__(self) -> None:
        if self.X.ndim != 2 or self.y.ndim != 1:
            raise ValueError("ожидаются y формы (n,) и X формы (n, k)")
        if self.X.shape != (len(self.names), len(self.columns)) or self.y.shape[0] != len(self.names):
            raise ValueError("размеры names, y, X и columns не согласованы")

    @property
    def n(self) -> int:
        """Число наблюдений."""
        return self.X.shape[0]

    @property
    def k(self) -> int:
        """Число регрессоров (без константы)."""
        return self.X.shape[1]

    def column(self, name: str) -> np.ndarray:
        """Столбец регрессора по имени."""
        return self.X[:, self.columns.index(name)]

    def index(self, name: str) -> int:
        """Номер вещества по названию."""
        return self.names.index(name)

    def subset(self, columns: Sequence[str]) -> "LipoData":
        """Та же выборка с частью регрессоров (порядок — как в ``columns``)."""
        missing = [c for c in columns if c not in self.columns]
        if missing:
            raise KeyError(f"нет регрессоров: {', '.join(missing)}")
        idx = [self.columns.index(c) for c in columns]
        return LipoData(self.names, self.y, self.X[:, idx], tuple(columns))

    def rows(self, index: Sequence[int]) -> "LipoData":
        """Часть наблюдений (для перекрёстной проверки)."""
        index = np.asarray(index)
        return LipoData(tuple(self.names[i] for i in index), self.y[index], self.X[index], self.columns)

    def design(self, intercept: bool = True) -> np.ndarray:
        """Матрица плана: первый столбец из единиц (константа :math:`\\beta_0`), затем ``X``."""
        if not intercept:
            return self.X.copy()
        return np.column_stack([np.ones(self.n), self.X])

    def labels(self, intercept: bool = True) -> Tuple[str, ...]:
        """Имена коэффициентов в порядке столбцов :meth:`design`."""
        return (("const",) if intercept else ()) + self.columns


def load_lipo(path: Optional[PathLike] = None) -> LipoData:
    """Прочитать ``lipo.csv`` и проверить его структуру.

    Raises
    ------
    FileNotFoundError
        файла нет;
    ValueError
        неожиданные столбцы, пустые или нечисловые значения.
    """
    path = Path(path) if path is not None else default_data_path()
    if not path.is_file():
        raise FileNotFoundError(f"не найден файл с выборкой: {path} (задайте путь или LIPO_CSV)")
    with open(path, encoding="utf-8-sig", newline="") as handle:
        rows = [row for row in csv.reader(handle) if row]
    if not rows:
        raise ValueError(f"{path}: пустой файл")
    header = [cell.strip() for cell in rows[0]]
    if len(header) < 3 or header[1] != "logP":
        raise ValueError(f"{path}: ожидается заголовок «Вещество,logP,<регрессоры>», получено {header[:3]}")
    columns = tuple(header[2:])
    names: List[str] = []
    values: List[List[float]] = []
    for line, row in enumerate(rows[1:], start=2):
        if len(row) != len(header):
            raise ValueError(f"{path}:{line}: {len(row)} значений вместо {len(header)}")
        try:
            numbers = [float(cell) for cell in row[1:]]
        except ValueError as exc:
            raise ValueError(f"{path}:{line}: нечисловое значение ({exc})") from None
        if not np.all(np.isfinite(numbers)):
            raise ValueError(f"{path}:{line}: пропуск или бесконечность")
        names.append(row[0].strip())
        values.append(numbers)
    table = np.asarray(values, dtype=float)
    return LipoData(tuple(names), table[:, 0].copy(), table[:, 1:].copy(), columns)


def duplicate_groups(data: LipoData) -> List[List[int]]:
    """Группы веществ с *одинаковыми* строками ``X`` (изомеры, неразличимые регрессорами).

    Внутри группы любая модель вида :math:`\\hat y = f(x)` даёт один прогноз,
    поэтому разброс :math:`\\log P` внутри групп — нижняя граница ошибки
    (так называемая «чистая ошибка»).

    >>> [[load_lipo().names[i] for i in g] for g in duplicate_groups(load_lipo())][0]
    ['Sulfadimethoxine', 'Sulfadoxine']
    """
    groups: Dict[bytes, List[int]] = {}
    for i, row in enumerate(data.X):
        groups.setdefault(row.tobytes(), []).append(i)
    return [g for g in groups.values() if len(g) > 1]

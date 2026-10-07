r"""Выборка ``lipo.csv`` и стандартизация признаков.

Файл тот же, что в ЛР3: 82 вещества, :math:`\log P` (десятичный логарифм,
см. отчёт ЛР3) и 24 регрессора из табл. 1 методички. Путь ищется так:
аргумент ``path``, переменная окружения ``LIPO_CSV``, ``lipo.csv`` в корне
репозитория.

Регрессоры измерены в разных единицах: индикаторы групп принимают значения
0–3, число атомов углерода — до 40, площадь PSA — до 200 Å². Для
градиентного спуска это плохо: шаг, подходящий для PSA, слишком мал для
индикаторов, и наоборот. Поэтому перед обучением каждый признак и отклик
приводятся к нулевому среднему и единичной дисперсии (:class:`Standardizer`),
а прогнозы переводятся обратно в единицы :math:`\log P`.

>>> data = load_lipo()
>>> data.n, data.k
(82, 24)
>>> scaler = Standardizer.fit(data.X, data.y)
>>> Z, t = scaler.transform(data.X, data.y)
>>> abs(float(Z.mean())) < 1e-12, round(float(t.std()), 6)
(True, 1.0)
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

__all__ = ["DESCRIPTORS", "LipoData", "Standardizer", "default_data_path", "duplicate_groups", "load_lipo"]

#: Регрессоры из табл. 1 методички: имя столбца → физический смысл.
DESCRIPTORS: Dict[str, str] = {
    "RNH2": "первичные аминогруппы", "R2NH": "вторичные аминогруппы", "R3N": "третичные аминогруппы",
    "ROPO3": "фосфатные группы", "ROH": "гидроксильные группы", "RCHO": "альдегидные группы",
    "RCOR": "кетоновые группы", "RCOOH": "карбоксильные группы", "RCOOR": "сложноэфирные группы",
    "ROR": "простые эфирные группы", "RSO2NR": "сульфонамидные группы", "RSR": "тиоэфирные группы",
    "RF": "фторалкильные группы", "RCl": "хлоралкильные группы", "RBr": "бромалкильные группы",
    "RSO2R": "сульфоновые группы", "C": "атомы углерода", "RINGS": "циклы", "AROMATIC": "ароматические кольца",
    "HBA1": "точки образования водородной связи", "HBA2": "атомы-акцепторы водородной связи",
    "HBD": "атомы-доноры водородной связи", "PSA": "площадь полярной поверхности, Å²",
    "MR": "мольная рефракция, см³/моль",
}

_ROOT = Path(__file__).resolve().parents[3]
PathLike = Union[str, "os.PathLike[str]"]


def default_data_path() -> Path:
    """``LIPO_CSV`` или ``lipo.csv`` в корне репозитория."""
    env = os.environ.get("LIPO_CSV")
    return Path(env) if env else _ROOT / "lipo.csv"


@dataclass(frozen=True)
class LipoData:
    """Названия веществ, отклик :math:`y = \\log P` (форма ``(n,)``) и регрессоры ``X`` (форма ``(n, k)``)."""

    names: Tuple[str, ...]
    y: np.ndarray
    X: np.ndarray
    columns: Tuple[str, ...]

    @property
    def n(self) -> int:
        return self.X.shape[0]

    @property
    def k(self) -> int:
        return self.X.shape[1]

    def index(self, name: str) -> int:
        return self.names.index(name)

    def column(self, name: str) -> np.ndarray:
        return self.X[:, self.columns.index(name)]


def load_lipo(path: Optional[PathLike] = None) -> LipoData:
    """Прочитать ``lipo.csv`` (структура проверяется так же, как в ЛР3)."""
    path = Path(path) if path is not None else default_data_path()
    if not path.is_file():
        raise FileNotFoundError(f"не найден файл с выборкой: {path} (задайте путь или LIPO_CSV)")
    with open(path, encoding="utf-8-sig", newline="") as handle:
        rows = [row for row in csv.reader(handle) if row]
    header = [cell.strip() for cell in rows[0]] if rows else []
    if len(header) < 3 or header[1] != "logP":
        raise ValueError(f"{path}: ожидается заголовок «Вещество,logP,<регрессоры>»")
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
    return LipoData(tuple(names), table[:, 0].copy(), table[:, 1:].copy(), tuple(header[2:]))


def duplicate_groups(X: np.ndarray) -> List[List[int]]:
    """Группы строк с одинаковыми признаками (изомеры, которые сеть не может различить)."""
    groups: Dict[bytes, List[int]] = {}
    for i, row in enumerate(np.asarray(X, dtype=float)):
        groups.setdefault(row.tobytes(), []).append(i)
    return [g for g in groups.values() if len(g) > 1]


@dataclass(frozen=True)
class Standardizer:
    """Линейное преобразование :math:`z = (x - \\mu)/\\sigma` для признаков и отклика.

    Параметры оцениваются только по обучающей части (при перекрёстной
    проверке — по обучающим блокам), иначе контрольные данные «подсмотрены».
    Признак с нулевой дисперсией (например, индикатор, равный нулю во всех
    обучающих веществах) не масштабируется.
    """

    x_mean: np.ndarray
    x_std: np.ndarray
    y_mean: float
    y_std: float

    @classmethod
    def fit(cls, X: np.ndarray, y: np.ndarray) -> "Standardizer":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        sx = X.std(axis=0)
        sx = np.where(sx > 0, sx, 1.0)
        sy = float(y.std()) or 1.0
        return cls(X.mean(axis=0), sx, float(y.mean()), sy)

    def transform_x(self, X: np.ndarray) -> np.ndarray:
        return (np.asarray(X, dtype=float) - self.x_mean) / self.x_std

    def transform_y(self, y: np.ndarray) -> np.ndarray:
        return (np.asarray(y, dtype=float) - self.y_mean) / self.y_std

    def transform(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return self.transform_x(X), self.transform_y(y)

    def inverse_y(self, t: np.ndarray) -> np.ndarray:
        """Из стандартизованного отклика обратно в единицы :math:`\\log P`."""
        return np.asarray(t, dtype=float) * self.y_std + self.y_mean

    def subset(self, index: Sequence[int]) -> "Standardizer":
        """Те же параметры для части признаков."""
        idx = np.asarray(index)
        return Standardizer(self.x_mean[idx], self.x_std[idx], self.y_mean, self.y_std)

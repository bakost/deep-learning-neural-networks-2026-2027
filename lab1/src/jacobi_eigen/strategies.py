r"""Стратегии выбора зануляемого элемента :math:`A_{pq}`.

Раздел 1.6 методички отмечает, что порядок обхода недиагональных элементов
можно выбирать произвольно, и разные порядки дают разную эффективность.
Здесь реализованы три классических варианта, оформленные паттерном
«Стратегия», так что решатель ничего не знает о конкретном порядке обхода.

===========================  ==========================================
Стратегия                    Идея
===========================  ==========================================
:class:`CyclicStrategy`      Обход по строкам верхнего треугольника
:class:`MaxElementStrategy`  Каждый раз ищется максимальный по модулю
                             недиагональный элемент (классический
                             «ручной» метод Якоби)
:class:`RoundRobinStrategy`  Турнирное разбиение: свип делится на
                             раунды непересекающихся пар — вариант,
                             упомянутый в сноске про распараллеливание
===========================  ==========================================

Понятие *свип* (sweep) — один полный проход по всем :math:`n(n-1)/2` парам
индексов; именно в свипах измеряется сходимость метода Якоби.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Iterator, List, Optional, Tuple, Type

import numpy as np

from .exceptions import InvalidParameterError

__all__ = [
    "PivotStrategy",
    "CyclicStrategy",
    "MaxElementStrategy",
    "RoundRobinStrategy",
    "get_strategy",
    "available_strategies",
]

Pivot = Tuple[int, int]


class PivotStrategy(ABC):
    """Абстрактная стратегия выбора пар индексов для зануления."""

    #: Короткое имя стратегии.
    name: str = "abstract"

    @abstractmethod
    def sweep(self, matrix: np.ndarray) -> Iterator[Pivot]:
        """Вернуть последовательность пар ``(p, q)``, ``p < q``, на один свип.

        Parameters
        ----------
        matrix:
            Текущая матрица :math:`A_i`. Стратегия может её читать (но не
            изменять) — например, чтобы найти максимальный элемент.

        Yields
        ------
        tuple of int
            Пара индексов ``p < q``.
        """

    @staticmethod
    def sweep_length(n: int) -> int:
        """Число вращений в одном полном свипе: :math:`n(n-1)/2`."""
        return n * (n - 1) // 2

    def __repr__(self) -> str:  # pragma: no cover - косметика
        return f"{type(self).__name__}()"


class CyclicStrategy(PivotStrategy):
    """Циклический обход верхнего треугольника по строкам.

    Порядок: ``(0,1), (0,2), ..., (0,n-1), (1,2), ..., (n-2,n-1)``.
    Стандартный выбор для программной реализации: не требует поиска
    максимума и обеспечивает квадратичную сходимость на поздних свипах.

    Examples
    --------
    >>> list(CyclicStrategy().sweep(np.zeros((3, 3))))
    [(0, 1), (0, 2), (1, 2)]
    """

    name = "cyclic"

    def sweep(self, matrix: np.ndarray) -> Iterator[Pivot]:
        n = matrix.shape[0]
        for p in range(n - 1):
            for q in range(p + 1, n):
                yield p, q


class MaxElementStrategy(PivotStrategy):
    r"""Классический метод Якоби: зануляется максимальный по модулю элемент.

    На каждом шаге ищется :math:`\max_{p<q} |A_{pq}|`, что стоит
    :math:`O(n^2)` операций — дороже самого вращения :math:`O(n)`. Зато
    число вращений минимально, поэтому стратегия удобна для ручного счёта и
    как эталон при отладке.

    Parameters
    ----------
    rotations_per_sweep:
        Сколько вращений считать одним «свипом». По умолчанию
        :math:`n(n-1)/2`, как и у циклической стратегии, — чтобы лимит
        ``max_sweeps`` означал одно и то же для всех стратегий.
    """

    name = "max"

    def __init__(self, rotations_per_sweep: Optional[int] = None) -> None:
        self.rotations_per_sweep = rotations_per_sweep

    def sweep(self, matrix: np.ndarray) -> Iterator[Pivot]:
        n = matrix.shape[0]
        count = self.rotations_per_sweep or self.sweep_length(n)
        for _ in range(count):
            yield self.largest_off_diagonal(matrix)

    @staticmethod
    def largest_off_diagonal(matrix: np.ndarray) -> Pivot:
        """Индексы максимального по модулю наддиагонального элемента."""
        magnitude = np.abs(np.triu(matrix, k=1))
        flat = int(np.argmax(magnitude))
        p, q = divmod(flat, matrix.shape[1])
        return p, q


class RoundRobinStrategy(PivotStrategy):
    """Турнирный («параллельный») порядок обхода.

    Свип разбивается на :math:`n - 1` раундов (для чётного :math:`n`), внутри
    каждого раунда пары индексов не пересекаются. Такой порядок упоминается
    в методичке как наиболее удобный для распараллеливания: вращения одного
    раунда затрагивают непересекающиеся строки и столбцы, поэтому их можно
    выполнять независимо.

    Реализована классическая схема «круглого стола»: один индекс
    зафиксирован, остальные циклически сдвигаются.

    Examples
    --------
    >>> pairs = list(RoundRobinStrategy().sweep(np.zeros((4, 4))))
    >>> len(pairs) == 4 * 3 // 2
    True
    >>> sorted(pairs) == [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    True
    """

    name = "round-robin"

    def sweep(self, matrix: np.ndarray) -> Iterator[Pivot]:
        n = matrix.shape[0]
        if n < 2:
            return
        # Для нечётного n добавляем фиктивного участника: его пара «отдыхает».
        players: List[int] = list(range(n))
        if n % 2 == 1:
            players.append(-1)
        size = len(players)
        for _ in range(size - 1):
            for k in range(size // 2):
                left = players[k]
                right = players[size - 1 - k]
                if left == -1 or right == -1:
                    continue
                yield (left, right) if left < right else (right, left)
            # Циклический сдвиг всех участников, кроме первого.
            players = [players[0]] + [players[-1]] + players[1:-1]

    def rounds(self, n: int) -> List[List[Pivot]]:
        """Разбиение свипа на раунды независимых пар (для распараллеливания).

        Returns
        -------
        list of list of tuple
            Каждый вложенный список — раунд, внутри которого все пары
            индексов попарно не пересекаются.
        """
        players: List[int] = list(range(n))
        if n % 2 == 1:
            players.append(-1)
        size = len(players)
        result: List[List[Pivot]] = []
        for _ in range(size - 1):
            round_pairs: List[Pivot] = []
            for k in range(size // 2):
                left, right = players[k], players[size - 1 - k]
                if left == -1 or right == -1:
                    continue
                round_pairs.append((left, right) if left < right else (right, left))
            result.append(round_pairs)
            players = [players[0]] + [players[-1]] + players[1:-1]
        return result


_STRATEGIES: Dict[str, Type[PivotStrategy]] = {
    CyclicStrategy.name: CyclicStrategy,
    MaxElementStrategy.name: MaxElementStrategy,
    RoundRobinStrategy.name: RoundRobinStrategy,
}


def available_strategies() -> tuple:
    """Имена всех доступных стратегий."""
    return tuple(_STRATEGIES)


def get_strategy(strategy: "str | PivotStrategy") -> PivotStrategy:
    """Получить экземпляр стратегии по имени или вернуть переданный объект.

    Raises
    ------
    InvalidParameterError
        Если имя неизвестно или объект не является стратегией.
    """
    if isinstance(strategy, PivotStrategy):
        return strategy
    if isinstance(strategy, str):
        try:
            return _STRATEGIES[strategy]()
        except KeyError:
            raise InvalidParameterError(
                f"Неизвестная стратегия {strategy!r}. Доступны: {', '.join(sorted(_STRATEGIES))}."
            ) from None
    raise InvalidParameterError(
        f"strategy должна быть строкой или экземпляром PivotStrategy, "
        f"получен {type(strategy).__name__}."
    )

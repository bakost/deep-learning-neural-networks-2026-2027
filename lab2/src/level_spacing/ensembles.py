r"""Ансамбли случайных вещественных симметричных матриц.

Каждый ансамбль задаётся двумя независимыми выборами:

* **распределение элементов** исходной матрицы :math:`A`:

  - ``"normal"`` — :math:`\mathcal{N}(0, 1)`, формула (1) методички с
    :math:`\sigma = 1`;
  - ``"rademacher"`` — значения :math:`\pm 1` с вероятностями :math:`1/2`
    (пункт 9 задания);

* **способ получить из** :math:`A` **симметричную матрицу** :math:`H`:

  - ``"sum"`` — :math:`H = A + A^{T}`, как в разделе 1.3 методички. Диагональ
    удваивается, внедиагональные элементы складываются попарно;
  - ``"mirror"`` — верхний треугольник :math:`A` (с диагональю) отражается
    в нижний: :math:`H_{ij} = H_{ji} = A_{ij}` при :math:`i \le j`. Элементы
    :math:`H` распределены *ровно так же*, как элементы :math:`A`.

Зарегистрированные ансамбли:

=================  ==============  ===========  ==================================
Имя                Элементы A      Конструкция  Где используется
=================  ==============  ===========  ==================================
``goe``            N(0, 1)         sum          пункты 1–8 (гауссов ортогональный)
``pm1``            ±1              mirror       пункт 9: элементы *матрицы* ±1
``pm1-sum``        ±1              sum          пункт 9: шаги 1–8 буквально
``normal-mirror``  N(0, 1)         mirror       сравнение: почему именно A + Aᵀ
=================  ==============  ===========  ==================================

Ортогональность ансамбля
------------------------
Гауссов *ортогональный* ансамбль называется так потому, что его
распределение не меняется при любом ортогональном преобразовании
:math:`H \to Q^T H Q`, :math:`Q^T Q = I`. У :math:`A + A^T` дисперсия
диагонали равна 4, а вне диагонали — 2, и плотность

.. math::
    P(H) \propto \exp\Bigl(-\sum_i \frac{H_{ii}^2}{8} - \sum_{i<j} \frac{H_{ij}^2}{4}\Bigr)
         = \exp\Bigl(-\frac{\operatorname{tr} H^2}{8}\Bigr)

зависит только от :math:`\operatorname{tr} H^2 = \operatorname{tr}(Q^T H Q)^2`.
Линейное отображение :math:`H \to Q^T H Q` сохраняет объём, поэтому
:math:`Q^T H Q` распределена так же, как :math:`H`. Соотношение дисперсий
2 : 1 здесь обязательно: у ``normal-mirror`` все дисперсии равны 1, и
инвариантности нет (для :math:`2 \times 2` догадка Вигнера перестаёт быть
точной, см. :func:`level_spacing.distributions.normal_mirror_2x2_pdf`).
Проверка — :func:`level_spacing.experiments.orthogonal_invariance`.

Случайные ортогональные матрицы для этой проверки даёт
:func:`random_orthogonal`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple, Union

import numpy as np

__all__ = [
    "ENSEMBLES",
    "Ensemble",
    "RandomState",
    "available_ensembles",
    "get_ensemble",
    "make_rng",
    "random_orthogonal",
    "rotation_2x2",
    "sample_matrices",
]

#: Всё, из чего можно получить генератор: зерно, последовательность зёрен,
#: готовый :class:`numpy.random.Generator` или ``None`` (случайное зерно).
RandomState = Union[None, int, Tuple[int, ...], np.random.Generator]

_ENTRY_SAMPLERS: Dict[str, Callable[[np.random.Generator, Tuple[int, ...]], np.ndarray]] = {
    "normal": lambda rng, shape: rng.standard_normal(shape),
    "rademacher": lambda rng, shape: 2.0 * rng.integers(0, 2, size=shape) - 1.0,
}

_CONSTRUCTIONS = ("sum", "mirror")


def make_rng(rng: RandomState = None) -> np.random.Generator:
    """Привести зерно (или готовый генератор) к :class:`numpy.random.Generator`.

    >>> make_rng(1).standard_normal() == make_rng(1).standard_normal()
    True
    """
    if isinstance(rng, np.random.Generator):
        return rng
    return np.random.default_rng(rng)


def check_int(name: str, value: object, minimum: int) -> int:
    """Проверить, что параметр — целое число не меньше ``minimum``."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} должен быть целым числом, получено {value!r}")
    if value < minimum:
        raise ValueError(f"{name} должен быть не меньше {minimum}, получено {value}")
    return int(value)


@dataclass(frozen=True)
class Ensemble:
    """Статистический ансамбль симметричных матриц.

    Attributes
    ----------
    name:
        короткое имя (ключ в :data:`ENSEMBLES`);
    entries:
        распределение элементов исходной матрицы: ``"normal"`` или
        ``"rademacher"``;
    construction:
        способ симметризации: ``"sum"`` (:math:`A + A^T`) или ``"mirror"``;
    title:
        подпись для графиков и таблиц.
    """

    name: str
    entries: str
    construction: str
    title: str

    def __post_init__(self) -> None:
        if self.entries not in _ENTRY_SAMPLERS:
            raise ValueError(
                f"неизвестное распределение элементов {self.entries!r}; "
                f"доступны: {', '.join(sorted(_ENTRY_SAMPLERS))}"
            )
        if self.construction not in _CONSTRUCTIONS:
            raise ValueError(
                f"неизвестный способ симметризации {self.construction!r}; "
                f"доступны: {', '.join(_CONSTRUCTIONS)}"
            )

    @property
    def is_discrete(self) -> bool:
        """Принимают ли элементы конечное число значений."""
        return self.entries == "rademacher"

    def sample(self, n: int, size: int, rng: RandomState = None) -> np.ndarray:
        """Сгенерировать ``size`` матриц ``n x n``; результат формы ``(size, n, n)``.

        >>> H = get_ensemble("pm1").sample(3, 2, rng=0)
        >>> H.shape, bool(np.all(H == np.swapaxes(H, -1, -2)))
        ((2, 3, 3), True)
        >>> sorted(set(H.ravel().tolist()))
        [-1.0, 1.0]
        """
        n = check_int("n", n, 1)
        size = check_int("size", size, 0)
        a = _ENTRY_SAMPLERS[self.entries](make_rng(rng), (size, n, n))
        if self.construction == "sum":
            return a + np.swapaxes(a, -1, -2)
        upper = np.triu(a)
        return upper + np.swapaxes(np.triu(a, 1), -1, -2)


ENSEMBLES: Dict[str, Ensemble] = {
    e.name: e
    for e in (
        Ensemble("goe", "normal", "sum", "N(0,1), H = A + Aᵀ (GOE)"),
        Ensemble("pm1", "rademacher", "mirror", "±1, симметричная матрица"),
        Ensemble("pm1-sum", "rademacher", "sum", "±1, H = A + Aᵀ"),
        Ensemble("normal-mirror", "normal", "mirror", "N(0,1), симметричная матрица"),
    )
}


def available_ensembles() -> Tuple[str, ...]:
    """Имена зарегистрированных ансамблей."""
    return tuple(ENSEMBLES)


def get_ensemble(ensemble: Union[str, Ensemble]) -> Ensemble:
    """Найти ансамбль по имени (объект :class:`Ensemble` возвращается как есть).

    >>> get_ensemble("goe").construction
    'sum'
    """
    if isinstance(ensemble, Ensemble):
        return ensemble
    try:
        return ENSEMBLES[ensemble]
    except (KeyError, TypeError):
        raise ValueError(
            f"неизвестный ансамбль {ensemble!r}; доступны: {', '.join(ENSEMBLES)}"
        ) from None


def sample_matrices(
    ensemble: Union[str, Ensemble], n: int, size: int, rng: RandomState = None
) -> np.ndarray:
    """Сгенерировать ``size`` матриц ``n x n`` из ансамбля (функциональная форма).

    >>> sample_matrices("goe", n=2, size=5, rng=1).shape
    (5, 2, 2)
    """
    return get_ensemble(ensemble).sample(n, size, rng)


def random_orthogonal(n: int, size: Optional[int] = None, rng: RandomState = None) -> np.ndarray:
    r"""Случайная ортогональная матрица, равномерно распределённая на :math:`O(n)` (мера Хаара).

    QR-разложение (то есть процесс Грама–Шмидта) гауссовой матрицы
    :math:`Z = QR` даёт ортогональную :math:`Q`. Чтобы распределение было
    именно равномерным, знаки столбцов :math:`Q` согласуются с диагональю
    :math:`R` (F. Mezzadri, Notices AMS, 2007): иначе результат зависит от
    соглашения о знаках в LAPACK.

    >>> q = random_orthogonal(4, rng=0)
    >>> bool(np.allclose(q.T @ q, np.eye(4)))
    True
    >>> random_orthogonal(3, size=5, rng=0).shape
    (5, 3, 3)
    """
    n = check_int("n", n, 1)
    shape = (n, n) if size is None else (check_int("size", size, 0), n, n)
    z = make_rng(rng).standard_normal(shape)
    q, r = np.linalg.qr(z)
    signs = np.sign(np.diagonal(r, axis1=-2, axis2=-1))
    signs[signs == 0] = 1.0
    return q * signs[..., None, :]


def rotation_2x2(angle: float) -> np.ndarray:
    """Матрица поворота плоскости на угол ``angle`` (радианы).

    >>> rotation_2x2(np.pi / 2).round(12)
    array([[ 0., -1.],
           [ 1.,  0.]])
    """
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s], [s, c]])

r"""Собственные числа, расстояния между соседними уровнями и их нормировка.

Это пункты 2–4 задания:

2. найти собственные числа каждой матрицы и упорядочить их по возрастанию —
   :func:`eigenvalues`;
3. найти разность между *фиксированной* парой соседних собственных чисел
   :math:`s = \lambda_{k+1} - \lambda_k` (сноски 2 и 4 методички) —
   :func:`pair_spacings`;
4. нормировать разности на среднее — :func:`normalize`.

Нумерация собственных чисел, как в методичке, с единицы:
:math:`\lambda_1 \le \lambda_2 \le \dots \le \lambda_n`. По умолчанию берётся
центральная пара :math:`k = n/2` (для :math:`n = 2, 4, 16` это пары
(1, 2), (2, 3), (8, 9)): в центре спектра плотность уровней максимальна и
почти постоянна, поэтому локальная статистика там чище всего.

Для больших ансамблей матрицы не хранятся целиком: :func:`simulate_eigenvalues`
генерирует их порциями и сохраняет только спектры. Порции берутся из одного
генератора подряд, поэтому результат не зависит от размера порции.
"""

from __future__ import annotations

from typing import Iterator, Optional, Union

import numpy as np

from .ensembles import Ensemble, RandomState, check_int, get_ensemble, make_rng

__all__ = [
    "BACKENDS",
    "DEFAULT_CHUNK_SIZE",
    "all_spacings",
    "central_pair",
    "eigenvalues",
    "normalize",
    "pair_spacings",
    "simulate_eigenvalues",
    "simulate_spacings",
]

#: Способы вычислить собственные числа: LAPACK через NumPy или метод Якоби из ЛР1.
BACKENDS = ("numpy", "jacobi")

#: Сколько матриц генерировать за раз (≈ 100 МБ для 16×16).
DEFAULT_CHUNK_SIZE = 50_000


def eigenvalues(matrices: np.ndarray, backend: str = "numpy") -> np.ndarray:
    """Собственные числа пачки симметричных матриц по возрастанию.

    Parameters
    ----------
    matrices:
        массив формы ``(n, n)`` или ``(size, n, n)``;
    backend:
        ``"numpy"`` — :func:`numpy.linalg.eigvalsh` (LAPACK ``syevd``), вся пачка
        за один вызов; ``"jacobi"`` — метод вращений Якоби из лабораторной
        работы 1 (пакет ``jacobi_eigen``), по одной матрице. Второй вариант в
        сотни раз медленнее и нужен для независимой проверки.

    Returns
    -------
    numpy.ndarray
        массив формы ``(..., n)``, в каждой строке — упорядоченный спектр.

    >>> eigenvalues(np.array([[2.0, 1.0], [1.0, 2.0]]))
    array([1., 3.])
    """
    matrices = np.asarray(matrices, dtype=float)
    if matrices.ndim < 2 or matrices.shape[-1] != matrices.shape[-2]:
        raise ValueError(f"ожидались квадратные матрицы, получен массив формы {matrices.shape}")
    if backend == "numpy":
        values = np.linalg.eigvalsh(matrices)
    elif backend == "jacobi":
        values = _jacobi_eigenvalues(matrices)
    else:
        raise ValueError(f"неизвестный бэкенд {backend!r}; доступны: {', '.join(BACKENDS)}")
    # eigvalsh уже возвращает спектр по возрастанию, но пункт 2 задания требует
    # упорядочить явно — и это делает результат независимым от бэкенда.
    return np.sort(values, axis=-1)


def _jacobi_eigenvalues(matrices: np.ndarray) -> np.ndarray:
    try:
        from jacobi_eigen import contracts_disabled, jacobi_eigh
    except ImportError as exc:
        raise ImportError(
            "бэкенд 'jacobi' использует пакет jacobi_eigen из лабораторной работы 1; "
            "установите его (pip install -e ../lab1) или добавьте ../lab1/src в PYTHONPATH"
        ) from exc
    flat = matrices.reshape(-1, *matrices.shape[-2:])
    # Контракты ЛР1 замедляют счёт почти вдвое, а входные матрицы здесь
    # заведомо корректны — отключаем их на время расчёта.
    with contracts_disabled():
        values = [jacobi_eigh(m, compute_eigenvectors=False).eigenvalues for m in flat]
    return np.asarray(values).reshape(matrices.shape[:-1])


def central_pair(n: int) -> int:
    """Номер ``k`` центральной пары :math:`(\\lambda_k, \\lambda_{k+1})` (нумерация с 1).

    >>> [central_pair(n) for n in (2, 3, 4, 16)]
    [1, 1, 2, 8]
    """
    n = check_int("n", n, 2)
    return n // 2


def _check_pair(k: Optional[int], n: int) -> int:
    if k is None:
        return central_pair(n)
    k = check_int("k", k, 1)
    if k > n - 1:
        raise ValueError(f"для матриц {n}x{n} номер пары k должен быть от 1 до {n - 1}, получено {k}")
    return k


def pair_spacings(eigvals: np.ndarray, k: Optional[int] = None) -> np.ndarray:
    """Разности :math:`\\lambda_{k+1} - \\lambda_k` для каждого спектра ансамбля.

    Parameters
    ----------
    eigvals:
        упорядоченные спектры, форма ``(size, n)`` (или ``(n,)``);
    k:
        номер пары с единицы, :math:`1 \\le k \\le n-1`; по умолчанию —
        центральная пара :func:`central_pair`.

    >>> pair_spacings(np.array([[0.0, 1.0, 3.0, 6.0]]))       # k = 2
    array([2.])
    >>> pair_spacings(np.array([[0.0, 1.0, 3.0, 6.0]]), k=3)
    array([3.])
    """
    eigvals = np.asarray(eigvals, dtype=float)
    if eigvals.ndim == 0 or eigvals.shape[-1] < 2:
        raise ValueError(f"нужно хотя бы два собственных числа, получен массив формы {eigvals.shape}")
    k = _check_pair(k, eigvals.shape[-1])
    return eigvals[..., k] - eigvals[..., k - 1]


def all_spacings(eigvals: np.ndarray) -> np.ndarray:
    """Все разности соседних собственных чисел, форма ``(size, n - 1)``.

    Столбец ``j`` (с нуля) — это пара ``k = j + 1``.
    """
    return np.diff(np.asarray(eigvals, dtype=float), axis=-1)


def normalize(spacings: np.ndarray) -> np.ndarray:
    """Поделить разности на их выборочное среднее (пункт 4 задания).

    Нормировка убирает масштаб: результат не зависит ни от :math:`\\sigma`
    элементов, ни от множителя при симметризации (:math:`A + A^T` или
    :math:`(A + A^T)/2`), ни от размера матрицы.

    >>> normalize(np.array([1.0, 2.0, 3.0]))
    array([0.5, 1. , 1.5])
    """
    spacings = np.asarray(spacings, dtype=float)
    if spacings.size == 0:
        raise ValueError("нельзя нормировать пустую выборку")
    mean = spacings.mean()
    if not np.isfinite(mean) or mean <= 0.0:
        raise ValueError(f"среднее расстояние должно быть положительным, получено {mean}")
    return spacings / mean


def _eigenvalue_chunks(
    ensemble: Ensemble, n: int, size: int, rng: np.random.Generator, chunk_size: int, backend: str
) -> Iterator[np.ndarray]:
    done = 0
    while done < size:
        m = min(chunk_size, size - done)
        yield eigenvalues(ensemble.sample(n, m, rng), backend=backend)
        done += m


def simulate_eigenvalues(
    ensemble: Union[str, Ensemble],
    n: int,
    size: int,
    rng: RandomState = None,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    backend: str = "numpy",
) -> np.ndarray:
    """Сгенерировать ансамбль и вернуть упорядоченные спектры, форма ``(size, n)``.

    >>> ev = simulate_eigenvalues("goe", n=4, size=3, rng=0)
    >>> ev.shape, bool(np.all(np.diff(ev, axis=1) >= 0))
    ((3, 4), True)
    """
    ensemble = get_ensemble(ensemble)
    n = check_int("n", n, 2)
    size = check_int("size", size, 1)
    chunk_size = check_int("chunk_size", chunk_size, 1)
    generator = make_rng(rng)
    return np.concatenate(
        list(_eigenvalue_chunks(ensemble, n, size, generator, chunk_size, backend))
    )


def simulate_spacings(
    ensemble: Union[str, Ensemble],
    n: int,
    size: int,
    rng: RandomState = None,
    *,
    k: Optional[int] = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    backend: str = "numpy",
) -> np.ndarray:
    """Пункты 1–3 целиком: ансамбль → спектры → разности пары ``k`` (без нормировки).

    Хранит только разности, поэтому годится для миллионов матриц.

    >>> s = simulate_spacings("goe", n=2, size=1000, rng=0)
    >>> s.shape, bool(np.all(s > 0))
    ((1000,), True)
    """
    ensemble = get_ensemble(ensemble)
    n = check_int("n", n, 2)
    size = check_int("size", size, 1)
    chunk_size = check_int("chunk_size", chunk_size, 1)
    k = _check_pair(k, n)
    generator = make_rng(rng)
    return np.concatenate(
        [
            pair_spacings(chunk, k)
            for chunk in _eigenvalue_chunks(ensemble, n, size, generator, chunk_size, backend)
        ]
    )

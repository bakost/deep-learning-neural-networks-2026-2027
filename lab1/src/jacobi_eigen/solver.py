r"""Метод Якоби (вращений) — решатель и его публичный интерфейс.

Модуль содержит два уровня API:

* :class:`JacobiEigensolver` — объектный интерфейс: настройки задаются один
  раз при создании, после чего один и тот же решатель применяется к разным
  матрицам;
* :func:`jacobi_eigh` — функциональная обёртка «одна строка — один вызов»
  для простых случаев.

Алгоритм (раздел 2 методички)
-----------------------------
1. :math:`A_0 = A`, :math:`V_0 = I`, :math:`S_0 = \sum_{r \neq s} A_{rs}^2`.
2. Выбирается недиагональный элемент :math:`A_{pq}`.
3. Считается :math:`C = \dfrac{A_{qq} - A_{pp}}{2 A_{pq}}` и

   .. math::
       \mathrm{tg}\,\varphi = \begin{cases}
           \dfrac{1}{C + \sqrt{C^2 + 1}}, & C > 0,\\[2ex]
           \dfrac{1}{C - \sqrt{C^2 + 1}}, & C < 0,
       \end{cases}
       \qquad
       \cos\varphi = \frac{1}{\sqrt{1 + \mathrm{tg}^2\varphi}},
       \qquad
       \sin\varphi = \mathrm{tg}\,\varphi \cos\varphi .

4. Обновляются :math:`A` и :math:`V` (см. :mod:`jacobi_eigen.kernels`).
5. :math:`S_{i+1} = S_i - 2 A_{pq}^2`; итерации продолжаются, пока
   :math:`S > \varepsilon`.

Отличия от буквального текста методички
---------------------------------------
Все они сделаны осознанно и не меняют сути метода:

* **Множитель 2 в пересчёте суммы.** В методичке :math:`S` определена как
  сумма по *всем* недиагональным элементам, но формула (19) вычитает
  :math:`(A^i_{pq})^2` — ровно один элемент. Так как одно вращение зануляет
  сразу два симметричных элемента, :math:`A_{pq}` и :math:`A_{qp}`,
  корректное обновление — :math:`S_{i+1} = S_i - 2(A^i_{pq})^2`.
* **Периодический пересчёт** :math:`S`. Инкрементальная формула накапливает
  ошибку округления, поэтому в конце каждого свипа сумма пересчитывается
  честно по матрице. Это стоит :math:`O(n^2)` против :math:`O(n^3)` на свип
  и полностью исключает ложную остановку.
* **Жёсткий лимит итераций.** ``max_sweeps`` гарантирует, что зацикливание
  невозможно даже при патологических данных: вместо бесконечного цикла
  возбуждается :class:`~jacobi_eigen.exceptions.ConvergenceError`.
* **Случай** :math:`C = 0` (равные диагональные элементы) в формуле (9) не
  определён; берётся ветвь :math:`C \ge 0`, дающая
  :math:`\mathrm{tg}\,\varphi = 1`, то есть поворот на :math:`45^\circ`.
"""

from __future__ import annotations

import math
import time
from typing import Any, Optional, Union

import numpy as np

from .contracts import ensure, invariant, require
from .exceptions import ConvergenceError, InvalidParameterError
from .kernels import RotationKernel, get_kernel
from .result import JacobiResult
from .strategies import PivotStrategy, get_strategy
from .validation import (
    DEFAULT_SYMMETRY_ATOL,
    DEFAULT_SYMMETRY_RTOL,
    off_diagonal_sq,
    validate_matrix,
    validate_positive_int,
    validate_tolerance,
)

__all__ = ["JacobiEigensolver", "jacobi_eigh", "rotation_parameters"]

#: Порог, за которым `sqrt(C^2 + 1)` рискует переполниться; там работает
#: асимптотика `tg(phi) ~ 1 / (2C)`.
_HUGE_C = 1e150

#: Безопасный диапазон модулей элементов. Критерий остановки оперирует
#: *квадратами* элементов, а значит вдвое сужает доступный диапазон порядков:
#: при элементах ~1e300 квадрат даёт `inf`, при ~1e-300 — машинный ноль. И то,
#: и другое ломает критерий остановки (сумма становится `inf` либо `0`), из-за
#: чего метод либо не делает ни одного вращения, либо считает мусор. Матрицы
#: вне этого диапазона предварительно нормируются.
_SAFE_MAX = 1e100
_SAFE_MIN = 1e-100


@require(lambda a_pp, a_qq, a_pq: math.isfinite(a_pp) and math.isfinite(a_qq) and math.isfinite(a_pq),
         "элементы матрицы должны быть конечными")
@ensure(lambda result: abs(result[0]) <= 1.0 and abs(result[1]) <= 1.0,
        "|cos| <= 1 и |sin| <= 1")
@ensure(lambda result: abs(result[0] ** 2 + result[1] ** 2 - 1.0) < 1e-12,
        "cos^2 + sin^2 = 1 (поворот сохраняет норму)")
@ensure(lambda result: abs(result[2]) <= 1.0 + 1e-12,
        "|tg(phi)| <= 1, то есть выбран меньший корень (угол не превышает 45 градусов)")
def rotation_parameters(a_pp: float, a_qq: float, a_pq: float) -> "tuple[float, float, float]":
    r"""Вычислить :math:`(\cos\varphi, \sin\varphi, \mathrm{tg}\,\varphi)` для зануления :math:`A_{pq}`.

    Реализует пункты (b)–(d) алгоритма: формулы (7), (9), (10), (11).
    Выбирается меньший корень квадратного уравнения, что соответствует углу
    :math:`|\varphi| \le 45^\circ` и даёт лучшую сходимость.

    Parameters
    ----------
    a_pp, a_qq, a_pq:
        Элементы :math:`A_{pp}`, :math:`A_{qq}`, :math:`A_{pq}` текущей матрицы.

    Returns
    -------
    tuple of float
        Тройка ``(cos, sin, tg)``. Если ``a_pq == 0``, возвращается
        ``(1.0, 0.0, 0.0)`` — тождественный поворот.

    Examples
    --------
    Пример из методички (зануляем ``A[1, 3]`` матрицы 5x5):

    >>> cos_phi, sin_phi, tan_phi = rotation_parameters(7.0, 19.0, 13.0)
    >>> round(tan_phi, 2), round(cos_phi, 3), round(sin_phi, 3)
    (0.64, 0.842, 0.539)

    Для равных диагональных элементов угол равен 45 градусам:

    >>> cos_phi, sin_phi, tan_phi = rotation_parameters(1.0, 1.0, 2.0)
    >>> round(tan_phi, 12), round(math.degrees(math.atan(tan_phi)))
    (1.0, 45)
    """
    if a_pq == 0.0:
        return 1.0, 0.0, 0.0

    c_value = (a_qq - a_pp) / (2.0 * a_pq)

    if abs(c_value) > _HUGE_C:
        # sqrt(C^2 + 1) ~ |C|, обе ветви формулы (9) вырождаются в 1 / (2C).
        tan_phi = 1.0 / (2.0 * c_value)
    elif c_value >= 0.0:
        tan_phi = 1.0 / (c_value + math.sqrt(c_value * c_value + 1.0))
    else:
        tan_phi = 1.0 / (c_value - math.sqrt(c_value * c_value + 1.0))

    cos_phi = 1.0 / math.sqrt(1.0 + tan_phi * tan_phi)
    sin_phi = tan_phi * cos_phi
    return cos_phi, sin_phi, tan_phi


def _safe_ldexp(value: float, exponent: int) -> float:
    """``value * 2**exponent`` с переполнением в ``inf`` вместо исключения."""
    try:
        return math.ldexp(value, exponent)
    except OverflowError:
        return math.inf


@invariant(lambda self: self.tol > 0 and math.isfinite(self.tol),
           "tol — конечное положительное число")
@invariant(lambda self: self.max_sweeps > 0,
           "max_sweeps положителен, поэтому цикл итераций заведомо конечен")
@invariant(lambda self: self.tol_mode in ("relative", "absolute"),
           "tol_mode имеет допустимое значение")
class JacobiEigensolver:
    r"""Решатель полной симметричной проблемы собственных значений.

    Parameters
    ----------
    tol:
        Требуемая точность. Итерации останавливаются, когда норма
        недиагональной части :math:`\mathrm{off}(A) = \sqrt{S}` становится
        меньше порога:

        * ``tol_mode="relative"`` (по умолчанию) — порог равен
          :math:`\mathrm{tol} \cdot \|A\|_F`;
        * ``tol_mode="absolute"`` — порог равен :math:`\mathrm{tol}`.

        В терминах методички это означает :math:`\varepsilon = \mathrm{tol}^2 \|A\|_F^2`
        и :math:`\varepsilon = \mathrm{tol}^2` соответственно.
    tol_mode:
        ``"relative"`` или ``"absolute"``. Относительный режим выбран по
        умолчанию, потому что он не зависит от масштаба матрицы: для матрицы
        с элементами порядка :math:`10^6` абсолютный порог :math:`10^{-12}`
        недостижим в двойной точности, и метод сообщил бы о расходимости на
        совершенно корректных данных.
    kernel:
        Ядро вращения: ``"vectorized"`` (по умолчанию), ``"scalar"``,
        ``"matrix"`` или собственный экземпляр
        :class:`~jacobi_eigen.kernels.RotationKernel`.
    strategy:
        Стратегия выбора элемента: ``"cyclic"`` (по умолчанию), ``"max"``,
        ``"round-robin"`` или экземпляр
        :class:`~jacobi_eigen.strategies.PivotStrategy`.
    max_sweeps:
        Максимальное число свипов. Метод Якоби для матриц разумного размера
        сходится за 6–12 свипов, значение по умолчанию (100) оставляет
        большой запас и одновременно исключает зацикливание.
    compute_eigenvectors:
        Считать ли матрицу собственных векторов. Отключение экономит время,
        если нужны только собственные числа.
    sort:
        Порядок собственных чисел на выходе: ``"asc"`` (по умолчанию, как у
        :func:`numpy.linalg.eigh`), ``"desc"`` или ``None`` (в порядке,
        полученном алгоритмом).
    symmetry_atol, symmetry_rtol:
        Допуски проверки симметричности входа.
    symmetrize:
        Приводить ли вход к строго симметричному виду ``(A + A.T) / 2``
        после успешной проверки.

    Raises
    ------
    InvalidParameterError
        Если какой-либо параметр имеет недопустимое значение.

    Examples
    --------
    >>> solver = JacobiEigensolver(tol=1e-12)
    >>> result = solver.solve([[4.0, 1.0], [1.0, 4.0]])
    >>> np.round(result.eigenvalues, 10)
    array([3., 5.])
    >>> result.converged
    True

    Один настроенный решатель можно применять к нескольким матрицам:

    >>> solver = JacobiEigensolver(kernel="scalar", strategy="max")
    >>> [round(float(solver.solve(np.diag([m, -m])).eigenvalues[0]), 1) for m in (1, 2)]
    [-1.0, -2.0]
    """

    def __init__(
        self,
        tol: float = 1e-12,
        *,
        tol_mode: str = "relative",
        kernel: Union[str, RotationKernel] = "vectorized",
        strategy: Union[str, PivotStrategy] = "cyclic",
        max_sweeps: int = 100,
        compute_eigenvectors: bool = True,
        sort: Optional[str] = "asc",
        symmetry_atol: float = DEFAULT_SYMMETRY_ATOL,
        symmetry_rtol: float = DEFAULT_SYMMETRY_RTOL,
        symmetrize: bool = True,
    ) -> None:
        if tol_mode not in ("relative", "absolute"):
            raise InvalidParameterError(
                f"tol_mode должен быть 'relative' или 'absolute', получено {tol_mode!r}."
            )
        if sort not in ("asc", "desc", None):
            raise InvalidParameterError(
                f"sort должен быть 'asc', 'desc' или None, получено {sort!r}."
            )

        self.tol = validate_tolerance(tol, "tol")
        self.tol_mode = tol_mode
        self.kernel = get_kernel(kernel)
        self.strategy = get_strategy(strategy)
        self.max_sweeps = validate_positive_int(max_sweeps, "max_sweeps")
        self.compute_eigenvectors = bool(compute_eigenvectors)
        self.sort = sort
        self.symmetry_atol = float(symmetry_atol)
        self.symmetry_rtol = float(symmetry_rtol)
        self.symmetrize = bool(symmetrize)

    # ------------------------------------------------------------------
    # Публичный интерфейс
    # ------------------------------------------------------------------

    @ensure(lambda result: result.converged,
            "успешный возврат означает достигнутую сходимость")
    @ensure(lambda result: result.off_norm <= result.threshold,
            "норма недиагональной части не превышает порог остановки")
    @ensure(lambda result, matrix: result.eigenvalues.shape[0] == np.shape(matrix)[0],
            "число собственных чисел равно размеру матрицы")
    @ensure(lambda result, self: (not self.compute_eigenvectors)
            or result.eigenvectors.shape == (result.size, result.size),
            "матрица собственных векторов квадратная и того же размера")
    def solve(self, matrix: Any) -> JacobiResult:
        r"""Найти собственные числа и векторы симметричной матрицы.

        Parameters
        ----------
        matrix:
            Вещественная симметричная матрица: ``numpy.ndarray``, список
            списков и т. п. Входные данные **не изменяются**.

        Returns
        -------
        JacobiResult
            Собственные числа, собственные векторы и статистика сходимости.

        Raises
        ------
        InvalidMatrixError
            Матрица не прошла валидацию (не квадратная, не вещественная, не
            симметричная, содержит ``NaN``/``inf`` и т. п.) — см.
            :mod:`jacobi_eigen.validation`.
        ConvergenceError
            Не удалось достичь требуемой точности за ``max_sweeps`` свипов.

        Examples
        --------
        >>> solver = JacobiEigensolver()
        >>> result = solver.solve(np.diag([3.0, 1.0, 2.0]))
        >>> result.rotations  # диагональная матрица: вращения не нужны
        0
        >>> np.round(result.eigenvalues, 10)
        array([1., 2., 3.])
        """
        started = time.perf_counter()
        a = validate_matrix(
            matrix,
            symmetry_atol=self.symmetry_atol,
            symmetry_rtol=self.symmetry_rtol,
            symmetrize=self.symmetrize,
        )
        n = a.shape[0]
        v = np.eye(n, dtype=np.float64) if self.compute_eigenvectors else None

        # Нормировка масштаба (см. `_SAFE_MAX`). Множитель — степень двойки,
        # поэтому деление и обратное умножение не теряют ни одного бита
        # мантиссы, а собственные векторы от масштаба вовсе не зависят.
        scale_exponent = self._scale_exponent(a)
        if scale_exponent:
            a = np.ldexp(a, -scale_exponent)

        threshold = self._threshold(a)
        threshold_sq = threshold * threshold

        # S_0 — сумма квадратов недиагональных элементов (пункт 2 алгоритма).
        s_value = off_diagonal_sq(a)
        rotations = 0
        sweeps = 0
        converged = s_value <= threshold_sq

        while not converged and sweeps < self.max_sweeps:
            for p, q in self.strategy.sweep(a):
                a_pq = a[p, q]
                if a_pq == 0.0:
                    continue

                cos_phi, sin_phi, tan_phi = rotation_parameters(a[p, p], a[q, q], a_pq)
                if tan_phi == 0.0:
                    # Вращение вырождено (элемент пренебрежимо мал на фоне
                    # разности диагональных элементов) — зануляем напрямую.
                    a[p, q] = 0.0
                    a[q, p] = 0.0
                    continue

                self.kernel.apply(a, v, p, q, cos_phi, sin_phi, tan_phi)
                rotations += 1
                # Формула (19) с поправкой на симметричную пару элементов.
                s_value -= 2.0 * a_pq * a_pq
                if s_value <= threshold_sq:
                    break

            sweeps += 1
            # Честный пересчёт S: инкрементальная формула накапливает ошибку.
            s_value = off_diagonal_sq(a)
            if s_value <= threshold_sq:
                converged = True

        off_norm = math.sqrt(max(s_value, 0.0))
        if scale_exponent:
            # Возвращаем собственные числа и метрики в исходный масштаб.
            # Переполнение здесь возможно и допустимо: если собственное число
            # само по себе не представимо в float64 (например, у матрицы из
            # элементов 1e308), результатом будет `inf` — ровно то же самое
            # возвращает и `numpy.linalg.eigvalsh`.
            with np.errstate(over="ignore"):
                a = np.ldexp(a, scale_exponent)
            off_norm = _safe_ldexp(off_norm, scale_exponent)
            threshold = _safe_ldexp(threshold, scale_exponent)

        result = self._make_result(
            a, v, rotations, sweeps, off_norm, threshold, converged,
            time.perf_counter() - started, n,
        )
        if not converged:
            raise ConvergenceError(
                sweeps=sweeps,
                rotations=rotations,
                off_norm=off_norm,
                threshold=threshold,
                partial=result,
            )
        return result

    def __call__(self, matrix: Any) -> JacobiResult:
        """Синоним :meth:`solve`: решатель можно использовать как функцию.

        >>> solver = JacobiEigensolver()
        >>> float(solver(np.diag([7.0])).eigenvalues[0])
        7.0
        """
        return self.solve(matrix)

    def eigenvalues(self, matrix: Any) -> np.ndarray:
        """Вернуть только собственные числа (без матрицы векторов).

        Быстрее, чем :meth:`solve`, если векторы не нужны.
        """
        saved = self.compute_eigenvectors
        self.compute_eigenvectors = False
        try:
            return self.solve(matrix).eigenvalues
        finally:
            self.compute_eigenvectors = saved

    # ------------------------------------------------------------------
    # Внутренняя кухня
    # ------------------------------------------------------------------

    @staticmethod
    def _scale_exponent(a: np.ndarray) -> int:
        """Показатель степени двойки для нормировки матрицы.

        Возвращает ``0``, если матрица уже в безопасном диапазоне и
        нормировка не нужна.
        """
        max_abs = float(np.max(np.abs(a))) if a.size else 0.0
        if max_abs == 0.0 or _SAFE_MIN <= max_abs <= _SAFE_MAX:
            return 0
        # frexp: max_abs = mantissa * 2**exponent, где 0.5 <= mantissa < 1,
        # поэтому после сдвига наибольший элемент попадает в [0.5, 1).
        _, exponent = math.frexp(max_abs)
        return exponent

    def _threshold(self, a: np.ndarray) -> float:
        """Абсолютный порог для ``off(A)`` с учётом режима ``tol_mode``."""
        if self.tol_mode == "absolute":
            return self.tol
        frobenius = float(np.linalg.norm(a))
        if frobenius == 0.0:
            # Нулевая матрица уже диагональна; порог не должен быть нулевым,
            # иначе критерий `<=` формально не сработал бы при S = 0.
            return self.tol
        return self.tol * frobenius

    def _make_result(
        self,
        a: np.ndarray,
        v: Optional[np.ndarray],
        rotations: int,
        sweeps: int,
        off_norm: float,
        threshold: float,
        converged: bool,
        elapsed: float,
        n: int,
    ) -> JacobiResult:
        """Собрать объект результата, применив требуемую сортировку."""
        eigenvalues = np.diagonal(a).copy()
        eigenvectors = v if v is not None else np.empty((n, 0), dtype=np.float64)

        if self.sort is not None:
            order = np.argsort(eigenvalues)
            if self.sort == "desc":
                order = order[::-1]
            eigenvalues = eigenvalues[order]
            if v is not None:
                eigenvectors = eigenvectors[:, order]

        return JacobiResult(
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            rotations=rotations,
            sweeps=sweeps,
            off_norm=off_norm,
            threshold=threshold,
            converged=converged,
            elapsed=elapsed,
            kernel=self.kernel.name,
            strategy=self.strategy.name,
            metadata={"tol": self.tol, "tol_mode": self.tol_mode, "size": n},
        )

    def __repr__(self) -> str:
        return (
            f"JacobiEigensolver(tol={self.tol:g}, tol_mode={self.tol_mode!r}, "
            f"kernel={self.kernel.name!r}, strategy={self.strategy.name!r}, "
            f"max_sweeps={self.max_sweeps})"
        )


def jacobi_eigh(
    matrix: Any,
    tol: float = 1e-12,
    *,
    tol_mode: str = "relative",
    kernel: Union[str, RotationKernel] = "vectorized",
    strategy: Union[str, PivotStrategy] = "cyclic",
    max_sweeps: int = 100,
    compute_eigenvectors: bool = True,
    sort: Optional[str] = "asc",
    symmetry_atol: float = DEFAULT_SYMMETRY_ATOL,
    symmetry_rtol: float = DEFAULT_SYMMETRY_RTOL,
    symmetrize: bool = True,
) -> JacobiResult:
    """Найти собственные числа и векторы методом Якоби (функциональный интерфейс).

    Тонкая обёртка над :class:`JacobiEigensolver`; все параметры совпадают с
    параметрами его конструктора и метода :meth:`JacobiEigensolver.solve`.

    Returns
    -------
    JacobiResult

    Examples
    --------
    >>> result = jacobi_eigh([[2.0, 0.0], [0.0, 5.0]])
    >>> np.round(result.eigenvalues, 10)
    array([2., 5.])

    Некорректные данные приводят к понятной ошибке, а не к падению:

    >>> jacobi_eigh([[1.0, 2.0], [3.0, 4.0]])
    Traceback (most recent call last):
        ...
    jacobi_eigen.exceptions.NotSymmetricError: ...
    """
    solver = JacobiEigensolver(
        tol=tol,
        tol_mode=tol_mode,
        kernel=kernel,
        strategy=strategy,
        max_sweeps=max_sweeps,
        compute_eigenvectors=compute_eigenvectors,
        sort=sort,
        symmetry_atol=symmetry_atol,
        symmetry_rtol=symmetry_rtol,
        symmetrize=symmetrize,
    )
    return solver.solve(matrix)

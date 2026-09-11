r"""Результат работы метода Якоби и средства его проверки."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np

from .validation import off_diagonal_norm, validate_matrix

__all__ = ["JacobiResult"]


@dataclass(frozen=True)
class JacobiResult:
    r"""Собственные числа, собственные векторы и статистика сходимости.

    Основное соотношение, которому удовлетворяет результат:

    .. math::
        A V = V \Lambda, \qquad V^{T} V = I, \qquad A = V \Lambda V^{T},

    где :math:`\Lambda = \mathrm{diag}(\lambda_1, \dots, \lambda_n)`.

    Attributes
    ----------
    eigenvalues:
        Одномерный массив собственных чисел (диагональ финальной матрицы
        :math:`A_i`).
    eigenvectors:
        Матрица собственных векторов; **столбец** ``eigenvectors[:, k]``
        отвечает собственному числу ``eigenvalues[k]``.
    rotations:
        Число выполненных вращений Гивенса.
    sweeps:
        Число полных свипов (проходов по всем парам индексов).
    off_norm:
        Достигнутая норма недиагональной части
        :math:`\mathrm{off}(A) = \sqrt{\sum_{r \neq s} A_{rs}^2}`.
    threshold:
        Порог, при достижении которого итерации остановлены.
    converged:
        Достигнут ли критерий остановки.
    elapsed:
        Время счёта в секундах.
    kernel, strategy:
        Имена использованного ядра вращения и стратегии выбора элемента.

    Examples
    --------
    >>> from jacobi_eigen import jacobi_eigh
    >>> result = jacobi_eigh([[2.0, 1.0], [1.0, 2.0]])
    >>> np.round(result.eigenvalues, 12)
    array([1., 3.])
    >>> result.orthogonality_error() < 1e-14
    True
    """

    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    rotations: int
    sweeps: int
    off_norm: float
    threshold: float
    converged: bool
    elapsed: float
    kernel: str = "vectorized"
    strategy: str = "cyclic"
    metadata: Dict[str, Any] = field(default_factory=dict, repr=False)

    # ------------------------------------------------------------------
    # Производные величины
    # ------------------------------------------------------------------

    @property
    def size(self) -> int:
        """Размер исходной матрицы :math:`n`."""
        return int(self.eigenvalues.shape[0])

    @property
    def has_eigenvectors(self) -> bool:
        """Вычислялись ли собственные векторы.

        При ``compute_eigenvectors=False`` решатель экономит время и память,
        и все методы, которым нужна матрица :math:`V`, сообщают об этом
        предусмотренной ошибкой.
        """
        return self.eigenvectors.shape == (self.size, self.size)

    def _require_eigenvectors(self, what: str) -> None:
        """Проверить наличие матрицы собственных векторов."""
        if not self.has_eigenvectors:
            from .exceptions import InvalidParameterError

            raise InvalidParameterError(
                f"Невозможно вычислить {what}: результат получен без собственных "
                f"векторов (compute_eigenvectors=False). Повторите расчёт с "
                f"compute_eigenvectors=True."
            )

    def diagonal_matrix(self) -> np.ndarray:
        r"""Диагональная матрица :math:`\Lambda` из собственных чисел."""
        return np.diag(self.eigenvalues)

    def reconstruct(self) -> np.ndarray:
        r"""Восстановить исходную матрицу как :math:`V \Lambda V^{T}`.

        Полезно для наглядной проверки: результат должен совпадать с
        исходной матрицей с точностью до ошибок округления.

        Raises
        ------
        InvalidParameterError
            Если собственные векторы не вычислялись.
        """
        self._require_eigenvectors("V L V^T")
        return self.eigenvectors @ self.diagonal_matrix() @ self.eigenvectors.T

    def orthogonality_error(self) -> float:
        r"""Мера неортогональности :math:`\|V^{T} V - I\|_F`.

        Для корректного результата величина должна быть порядка машинной
        точности (``~1e-15``).

        Raises
        ------
        InvalidParameterError
            Если собственные векторы не вычислялись.
        """
        self._require_eigenvectors("ошибку ортогональности")
        n = self.size
        return float(np.linalg.norm(self.eigenvectors.T @ self.eigenvectors - np.eye(n)))

    def residual_norm(self, matrix: Any) -> float:
        r"""Невязка :math:`\|A V - V \Lambda\|_F` для исходной матрицы ``A``.

        Parameters
        ----------
        matrix:
            Исходная матрица (проходит ту же валидацию, что и вход решателя).

        Returns
        -------
        float
            Норма Фробениуса невязки. Малое значение подтверждает, что
            найденные пары «число — вектор» действительно собственные.
        """
        self._require_eigenvectors("невязку A V - V L")
        a = validate_matrix(matrix)
        if a.shape[0] != self.size:
            from .exceptions import InvalidParameterError

            raise InvalidParameterError(
                f"Размер матрицы {a.shape[0]} не совпадает с размером результата {self.size}."
            )
        return float(np.linalg.norm(a @ self.eigenvectors - self.eigenvectors * self.eigenvalues))

    def reconstruction_error(self, matrix: Any) -> float:
        r"""Ошибка восстановления :math:`\|A - V \Lambda V^{T}\|_F`.

        Raises
        ------
        InvalidParameterError
            Если собственные векторы не вычислялись.
        """
        self._require_eigenvectors("ошибку восстановления")
        a = validate_matrix(matrix)
        return float(np.linalg.norm(a - self.reconstruct()))

    def trace_error(self, matrix: Any) -> float:
        r"""Отклонение суммы собственных чисел от следа матрицы.

        След — инвариант подобия, поэтому :math:`\sum \lambda_k = \mathrm{tr}\,A`
        для точного решения.
        """
        a = validate_matrix(matrix)
        return abs(float(np.sum(self.eigenvalues)) - float(np.trace(a)))

    def sorted(self, order: str = "asc") -> "JacobiResult":
        """Вернуть копию результата с отсортированными собственными числами.

        Parameters
        ----------
        order:
            ``"asc"`` — по возрастанию (как у :func:`numpy.linalg.eigh`),
            ``"desc"`` — по убыванию.
        """
        from .exceptions import InvalidParameterError

        if order not in ("asc", "desc"):
            raise InvalidParameterError(
                f"order должен быть 'asc' или 'desc', получено {order!r}."
            )
        index = np.argsort(self.eigenvalues)
        if order == "desc":
            index = index[::-1]
        return JacobiResult(
            eigenvalues=self.eigenvalues[index],
            eigenvectors=self.eigenvectors[:, index] if self.has_eigenvectors
            else self.eigenvectors,
            rotations=self.rotations,
            sweeps=self.sweeps,
            off_norm=self.off_norm,
            threshold=self.threshold,
            converged=self.converged,
            elapsed=self.elapsed,
            kernel=self.kernel,
            strategy=self.strategy,
            metadata=dict(self.metadata),
        )

    # ------------------------------------------------------------------
    # Представление
    # ------------------------------------------------------------------

    def summary(self, matrix: Optional[Any] = None, precision: int = 6) -> str:
        """Человекочитаемый отчёт о решении.

        Parameters
        ----------
        matrix:
            Если передана исходная матрица, в отчёт добавляются невязка и
            ошибка восстановления.
        precision:
            Число знаков после запятой при печати чисел.
        """
        with np.printoptions(precision=precision, suppress=True):
            lines = [
                f"Метод Якоби: n = {self.size}, ядро = {self.kernel}, стратегия = {self.strategy}",
                f"Сходимость:  {'достигнута' if self.converged else 'НЕ достигнута'} "
                f"за {self.sweeps} свип(ов), {self.rotations} вращений",
                f"off(A):      {self.off_norm:.3e} (порог {self.threshold:.3e})",
                f"Время:       {self.elapsed * 1000:.3f} мс",
                f"Собственные числа:\n{self.eigenvalues}",
            ]
            if self.has_eigenvectors:
                lines.append(f"Собственные векторы (по столбцам):\n{self.eigenvectors}")
                lines.append(
                    f"Ортогональность ||V^T V - I||_F: {self.orthogonality_error():.3e}"
                )
            else:
                lines.append("Собственные векторы не вычислялись (compute_eigenvectors=False).")
            if matrix is not None and self.has_eigenvectors:
                lines.append(f"Невязка ||A V - V L||_F:        {self.residual_norm(matrix):.3e}")
                lines.append(
                    f"Восстановление ||A - V L V^T||_F: {self.reconstruction_error(matrix):.3e}"
                )
            elif matrix is not None:
                lines.append(f"Ошибка следа |sum(L) - tr(A)|:   {self.trace_error(matrix):.3e}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Представление результата в виде обычных типов Python (для JSON)."""
        return {
            "eigenvalues": self.eigenvalues.tolist(),
            "eigenvectors": self.eigenvectors.tolist(),
            "rotations": self.rotations,
            "sweeps": self.sweeps,
            "off_norm": self.off_norm,
            "threshold": self.threshold,
            "converged": self.converged,
            "elapsed": self.elapsed,
            "kernel": self.kernel,
            "strategy": self.strategy,
        }

    def __repr__(self) -> str:
        return (
            f"JacobiResult(n={self.size}, converged={self.converged}, "
            f"sweeps={self.sweeps}, rotations={self.rotations}, "
            f"off_norm={self.off_norm:.3e}, kernel={self.kernel!r}, "
            f"strategy={self.strategy!r})"
        )

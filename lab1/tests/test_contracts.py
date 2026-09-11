"""Тесты механизма контрактов и контрактов самого решателя."""

from __future__ import annotations

import math

import numpy as np
import pytest

from conftest import random_symmetric
from jacobi_eigen import (
    ContractViolation,
    InvariantViolation,
    JacobiEigensolver,
    JacobiError,
    PostconditionViolation,
    PreconditionViolation,
    contracts,
    contracts_disabled,
    ensure,
    invariant,
    jacobi_eigh,
    require,
    rotation_parameters,
)


class TestContractMechanism:
    """Базовые возможности мини-фреймворка DbC."""

    def test_precondition_passes(self):
        @require(lambda x: x > 0, "x положителен")
        def sqrt(x):
            return math.sqrt(x)

        assert sqrt(4) == 2.0

    def test_precondition_fails(self):
        @require(lambda x: x > 0, "x положителен")
        def sqrt(x):
            return math.sqrt(x)

        with pytest.raises(PreconditionViolation, match="x положителен"):
            sqrt(-1)

    def test_postcondition_sees_result(self):
        @ensure(lambda result: result >= 0, "результат неотрицателен")
        def broken_abs(x):
            return x  # намеренная ошибка реализации

        assert broken_abs(5) == 5
        with pytest.raises(PostconditionViolation, match="неотрицателен"):
            broken_abs(-5)

    def test_postcondition_can_use_arguments(self):
        @ensure(lambda result, x: result == 2 * x, "результат вдвое больше аргумента")
        def double(x):
            return x + x

        assert double(3) == 6

    def test_defaults_are_visible_to_contracts(self):
        @require(lambda step: step != 0, "шаг ненулевой")
        def advance(value, step=1):
            return value + step

        assert advance(1) == 2
        with pytest.raises(PreconditionViolation):
            advance(1, step=0)

    def test_invariant_checked_after_init_and_methods(self):
        @invariant(lambda self: self.balance >= 0, "баланс неотрицателен")
        class Account:
            def __init__(self, balance):
                self.balance = balance

            def withdraw(self, amount):
                self.balance -= amount

        account = Account(100)
        account.withdraw(40)
        assert account.balance == 60

        with pytest.raises(InvariantViolation, match="баланс"):
            account.withdraw(1000)
        with pytest.raises(InvariantViolation):
            Account(-1)

    def test_unknown_parameter_in_contract_is_reported(self):
        @require(lambda missing: missing > 0)
        def func(x):
            return x

        with pytest.raises(TypeError, match="неизвестный параметр"):
            func(1)

    def test_contract_preserves_metadata(self):
        @require(lambda x: True)
        def documented(x):
            """Документация функции."""
            return x

        assert documented.__name__ == "documented"
        assert documented.__doc__ == "Документация функции."

    def test_violations_are_jacobi_errors(self):
        """Нарушение контракта тоже ловится общим `except JacobiError`."""
        assert issubclass(ContractViolation, JacobiError)
        assert issubclass(PreconditionViolation, ContractViolation)
        assert issubclass(ContractViolation, AssertionError)


class TestContractToggling:
    """Контракты можно отключать — например, для замеров производительности."""

    def test_context_manager_disables_checks(self):
        @require(lambda x: x > 0, "x положителен")
        def func(x):
            return x

        with contracts_disabled():
            assert func(-1) == -1
        with pytest.raises(PreconditionViolation):
            func(-1)

    def test_state_restored_after_exception(self):
        was_enabled = contracts.enabled()
        with pytest.raises(RuntimeError):
            with contracts_disabled():
                raise RuntimeError("сбой внутри блока")
        assert contracts.enabled() == was_enabled

    def test_enable_disable(self):
        was_enabled = contracts.enabled()
        try:
            contracts.disable()
            assert not contracts.enabled()
            contracts.enable()
            assert contracts.enabled()
        finally:
            (contracts.enable if was_enabled else contracts.disable)()

    def test_algorithm_result_is_identical_without_contracts(self):
        """Контракты не влияют на вычисления — только проверяют их."""
        a = random_symmetric(8, seed=123)
        with_contracts = jacobi_eigh(a).eigenvalues
        with contracts_disabled():
            without_contracts = jacobi_eigh(a).eigenvalues
        np.testing.assert_array_equal(with_contracts, without_contracts)


class TestSolverContracts:
    """Контракты, объявленные в самом пакете."""

    def test_rotation_parameters_postconditions_hold(self):
        r"""Для любых входов выполняется :math:`\cos^2 + \sin^2 = 1` и :math:`|\mathrm{tg}| \le 1`."""
        generator = np.random.default_rng(0)
        for _ in range(500):
            a_pp, a_qq, a_pq = generator.normal(scale=10.0, size=3)
            cos_phi, sin_phi, tan_phi = rotation_parameters(a_pp, a_qq, a_pq)
            assert cos_phi**2 + sin_phi**2 == pytest.approx(1.0, abs=1e-12)
            assert abs(tan_phi) <= 1.0 + 1e-12

    def test_rotation_parameters_rejects_non_finite(self):
        with pytest.raises(PreconditionViolation):
            rotation_parameters(float("nan"), 1.0, 1.0)
        with pytest.raises(PreconditionViolation):
            rotation_parameters(1.0, float("inf"), 1.0)

    def test_solver_invariants_are_declared(self):
        assert getattr(JacobiEigensolver, "__contract_invariants__", None)

    def test_invariant_catches_broken_state(self):
        """Порча состояния решателя обнаруживается при следующем вызове."""
        solver = JacobiEigensolver()
        solver.max_sweeps = 0  # нарушение инварианта «цикл конечен»
        with pytest.raises(InvariantViolation):
            solver.solve(np.eye(3))

    def test_solve_postcondition_sizes(self):
        result = JacobiEigensolver().solve(random_symmetric(6, seed=4))
        assert result.eigenvalues.shape == (6,)
        assert result.eigenvectors.shape == (6, 6)

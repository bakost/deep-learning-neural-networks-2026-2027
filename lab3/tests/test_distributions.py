"""Плотности и квантили Стьюдента и Фишера по формулам лекции 3."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import integrate, stats

from lipo_ols import fisher_pdf, fisher_quantile, student_pdf, student_quantile


@pytest.mark.parametrize("nu", [1, 3, 10, 57, 300])
def test_student_pdf_matches_scipy(nu):
    x = np.linspace(-6, 6, 61)
    assert np.allclose(student_pdf(x, nu), stats.t.pdf(x, nu), rtol=1e-12)


@pytest.mark.parametrize("nu", [2, 57])
def test_student_pdf_normalized(nu):
    total, _ = integrate.quad(lambda x: float(student_pdf(x, nu)), -np.inf, np.inf)
    assert total == pytest.approx(1.0, abs=1e-9)


@pytest.mark.parametrize("alpha, nu", [(0.95, 57), (0.99, 57), (0.9, 5), (0.95, 74)])
def test_student_quantile(alpha, nu):
    assert student_quantile(alpha, nu) == pytest.approx(stats.t.ppf(0.5 + alpha / 2, nu), abs=1e-9)


@pytest.mark.parametrize("d1, d2", [(1, 57), (2, 57), (5, 57), (24, 57), (17, 57)])
def test_fisher_pdf_matches_scipy(d1, d2):
    x = np.linspace(0.05, 6, 60)
    assert np.allclose(fisher_pdf(x, d1, d2), stats.f.pdf(x, d1, d2), rtol=1e-10)


@pytest.mark.parametrize("d1, d2", [(24, 57), (5, 57), (2, 57)])
def test_fisher_quantile(d1, d2):
    assert fisher_quantile(0.95, d1, d2) == pytest.approx(stats.f.ppf(0.95, d1, d2), abs=1e-8)


def test_fisher_is_square_of_student():
    """F(1, ν) — распределение квадрата t(ν): F_α(1, ν) = (t_α^ν)²."""
    assert fisher_quantile(0.95, 1, 57) == pytest.approx(student_quantile(0.95, 57) ** 2, abs=1e-8)


def test_invalid_arguments():
    with pytest.raises(ValueError):
        student_quantile(1.2, 5)
    with pytest.raises(ValueError):
        student_pdf(0.0, 0)
    with pytest.raises(ValueError):
        fisher_pdf(1.0, 0, 3)
    assert fisher_pdf(np.array([-1.0, 0.0]), 3, 4).tolist() == [0.0, 0.0]

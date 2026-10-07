"""Воспроизведение статистик моделей MLR-1, MLR-2, MLR-3 из работы [2]."""

from __future__ import annotations

import pytest

from lipo_ols import PAPER_MODELS, PAPER_PAIRWISE_R2, fit_ols_data, pairwise_correlations, paper_columns


def _round_like(value: float, printed: float) -> float:
    text = f"{printed}"
    digits = len(text.split(".")[1]) if "." in text else 0
    return round(value, digits)


@pytest.mark.parametrize("model", list(PAPER_MODELS))
def test_paper_statistics_reproduced(data, model):
    ref = PAPER_MODELS[model]
    fit = fit_ols_data(data, paper_columns(model))
    s = fit.summary()
    assert fit.K == ref["k"] + 1
    for key in ("r2", "r2_adj", "rmse", "s", "F"):
        assert _round_like(s[key], ref[key]) == ref[key], key
    assert s["F_p"] == pytest.approx(ref["F_p"], rel=0.005)


def test_paper_pairwise_r2(data):
    r = pairwise_correlations(data)
    for name, value in PAPER_PAIRWISE_R2.items():
        assert round(r[name] ** 2, 2) == value


def test_paper_columns():
    assert paper_columns("MLR-1")[-1] == "AROMATIC" and len(paper_columns("MLR-2")) == 22

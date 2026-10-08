"""Свой расчёт совпадает со statsmodels."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("statsmodels")

from lipo_ols.package_compare import compare_with_statsmodels


def test_all_quantities_agree(data):
    res = compare_with_statsmodels(data)
    assert len(res["rows"]) >= 15
    for row in res["rows"]:
        assert row["diff"] < 1e-9, row["item"]


def test_special_points_have_unit_leverage(data):
    special = compare_with_statsmodels(data)["special"]
    assert {s["name"] for s in special} == {"Benzaldehyde", "Bromobenzene", "Fosamprenavir"}
    for s in special:
        assert s["leverage_ours"] == pytest.approx(1.0) and s["leverage_sm"] == pytest.approx(1.0)
    # хотя бы для одной из них statsmodels выдаёт конечное (бессмысленное) число вместо «не определено»
    assert any(np.isfinite(s["cook_sm"]) for s in special)

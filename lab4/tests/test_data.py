"""Выборка и стандартизация."""

from __future__ import annotations

import numpy as np
import pytest

from lipo_mlp import DESCRIPTORS, Standardizer, duplicate_groups, load_lipo


def test_load(data):
    assert (data.n, data.k) == (82, 24) and data.columns == tuple(DESCRIPTORS)
    assert data.y[data.index("Benzene")] == pytest.approx(2.13)


def test_standardizer_roundtrip(data, std):
    Z, t, scaler = std
    assert np.allclose(Z.mean(axis=0), 0, atol=1e-12) and np.allclose(Z.std(axis=0), 1)
    assert np.allclose(scaler.inverse_y(t), data.y)


def test_zero_variance_column_is_not_scaled():
    X = np.column_stack([np.zeros(5), np.arange(5.0)])
    s = Standardizer.fit(X, np.arange(5.0))
    Z = s.transform_x(X)
    assert np.all(Z[:, 0] == 0) and np.all(np.isfinite(Z))
    assert s.subset([1]).x_mean.shape == (1,)


def test_duplicate_groups(data):
    assert sorted(len(g) for g in duplicate_groups(data.X)) == [2, 2, 3]


def test_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_lipo(tmp_path / "none.csv")
    bad = tmp_path / "bad.csv"
    bad.write_text("Вещество,logP,A\nx,1,abc\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_lipo(bad)

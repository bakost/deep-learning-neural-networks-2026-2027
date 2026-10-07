"""Чтение и проверка lipo.csv (пункт 1)."""

from __future__ import annotations

import numpy as np
import pytest

from lipo_ols import DESCRIPTOR_NAMES, duplicate_groups, load_lipo


def test_shape_and_columns(data):
    assert (data.n, data.k) == (82, 24)
    assert data.columns == DESCRIPTOR_NAMES
    assert np.all(np.isfinite(data.X)) and np.all(np.isfinite(data.y))


def test_names_are_stripped(data):
    assert "Benzoic acid" in data.names and "Ethanol" in data.names
    assert all(name == name.strip() for name in data.names)


def test_logp_is_decimal_logarithm(data):
    """log P бензола 2.13 — это десятичный логарифм (ln P было бы 2.13·ln 10 ≈ 4.9)."""
    assert data.y[data.index("Benzene")] == pytest.approx(2.13)
    assert data.y[data.index("Phenol")] - data.y[data.index("Benzene")] == pytest.approx(-0.67)


def test_design_has_intercept(data):
    X = data.design()
    assert X.shape == (82, 25) and np.all(X[:, 0] == 1.0)
    assert data.labels()[0] == "const" and len(data.labels()) == 25


def test_subset_and_rows(data):
    sub = data.subset(["C", "PSA"])
    assert sub.columns == ("C", "PSA") and sub.X.shape == (82, 2)
    assert np.array_equal(sub.column("PSA"), data.column("PSA"))
    part = data.rows([0, 5, 7])
    assert part.n == 3 and part.names[1] == data.names[5]
    with pytest.raises(KeyError):
        data.subset(["NOPE"])


def test_rare_descriptors(data):
    counts = dict(zip(data.columns, np.count_nonzero(data.X, axis=0)))
    assert counts["ROPO3"] == counts["RCHO"] == counts["RBr"] == 1
    assert counts["RSO2R"] == 2 and counts["C"] == 82


def test_duplicate_groups(data):
    groups = [{data.names[i] for i in g} for g in duplicate_groups(data)]
    assert {"Sulfadimidine", "Sulfisomidine"} in groups
    assert {"Sulfamethoxydiazine", "Sulfamethoxypyridazine", "Sulfametopyrazine"} in groups
    assert len(groups) == 3


def test_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_lipo(tmp_path / "nope.csv")


def test_env_variable(tmp_path, monkeypatch, data):
    path = tmp_path / "small.csv"
    path.write_text("Вещество,logP,A,B\nx,1.0,1,2\ny,2.0,3,5\n", encoding="utf-8")
    monkeypatch.setenv("LIPO_CSV", str(path))
    small = load_lipo()
    assert small.columns == ("A", "B") and small.n == 2


@pytest.mark.parametrize("content, message", [
    ("Вещество,lgP,A\nx,1,2\n", "заголовок"),
    ("Вещество,logP,A\nx,1\n", "значений"),
    ("Вещество,logP,A\nx,1,abc\n", "нечисловое"),
    ("Вещество,logP,A\nx,1,nan\n", "пропуск"),
])
def test_bad_files(tmp_path, content, message):
    path = tmp_path / "bad.csv"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_lipo(path)

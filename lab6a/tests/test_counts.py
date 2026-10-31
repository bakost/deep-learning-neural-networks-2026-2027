"""Считаем реальные элементы массивов, а не повторяем формулу подсчёта."""

import numpy as np
import pytest

from transformer_lab import init_params, parameter_counts


@pytest.mark.parametrize("d,n,h", [(1, 1, 1), (3, 4, 7), (8, 16, 128)])
def test_counts_match_parameter_arrays(d, n, h):
    r = parameter_counts(d, n, h)
    params = init_params(d)
    assert r["qkv_total"] == sum(a.size for a in vars(params).values())
    mlp = [np.empty((h, d * n)), np.empty(h), np.empty((d * n, h)), np.empty(d * n)]
    assert r["mlp_total"] == sum(a.size for a in mlp)
    assert r["attention_trainable"] == 0
    assert r["attention_entries"] == np.empty((n, n)).size


def test_default_width_and_no_dependence_on_sequence_length():
    a, b = parameter_counts(3, 4), parameter_counts(3, 17)
    assert a["H"] == 12
    assert a["mlp_total"] == 312
    assert a["qkv_total"] == b["qkv_total"] == 36


@pytest.mark.parametrize("args", [(0, 1), (1, -1), (2.5, 3), (True, 4), (2, 3, 0)])
def test_invalid_dimensions(args):
    with pytest.raises(ValueError):
        parameter_counts(*args)

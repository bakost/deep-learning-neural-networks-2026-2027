"""Общие фикстуры. Все генераторы инициализируются фиксированными зёрнами."""

from __future__ import annotations

import warnings

import pytest

from sgd_lab import MODELS


@pytest.fixture(params=list(MODELS))
def model(request):
    return MODELS[request.param]


@pytest.fixture(autouse=True)
def _quiet():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        yield

"""Интерфейс командной строки."""

from __future__ import annotations

import json

import pytest

from level_spacing.__main__ import EXIT_ERROR, EXIT_OK, main


def test_run_table(capsys):
    assert main(["run", "--sizes", "2", "4", "--matrices", "2000"]) == EXIT_OK
    out = capsys.readouterr().out
    assert "goe" in out and "(2,3)" in out and "0.2732" in out


def test_run_json(capsys):
    assert main(["run", "-e", "pm1", "-n", "2", "-m", "1000", "--json"]) == EXIT_OK
    data = json.loads(capsys.readouterr().out)
    assert data[0]["ensemble"] == "pm1" and data[0]["size"] == 1000


def test_run_plot(tmp_path, capsys):
    path = tmp_path / "h.png"
    assert main(["run", "-n", "4", "-m", "1000", "--plot", str(path)]) == EXIT_OK
    assert path.stat().st_size > 10_000


def test_run_custom_pair(capsys):
    assert main(["run", "-n", "16", "-m", "500", "-k", "1", "--json"]) == EXIT_OK
    assert json.loads(capsys.readouterr().out)[0]["k"] == 1


def test_exact(capsys):
    assert main(["exact", "--size", "2", "--construction", "sum"]) == EXIT_OK
    out = capsys.readouterr().out
    assert "перебрано 16 матриц" in out and "0.25" in out


def test_list(capsys):
    assert main(["list"]) == EXIT_OK
    assert "pm1-sum" in capsys.readouterr().out


@pytest.mark.parametrize("argv", [
    ["run", "-n", "1", "-m", "10"],            # нет пары соседних уровней
    ["run", "-n", "4", "-m", "10", "-k", "4"], # пары (λ4, λ5) у 4×4 нет
    ["run", "-m", "0"],                        # пустой ансамбль
    ["exact", "--size", "7"],                  # перебор 2^28 матриц
])
def test_errors_are_reported(argv, capsys):
    assert main(argv) == EXIT_ERROR
    err = capsys.readouterr().err
    assert err.startswith("Ошибка:") and "Traceback" not in err


def test_unknown_ensemble_rejected_by_argparse(capsys):
    with pytest.raises(SystemExit):
        main(["run", "-e", "gue"])

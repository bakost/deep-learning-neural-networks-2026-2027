# transformer-lab — самовнимание и слои трансформеров

Лабораторная работа **6а**, все три задачи из [lab6a.pdf](../lab6a.pdf).
Обозначения взяты из [лекции о самовнимании](../lecture5aa.pdf) и
[лекции о слоях трансформеров](../lecture6a.pdf): входы — **столбцы**
матрицы $X$ размера $D \times N$, softmax нормирует каждый столбец $K^TQ$.

**Решения и выводы — [REPORT.md](REPORT.md), версия для сдачи — [REPORT.pdf](REPORT.pdf).**

1. Число весов и смещений Q/K/V, коэффициентов внимания и параметров MLP.
2. Доказательство $\mathrm{Sa}(XP)=\mathrm{Sa}(X)P$ и численная проверка.
3. Все 25 производных softmax для $z=(-3,1,100,5,-1)$, проверка повышенной
   точностью и объяснение насыщения.

## Установка и запуск

Python ≥ 3.9, NumPy и Matplotlib; дополнительные данные и сеть при запуске
не нужны. Команды выполняются из папки `lab6a`.

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[test,pdf,dev]"
source .venv/bin/activate

python -m transformer_lab counts -D 64 -N 16
python -m transformer_lab counts -D 64 -N 16 -H 128  # произвольная скрытая ширина MLP
python -m transformer_lab equivariance              # матрица P из задания
python -m transformer_lab equivariance --scaled     # деление KᵀQ на √D
python -m transformer_lab softmax                   # вектор y и все 25 производных
python -m transformer_lab softmax --json            # также эталон Decimal, 110 знаков
```

```python
import numpy as np
from transformer_lab import init_params, self_attention, permutation_matrix, softmax_jacobian

X = np.random.default_rng(0).normal(size=(3, 4))
params = init_params(3, seed=2026)
P = permutation_matrix([1, 3, 0, 2])
Y, A = self_attention(X, params)
Yp, Ap = self_attention(X @ P, params)
np.allclose(Yp, Y @ P)                       # True
J = softmax_jacobian([-3, 1, 100, 5, -1])
J[2, 2]                                     # ≈ 5.6377e-42, не ноль
```

## Воспроизведение отчёта

```bash
python -m transformer_lab report             # results/* и figures/*, несколько секунд
python tools/fill_tables.py                  # обновить 5 таблиц в REPORT.md
python tools/build_pdf.py                    # REPORT.pdf; нужен Google Chrome/Chromium
python tools/fill_tables.py --check          # проверить актуальность таблиц
```

PDF собирается так же, как в остальных работах: pandoc преобразует формулы
в MathML, Chrome печатает локальный HTML. При нестандартном расположении
браузера задайте переменную `CHROME`. Скрипт использует временный профиль.
Можно сохранить промежуточную страницу: `python tools/build_pdf.py --html /tmp/lab6a.html`.

Результаты воспроизводимы: генераторы случайных чисел фиксированы; в
`results/summary.json` сохранены входы, перестановки, результаты проверок,
якобиан и его эталон с 110 десятичными знаками. Все 25 производных доступны
также в [CSV](results/jacobian.csv) и [таблицах](results/tables.md).
`report --output DIR` позволяет сохранить результаты в другую папку.

## Проверки

```bash
python -m pytest
ruff check src tests tools
```

| Файл | Что проверяется |
|---|---|
| `test_counts.py` | число параметров по размерам реальных массивов, произвольная ширина MLP |
| `test_attention.py` | скалярная формула внимания, ориентация P, все 24 перестановки, ненулевые смещения, масштабирование, один токен, влияние фиксированных позиций |
| `test_softmax.py` | нормировка, сдвиг логитов, якобиан против обычных и высокоточных конечных разностей, все 25 элементов против Decimal с нулевым абсолютным допуском |
| `test_cli.py` | CLI, ошибки входных данных, артефакты отчёта и воспроизводимость расчётов |

## Структура

```text
lab6a/
├── REPORT.md / REPORT.pdf       решения всех трёх задач
├── README.md / pyproject.toml
├── src/transformer_lab/
│   ├── counts.py                задача 1: параметры Q/K/V, A(X), MLP
│   ├── attention.py             задача 2: самовнимание, матрицы перестановок
│   ├── softmax.py               задача 3: softmax, устойчивый якобиан, Decimal
│   ├── experiments.py           фиксированные численные эксперименты
│   ├── report.py                таблицы, CSV, JSON и три рисунка
│   └── __main__.py              командная строка
├── tests/
├── tools/                      fill_tables.py, build_pdf.py
├── figures/                    сравнение параметров, перестановка, якобиан
└── results/                    summary.json, tables.md, jacobian.csv
```

В условии нет обучения на выборке: основа работы — аналитические решения.
Код проверяет формулы и численные выводы. Для двухслойного MLP приведена
общая формула с шириной $H$ и отдельно случай $H=DN$, поскольку скрытая
ширина в условии не задана. Коэффициенты $A(X)$ не считаются дополнительными
обучаемыми параметрами. В задаче 3 диагональ вычисляется без вычитания
почти равных чисел, чтобы не потерять производную для третьего выхода.

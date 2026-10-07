# lipo-mlp — липофильность и двухслойный перцептрон

Лабораторная работа №4. Пакет строит двухслойный перцептрон (2) методички

$$\log P = b^2_1 + (w^2)^T \mathrm{relu}(w^1 x + b^1)$$

для 2, 4, 8, 16 и 32 нейронов скрытого слоя и обучает его градиентным
спуском. Градиент выведен вручную по формулам лекции 4 и проверен конечными
разностями. Пакет исследует сходимость, уменьшение скорости обучения,
обобщающую способность и вклад регрессоров, а результаты сравнивает с
`MLPRegressor` из scikit-learn и с линейной регрессией из [ЛР3](../lab3).

**Отчёт с результатами и ответами на вопросы пункта 5 — [REPORT.md](REPORT.md),
он же в PDF — [REPORT.pdf](REPORT.pdf).**

```python
from lipo_mlp import load_lipo, Standardizer, init_params, gradient_descent

data = load_lipo()                                    # пункт 1: 82 вещества, 24 регрессора
scaler = Standardizer.fit(data.X, data.y)             # признаки и logP → среднее 0, дисперсия 1
Z, t = scaler.transform(data.X, data.y)
params = init_params(24, hidden=16, seed=0)           # пункт 2: H ∈ {2, 4, 8, 16, 32}
res = gradient_descent(params, Z, t, lr=0.1,          # пункт 3: ε = 0.1, δ = 1e-4
                       delta=1e-4, max_iter=300_000)  # пункт 4: градиентный спуск
res.status, res.iterations                            # ('converged', ...)
```

## Установка

Нужен Python ≥ 3.9, NumPy, SciPy, Matplotlib; для сравнения — scikit-learn.

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[test]"        # вместе с pytest и scikit-learn
```

Файл с выборкой ищется в корне репозитория (`../lipo.csv`) или по переменной
окружения `LIPO_CSV`. На macOS 26 (Apple M1) нужен SciPy 1.14.1
(`pip install "scipy==1.14.1"`), как в ЛР2 и ЛР3.

## Командная строка

```bash
python -m lipo_mlp train -H 16                          # один перцептрон: статус, итерации, RMSE
python -m lipo_mlp train -H 16 --schedule inverse --tau 10000   # εⱼ = ε₀/(1 + j/τ)
python -m lipo_mlp train -H 4 --schedule exponential --gamma 0.99999 --plot h4.png
python -m lipo_mlp train -H 16 --batch-size 16 --schedule inverse --tau 20000  # мини-пакеты
python -m lipo_mlp sweep --seeds 3                      # все размеры скрытого слоя (параллельно)
python -m lipo_mlp sklearn -H 8                         # сверка с MLPRegressor
python -m lipo_mlp report                               # все рисунки и таблицы отчёта (≈ 6 мин на 8 ядрах)
python -m lipo_mlp report --quick                       # уменьшенный вариант (≈ 20 с)
```

Пример вывода `train -H 16`:

```
Перцептрон H = 16 (417 параметров), постоянная ε = 0.1, δ = 0.0001
  сошёлся (‖DJ‖ < δ): итераций 87120, ‖DJ‖ = 1.000e-04, 3.8 с
  RMSE на обучающей выборке = 0.1237, R² = 0.9940 (нижняя граница RMSE 0.1237)
```

## PDF-версия отчёта

```bash
pip install -e ".[pdf]"
python -m lipo_mlp report        # пересчитать рисунки и results/tables.md
python tools/fill_tables.py      # подставить таблицы в REPORT.md
python tools/build_pdf.py        # REPORT.md -> REPORT.pdf (нужен Google Chrome)
```

## Тесты

```bash
python -m pytest                       # ≈ 30 с
python -m pytest --doctest-modules src
ruff check src tests
```

| Файл | Что проверяет |
|---|---|
| `test_network.py` | формула модели, число параметров, ручной градиент против конечных разностей для H = 1…32, производная по входам, инициализация |
| `test_training.py` | убывание потерь, остановка по δ, расходимость при большом ε, выход на нижнюю границу ошибки, мини-пакеты, законы ε |
| `test_schedules.py` | значения и суммы законов уменьшения скорости обучения |
| `test_sklearn.py` | траектория полнопакетного SGD в scikit-learn совпадает с нашей до 10⁻¹⁰ |
| `test_experiments.py` | граница устойчивости, разбиения перекрёстной проверки, детерминированность параллельных запусков |
| `test_importance.py`, `test_data.py`, `test_cli.py` | меры вклада регрессоров, данные, командная строка, генерация отчёта |

## Структура

```
lab4/
├── REPORT.md / REPORT.pdf     отчёт (пункт 5)
├── README.md                  этот файл
├── pyproject.toml
├── src/lipo_mlp/
│   ├── data.py                чтение lipo.csv, стандартизация                — пункт 1
│   ├── network.py             модель (2), функция потерь, ручной градиент    — пункт 2
│   ├── training.py            градиентный спуск с остановкой по ‖DJ‖ < δ     — пункты 3–4
│   ├── schedules.py           законы уменьшения скорости обучения            — пункт 5
│   ├── experiments.py         запуски (параллельно), перекрёстная проверка   — пункт 5
│   ├── importance.py          вклад регрессоров                              — пункт 5
│   ├── sklearn_compare.py     сравнение с MLPRegressor                       — пункт 5
│   ├── plots.py, report.py    рисунки и пересчёт всего отчёта
│   └── __main__.py            командная строка
├── tools/                     fill_tables.py, build_pdf.py
├── tests/
├── figures/                   рисунки отчёта (генерируются)
└── results/                   summary.json, tables.md (генерируются)
```

## Что сделано по пунктам задания

| Пункт | Где |
|---|---|
| 1. Изучить `lipo.csv` | `data.py`; особенности выборки разобраны в отчёте ЛР3, здесь — то, что важно для сети (масштабы, повторы X) |
| 2. Модели с 2, 4, 8, 16, 32 нейронами | `network.py`; отчёт, раздел 1 |
| 3. Зафиксировать ε и δ | ε = 0.1, δ = 10⁻⁴ (норма градиента), предел 3·10⁵ итераций; выбор обоснован в разделе 2 |
| 4. Градиентный спуск | `training.gradient_descent` по алгоритму лекции 4; градиент сверен с конечными разностями и с scikit-learn |
| 5. Число итераций, сошёлся ли | раздел 3: табл. 1–2, рис. 1–3 |
| 5. Уменьшение скорости обучения | раздел 4: табл. 3, рис. 4 |
| 5. Сравнение с библиотекой | раздел 5: табл. 5 |
| 5. Адекватность модели | раздел 6: перекрёстная проверка, табл. 4, рис. 5 |
| 5. Вклад регрессоров | раздел 7: табл. 6, рис. 6 |
| 5. Сравнение с линейной регрессией | раздел 8 |

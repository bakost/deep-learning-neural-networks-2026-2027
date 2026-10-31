# sgd-lab — стохастический градиентный спуск и его модификации

Лабораторная работа №6 (задание — [lab6b.pdf](../lab6b.pdf), лекция —
[lecture6.pdf](../lecture6.pdf)). Пакет решает все пять задач:

1. минимум функции $f(z) = 1 - \tfrac12 e^{-10(z-0.65)^2} - 0.45 e^{-50(z-0.35)^2}$
   методами **золотого сечения** и **Брента** (Numerical Recipes, разд. 10.1–10.3);
2. обучение трёх моделей — $\hat y = w_1x + b_1$, $\hat y = w_1x^2 + w_2x + b_1$ и сети
   $\mathrm{ReLU}(w_{21}\mathrm{ReLU}(w_{11}x + b_{11}) + w_{22}\mathrm{ReLU}(w_{12}x + b_{12}) + b_{21})$ —
   **классическим градиентным спуском** с *численными* производными (NR, разд. 5.7);
3. **СГС с мини-пакетами** размера 1, 2, 3, 4, 6, 12;
4. **импульс Нестерова** и **Adam** (формулы лекции 6) и сравнение задач 2–4;
5. то же с **квадратичным штрафом** $\lambda\sum w^2$ (задача 5*).

**Отчёт с результатами и комментариями — [REPORT.md](REPORT.md), он же в PDF —
[REPORT.pdf](REPORT.pdf).**

```python
from sgd_lab import MODELS, X, Y, Objective, TrainConfig, train, init_params
from sgd_lab import bracket_minimum, brent, golden_section, f_task

trip, _ = bracket_minimum(f_task, 0.0, 1.0)            # задача 1: отрезок, содержащий минимум
brent(f_task, trip).x                                   # 0.380321528…

obj = Objective(MODELS["quadratic"], X, Y)              # задача 2: J = Σ(yᵢ − ŷᵢ)²
res = train(obj, init_params(MODELS["quadratic"], 0),
            TrainConfig(method="gd", lr=0.0174, epochs=100_000, delta=1e-6))
res.status, res.epochs                                  # ('converged', 2417)

res = train(obj, init_params(MODELS["quadratic"], 0),   # задачи 3–4: мини-пакеты, Adam
            TrainConfig(method="adam", lr=0.05, batch_size=4, epochs=2000, tau=70))
obj5 = Objective(MODELS["relu"], X, Y, l2=0.1)          # задача 5: J̃ = J + λΣw²
```

## Установка

Нужен Python ≥ 3.9, NumPy, SciPy, Matplotlib.

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[test]"
```

На macOS 26 (Apple M1) нужен SciPy 1.14.1 (`pip install "scipy==1.14.1"`), как в предыдущих ЛР.

## Командная строка

```bash
python -m sgd_lab minimize                         # задача 1, метод Брента
python -m sgd_lab minimize -m golden --plot f.png  # золотое сечение и график f(z)
python -m sgd_lab train linear -e 0.037 -d 1e-6    # задача 2: ГС до ‖DJ‖ < δ
python -m sgd_lab train quadratic -b 1 -e 0.0025 -n 2000 --tau 200   # задача 3: СГС, m = 1
python -m sgd_lab train relu -m adam -e 0.03 -n 3000 -s 5 --plot fit.png  # задача 4
python -m sgd_lab train relu -m adam -e 0.01 --l2 0.1 -n 3000      # задача 5
python -m sgd_lab report                           # все рисунки и таблицы отчёта (≈ 5 мин на 8 ядрах)
python -m sgd_lab report --quick                   # уменьшенный вариант (≈ 20 с)
```

Пример вывода `train linear -e 0.037 -d 1e-6`:

```
Модель ŷ = w₁x + b₁; градиентный спуск, ε = 0.037, мини-пакет 12
  сошёлся (‖DJ‖ < δ): эпох 110, обновлений 110, ‖DJ‖ = 9.26e-07
  сумма квадратов ошибок 0.199866
  w1 = 0.52223, b1 = 0.82595
```

## PDF-версия отчёта

```bash
pip install -e ".[pdf]"
python -m sgd_lab report         # пересчитать рисунки и results/tables.md
python tools/fill_tables.py      # подставить таблицы в REPORT.md
python tools/build_pdf.py        # REPORT.md -> REPORT.pdf (нужен Google Chrome)
```

## Тесты

```bash
python -m pytest                       # ≈ 30 с
python -m pytest --doctest-modules src
ruff check src tests tools
```

| Файл | Что проверяет |
|---|---|
| `test_linesearch.py` | единственность минимума f, корректность отрезка `mnbrak`, точность обоих методов, совпадение числа итераций со SciPy, сокращение отрезка в φ раз, параболические шаги Брента, предел точности √ε, «плато» вдали от минимума |
| `test_numdiff.py` | численный градиент против аналитического для трёх моделей, векторизация, точно представимый шаг, метод Риддерса |
| `test_training.py` | несмещённость оценки по мини-пакету, сходимость ГС к МНК, граница устойчивости 2/λmax, формулы Нестерова и Adam по шагам, векторизованное обучение = поштучное (побитово), обход выборки без повторений, уменьшение ε против шума, сходимость к гребневой регрессии, «мёртвая» сеть |
| `test_reference.py` | МНК и гребневая регрессия, числа обусловленности, граница ε для мини-пакетов, скользящий контроль, классификация исходов сети |
| `test_cli.py` | командная строка и генерация отчёта |

## Структура

```
lab6/
├── REPORT.md / REPORT.pdf     отчёт
├── README.md                  этот файл
├── pyproject.toml
├── src/sgd_lab/
│   ├── linesearch.py          mnbrak, золотое сечение, метод Брента        — задача 1
│   ├── models.py              выборка и три модели (векторизованы по весам) — задача 2
│   ├── numdiff.py             центральные разности, метод Риддерса (NR 5.7) — задача 2
│   ├── training.py            J̃, ГС / СГС / Нестеров / Adam, L2-штраф       — задачи 2–5
│   ├── reference.py           МНК, гребневая регрессия, κ, эталон сети
│   ├── experiments.py         серии запусков (векторизованно и параллельно)
│   ├── report.py, tables.py, plots.py   пересчёт отчёта
│   └── __main__.py            командная строка
├── tools/                     fill_tables.py, build_pdf.py
├── tests/
├── figures/                   рисунки отчёта (генерируются)
└── results/                   summary.json, tables.md (генерируются)
```

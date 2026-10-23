# Базовый эксперимент

Сравнение RF, LIME, PCA и прямого анализа корреляций при изменении разности средних дополнительного признака. Группы заданы заранее.

## Запуск

Python 3.13. Команды выполняются из корня репозитория:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r code/requirements.txt
.\.venv\Scripts\python.exe code/experiment.py
```

По умолчанию выполняются 20 повторов для 6 значений сдвига. Ориентировочное время — 13 минут на CPU. Повторный запуск перезаписывает данные и результаты в `code/`.

Короткая проверка в отдельной папке:

```powershell
.\.venv\Scripts\python.exe code/experiment.py --reps 1 --output .smoke
```

## Файлы

- `experiment.py` — генерация данных, обучение и запуск сравнения.
- `methods.py` — способы оценки важности признаков.
- `visualization.py` — сводная таблица и графики.
- `data/` — исходные наборы в NPZ и примеры в CSV.
- `results/runs.csv` — результаты отдельных запусков.
- `results/summary.csv` — средние значения и 95%-ные интервалы Wilson.
- `figures/` — графики результатов и пример данных.
- `RESULTS.md` — подробное описание первых результатов.
- `protocol.json` — параметры базового эксперимента.
- `requirements-lock.txt` — версии пакетов исходного окружения.

Сохранённые результаты получены до переноса кода в шаблон. Их `results/manifest.json` содержит хеши исходных скриптов; изменения путей при переносе зафиксированы в `migration.json`. Настройки генератора и методов сохранены.

Перестроить графики по сохранённой таблице без обучения:

```powershell
.\.venv\Scripts\python.exe -c "import sys; from pathlib import Path; import pandas as pd; sys.path.insert(0, 'code'); from visualization import make_report; make_report(pd.read_csv('code/results/runs.csv'), Path('code'))"
```

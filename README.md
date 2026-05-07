# Прогноз weekly KPI бизнес-процессов через Transformer

Финальный проект по дисциплине «ЦОС».

## Задача

Прогноз недельных KPI (cycle time, новые кейсы) для двух event-логов:
`142_БЗ` (внутренний) и `BPI Challenge 2017` (публичный). Метод -
encoder-only Transformer, обучаемый на нормализованных дельтах ряда.

## Архитектура

```
[batch, 24]                                 # delta_y (стандартизованные)
   -> Linear(1->64) + Positional Embedding
   -> 2x TransformerEncoderLayer (nhead=4, ffn=128, GELU, dropout=0.1)
   -> mean pool по времени
   -> MLP(64->64->1)
[batch]                                     # delta_y_{t+1}
```

Параметры: 72 833. Оптимизатор AdamW (lr=1e-3, wd=1e-4), SmoothL1 loss,
gradient clip 1.0, 80 эпох на CPU.

## Сравнение

7 моделей: naive_last, seasonal_naive, moving_avg, ridge_lags, xgb_lags,
наш Transformer, Chronos zero-shot.

Лучший на `bpi2017 cycle_time` (test=8 недель):

| модель | MAE (ч) | sMAPE | R² |
|---|---:|---:|---:|
| naive_last | 25.04 | 0.051 | -0.125 |
| moving_avg | 21.49 | 0.044 | -0.027 |
| **Transformer (наш)** | **20.16** | **0.041** | **+0.126** |
| chronos zero-shot | 27.86 | 0.057 | -0.631 |

## Запуск

```bash
pip install -r requirements.txt   # см. ниже
python src/train_transformer.py --dataset bpi2017 --kpi cycle_time --epochs 80
python src/make_plots.py
```

Зависимости: `torch`, `numpy`, `pandas`, `scikit-learn`, `xgboost`,
`pm4py`, `matplotlib`, `chronos-forecasting`.

Загрузка данных в `src/datasets.py` ожидает родительскую папку `data/`
с подпапками `142_БЗ/` и `bpi2017/`. BPI 2017 публично доступен на
4TU.ResearchData.

## Состав

```
src/
  datasets.py            загрузка XES/CSV
  features.py            split, feature engineering
  forecasting.py         6 baselines + наш Transformer + Chronos
  train_transformer.py   standalone обучение
  make_plots.py          генерация всех графиков
slides/
  slides.pdf             презентация
  slides.pptx            редактируемая версия
```

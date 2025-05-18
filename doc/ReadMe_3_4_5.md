# Credit Risk Pipeline

Этот репозиторий реализует сквозной конвейер для кредитного скоринга: от подготовки данных до интерпретации и мониторинга дрейфа. Фокус — разделы 3–5 задания.

## Структура проекта

```
├── preprocessing.py    # модуль подготовки данных (этап 3)
├── model.py            # модуль обучения и сохранения модели (этап 4)
├── validation.py       # модуль валидации, интерпретации, дрейф-мониторинга (этап 5)
├── main.py             # пример скрипта-конвейера
└── README.md           # это руководство
```

---

## preprocessing.py

Класс `CreditDataPreprocessor` отвечает за этап 3 — подготовку данных.

```python
preprocessor = CreditDataPreprocessor(config)
X, y = preprocessor.fit_transform(df)
# — обучает пайплайн, обрабатывает NaN, масштабирует,
#   кодирует категориальные и извлекает дату.

X_new = preprocessor.transform(df_new)
# — применяет уже обученный пайплайн.

preprocessor.save(path)
# — сохраняет pipeline и имена признаков в pickle.

preprocessor.load(path)
# — загружает ранее сохранённый pipeline и feature_names.
```

**Параметры config** (минимум):

* `target_column`: имя целевой переменной.

---

## model.py

Класс `CreditModel` отвечает за этап 4 — обучение/дообучение и сохранение модели.

```python
model = CreditModel(config)
model.train(X_train, y_train, update=False)
# — обучает новую модель XGBoost.

model.train(X_update, y_update, update=True)
# — дообучает ранее сохранённую модель (warm-start).

preds = model.predict(X_test)

metrics = model.evaluate(X_test, y_test)
# — возвращает словарь {'mae', 'r2'}.

model.save_model()
# — сохраняет модель и metadata (timestamp, input_shape, config).
# — старые модели так же созраняются и могут быть использованы.

model.load_model()
# — загружает модель из файла.
```

**Config**:

* `model_storage`: путь к папке для моделей.
* (опции гиперпараметров XGBoost — см. конструктор).

---

## validation.py

Класс `ModelValidator` и вспомогательные функции реализуют этап 5 — валидацию, интерпретацию прогнозов и мониторинг дрейфа.

```python
validator = ModelValidator(config)
```

* **`validator.validate(df, method='holdout'|'cv'|'timeseries')`**

  * Hold-out: train/test split + обучение + оценка.
  * CV: k-fold cross\_val\_score.
  * TimeSeriesCV: TimeSeriesSplit + cross\_val\_score.
    Возвращает словарь метрик.

```python
validator.save_metrics()
```

* Сохраняет текущие метрики в `model_storage/metrics/*.json` с таймстампом.

### Интерпретация

```python
explain_with_shap(model, X_test, feature_names)
# — генерирует SHAP summary и waterfall, сохраняет в models/shap/.

explain_with_lime(X_train, X_test, model.predict, feature_names)
# — создаёт LIME-отчёт HTML, сохраняет как models/lime.html.
```

### Мониторинг дрейфа

```python
detect_drift(baseline_X, X_new, feature_names, alpha=0.05)
# — KS-тест по каждому признаку, логгирует drift.
Возвращает 1 если найден дрифт и 0 если не найден.
```

**Config** для `ModelValidator`:

* `model_storage`, `random_state`, `test_size`, `cv_folds`, `n_splits`, `scoring`, `target_column`.

---

## main.py

Пример полноценного конвейера:

1. Сбор и эмуляция потоковых батчей (`DataLoader`).
2. Препроцессинг на всей выборке + сохранение.
3. Обучение XGBoost + степень дообучения на поступивших батчах.
4. Валидация (hold-out, CV, TimeSeriesCV) + сохранение метрик.
5. Интерпретация (SHAP, LIME) + мониторинг дрейфа.
6. Сохранение всех артефактов (модель, прeпроцессор, конфиг, метрики).


---

## Зависимости

```text
pandas
numpy
scikit-learn
xgboost
shap
lime
scipy
matplotlib
```

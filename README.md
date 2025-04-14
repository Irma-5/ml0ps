# ml0ps
# Artifact structure  
model_artifacts  
├── model.xgb                # Сериализованная модель XGBoost  
├── preprocessor.joblib      # Состояние препроцессора. 
├── config.json              # Параметры конфигурации  
├── validation_artifacts.joblib # Метрики и валидатор  
└── requirements.txt         # Версии зависимостей  

# Credit Risk Model Pipeline

## Описание
Конвейер для прогнозирования сроков кредитов с возможностью:
- Предсказаний (inference)
- Обновления модели (update)
- Получения статистики (summary)

## Требования
- Python 3.8+
- Зависимости (см. requirements.txt)

## Установка
```bash
mkdir mlops
cd mlops
cd ml0ps
git clone https://github.com/Irma-5/ml0ps
pip install -r requirements.txt
```

## Использование

### Предсказания
```bash
python pipeline_manager.py inference --data path/to/data.csv
```
Результат сохраняется в predictions_<timestamp>.csv

### Обновление модели
```bash
python pipeline_manager.py update --data path/to/new_data.csv
```

### Получение статистики
```bash
python pipeline_manager.py summary
```
### Изменение конфигурации данных напрямую
Path указывает на папку в котором лежит config.py. В нем реализован словарь DATA_LOADER_PARAMS с парами path->str,year_to_split->int, num_batch->int. year_to_split>= 2000 и на момент реализации <=2023.
```bash
python3 pipeline_manager.py setconfig --param path --value "cofig_folder"
python3 pipeline_manager.py setconfig --param year_to_split --value 2010
python3 pipeline_manager.py setconfig --param num_batch --value 10
```
## Структура артефактов
```
model_artifacts/
├── model.xgb
├── preprocessor.joblib
├── config.json
└── validation_artifacts.joblib
```

## Поддерживаемые форматы данных
- CSV с колонками согласно обучающим данным
- Кодировка: UTF-8
- Разделитель: запятая

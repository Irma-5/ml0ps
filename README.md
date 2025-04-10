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

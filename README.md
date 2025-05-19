# ml0ps
# Artifact structure  
model_artifacts  
├── current_model.json           # Сериализованная модель XGBoost  
├── preprocessor.pkl      # Состояние препроцессора. 
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

## Создание модели  
python3 pipeline_manager.py inference create_model

## Предсказания
```bash
python3 pipeline_manager.py inference
```
Результат сохраняется в predictions_<timestamp>.csv

## Обновление модели
```bash
python3 pipeline_manager.py update 
```

## Получение статистики
```bash
python3 pipeline_manager.py summary
```
## Изменение конфигурации данных напрямую
config.json хранит все изменяемые параметры

```bash
python3 pipeline_manager.py setconfig --param year_to_split --value 2010
```
```bash
python3 pipeline_manager.py setconfig --param num_batch --value 10
```
## Структура артефактов
```
model_artifacts/
├── model.xgb - лучшая модель на валидации по метрике мае
├── preprocessor.joblib
├── metrics
├──metric_his.json
└── model_metadata.json
```

## Поддерживаемые форматы данных
- CSV с колонками согласно обучающим данным
- Кодировка: UTF-8
- Разделитель: запятая

## Подробнее про работу модели написано в документации в папке doc

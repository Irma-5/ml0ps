import pickle
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from zipfile import ZipFile
from typing import List
import re
import io
from typing import Dict
import warnings
from DataClasses_пон import DataLoader
# data_loader = DataLoader()
# df1 = data_loader.first() # загрузка train датасета
# print(df1.shape)
# df1 = data_loader.step_(df1, verbose=2) # новая порция данных за 2010-2011 (кредиты, начатые в 1999 и закрытые в 2010-2011)
# df1 = data_loader.step_(df1, verbose=2)
# df1 = data_loader.step_(df1, verbose=2)
# df1 = data_loader.step_(df1, verbose=2)
# df1 = data_loader.step_(df1, verbose=2)
# df1 = data_loader.step_(df1, verbose=2)
# df1 = data_loader.step_(df1, verbose=2)
# df1.to_csv('kal.csv', index=False)
# print(df1.shape)

import sys
import numpy as np
from datetime import datetime
from sklearn.metrics import mean_absolute_error
from pathlib import Path
import logging
import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split
from preprocessing_пон import CreditDataPreprocessor
import json
from sklearn.model_selection import RandomizedSearchCV
from model_пон import CreditModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__) # чтобы всё отображались

config = {
    'model_storage': './models',
    'preprocessor_path': './preprocessors',
    'random_state': 42,
    'test_size': 0.2,
    'target_column': 'time',
    'hyperparam_tuning': False,  # Флаг для включения/выключения подбора параметров
    'cv_folds': 3,  # Количество фолдов для кросс-валидации
    'n_iter': 20  # Количество итераций для RandomizedSearch
}
preprocessor = CreditDataPreprocessor(config)

try:
    # Загрузка и обработка данных
    raw_df = pd.read_csv('./kal.csv')
    logging.info(f"Raw data shape: {raw_df.shape}")
    mid_index = len(raw_df) // 2
    # Разделяем DataFrame на две части
    df1 = raw_df.iloc[:mid_index].reset_index(drop=True)
    df2 = raw_df.iloc[mid_index:].reset_index(drop=True)
    raw_df = df1
    # Препроцессинг данных
    X_processed, y = preprocessor.fit_transform(raw_df)
    config['n_features'] = X_processed.shape[1]  # Обновляем количество признаков
    # Разделение данных
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y,
        test_size=config['test_size'],
        random_state=config['random_state'], shuffle=False
    )
    
    # Сохранение препроцессора
    Path(config['preprocessor_path']).mkdir(exist_ok=True)
    preprocessor.save(config['preprocessor_path'])
    logging.info("Preprocessor saved successfully")
    # Инициализация и обучение модели
    model = CreditModel(config)

    if config['hyperparam_tuning']:
        logging.info("Starting hyperparameter tuning...")
        
        # Сетка параметров для поиска
        param_grid = {
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 5, 7],
            'n_estimators': [100, 200, 300],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0],
            'reg_alpha': [0, 0.1, 1],
            'reg_lambda': [1]
        }
        # Создание и настройка поисковика
        search = RandomizedSearchCV(
            estimator=model.model,
            param_distributions=param_grid,
            n_iter=config['n_iter'],
            scoring='neg_mean_absolute_error',
            cv=config['cv_folds'],
            verbose=3,
            random_state=config['random_state'],
            n_jobs=-1
        )
        # Запуск поиска
        search.fit(X_train, y_train)
        
        logging.info(f"Best parameters: {search.best_params_}")
        logging.info(f"Best MAE: {-search.best_score_:.2f}")
        # Обновляем модель с лучшими параметрами
        model.model = search.best_estimator_
        
        # Сохраняем лучшие параметры в конфиг
        config.update(search.best_params_)
    else:
        logging.info("Skipping hyperparameter tuning...")
    # Базовая оценка
    model.train(X_train, y_train, update=False)
    metrics = model.evaluate(X_train, y_train)
    logging.info("\nInitial test metrics:")
    logging.info(f"MAE: {metrics['mae']:.2f} months")
    logging.info(f"R²: {metrics['r2']:.4f}")
    model.save_model()
    
    # Пример дообучения с новыми данными
    logging.info("\nPreparing update batch...")
    X_update, y_update = preprocessor.transform(df2), df2[config['target_column']].values
    
    logging.info("Updating model...")
    model.train(X_test, y_test, update=True)
    
    # Оценка после дообучения
    updated_metrics = model.evaluate(X_test, y_test)
    logging.info("\nMetrics after update:")
    logging.info(f"MAE: {updated_metrics['mae']:.2f} months")
    logging.info(f"R²: {updated_metrics['r2']:.4f}")
    
    # Сохранение итоговой модели
    model.save_model()
    logging.info("Final model saved successfully")
except Exception as e:
    logging.error(f"Pipeline failed: {str(e)}")
    sys.exit(1)


###Validation
from validation_пон import ModelValidator
config = {
        'model_storage': './test_models',
        'reports_dir': './test_reports',
        'n_splits': 3,
        'improvement_threshold': 0.05,  # 5% улучшение для обновления модели
        'model_storage': './models',
        'preprocessor_path': './preprocessors',
        'random_state': 42,
        'test_size': 0.2,
        'target_column': 'time',
        'hyperparam_tuning': False,  # Флаг для включения/выключения подбора параметров
        'cv_folds': 3,  # Количество фолдов для кросс-валидации
        'n_iter': 20  # Количество итераций для RandomizedSearch
    }

    # Создаем необходимые директории
Path(config['model_storage']).mkdir(parents=True, exist_ok=True)
Path(config['reports_dir']).mkdir(parents=True, exist_ok=True)
# Генерируем тестовые данные
raw_df = pd.read_csv('./kal.csv')
preprocessor = CreditDataPreprocessor(config)
X_test, y_test = preprocessor.fit_transform(raw_df)
# Инициализируем валидатор
validator = ModelValidator(config)
try:
    print("=== Тест кросс-валидации ===")
    avg_mae = validator.cross_validate(X_test, y_test)
    print(f"Средний MAE по кросс-валидации: {avg_mae:.2f}")
    print("\n=== Тест сравнения моделей ===")
    # Создаем тестовые метрики новой модели
    new_metrics = {'mae': avg_mae * 0.9}  # Имитируем улучшение на 10%
    
    if validator.validate_new_model(new_metrics):
        print("Новая модель лучше! Произошло обновление.")
    else:
        print("Новая модель не лучше текущей.")
    print("\n=== Проверка генерации отчетов ===")
    report_files = list(Path(config['reports_dir']).glob('*.json'))
    print(f"Найдено отчетов: {len(report_files)}")
    if report_files:
        with open(report_files[0], 'r') as f:
            print("Пример отчета:", json.load(f))
except Exception as e:
    print(f"Ошибка при выполнении тестов: {str(e)}")
    raise

import mlflow
import xgboost as xgb
from sklearn.metrics import mean_absolute_error

import joblib
import json
from pathlib import Path

# После обучения модели добавить:

def save_artifacts(model, preprocessor, config, validator):
    # Создаем директории
    artifact_dir = Path("model_artifacts")
    artifact_dir.mkdir(exist_ok=True)
    
    # 1. Сохраняем модель XGBoost
    model.model.save_model(artifact_dir/"model.xgb")
    
    # 2. Сохраняем препроцессор
    joblib.dump(preprocessor, artifact_dir/"preprocessor.joblib")
    
    # 3. Сохраняем конфигурацию
    with open(artifact_dir/"config.json", "w") as f:
        json.dump(config, f)
    
    # 4. Сохраняем валидатор и метрики
    joblib.dump({
        'validator': validator,
        'metrics': {
            'initial_metrics': metrics,
            'updated_metrics': updated_metrics
        }
    }, artifact_dir/"validation_artifacts.joblib")

    # 5. Сохраняем информацию о версиях
    with open(artifact_dir/"requirements.txt", "w") as f:
        f.write("\n".join([
            "xgboost==1.7.6",
            "scikit-learn==1.2.2",
            "pandas==2.0.3",
            "mlflow==2.6.0"
        ]))

# Вызываем после обучения:
save_artifacts(model, preprocessor, config, validator)
import sys
import numpy as np
from datetime import datetime
from sklearn.metrics import mean_absolute_error
from pathlib import Path
import logging
import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split
from preprocessing import CreditDataPreprocessor
import json
from sklearn.model_selection import RandomizedSearchCV


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__) # чтобы всё отображались

class CreditModel:
    def __init__(self, config):
        self.model = None
        self.config = config
        self.model_path = Path(config['model_storage']) / 'current_model.json'
        self._init_model()
        model_dir = Path(self.config['model_storage'])
        model_dir.mkdir(parents=True, exist_ok=True)

    def _init_model(self):
        """Инициализация модели с параметрами из конфига"""
        if self.model_path.exists():
            self.load_model()
        else:
            self.model = xgb.XGBRegressor(
                objective='reg:squarederror',
                learning_rate = 0.01,
                max_depth = 7,
                n_estimators=200,
                subsample=0.8,
                colsample_bytree=0.9,
                reg_alpha=0.3,
                reg_lambda=0.9,
                random_state=42
            )

    def train(self, X, y, update=False, save=False):
        """Обучение/дообучение модели"""
        if update and self.model is not None:
            self.model = xgb.XGBRegressor(learning_rate = 0.05)
            self.model.load_model(self.model_path)
            print(self.model_path)
            self.model.fit(X, y, 
                         xgb_model=self.model)
        else:
            self.model.fit(X, y)
        if save:
            self._save_model()
            logger.info(f"Model {'updated' if update else 'trained'} successfully")

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X_test, y_test):
        """Оценка качества модели""" # для удобства реализована и тут
        y_pred = self.predict(X_test)
        return {
            'mae': mean_absolute_error(y_test, y_pred),
            'r2': self.model.score(X_test, y_test)
        }

    def _save_model(self):
        """Сохранение модели в native XGBoost формате"""
        self.model.save_model(self.model_path)

    def load_model(self):
        """Загрузка модели из файла"""
        self.model = xgb.XGBRegressor()
        self.model.load_model(self.model_path)
    
    def save_model(self, custom_path=None):
        """
        Сохранение модели с возможностью указать кастомный путь
        :param custom_path: Опциональный путь для сохранения
        """
        try:
            save_path = Path(custom_path) if custom_path else self.model_path
            save_dir = save_path.parent
            save_dir.mkdir(parents=True, exist_ok=True)
            
            self.model.save_model(save_path)
            
            metadata = {
                'saved_at': datetime.now().isoformat(), # дополнительные метаданные
                'input_shape': self.model.n_features_in_ if hasattr(self.model, 'n_features_in_') else None,
                'model_type': str(type(self.model)),
                'config': self.config
            }
            with open(save_dir / 'model_metadata.json', 'w') as f: # метаданные в отдельный файл
                json.dump(metadata, f, indent=4)
            logger.info(f"Model saved successfully to {save_path}")
            
        except Exception as e:
            logger.error(f"Failed to save model: {str(e)}")
            raise

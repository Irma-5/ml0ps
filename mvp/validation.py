import json
from datetime import datetime
from pathlib import Path
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from model import CreditModel
import pickle
import pandas as pd


class ModelValidator:
    def __init__(self, config):
        self.config = config
        self.storage_path = Path(config['model_storage'])
        self.reports_path = Path(config['reports_dir'])
    
    def cross_validate(self, X, y):
        """Кросс-валидация с временным разделением"""
        tscv = TimeSeriesSplit(n_splits=self.config['n_splits'])
        metrics = []
        
        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            model = CreditModel(self.config)
            model.train(X_train, y_train)
            metrics.append(model.evaluate(X_test, y_test))
        
        self._save_validation_report(metrics)
        return np.mean([m['mae'] for m in metrics])

    def validate_new_model(self, new_model_metrics):
        """Сравнение новой модели с текущей лучшей"""
        best_model_path = self.storage_path / 'best_model.pkl'
        
        if not best_model_path.exists():
            return True
            
        with open(best_model_path, 'rb') as f:
            best_model = pickle.load(f)
        
        improvement = (best_model['metadata']['mae'] - new_model_metrics['mae']) / best_model['metadata']['mae']
        
        if improvement > self.config['improvement_threshold']:
            self._update_best_model(new_model_metrics)
            return True
        return False

    def _save_validation_report(self, metrics):
        """Сохранение отчета о валидации"""
        report = {
            'created_at': datetime.now().isoformat(),
            'avg_mae': np.mean([m['mae'] for m in metrics]),
            'avg_r2': np.mean([m['r2'] for m in metrics]),
            'n_folds': len(metrics)
        }
        
        filename = self.reports_path / f"validation_{datetime.now().strftime('%Y%m%d%H%M')}.json"
        with open(filename, 'w') as f:
            json.dump(report, f)

    def _update_best_model(self, metrics):
        """Обновление лучшей модели"""
        current_model_path = self.storage_path / 'current_model.pkl'
        best_model_path = self.storage_path / 'best_model.pkl'
        
        with open(current_model_path, 'rb') as src, open(best_model_path, 'wb') as dst:
            model_data = pickle.load(src)
            model_data['metadata'].update(metrics)
            pickle.dump(model_data, dst)

import json
import logging
from pathlib import Path
from datetime import datetime

import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from preprocessing import CreditDataPreprocessor
from model import CreditModel

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ModelValidator:
    def __init__(self, config):
        self.config = config
        self.preprocessor = CreditDataPreprocessor(config)
        self.model = CreditModel(config)
        self.metrics = {}

    def load_data(self, data_path):
        """Загрузка данных из CSV-файла"""
        df = pd.read_csv(data_path)
        return df

    def validate(self, df, method='holdout'):
        """
        Проверка модели.
        method: 'holdout', 'cv' или 'timeseries'
        """
        X, y = self.preprocessor.fit_transform(df)

        if method == 'holdout':
            test_size = self.config.get('test_size', 0.2)
            random_state = self.config.get('random_state', 42)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
            self.model.train(X_train, y_train)
            self.metrics = self.model.evaluate(X_test, y_test)

        elif method == 'cv':
            cv = self.config.get('cv_folds', 5)
            scoring = self.config.get('scoring', 'neg_mean_absolute_error')
            scores = cross_val_score(self.model.model, X, y, cv=cv, scoring=scoring)
            self.metrics = {
                'cv_scores': scores.tolist(),
                'cv_mean_score': float(scores.mean())
            }

        elif method == 'timeseries':
            n_splits = self.config.get('n_splits', 5)
            tscv = TimeSeriesSplit(n_splits=n_splits)
            scoring = self.config.get('scoring', 'neg_mean_absolute_error')
            scores = cross_val_score(self.model.model, X, y, cv=tscv, scoring=scoring)
            self.metrics = {
                'tscv_scores': scores.tolist(),
                'tscv_mean_score': float(scores.mean())
            }

        else:
            raise ValueError(f"Unknown validation method: {method}")

        logger.info(f"Validation metrics: {self.metrics}")
        return self.metrics

    def save_metrics(self):
        """Сохранение метрик в JSON с временной меткой"""
        metrics_dir = Path(self.config['model_storage']) / 'metrics'
        metrics_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        metrics_path = metrics_dir / f"metrics_{timestamp}.json"
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics, f, indent=4)
        logger.info(f"Metrics saved to {metrics_path}")
        return metrics_path

if __name__ == "__main__":
    import json
    # Пример запуска из консоли:
    config_path = "config.json"
    with open(config_path, 'r') as f:
        config = json.load(f)
    data_path = config['data_path']

    validator = ModelValidator(config)
    df = validator.load_data(data_path)
    metrics = validator.validate(df, method=config.get('validation_method', 'holdout'))
    metrics_path = validator.save_metrics()
    print(f"Validation complete. Metrics: {metrics}")
    print(f"Metrics saved at: {metrics_path}")

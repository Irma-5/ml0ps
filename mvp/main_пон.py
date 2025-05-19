import os
import sys
import json
import logging
from pathlib import Path
from sklearn.model_selection import train_test_split
import joblib
import sys
from datetime import datetime
import time
import psutil
from DataLoader.DataClasses import DataLoader
from DataLoader.config import DATA_LOADER_PARAMS
from preprocessing import CreditDataPreprocessor
from model import CreditModel
from validation import ModelValidator, detect_drift, explain_with_lime, explain_with_shap
import pickle
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global config (could also be loaded/saved)
with open('mvp/model_artifacts/config.json', 'r') as f:
    CONFIG = json.load(f)
del CONFIG["year_to_split"]
del CONFIG["num_batch"]
def initialize_pipeline():
    """Initial setup and data loading"""
    process = psutil.Process()
    start_time = time.time()
    start_memory = process.memory_info().rss / 1024**2
    data_loader = DataLoader(DATA_LOADER_PARAMS)
    data_loader.create_datasets(verbose=True)
    elapsed_time = time.time() - start_time
    logger.info(f"Dataset creation time: {elapsed_time:.4f} seconds")
    end_memory = process.memory_info().rss / 1024**2
    logging.info(f"Memory usage: {end_memory - start_memory:.2f} MB (Delta) | ")
    return data_loader

# def load_batches(data_loader):
#     """Load all available batches"""
#     batches = []
#     for i in range(DATA_LOADER_PARAMS['num_batch']):
#         batch = data_loader.get_data(verbose=0)
#         batches.append(batch)
#         logger.info(f"Loaded batch {i+1}/{DATA_LOADER_PARAMS['num_batch']}")
#     return batches

def train_initial_model(data_loader, flag=True):
    """Initial model training"""
    process = psutil.Process()
    start_time = time.time()
    start_memory = process.memory_info().rss / 1024**2
    if flag:
        train_df = data_loader.get_data()
    else:
        train_df = data_loader
    logger.info(f"Initial training data shape: {train_df.shape}")

    preprocessor = CreditDataPreprocessor(CONFIG)
    X_processed, y = preprocessor.fit_transform(train_df.copy())
    CONFIG['n_features'] = X_processed.shape[1]

    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y,
        test_size=CONFIG['test_size'],
        random_state=CONFIG['random_state'],
        shuffle=False
    )

    Path(CONFIG['preprocessor_path']).mkdir(parents=True, exist_ok=True)
    # preprocessor.save(CONFIG['preprocessor_path'])
    logger.info("Preprocessor saved") 

    model = CreditModel(CONFIG)
    
    if CONFIG['hyperparam_tuning']:
        from sklearn.model_selection import RandomizedSearchCV
        logger.info("Starting hyperparameter tuning...")
        param_grid = {
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 5, 7],
            'n_estimators': [100, 200, 300],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0],
            'reg_alpha': [0, 0.1, 1],
            'reg_lambda': [1]
        }
        search = RandomizedSearchCV(
            estimator=model.model,
            param_distributions=param_grid,
            n_iter=CONFIG['n_iter'],
            scoring='neg_mean_absolute_error',
            cv=CONFIG['cv_folds'],
            verbose=2,
            random_state=CONFIG['random_state'],
            n_jobs=-1
        )
        search.fit(X_train, y_train)
        model.model = search.best_estimator_
        CONFIG.update(search.best_params_)

    model.train(X_train, y_train, update=False)
    model.save_model()
    logger.info("Initial model trained and saved")
    elapsed_time = time.time() - start_time
    logger.info(f"Trainig time: {elapsed_time:.4f} seconds")
    end_memory = process.memory_info().rss / 1024**2
    logging.info(f"Memory usage: {end_memory - start_memory:.2f} MB (Delta) | ")
    
    return model, preprocessor, X_test, y_test

def update_model(batch, model, preprocessor):
    """Update model with a single batch"""
    process = psutil.Process()
    start_time = time.time()
    start_memory = process.memory_info().rss / 1024**2
    logger.info(f"Processing batch of shape {batch.shape}")
    batch = batch[batch['zero_balance_code'] == 1.0]
    
    X_update = preprocessor.transform(batch)
    y_update = batch[CONFIG['target_column']].values
    
    X_up_train, X_up_test, y_up_train, y_up_test = train_test_split(
        X_update, y_update,
        test_size=0.15,
        random_state=CONFIG['random_state'],
        shuffle=True
    )
    
    model.train(X_up_train, y_up_train, update=True)
    metrics = model.evaluate(X_up_test, y_up_test)
    model.save_model()
    logger.info(f"Update complete. MAE: {metrics['mae']:.2f}, R2: {metrics['r2']:.4f}")
    elapsed_time = time.time() - start_time
    logger.info(f"Fine-tuning time: {elapsed_time:.4f} seconds")
    end_memory = process.memory_info().rss / 1024**2
    logging.info(f"Memory usage: {end_memory - start_memory:.2f} MB (Delta) | ")
    
    return metrics

def validate_model(validator, data):
    """Run validation routines"""
    process = psutil.Process()

    print("=== Hold-out validation ===")
    start_time = time.time()
    start_memory = process.memory_info().rss / 1024**2
    metrics_hold = validator.validate(data, method='holdout')
    validator.save_metrics()
    elapsed_time1 = time.time() - start_time
    end_memory1 = process.memory_info().rss / 1024**2
    
    print("=== K-Fold CV ===")
    start_time = time.time()
    start_memory = process.memory_info().rss / 1024**2
    metrics_cv = validator.validate(data, method='cv')
    validator.save_metrics()
    elapsed_time2 = time.time() - start_time
    end_memory2 = process.memory_info().rss / 1024**2
    
    print("=== TimeSeries CV ===")
    start_time = time.time()
    start_memory = process.memory_info().rss / 1024**2
    metrics_ts = validator.validate(data, method='timeseries')
    validator.save_metrics()
    elapsed_time3 = time.time() - start_time
    end_memory3 = process.memory_info().rss / 1024**2

    logging.info(
            f"Hold-out validation: {elapsed_time1:.2f}sec and {end_memory1:.2f} MB (Delta) | "
            f"K-Fold CV: {elapsed_time2:.2f}sec and {end_memory2:.2f} MB (Delta) | "
            f"TimeSeries CV: {elapsed_time3:.2f}sec and {end_memory3:.2f} MB (Delta) | "
        )
    
    return {
        'holdout': metrics_hold,
        'cv': metrics_cv,
        'timeseries': metrics_ts
    }


def save_artifacts(model_obj, preprocessor_obj, config_obj, validator_obj,metrics, artifacts_dir="model_artifacts", batch_num = 10000):
    artifact_dir = Path(artifacts_dir)
    artifact_dir.mkdir(exist_ok=True)
    model_obj.model.save_model(artifact_dir / f"model_batch{batch_num}.xgb")
    joblib.dump(preprocessor_obj, artifact_dir / "preprocessor.joblib")

    metrics_entry = {
        "num": batch_num,
        **metrics
    }

    # Загрузка существующих данных
    history = []
    best_mae = float("inf")
    if Path(artifact_dir/"metric_his.json").exists():
        with open(artifact_dir/"metric_his.json", "r") as f:
            try:
                history = json.load(f)
                best_mae = min([entry["holdout"]["mae"] for entry in history])
            except json.JSONDecodeError:
                history = []

    if metrics["holdout"]["mae"] < best_mae:
        model_obj.model.save_model(artifact_dir / "model.xgb")
    # Добавление новой записи
    history.append(metrics_entry)

    with open(artifact_dir/"metric_his.json", "w") as f:
        json.dump(history, f, indent=2)

    # with open(artifact_dir / "config.json", "w") as f:
    #     json.dump(config_obj, f, indent=4)
    # joblib.dump({'validator': validator_obj,
    #              'metrics': {
    #                  'holdout': metrics['holdout'],
    #                  'cv': metrics['cv'],
    #                  'timeseries': metrics['timeseries']
    #              }}, artifact_dir / "validation_artifacts.joblib")
    # with open(artifact_dir / "requirements.txt", "w") as f:
    #     f.write("\n".join([
    #         "xgboost==1.7.6",
    #         "scikit-learn==1.2.2",
    #         "pandas==2.0.3",
    #         "shap==0.42.1",
    #         "lime==0.2.0",
    #         "scipy==1.10.1"
    #     ]))
    # logger.info(f"Artifacts saved to {artifact_dir}")

    # save_artifacts(model, preprocessor, config, validator)

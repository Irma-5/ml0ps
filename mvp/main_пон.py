import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
from typing import List, Dict
import warnings
from sklearn.model_selection import train_test_split

from DataLoader.DataClasses import DataLoader
from DataLoader.config import DATA_LOADER_PARAMS
from preprocessing import CreditDataPreprocessor
from model import CreditModel
from validation import ModelValidator, detect_drift, explain_with_lime, explain_with_shap

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    data_loader = DataLoader(DATA_LOADER_PARAMS)
    if not os.path.exists("DataLoader/datasets"):
        data_loader.create_datasets(verbose=True)
    initial_train_df = data_loader.get_data()
    logger.info(f"Initial training data shape: {initial_train_df.shape}")

    fine_tune_batches = []
    for i in range(DATA_LOADER_PARAMS['num_batch']):
        batch = data_loader.get_data(verbose=0)
        fine_tune_batches.append(batch)
        logger.info(f"Loaded fine-tuning batch {i+1}/{DATA_LOADER_PARAMS['num_batch']} with shape {batch.shape}")


    config = {
        'model_storage':     './models',
        'preprocessor_path': './preprocessors',
        'random_state':      42,
        'test_size':         0.2,
        'target_column':     'time',
        'hyperparam_tuning': False,
        'cv_folds':          3,
        'n_iter':            20
    }

    preprocessor = CreditDataPreprocessor(config)
    raw_df = initial_train_df.copy()
    X_processed, y = preprocessor.fit_transform(raw_df)
    config['n_features'] = X_processed.shape[1]


    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y,
        test_size=config['test_size'],
        random_state=config['random_state'],
        shuffle=False
    )


    Path(config['preprocessor_path']).mkdir(parents=True, exist_ok=True)
    preprocessor.save(config['preprocessor_path'])
    logger.info("Preprocessor saved successfully")

    model = CreditModel(config)

    if config['hyperparam_tuning']:
        from sklearn.model_selection import RandomizedSearchCV
        logger.info("Starting hyperparameter tuning...")
        param_grid = {
            'learning_rate':   [0.01, 0.05, 0.1],
            'max_depth':       [3, 5, 7],
            'n_estimators':    [100, 200, 300],
            'subsample':       [0.8, 0.9, 1.0],
            'colsample_bytree':[0.8, 0.9, 1.0],
            'reg_alpha':       [0, 0.1, 1],
            'reg_lambda':      [1]
        }
        search = RandomizedSearchCV(
            estimator=model.model,
            param_distributions=param_grid,
            n_iter=config['n_iter'],
            scoring='neg_mean_absolute_error',
            cv=config['cv_folds'],
            verbose=2,
            random_state=config['random_state'],
            n_jobs=-1
        )
        search.fit(X_train, y_train)
        logger.info(f"Best params: {search.best_params_}")
        model.model = search.best_estimator_
        config.update(search.best_params_)
    else:
        logger.info("Skipping hyperparameter tuning...")

    model.train(X_train, y_train, update=False)
    metrics_init = model.evaluate(X_train, y_train)
    logger.info(f"Initial train MAE: {metrics_init['mae']:.2f}, R2: {metrics_init['r2']:.4f}")
    model.save_model()

    for idx, batch_df in enumerate(fine_tune_batches, start=1):
        logger.info(f"=== Fine-tuning batch {idx}/{len(fine_tune_batches)} ===")
        batch_df = batch_df[batch_df['zero_balance_code'] == 1.0]
        X_update = preprocessor.transform(batch_df)
        y_update = batch_df[config['target_column']].values
        X_up_train, X_up_test, y_up_train, y_up_test = train_test_split(
            X_update, y_update,
            test_size=0.15,
            random_state=config['random_state'],
            shuffle=True
        )
        model.train(X_up_train, y_up_train, update=True)
        metrics_up = model.evaluate(X_up_test, y_up_test)
        logger.info(f"After batch {idx} MAE: {metrics_up['mae']:.2f}, R2: {metrics_up['r2']:.4f}")

    model.save_model()
    logger.info("Final model saved successfully")

    val_config = {
        'model_storage':  config['model_storage'],
        'random_state':   config['random_state'],
        'test_size':      config['test_size'],
        'cv_folds':       config['cv_folds'],
        'n_splits':       config.get('n_splits', 5),
        'scoring':        'neg_mean_absolute_error',
        'target_column':  config['target_column']
    }
    validator = ModelValidator(val_config)

    print("=== Hold-out validation ===")
    metrics_hold = validator.validate(initial_train_df, method='holdout')
    print(metrics_hold)
    validator.save_metrics()

    print("=== K-Fold CV ===")
    metrics_cv = validator.validate(initial_train_df, method='cv')
    print(metrics_cv)
    validator.save_metrics()

    print("=== TimeSeries CV ===")
    metrics_ts = validator.validate(initial_train_df, method='timeseries')
    print(metrics_ts)
    validator.save_metrics()


    explain_with_shap(model, X_test, preprocessor.feature_names)
    explain_with_lime(X_train, X_test, model.predict, preprocessor.feature_names)

    baseline_X = X_train.copy()
    last_batch = fine_tune_batches[-1]
    X_new = preprocessor.transform(last_batch[last_batch['zero_balance_code']==1.0])
    detect_drift(baseline_X, X_new, preprocessor.feature_names)

    def save_artifacts(model_obj, preprocessor_obj, config_obj, validator_obj, artifacts_dir="model_artifacts"):
        import joblib
        artifact_dir = Path(artifacts_dir)
        artifact_dir.mkdir(exist_ok=True)
        model_obj.model.save_model(artifact_dir / "model.xgb")
        joblib.dump(preprocessor_obj, artifact_dir / "preprocessor.joblib")
        with open(artifact_dir / "config.json", "w") as f:
            json.dump(config_obj, f, indent=4)
        joblib.dump({'validator': validator_obj,
                     'metrics': {
                         'holdout': metrics_hold,
                         'cv': metrics_cv,
                         'timeseries': metrics_ts
                     }}, artifact_dir / "validation_artifacts.joblib")
        with open(artifact_dir / "requirements.txt", "w") as f:
            f.write("\n".join([
                "xgboost==1.7.6",
                "scikit-learn==1.2.2",
                "pandas==2.0.3",
                "shap==0.42.1",
                "lime==0.2.0",
                "scipy==1.10.1"
            ]))
        logger.info(f"Artifacts saved to {artifact_dir}")

    save_artifacts(model, preprocessor, config, validator)

except Exception as e:
    logger.error(f"Pipeline failed: {e}")
    sys.exit(1)
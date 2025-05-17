import os
import sys
import json
import logging
from pathlib import Path
from sklearn.model_selection import train_test_split
import joblib
import sys
from DataLoader.DataClasses import DataLoader
from DataLoader.config import DATA_LOADER_PARAMS
from preprocessing import CreditDataPreprocessor
from model import CreditModel
from validation import ModelValidator, detect_drift, explain_with_lime, explain_with_shap
import pickle
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s",
handlers=[logging.FileHandler('logfile.log', encoding='utf-8'),
          logging.StreamHandler()])
logger = logging.getLogger(__name__)

# Global config (could also be loaded/saved)
CONFIG = {
    'model_storage': './model_artifacts',
    'preprocessor_path': './model_artifacts',
    # 'model_path': './model_artifacts/model',
    'random_state': 42,
    'test_size': 0.2,
    'target_column': 'time',
    'hyperparam_tuning': False,
    'cv_folds': 3,
    'n_iter': 20
}

def initialize_pipeline():
    """Initial setup and data loading"""
    data_loader = DataLoader(DATA_LOADER_PARAMS)
    data_loader.create_datasets(verbose=True)
    return data_loader

# def load_batches(data_loader):
#     """Load all available batches"""
#     batches = []
#     for i in range(DATA_LOADER_PARAMS['num_batch']):
#         batch = data_loader.get_data(verbose=0)
#         batches.append(batch)
#         logger.info(f"Loaded batch {i+1}/{DATA_LOADER_PARAMS['num_batch']}")
#     return batches

def train_initial_model(data_loader):
    """Initial model training"""
    train_df = data_loader.get_data()
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
    return model, preprocessor, X_test, y_test

def update_model(batch, model, preprocessor):
    """Update model with a single batch"""
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
    return metrics

def validate_model(validator, data):
    """Run validation routines"""
    print("=== Hold-out validation ===")
    metrics_hold = validator.validate(data, method='holdout')
    validator.save_metrics()
    print("=== K-Fold CV ===")
    metrics_cv = validator.validate(data, method='cv')
    validator.save_metrics()
    print("=== TimeSeries CV ===")
    metrics_ts = validator.validate(data, method='timeseries')
    validator.save_metrics()
    return {
        'holdout': metrics_hold,
        'cv': metrics_cv,
        'timeseries': metrics_ts
    }

def save_artifacts(model_obj, preprocessor_obj, config_obj, validator_obj,metrics, artifacts_dir="model_artifacts"):
    import joblib
    artifact_dir = Path(artifacts_dir)
    artifact_dir.mkdir(exist_ok=True)
    model_obj.model.save_model(artifact_dir / "model.xgb")
    joblib.dump(preprocessor_obj, artifact_dir / "preprocessor.joblib")
    with open(artifact_dir / "config.json", "w") as f:
        json.dump(config_obj, f, indent=4)
    joblib.dump({'validator': validator_obj,
                 'metrics': {
                     'holdout': metrics['holdout'],
                     'cv': metrics['cv'],
                     'timeseries': metrics['timeseries']
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

    # save_artifacts(model, preprocessor, config, validator)

if __name__ == "__main__":
    try:
        # Initial pipeline setup
        data_loader = initialize_pipeline()
        # main_data = data_loader.get_data()
        model, preprocessor, X_test, y_test = train_initial_model(data_loader)
        
        # Initialize validator
        # validator = ModelValidator({
        #     'model_storage': CONFIG['model_storage'],
        #     'random_state': CONFIG['random_state'],
        #     'target_column': CONFIG['target_column']
        # })
        
        # # Initial validation
        # val_metrics = validate_model(validator, data_loader.get_data())
        
        # Save initial artifacts
        # save_artifacts(model, preprocessor, validator, val_metrics)

        
    except Exception as e:
        logger.error(f"Initial pipeline failed: {e}")
        sys.exit(1)

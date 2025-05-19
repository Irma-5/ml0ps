#!/usr/bin/env python3
import argparse
import logging
import joblib
import pickle
import sys
from pathlib import Path
from main_пон import initialize_pipeline, update_model,validate_model, save_artifacts,train_initial_model, CONFIG
from model import CreditModel
from preprocessing import CreditDataPreprocessor
from validation import ModelValidator, detect_drift
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s",
handlers=[logging.FileHandler('logfile.log', encoding='utf-8'),
          logging.StreamHandler()])
logger = logging.getLogger(__name__)

def load_components():
    """Load existing model and preprocessor"""
    try:
        preprocessor = joblib.load(Path(CONFIG['preprocessor_path']) / "preprocessor.joblib")
        model = CreditModel(CONFIG)
        model.load_model()
        return model, preprocessor
    except Exception as e:
        logger.error(f"Failed to load model components: {e}")
        raise

def create_model():
    """Run model inference on new data"""
    try:
        # Initial pipeline setup
        data_loader = initialize_pipeline()
        # main_data = data_loader.get_data()
        model, preprocessor, X_test, y_test = train_initial_model(data_loader)
        preds = model.predict(X_test)
        metrics_init = model.evaluate(X_test, y_test)
        # with open('model.pkl', 'wb') as f:
        #     pickle.dump(model, f)
        # with open('processor.pkl', 'wb') as f:
        #     pickle.dump(preprocessor, f)
        model.save_model()
        preprocessor.save("")
        logger.info(f"Initial train MAE: {metrics_init['mae']:.2f}, R2: {metrics_init['r2']:.4f}")
        # # Initialize validator
        val_config = {
            'model_storage':  CONFIG['model_storage'],
            'random_state':   CONFIG['random_state'],
            'test_size':      CONFIG['test_size'],
            'cv_folds':       CONFIG['cv_folds'],
            'n_splits':       CONFIG.get('n_splits', 5),
            'scoring':        'neg_mean_absolute_error',
            'target_column':  CONFIG['target_column']
        }
        validator = ModelValidator(val_config)
        
        # Initial validation
        val_metrics = validate_model(validator, data_loader.get_data())
        
        # Save initial artifacts
        save_artifacts(model, preprocessor, CONFIG, validator,val_metrics)

        with open('dataloader.pkl', 'wb') as f:
            pickle.dump(data_loader, f)

    except Exception as e:
        logger.error(f"Initial pipeline failed: {e}")
        sys.exit(1)

def inference(data_path):
    """Run model inference on new data"""
    try:
        with open('dataloader.pkl', 'rb') as f:
            data_loader = pickle.load(f)
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('processor.pkl', 'rb') as f:
            preprocessor = pickle.load(f)
        # model, preprocessor = load_components()
        data = data_loader.get_data(freeze = True).copy()
        data = data[data['zero_balance_code'] == 1.0].copy() 
        # model, preprocessor, X_test, y_test = train_initial_model(data_loader)
        X_processed= preprocessor.transform(data)
        if CONFIG['target_column'] in data.columns:
            y = data[CONFIG['target_column']].values
            preds = model.predict(X_processed)
            metrics_init = model.evaluate(X_processed, y)
            logger.info(f"Inference MAE: {metrics_init['mae']:.2f}, R2: {metrics_init['r2']:.4f}")
        else:
            preds = model.predict(X_processed)
            logger.info("Inference completed. No target column for evaluation.")

        with open('dataloader.pkl', 'wb') as f:
            pickle.dump(data_loader, f)
        return preds


    except Exception as e:
        logger.error(f"Inference failed: {e}")
        raise



def update(batch_number=None):
    """Update model with a specific or next batch"""
    try:
        print("Updating model...")
        with open('dataloader.pkl', 'rb') as f:
            data_loader = pickle.load(f)
        # with open('model.pkl', 'rb') as f:
        #     model = pickle.load(f)
        # with open('processor.pkl', 'rb') as f:
        #     preprocessor = pickle.load(f)

        # batches = load_batches(data_loader)
        # model, preprocessor = load_components()
        model = CreditModel(CONFIG)
        model.load_model()
        preprocessor = CreditDataPreprocessor(CONFIG)
        preprocessor.load("preprocessors")
        temp = data_loader.get_data()
        metrics = update_model(temp, model, preprocessor)
        print('Reached metrics')
        # Update artifacts
        # logger.info(f"Initial train MAE: {metrics_init['mae']:.2f}, R2: {metrics_init['r2']:.4f}")
        # # Initialize validator
        val_config = {
            'model_storage':  CONFIG['model_storage'],
            'random_state':   CONFIG['random_state'],
            'test_size':      CONFIG['test_size'],
            'cv_folds':       CONFIG['cv_folds'],
            'n_splits':       CONFIG.get('n_splits', 5),
            'scoring':        'neg_mean_absolute_error',
            'target_column':  CONFIG['target_column']
        }
        validator = ModelValidator(val_config)
        
        # Initial validation
        val_metrics = validate_model(validator, data_loader.get_data())
        
        save_artifacts(model, preprocessor, CONFIG, validator,val_metrics)
        with open('model.pkl', 'wb') as f:
            pickle.dump(model, f)
        with open('processor.pkl', 'wb') as f:
            pickle.dump(preprocessor, f)
        with open('dataloader.pkl', 'wb') as f:
            pickle.dump(data_loader, f)
        # Save initial artifacts
        
        return metrics
    except Exception as e:
        logger.error(f"Update failed: {e}")
        raise

def summary():
    """Show model summary"""
    try:
        artifacts = joblib.load("model_artifacts/validation_artifacts.joblib")
        print("\nModel Summary:")
        print(f"Latest MAE: {artifacts['metrics']['holdout']['mae']:.2f}")
        print(f"Latest R2: {artifacts['metrics']['holdout']['r2']:.4f}")
        print("\nValidation Metrics:")
        for k, v in artifacts['metrics'].items():
            print(f"{k.upper()}: MAE={v['mae']:.2f}, R2={v['r2']:.4f}")
    except Exception as e:
        logger.error(f"Failed to generate summary: {e}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ML Pipeline Manager")
    subparsers = parser.add_subparsers(dest='command')
    #create_model
    subparsers.add_parser('create_model', help='Create initial model')
    # Inference command
    infer_parser = subparsers.add_parser('inference', help='Run model inference')
    infer_parser.add_argument('--data', type=str, required=False, help='Path to input data')
    
    # Update command
    update_parser = subparsers.add_parser('update', help='Update model with a batch')
    update_parser.add_argument('--batch', type=int, required=False, help='Specific batch number to use')
    
    # Summary command
    subparsers.add_parser('summary', help='Show model summary')
    
    args = parser.parse_args()
    
    try:
        if args.command == 'inference':
            inference(args.data)
        elif args.command == 'create_model':
            create_model()
        elif args.command == 'update':
            update(args.batch)
        elif args.command == 'summary':
            summary()
        else:
            parser.print_help()
    except Exception as e:
        logger.error(f"Command failed: {e}")
        sys.exit(1)

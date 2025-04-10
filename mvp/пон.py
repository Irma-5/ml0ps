import argparse
import json
import joblib
import pandas as pd
from pathlib import Path
import logging
from typing import Dict, Any
from xgboost import XGBRegressor

# Инициализация логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PipelineManager:
    def __init__(self, artifacts_dir: str = "model_artifacts"):
        self.artifacts_dir = Path(artifacts_dir)
        self.model = None
        self.preprocessor = None
        self.config = None
        self.metrics = None
        
    def load_artifacts(self):
        """Загрузка артефактов модели"""
        try:
            self.model = XGBRegressor()
            self.model.load_model(self.artifacts_dir / "model.xgb")
            self.preprocessor = joblib.load(self.artifacts_dir / "preprocessor.joblib")
            with open(self.artifacts_dir / "config.json") as f:
                self.config = json.load(f)
            validation_data = joblib.load(self.artifacts_dir / "validation_artifacts.joblib")
            self.metrics = validation_data['metrics']
            logger.info("Artifacts loaded successfully")
        except Exception as e:
            logger.error(f"Error loading artifacts: {str(e)}")
            raise

    def inference(self, data_path: str) -> pd.DataFrame:
        """Выполнение предсказаний"""
        try:
            data = pd.read_csv(data_path)
            processed_data = self.preprocessor.transform(data)
            predictions = self.model.predict(processed_data)
            data['predicted_time'] = predictions
            return data
        except Exception as e:
            logger.error(f"Inference failed: {str(e)}")
            raise

    def update(self, new_data_path: str):
        """Обновление модели новыми данными"""
        try:
            new_data = pd.read_csv(new_data_path)
            X, y = self.preprocessor.transform(new_data), new_data[self.config['target_column']].values
            self.model.fit(X, y, xgb_model=self.model.get_booster())
            self.model.save_model(self.artifacts_dir / "model.xgb")
            logger.info("Model updated successfully")
        except Exception as e:
            logger.error(f"Model update failed: {str(e)}")
            raise

    def get_summary(self) -> Dict[str, Any]:
        """Получение информации о модели"""
        return {
            "model_type": type(self.model).__name__,
            "features_used": self.config.get('n_features'),
            "metrics": self.metrics,
            "config": self.config
        }

def main():
    parser = argparse.ArgumentParser(description="Credit Model Pipeline Manager")
    subparsers = parser.add_subparsers(dest='command')

    # Inference command
    infer_parser = subparsers.add_parser('inference', help='Make predictions')
    infer_parser.add_argument('--data', type=str, required=True, help='Path to input data')

    # Update command
    update_parser = subparsers.add_parser('update', help='Update model with new data')
    update_parser.add_argument('--data', type=str, required=True, help='Path to new training data')

    # Summary command
    subparsers.add_parser('summary', help='Get model summary')

    args = parser.parse_args()
    manager = PipelineManager()
    
    try:
        manager.load_artifacts()
        if args.command == 'inference':
            result = manager.inference(args.data)
            output_path = f"predictions_{datetime.now().strftime('%Y%m%d%H%M%S')}.csv"
            result.to_csv(output_path, index=False)
            logger.info(f"Predictions saved to {output_path}")
            
        elif args.command == 'update':
            manager.update(args.data)
            
        elif args.command == 'summary':
            summary = manager.get_summary()
            print("\nModel Summary:")
            print(f"Model type: {summary['model_type']}")
            print(f"Features used: {summary['features_used']}")
            print("\nMetrics:")
            print(f"Initial MAE: {summary['metrics']['initial_metrics']['mae']:.2f}")
            print(f"Updated MAE: {summary['metrics']['updated_metrics']['mae']:.2f}")
            
    except Exception as e:
        logger.error(f"Command failed: {str(e)}")
        exit(1)

if __name__ == "__main__":
    main()

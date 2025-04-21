import argparse
import json
import joblib
import pandas as pd
from pathlib import Path
import logging
from typing import Dict, Any
from xgboost import XGBRegressor
import re
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
    
    @staticmethod
    def update_config(param: str, value: str):
        """Обновление параметров в DataLoader/config.py с полной заменой значения"""
        config_path = Path("DataLoader/config.py")
        try:
            with open(config_path, 'r') as f:
                content = f.read()

            # Шаблон для поиска параметра в формате "param": value
            pattern = re.compile(
                rf'("{param}"\s*:\s*)([^,\n]+)(,?\s*#.*?)?',
                re.DOTALL
            )

            match = pattern.search(content)
            if not match:
                raise ValueError(f"Параметр {param} не найден в config.py")

            # Обработка значений по типам
            if param in ['year_to_split', 'num_batch']:
                try:
                    new_value = int(value)
                except ValueError:
                    raise ValueError(f"{param} должен быть целым числом")
                replacement = f'{new_value}'
            else:
                # Сохраняем raw string для path
                if '\\' in value and not value.startswith('r"'):
                    replacement = f'r"{value}"'
                else:
                    replacement = f'"{value}"'

            # Замена только значения параметра
            updated_content = content[:match.start(2)] + replacement + content[match.end(2):]

            with open(config_path, 'w') as f:
                f.write(updated_content)

            logger.info(f"Параметр {param} обновлен: {match.group(2)} → {replacement}")

        except Exception as e:
            logger.error(f"Ошибка обновления конфига: {e}")
            raise
def main():
    parser = argparse.ArgumentParser(description="Credit Model Pipeline Manager")
    subparsers = parser.add_subparsers(dest='command')


    # Inference command
    infer_parser = subparsers.add_parser('inference', help='Make predictions')
    infer_parser.add_argument('--data', type=str, required=False, help='Path to input data')

    # Update command
    update_parser = subparsers.add_parser('update', help='Update model with new data')
    update_parser.add_argument('--data', type=str, required=False, help='Path to new training data')

    # Summary command
    subparsers.add_parser('summary', help='Get model summary')

    #Config command
    config_parser = subparsers.add_parser('setconfig', help='Обновление параметров конфигурации')
    config_parser.add_argument('--param', 
                               type=str, 
                               required=True, 
                               choices=['path', 'year_to_split', 'num_batch'],
                               help='Название параметра для обновления')
    config_parser.add_argument('--value',
                              type=str,
                              required=True,
                              help='Новое значение параметра')
    
    args = parser.parse_args()
    manager = PipelineManager()
    
    try:
        if args.command == 'setconfig':
            manager.update_config(args.param, args.value)
        else:
            manager.load_artifacts()

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

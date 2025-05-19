# config.py
import json
with open('config.json', 'r') as f:
    CONFIG = json.load(f)
DATA_LOADER_PARAMS = {
    "path": r"DataLoader",  # Путь к модулю данных
    "year_to_split":CONFIG["year_to_split"],  # Год, по которому разделяем данные
    "num_batch": CONFIG["num_batch"],  # Количество кварталов для обработки
}

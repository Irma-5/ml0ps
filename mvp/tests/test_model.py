import unittest
from model import CreditModel
from preprocessing import CreditDataPreprocessor
from main_пон import CONFIG

class TestModel(unittest.TestCase):
    def test_model_loading(self):
        model = CreditModel(CONFIG)
        model.load_model()
        self.assertIsNotNone(model)

    def test_preprocessor(self):
        preprocessor = CreditDataPreprocessor(CONFIG)
        preprocessor.load("preprocessors")
        self.assertIsNotNone(preprocessor)

if __name__ == "__main__":
    unittest.main()
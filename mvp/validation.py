import json
import logging
from pathlib import Path
from datetime import datetime
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from preprocessing import CreditDataPreprocessor
from model import CreditModel
from lime.lime_tabular import LimeTabularExplainer
import shap
import matplotlib.pyplot as plt
import os
from scipy.stats import ks_2samp

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ModelValidator:
    def __init__(self, config):
        self.config = config
        self.preprocessor = CreditDataPreprocessor(config)
        self.model = CreditModel(config)
        self.metrics = {}

    def load_data(self, data_path):
        df = pd.read_csv(data_path)
        return df

    def validate(self, df, method='holdout'):
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
        metrics_dir = Path(self.config['model_storage']) / 'metrics'
        metrics_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        metrics_path = metrics_dir / f"metrics_{timestamp}.json"
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics, f, indent=4)
        logger.info(f"Metrics saved to {metrics_path}")
        return metrics_path


def explain_with_shap(model_obj, X, feature_names, save_dir="models/shap"):
    os.makedirs(save_dir, exist_ok=True)
    explainer = shap.TreeExplainer(model_obj.model)
    shap_vals = explainer.shap_values(X)
    shap.summary_plot(shap_vals, X, feature_names=feature_names, show=False)
    plt.savefig(f"{save_dir}/summary.png", dpi=200)
    plt.close()
    shap.plots.waterfall(shap.Explanation(values=shap_vals[0],
                                            base_values=explainer.expected_value,
                                            data=X[0],
                                            feature_names=feature_names), show=False)
    plt.savefig(f"{save_dir}/waterfall_0.png", dpi=200)
    plt.close()

def explain_with_lime(X_train, X_test, predict_fn, feature_names, save_file="models/lime.html"):
    explainer = LimeTabularExplainer(training_data=X_train,
                                        feature_names=feature_names,
                                        mode='regression')
    exp = explainer.explain_instance(data_row=X_test[0],
                                        predict_fn=predict_fn,
                                        num_features=10)
    exp.save_to_file(save_file)

def detect_drift(baseline, new_X, feature_names, alpha=0.05):
        drifted = []
        for i, name in enumerate(feature_names):
            _, p = ks_2samp(baseline[:, i], new_X[:, i])
            if p < alpha:
                drifted.append(name)
        if drifted:
            logger.warning(f"Drift detected in features: {drifted}")
        else:
            logger.info("No drift detected")
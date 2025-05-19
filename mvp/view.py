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


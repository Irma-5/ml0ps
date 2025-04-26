import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import pickle
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class CreditDataPreprocessor:
    def __init__(self, config: dict):
        self.config = config
        self.preprocessor: ColumnTransformer = None
        self.feature_names: np.ndarray = None
        self.target_column = config['target_column']
        self.numeric_features = [
            'DTI_ratio', 'CLTV', 'LTV', 'orig_UPB',
            'orig_interest_rate', 'orig_loan_term', 'MI_%', 'units_numb'
        ]
        self.categorical_features = [
            'occupancy_status', 'first_time_homebuyer_flag', 'loan_purpose',
            'property_val_method', 'PPM_flag', 'channel', 'amortization_type',
            'property_type', 'property_state', 'int_only_flag', 'MI_cancel_flag', 'cens'
        ]
        self.date_features = [
            'first_payment_date', 'maturity_date', 'period'
        ]

    def _build_pipeline(self):
        safe_imputer = SimpleImputer(strategy='constant', fill_value=0)
        numeric_transformer = Pipeline([
            ('imputer', safe_imputer),
            ('scaler', StandardScaler())
        ])
        categorical_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='UNKNOWN')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        date_transformer = Pipeline([
            ('date_parser', self.DateParser()),
            ('imputer', safe_imputer)
        ])

        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self.numeric_features),
                ('cat', categorical_transformer, self.categorical_features),
                ('date', date_transformer, self.date_features)
            ],
            remainder='drop'
        )

    class DateParser(BaseEstimator, TransformerMixin):
        def fit(self, X, y=None):
            return self

        def transform(self, X):
            date_df = pd.DataFrame(index=X.index)
            for col in X.columns:
                dt = pd.to_datetime(X[col], errors='coerce')
                date_df[f'{col}_year']  = dt.dt.year.fillna(0).astype(int)
                date_df[f'{col}_month'] = dt.dt.month.fillna(0).astype(int)
                date_df[f'{col}_day']   = dt.dt.day.fillna(0).astype(int)
                date_df[f'{col}_doy']   = dt.dt.dayofyear.fillna(0).astype(int)
            return date_df

    def fit_transform(self, df: pd.DataFrame):
        self._build_pipeline()

        df = df[df['zero_balance_code'] == 1.0].copy()
        y = df[self.target_column].values
        X = df.drop(columns=[
            'Unnamed: 0', 'id_loan', 'postal_code', 'id_loan_preharp', self.target_column
        ], errors='ignore')

        X_processed = self.preprocessor.fit_transform(X)

        num_names = self.numeric_features

        ohe = self.preprocessor.named_transformers_['cat'].named_steps['onehot']
        cat_names = ohe.get_feature_names_out(self.categorical_features)

        date_parser = self.preprocessor.named_transformers_['date'].named_steps['date_parser']
        date_df = date_parser.transform(X[self.date_features])
        date_names = date_df.columns.values

        self.feature_names = np.concatenate([num_names, cat_names, date_names])

        logger.info(f"Processed data shape: {X_processed.shape}")
        return X_processed, y

    def transform(self, df: pd.DataFrame):
        if self.preprocessor is None:
            raise ValueError("Preprocessor not fitted yet!")

        df = df[df['zero_balance_code'] == 1.0].copy()
        X = df.drop(columns=[
            'Unnamed: 0', 'id_loan', 'postal_code', 'id_loan_preharp', self.target_column
        ], errors='ignore')

        return self.preprocessor.transform(X)

    def save(self, path: str):
        Path(path).mkdir(parents=True, exist_ok=True)
        with open(Path(path) / 'preprocessor.pkl', 'wb') as f:
            pickle.dump({
                'preprocessor': self.preprocessor,
                'feature_names': self.feature_names,
                'target_column': self.target_column
            }, f)
        logger.info(f"Preprocessor saved to {path}/preprocessor.pkl")

    def load(self, path: str):
        with open(Path(path) / 'preprocessor.pkl', 'rb') as f:
            data = pickle.load(f)
        self.preprocessor = data['preprocessor']
        self.feature_names = data['feature_names']
        self.target_column = data.get('target_column', self.target_column)
        logger.info(f"Preprocessor loaded from {path}/preprocessor.pkl")

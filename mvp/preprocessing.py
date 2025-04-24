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
    def __init__(self, config):
        self.config = config
        self.preprocessor = None
        self.feature_names = []

        self.target_column = config['target_column']
    def _build_pipeline(self):
        numeric_features = [
            'DTI_ratio', 'CLTV', 'LTV', 'orig_UPB',
            'orig_interest_rate', 'orig_loan_term', 'MI_%', 'units_numb'
        ]
        
        categorical_features = [
            'occupancy_status', 'first_time_homebuyer_flag', 'loan_purpose',
            'property_val_method', 'PPM_flag', 'channel', 'amortization_type',
            'property_type', 'property_state', 'int_only_flag', 'MI_cancel_flag', 'cens'
        ]
        
        date_features = [
            'first_payment_date', 'maturity_date', 'period'
        ]


        safe_imputer = SimpleImputer(
            strategy='constant', 
            fill_value=0,  # Для числовых и временных признаков
            missing_values=np.nan
        )

        numeric_transformer = Pipeline(steps=[
            ('imputer', safe_imputer),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='UNKNOWN')),
            ('onehot', OneHotEncoder(handle_unknown='error', sparse_output=False))
        ])

        date_transformer = Pipeline(steps=[
            ('date_parser', self.DateParser()),
            ('imputer', safe_imputer)
        ])

        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features),
                ('date', date_transformer, date_features)
            ],
            remainder='drop'
        )

    class DateParser(BaseEstimator, TransformerMixin):
        def fit(self, X, y=None):
            return self
            
        def transform(self, X):
            date_df = pd.DataFrame()
            for col in X.columns:
                dt_series = pd.to_datetime(X[col], errors='coerce')
                date_df[f'{col}_year'] = dt_series.dt.year
                date_df[f'{col}_month'] = dt_series.dt.month
                date_df[f'{col}_day'] = dt_series.dt.day
                date_df[f'{col}_doy'] = dt_series.dt.dayofyear
            return date_df

    def fit_transform(self, df):
        """Обучение и преобразование данных"""
        self._build_pipeline()


        df = df[df['zero_balance_code'] == 1.0]
        y = df[self.target_column].values
        X = df.drop(columns=[
            'Unnamed: 0', 'id_loan', 'postal_code', 
            'id_loan_preharp', self.target_column
        ], errors='ignore')

        X_processed = self.preprocessor.fit_transform(X)
        
        logger.info(f"Processed data shape: {X_processed.shape}")
        return X_processed, y
    

    def transform(self, df):
        """Преобразование новых данных"""
        if self.preprocessor is None:
            raise ValueError("Preprocessor not fitted yet!")
        

        df = df.drop(columns=[
            'Unnamed: 0', 'id_loan', 'postal_code', 
            'id_loan_preharp', self.target_column
        ], errors='ignore')
        df = df[df['zero_balance_code'] == 1.0]


        return self.preprocessor.transform(df)


    def save(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)
        with open(Path(path) / 'preprocessor.pkl', 'wb') as f:
            pickle.dump({
                'preprocessor': self.preprocessor,
                'feature_names': self.feature_names,
                'target_column': self.target_column
            }, f)


    def load(self, path):
        with open(Path(path) / 'preprocessor.pkl', 'rb') as f:
            data = pickle.load(f)
            self.preprocessor = data['preprocessor']
            self.feature_names = data['feature_names']
            self.target_column = data.get('target_column', 'time')

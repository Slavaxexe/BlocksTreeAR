"""Keep categorical columns native for CatBoost, LightGBM and XGBoost."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class NativeCategories(TransformerMixin, BaseEstimator):
    def __init__(self, backend='catboost', numeric_imputer='median'):
        self.backend = backend
        self.numeric_imputer = numeric_imputer

    def fit(self, frame, y=None):
        if self.backend not in {'catboost', 'lightgbm', 'xgboost'}:
            raise ValueError('native_backend must be catboost, lightgbm or xgboost')
        self.feature_names_in_ = np.asarray(frame.columns, dtype=object)
        self.numeric_columns_ = list(frame.select_dtypes(include='number'))
        self.categorical_columns_ = [column for column in frame if column not in self.numeric_columns_]
        self.categories_ = {column: sorted(frame[column].astype('string').dropna().unique())
                            for column in self.categorical_columns_}
        self.fill_ = {column: (0. if self.numeric_imputer == 'constant' else
                      getattr(pd.to_numeric(frame[column], errors='coerce'), self.numeric_imputer)())
                      for column in self.numeric_columns_}
        return self

    def transform(self, frame):
        result = frame.loc[:, self.feature_names_in_].copy()
        for column in self.numeric_columns_:
            result[column] = pd.to_numeric(result[column], errors='coerce').replace([np.inf, -np.inf], np.nan).fillna(self.fill_[column])
        for column in self.categorical_columns_:
            values = result[column].astype('string')
            result[column] = (values.fillna('__missing__').astype(object) if self.backend == 'catboost'
                              else pd.Categorical(values, categories=self.categories_[column]))
        return result

    def estimator_parameters(self, estimator, parameters):
        if estimator.split('.')[0] != self.backend:
            raise ValueError('Native categorical preprocessing requires a matching boosting library')
        parameters = dict(parameters)
        if self.backend == 'catboost':
            parameters.setdefault('cat_features', self.categorical_columns_)
        elif self.backend == 'xgboost':
            parameters.setdefault('enable_categorical', True)
            parameters.setdefault('tree_method', 'hist')
        return parameters

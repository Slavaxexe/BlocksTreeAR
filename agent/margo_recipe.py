"""Replay an unmodified margo feature recipe in the correct fitting partition."""
import hashlib
import importlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


def register_margo(preparation):
    root = Path(preparation['source_root']).resolve()
    expected = preparation['source_hashes']
    if not expected:
        raise ValueError('Margo source snapshot is missing')
    for relative, digest in expected.items():
        path = (root/relative).resolve()
        if not path.is_relative_to(root/'data_preparation') or not path.is_file():
            raise ValueError('Invalid or missing margo source file: '+relative)
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError('Margo source changed since the handoff: '+relative)
    cached = sys.modules.get('data_preparation')
    if cached and not Path(cached.__file__).resolve().is_relative_to(root):
        raise ValueError('A different data_preparation package is already imported')
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    sys.dont_write_bytecode = True
    return importlib.import_module('data_preparation.transformer'), importlib.import_module('data_preparation.schemas')


def training_source_paths(config, root):
    paths = {role: root/value for role, value in config['files'].items() if value}
    if config.get('outer_split'):
        paths['outer_split'] = root/config['outer_split']
    if config.get('preparation'):
        paths.update(preparation_recipe=root/config['preparation']['recipe'],
                     preparation_split=root/config['preparation']['split'])
    return paths


class MargoRecipeTransformer(TransformerMixin, BaseEstimator):
    """Cloneable sklearn step; fit never sees outer or inner validation rows."""
    def __init__(self, preparation, target, id_columns=()):
        self.preparation, self.target, self.id_columns = preparation, target, id_columns

    def fit(self, x, y=None):
        module, schemas = register_margo(self.preparation)
        plan = schemas.FeaturePlanPayload.model_validate_json(Path(self.preparation['recipe']).read_text(encoding='utf-8'))
        self.transformer_ = module.DataTransformer(plan, self.target, list(self.id_columns))
        self.transformer_.fit(x)
        self.feature_names_ = list(self.transformer_.transform(x).columns)
        self.fit_rows_ = len(x)
        if not self.feature_names_:
            raise ValueError('Margo recipe produced no usable model features')
        return self

    def transform(self, x):
        result = self.transformer_.transform(x, is_test=True)
        if len(result) != len(x) or not result.index.equals(x.index):
            raise ValueError('Margo recipe changed row count/order')
        result = result.reindex(columns=self.feature_names_, fill_value=0)
        numeric = result.select_dtypes(include=np.number).columns
        result.loc[:, numeric] = result.loc[:, numeric].replace([np.inf, -np.inf], np.nan)
        return result


class PreparedAutoMLRecipe:
    """Selected AutoML estimator + freshly fitted external feature recipe."""
    def __init__(self, preparation, target, id_columns, family, estimator):
        self.preparation, self.target, self.id_columns = preparation, target, id_columns
        self.family, self.estimator = family, estimator

    def fit(self, x, y, deadline, output):
        self.features_ = MargoRecipeTransformer(self.preparation, self.target, self.id_columns).fit(x)
        transformed = self.features_.transform(x)
        if self.family == 'autogluon':
            self.estimator.fit(transformed, y, deadline, output)
        else:
            self.estimator.fit(transformed, y)
        return self

    def predict(self, x):
        return self.estimator.predict(self.features_.transform(x))

"""One registry links executable templates to the exact sources shown to Qwen."""
import hashlib
import inspect
import re

from .training_template_sklearn import (ridge, huber_linear, knn, elasticnet, sgd, random_forest, extra_trees,
                                       hist_gradient_boosting, mlp)
from .training_template_boosting import lightgbm, catboost, xgboost
from .training_template_automl import flaml, autogluon


TEMPLATES = {
    'ridge': (ridge, 'scikit-learn', {'alpha': 1., 'solver': 'lsqr'}),
    'huber_linear': (huber_linear, 'scikit-learn', {'epsilon': 1.35, 'alpha': .0001, 'max_iter': 300}),
    'knn': (knn, 'scikit-learn', {'n_neighbors': 5, 'weights': 'distance'}),
    'elasticnet': (elasticnet, 'scikit-learn', {'alpha': .01, 'l1_ratio': .5, 'max_iter': 2000}),
    'sgd': (sgd, 'scikit-learn', {'loss': 'squared_error', 'penalty': 'elasticnet',
                                'alpha': .0001, 'learning_rate': 'invscaling', 'eta0': .01}),
    'random_forest': (random_forest, 'scikit-learn', {'n_estimators': 150, 'min_samples_leaf': 2}),
    'extra_trees': (extra_trees, 'scikit-learn', {'n_estimators': 150, 'min_samples_leaf': 2}),
    'hist_gradient_boosting': (hist_gradient_boosting, 'scikit-learn',
                               {'max_iter': 200, 'learning_rate': .1, 'l2_regularization': 1.}),
    'lightgbm': (lightgbm, 'lightgbm', {'n_estimators': 300, 'learning_rate': .05, 'num_leaves': 31,
                                      'reg_lambda': 1., 'subsample': .8, 'subsample_freq': 1}),
    'catboost': (catboost, 'catboost', {'iterations': 300, 'depth': 6, 'learning_rate': .05, 'l2_leaf_reg': 3.}),
    'xgboost': (xgboost, 'xgboost', {'n_estimators': 300, 'max_depth': 6, 'learning_rate': .05, 'reg_lambda': 1.}),
    'mlp': (mlp, 'scikit-learn', {'hidden_layer_sizes': [64, 32], 'solver': 'adam',
                                'alpha': .001, 'learning_rate_init': .001, 'max_iter': 300}),
    'flaml': (flaml, 'flaml', {'estimator_list': ['lgbm', 'extra_tree'], 'max_iter': 30}),
    'autogluon': (autogluon, 'autogluon.tabular',
                  {'hyperparameters': {'GBM': {}, 'RF': {'n_estimators': 150}}, 'presets': 'medium_quality'}),
}


def plain_definition(family, seed):
    if family not in TEMPLATES or family in {'flaml', 'autogluon'}:
        raise ValueError(f'Unsupported regression model family: {family}')
    factory, _, _ = TEMPLATES[family]
    return factory(seed)


def template_context(selected_models, requirements_text, installed):
    """Only selected, allowlisted and installed libraries contribute Python sources.

    Reading a template never imports an optional ML library. Imports happen only
    when the worker actually instantiates that model.
    """
    allowed = {}
    for line in requirements_text.splitlines():
        match = re.match(r'^\s*([A-Za-z0-9_.-]+)(?:\[[^\]]+\])?==([^\s;#]+)', line)
        if match:
            allowed[match[1].lower()] = match[2]
    result = []
    for family in dict.fromkeys(model['family'] for model in selected_models):
        factory, distribution, example = TEMPLATES[family]
        item = {'family': family, 'distribution': distribution,
                'required_version': allowed.get(distribution.lower()),
                'installed_version': installed.get(distribution)}
        item['available'] = item['required_version'] is not None and item['installed_version'] is not None
        if item['available']:
            source = inspect.getsource(factory)
            item.update(source_file=inspect.getsourcefile(factory), python_source=source,
                        source_sha256=hashlib.sha256(source.encode('utf-8')).hexdigest(),
                        example_parameters=example)
        else:
            item['unavailable_reason'] = 'Library must be listed in requirements.txt and installed; do not install it at runtime'
        result.append(item)
    return {'adaptation': 'Return JSON parameter overrides for selected_models; never rewrite these Python files. '
                          'Examples are starting points, not mandatory values or a closed search space. '
                          'All supported estimator parameters remain available after validation. '
                          'Upstream native loss parameters and target_transform are locked; tune only other settings. '
                          'Shared backend owns preprocessing inside folds, fit, predict, primary scoring, '
                          'deadlines and all-row final refit. AutoML templates define adapter settings.',
            'models': result}

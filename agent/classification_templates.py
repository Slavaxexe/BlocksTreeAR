"""Classification equivalents of the executable regression template registry."""
import hashlib
import inspect
import re

from . import training_template_classification as factories


TEMPLATES = {
    name: (getattr(factories, name), distribution, example)
    for name, distribution, example in [
        ('logistic_regression', 'scikit-learn', {'C': 1., 'max_iter': 1000}),
        ('sgd', 'scikit-learn', {'loss': 'log_loss', 'alpha': .0001}),
        ('random_forest', 'scikit-learn', {'n_estimators': 200, 'min_samples_leaf': 2}),
        ('extra_trees', 'scikit-learn', {'n_estimators': 200, 'min_samples_leaf': 2}),
        ('hist_gradient_boosting', 'scikit-learn', {'max_iter': 200, 'l2_regularization': 1.}),
        ('lightgbm', 'lightgbm', {'n_estimators': 300, 'learning_rate': .05, 'reg_lambda': 1.}),
        ('catboost', 'catboost', {'iterations': 300, 'depth': 6, 'l2_leaf_reg': 3.}),
        ('xgboost', 'xgboost', {'n_estimators': 300, 'max_depth': 6, 'reg_lambda': 1.}),
        ('mlp', 'scikit-learn', {'hidden_layer_sizes': [64, 32], 'solver': 'adam', 'alpha': .001}),
        ('knn', 'scikit-learn', {'n_neighbors': 5, 'weights': 'distance'}),
        ('flaml', 'flaml', {'estimator_list': ['lgbm', 'extra_tree'], 'max_iter': 20}),
        ('autogluon', 'autogluon.tabular', {'hyperparameters': {'GBM': {}, 'RF': {'n_estimators': 150}}}),
    ]
}


def plain_definition(family, seed):
    if family not in TEMPLATES or family in {'flaml', 'autogluon'}:
        raise ValueError(f'Unsupported classifier family: {family}')
    return TEMPLATES[family][0](seed)


def template_context(selected_models, requirements_text, installed):
    allowed = {}
    for line in requirements_text.splitlines():
        match = re.match(r'^\s*([A-Za-z0-9_.-]+)(?:\[[^\]]+\])?==([^\s;#]+)', line)
        if match:
            allowed[match[1].lower()] = match[2]
    result = []
    for family in dict.fromkeys(model['family'] for model in selected_models):
        factory, distribution, example = TEMPLATES[family]
        item = dict(family=family, distribution=distribution, required_version=allowed.get(distribution.lower()),
                    installed_version=installed.get(distribution))
        item['available'] = item['required_version'] is not None and item['installed_version'] is not None
        if item['available']:
            source = inspect.getsource(factory)
            item.update(source_file=inspect.getsourcefile(factory), python_source=source,
                        source_sha256=hashlib.sha256(source.encode('utf-8')).hexdigest(), example_parameters=example)
        else:
            item['unavailable_reason'] = 'Library must be listed in requirements.txt and installed; never install at runtime'
        result.append(item)
    return {'adaptation': 'Classification: return JSON parameter overrides, preserve locked native loss and task metric. '
            'The backend fits preprocessing inside each fold, owns label encoding and class-aligned predict_proba, '
            'scores the fixed outer holdout and refits the selected configuration on every labelled row. '
            'SGD must use loss=log_loss or modified_huber when probability output is needed. '
            'XGBoost class_weight is a backend policy (balanced or encoded-class dictionary) converted into '
            'sample_weight independently inside each training fit and final all-row refit. '
            'HistGradientBoosting and MLP have no native n_jobs parameter; the backend owns two CPU threads. '
            'Inner CV may stop after measured folds to reserve time for the common outer-fit validation. '
            'Do not apply regression losses or target transformations to classifiers.', 'models': result}

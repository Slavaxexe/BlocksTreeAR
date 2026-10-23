"""Engine-specific model identifiers, independent of the installed AutoML imports."""
import re


MODELS = {
    'flaml': ['xgboost', 'xgb_limitdepth', 'rf', 'lgbm', 'lgbm_spark', 'rf_spark', 'lrl1', 'lrl2',
              'catboost', 'extra_tree', 'kneighbor', 'transformer', 'transformer_ms', 'histgb', 'svc',
              'sgd', 'nb_spark', 'enet', 'lassolars', 'glr_spark', 'lr_spark', 'svc_spark', 'gbt_spark', 'aft_spark'],
    'autogluon': ['RF', 'XT', 'KNN', 'GBM', 'CAT', 'XGB', 'NN_TORCH', 'LR', 'FASTAI', 'AG_TEXT_NN',
                 'AG_IMAGE_NN', 'AG_AUTOMM', 'FT_TRANSFORMER', 'TABPFN', 'TABPFNMIX', 'FASTTEXT',
                 'ENS_WEIGHTED', 'SIMPLE_ENS_WEIGHTED', 'IM_RULEFIT', 'IM_GREEDYTREE', 'IM_FIGS',
                 'IM_HSTREE', 'IM_BOOSTEDRULES', 'DUMMY'],
}

ALIASES = {
    'flaml': {'lightgbm': 'lgbm', 'gbm': 'lgbm', 'cat': 'catboost', 'xgb': 'xgboost',
              'randomforest': 'rf', 'extratrees': 'extra_tree', 'xt': 'extra_tree',
              'knn': 'kneighbor', 'histgradientboosting': 'histgb', 'elasticnet': 'enet'},
    'autogluon': {'lightgbm': 'GBM', 'lgbm': 'GBM', 'catboost': 'CAT', 'xgboost': 'XGB',
                  'randomforest': 'RF', 'extratrees': 'XT', 'extratree': 'XT',
                  'kneighbor': 'KNN', 'kneighbors': 'KNN', 'logisticregression': 'LR'},
}

CAPABILITIES = {
    engine: {'model_ids': names, 'notes': [
        'Use engine model IDs in models, model_parameters and search_space, not Python class names.',
        'Optional model families still require their installed dependencies; pretrained downloads are unavailable.',
        'is_unbalance and scale_pos_weight are binary-classification settings, not generic multiclass balancing.',
        'CV stays within fit; the shared outer split and metric are fixed.',
    ]} for engine, names in MODELS.items()
}


def model_id(engine, name):
    token = re.sub(r'[^a-z0-9]', '', name.rsplit('.', 1)[-1].lower())
    token = re.sub(r'(classifier|regressor)$', '', token)
    known = {re.sub(r'[^a-z0-9]', '', value.lower()): value for value in MODELS[engine]}
    selected = ALIASES[engine].get(token) or known.get(token)
    if selected is None:
        raise ValueError(f'Unknown {engine} model {name!r}; model IDs: {MODELS[engine]}')
    return selected


def normalize_models(value):
    if not isinstance(value, dict) or value.get('engine') not in MODELS:
        return value
    value = dict(value)
    engine = value['engine']
    if isinstance(value.get('models'), list) and all(isinstance(name, str) for name in value['models']):
        value['models'] = list(dict.fromkeys(model_id(engine, name) for name in value['models']))
    for field in ('model_parameters', 'search_space'):
        if field not in value or not isinstance(value[field], dict):
            continue
        normalized = {}
        for name, settings in value[field].items():
            name = model_id(engine, name)
            if name in normalized:
                raise ValueError(f'{field} repeats model {name} under different names')
            normalized[name] = settings
        value[field] = normalized
    return value

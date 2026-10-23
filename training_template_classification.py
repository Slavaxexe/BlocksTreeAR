"""Classifier factories using only libraries in the organizer requirements."""


def logistic_regression(seed):
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression, dict(C=1., max_iter=1000, solver='lbfgs', random_state=seed, n_jobs=2)


def sgd(seed):
    from sklearn.linear_model import SGDClassifier
    return SGDClassifier, dict(loss='log_loss', alpha=.0001, max_iter=1500, tol=.001, random_state=seed)


def random_forest(seed):
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier, dict(n_estimators=200, min_samples_leaf=2, n_jobs=2, random_state=seed)


def extra_trees(seed):
    from sklearn.ensemble import ExtraTreesClassifier
    return ExtraTreesClassifier, dict(n_estimators=200, min_samples_leaf=2, n_jobs=2, random_state=seed)


def hist_gradient_boosting(seed):
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier, dict(max_iter=200, learning_rate=.1, l2_regularization=1., random_state=seed)


def lightgbm(seed):
    from lightgbm import LGBMClassifier
    return LGBMClassifier, dict(n_estimators=300, learning_rate=.05, num_leaves=31,
                               reg_lambda=1., verbosity=-1, n_jobs=2, random_state=seed)


def catboost(seed):
    from catboost import CatBoostClassifier
    return CatBoostClassifier, dict(iterations=300, depth=6, learning_rate=.05, l2_leaf_reg=3.,
                                    verbose=False, thread_count=2, task_type='CPU', allow_writing_files=False, random_seed=seed)


def xgboost(seed):
    from xgboost import XGBClassifier
    return XGBClassifier, dict(n_estimators=300, max_depth=6, learning_rate=.05, reg_lambda=1.,
                               tree_method='hist', device='cpu', n_jobs=2, random_state=seed)


def mlp(seed):
    from sklearn.neural_network import MLPClassifier
    return MLPClassifier, dict(hidden_layer_sizes=(64, 32), solver='adam', alpha=.001,
                               learning_rate_init=.001, max_iter=300, random_state=seed)


def knn(seed):
    from sklearn.neighbors import KNeighborsClassifier
    return KNeighborsClassifier, dict(n_neighbors=5, weights='distance', n_jobs=2)


def flaml(parameters, *, metric, time_budget, log_file):
    """Search classifiers on fit-only inner holdout. Final fit reuses one estimator."""
    allowed = {'estimator_list', 'metric', 'early_stop', 'max_iter', 'starting_points'}
    unknown = sorted(set(parameters) - allowed)
    if unknown:
        raise ValueError(f'Unsupported FLAML parameters: {unknown}')
    if parameters.get('metric', metric) != metric:
        raise ValueError('FLAML metric cannot override the upstream primary metric')
    permitted = {'lgbm', 'catboost', 'xgboost', 'extra_tree', 'rf'}
    settings = dict(parameters)
    settings.setdefault('estimator_list', ['lgbm', 'extra_tree'])
    if not isinstance(settings['estimator_list'], list) or not settings['estimator_list'] or set(settings['estimator_list']) - permitted:
        raise ValueError('FLAML classifier backends permitted: lgbm, catboost, xgboost, extra_tree, rf')
    settings.update(task='classification', metric=metric, time_budget=max(1., time_budget * .65),
                    n_jobs=2, verbose=0, log_file_name=str(log_file), eval_method='holdout', retrain_full=False)
    return settings


def autogluon(parameters, *, metric):
    """Only required native classifier backends; no ensemble or runtime installs."""
    import copy
    unknown = sorted(set(parameters) - {'hyperparameters', 'presets'})
    if unknown:
        raise ValueError(f'Unsupported AutoGluon parameters: {unknown}')
    hyperparameters = copy.deepcopy(parameters.get('hyperparameters', {'GBM': {}, 'RF': {'n_estimators': 150}}))
    if not isinstance(hyperparameters, dict) or not hyperparameters or set(hyperparameters) - {'GBM', 'CAT', 'XGB', 'RF', 'XT'}:
        raise ValueError('AutoGluon classifier backends permitted: GBM, CAT, XGB, RF, XT')
    for name, configurations in hyperparameters.items():
        for configuration in configurations if isinstance(configurations, list) else [configurations]:
            if not isinstance(configuration, dict):
                raise ValueError('AutoGluon model configurations must be dictionaries')
            configuration[{'GBM': 'num_threads', 'CAT': 'thread_count', 'XGB': 'n_jobs', 'RF': 'n_jobs', 'XT': 'n_jobs'}[name]] = 2
    return dict(eval_metric=metric, hyperparameters=hyperparameters,
                presets=parameters.get('presets', 'medium_quality'))

"""Executable settings templates for FLAML and AutoGluon from requirements.txt.

The backend supplies fit-only internal folds, real data, deadline, resources,
and the upstream task metric. These are not estimator constructor parameters.
"""


def flaml(parameters, *, metric, time_budget, log_file):
    """FLAML AutoML: adapt estimator_list, max_iter, early_stop, starting_points.

    Supports backend-supplied group/time/random holdout or CV. Never change the
    upstream metric. Final refit reuses the selected estimator without new search.
    """
    allowed = {'estimator_list', 'metric', 'early_stop', 'max_iter', 'starting_points'}
    unknown = sorted(set(parameters) - allowed)
    if unknown:
        raise ValueError(f'Unsupported FLAML parameters: {unknown}; supported: {sorted(allowed)}')
    if 'metric' in parameters and parameters['metric'] != metric:
        raise ValueError('FLAML metric must match primary_metric')
    settings = dict(parameters)
    settings.update(task='regression', metric=metric, time_budget=max(1., time_budget * .75),
                    n_jobs=2, verbose=0, log_file_name=str(log_file))
    settings.setdefault('estimator_list', ['lgbm', 'catboost', 'xgboost', 'extra_tree'])
    return settings


def autogluon(parameters, *, metric):
    """TabularPredictor: adapt hyperparameters and presets; this adapter requires holdout.

    hyperparameters uses AG family names GBM/CAT/XGB/RF/XT, not sklearn class names.
    Bagging and weighted ensembles are disabled; final refit uses one saved base model.
    """
    import copy
    allowed = {'hyperparameters', 'presets'}
    unknown = sorted(set(parameters) - allowed)
    if unknown:
        raise ValueError(f'Unsupported AutoGluon parameters: {unknown}; supported: {sorted(allowed)}')
    metrics = {'rmse': 'root_mean_squared_error', 'mse': 'mean_squared_error',
               'mae': 'mean_absolute_error', 'r2': 'r2', 'mape': 'mean_absolute_percentage_error'}
    hyperparameters = copy.deepcopy(parameters.get('hyperparameters',
                                                   {'GBM': {}, 'CAT': {}, 'XGB': {}, 'RF': {}, 'XT': {}}))
    if not isinstance(hyperparameters, dict):
        raise ValueError('AutoGluon hyperparameters must be a dictionary of model families')
    for name, configurations in hyperparameters.items():
        for configuration in configurations if isinstance(configurations, list) else [configurations]:
            if not isinstance(configuration, dict):
                raise ValueError('AutoGluon model parameters must be dictionaries')
            key = {'RF': 'n_jobs', 'XT': 'n_jobs', 'GBM': 'num_threads',
                   'CAT': 'thread_count', 'XGB': 'n_jobs'}.get(name)
            if key:
                configuration[key] = 2
    return {'eval_metric': metrics[metric], 'hyperparameters': hyperparameters,
            'presets': parameters.get('presets', 'medium_quality')}

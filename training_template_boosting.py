"""Executable templates for the pinned LightGBM, CatBoost and XGBoost packages.

The backend owns categorical encoding, CPU limits, deadline callbacks and refit.
Use estimator constructor parameters, never fit-time callbacks in Qwen JSON.
"""


def lightgbm(seed):
    """Tune leaves/depth, min_child_samples, learning_rate, reg_alpha/reg_lambda.

    subsample below 1 requires subsample_freq > 0. Do not pass original
    categorical column names: backend preprocessing already encodes them.
    """
    from lightgbm import LGBMRegressor
    return LGBMRegressor, {'n_estimators': 300, 'verbosity': -1,
                           'random_state': seed, 'n_jobs': 2}


def catboost(seed):
    """Tune depth, learning_rate, iterations and l2_leaf_reg; thread_count, not n_jobs.

    Do not pass cat_features: this adapter receives encoded numeric features.
    """
    from catboost import CatBoostRegressor
    return CatBoostRegressor, {'iterations': 300, 'verbose': False, 'random_seed': seed,
                              'thread_count': 2, 'allow_writing_files': False, 'task_type': 'CPU'}


def xgboost(seed):
    """Tune depth, min_child_weight, subsample, colsample_bytree and regularization.

    This CPU adapter uses tree_method='hist'; GPU settings are unavailable.
    """
    from xgboost import XGBRegressor
    return XGBRegressor, {'n_estimators': 300, 'tree_method': 'hist',
                         'random_state': seed, 'n_jobs': 2, 'device': 'cpu'}

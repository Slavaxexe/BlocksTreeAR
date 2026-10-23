"""Executable regression templates for scikit-learn from requirements.txt.

Qwen adapts estimator parameters through JSON. Data loading, fold-local
preprocessing, fitting, scoring and full-data refit belong to the backend.
These are starting defaults, not a fixed hyperparameter search space.
"""


def ridge(seed):
    """Tune alpha; scale features. lsqr supports the backend's sparse matrix."""
    from sklearn.linear_model import Ridge
    return Ridge, {'alpha': 1., 'solver': 'lsqr'}


def huber_linear(seed):
    """Robust linear regression; tune epsilon, alpha and max_iter after scaling."""
    from sklearn.linear_model import HuberRegressor
    return HuberRegressor, {'epsilon': 1.35, 'alpha': .0001, 'max_iter': 300}


def knn(seed):
    """Distance-based regression; tune neighbors, weights and p on scaled features."""
    from sklearn.neighbors import KNeighborsRegressor
    return KNeighborsRegressor, {'n_neighbors': 5, 'weights': 'distance', 'n_jobs': 2}


def elasticnet(seed):
    """Tune alpha and l1_ratio; increase max_iter if convergence is insufficient."""
    from sklearn.linear_model import ElasticNet
    return ElasticNet, {'max_iter': 2000, 'random_state': seed}


def sgd(seed):
    """Tune loss, penalty, alpha, learning_rate, eta0 and averaging; scaling is essential."""
    from sklearn.linear_model import SGDRegressor
    return SGDRegressor, {'max_iter': 2000, 'random_state': seed}


def random_forest(seed):
    """Tune depth, min_samples_leaf, max_features and n_estimators; CPU count is fixed."""
    from sklearn.ensemble import RandomForestRegressor
    return RandomForestRegressor, {'n_estimators': 150, 'min_samples_leaf': 2,
                                   'random_state': seed, 'n_jobs': 2}


def extra_trees(seed):
    """Tune depth, min_samples_leaf, max_features and n_estimators; CPU count is fixed."""
    from sklearn.ensemble import ExtraTreesRegressor
    return ExtraTreesRegressor, {'n_estimators': 150, 'min_samples_leaf': 2,
                                'random_state': seed, 'n_jobs': 2}


def hist_gradient_boosting(seed):
    """Tune learning_rate, max_leaf_nodes and l2_regularization; backend uses dense features."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    return HistGradientBoostingRegressor, {'max_iter': 200, 'random_state': seed}


def mlp(seed):
    """Tune layers, alpha, learning_rate_init; momentum/Nesterov apply only to solver='sgd'."""
    from sklearn.neural_network import MLPRegressor
    return MLPRegressor, {'hidden_layer_sizes': (64, 32), 'max_iter': 300, 'random_state': seed}

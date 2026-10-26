"""Small shared helpers for weighted fits and averages across random seeds."""
import inspect
import time

import numpy as np
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor


def fit_model(model, features, targets, weights=None):
    if weights is None:
        return model.fit(features, targets)
    if isinstance(model, TransformedTargetRegressor):
        if 'sample_weight' not in inspect.signature(model.regressor.fit).parameters:
            raise ValueError('The selected target-transform estimator does not support sample_weight')
        return model.fit(features, targets, sample_weight=weights)
    if hasattr(model, 'steps'):
        name, final = model.steps[-1]
        if 'sample_weight' not in inspect.signature(final.fit).parameters:
            raise ValueError('The selected pipeline estimator does not support sample_weight')
        return model.fit(features, targets, **{name + '__sample_weight': weights})
    if 'sample_weight' not in inspect.signature(model.fit).parameters:
        raise ValueError('The selected estimator does not support sample_weight')
    return model.fit(features, targets, sample_weight=weights)


class SeedAverage:
    def __init__(self, models, classification):
        self.models, self.classification = models, classification
        if classification:
            self.classes_ = models[0].classes_

    @property
    def predict_proba(self):
        if not all(callable(getattr(model, 'predict_proba', None)) for model in self.models):
            raise AttributeError('Seed members do not expose probabilities')
        return self._probabilities

    def _probabilities(self, features):
        return np.mean([model.predict_proba(features) for model in self.models], axis=0)

    def predict(self, features):
        if not self.classification:
            return np.mean([model.predict(features) for model in self.models], axis=0)
        if callable(getattr(self, 'predict_proba', None)):
            return self.classes_[self.predict_proba(features).argmax(axis=1)]
        predictions = np.stack([np.asarray(model.predict(features)).reshape(-1) for model in self.models])
        votes = [np.unique(column, return_counts=True) for column in predictions.T]
        return np.array([labels[counts.argmax()] for labels, counts in votes])


def fit_seeds(model, features, targets, weights, seeds, classification, runtime):
    if not seeds:
        return fit_model(model, features, targets, weights)
    models = []
    for seed in dict.fromkeys(seeds):
        if time.monotonic() >= runtime.get('deadline', float('inf')):
            raise TimeoutError('Node budget exhausted before the next seed')
        candidate = clone(model)
        options = candidate.get_params(deep=False)
        key = 'random_seed' if type(candidate).__module__.startswith('catboost') else 'random_state'
        if key == 'random_seed' and 'random_state' in options:
            key = 'random_state'
        if key not in options and key != 'random_seed':
            raise ValueError('Seed averaging requires an estimator with random_state or random_seed')
        candidate.set_params(**{key: seed})
        models.append(fit_model(candidate, features, targets, weights))
    return SeedAverage(models, classification)

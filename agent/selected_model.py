"""Frozen AutoML choices: refit selected native models without repeating search."""
from pathlib import Path

import numpy as np


class SelectedModel:
    def __init__(self, engine, models, weights, transformer, labels, classes=None):
        self.engine, self.models, self.weights = engine, models, np.asarray(weights, dtype=float)
        self.transformer, self.labels = transformer, labels
        if classes is not None:
            self.classes_ = np.asarray(classes)

    def refit(self, frame, target, runtime, weights=None):
        from .automl import _remaining
        from .finalize import clone_estimator
        from .model_fitting import fit_model
        features = self.transformer.transform(frame) if self.transformer is not None else frame
        encoded = self.labels.transform(target) if self.labels is not None else target
        fitted = []
        for index, original in enumerate(self.models):
            remaining = _remaining(runtime) / (len(self.models) - index)
            if self.engine == 'flaml':
                model = clone_estimator(original)
                if type(model).__module__.startswith('flaml'):
                    model.fit(features, encoded, budget=remaining,
                              **({'sample_weight': weights} if weights is not None else {}))
                else:
                    model = fit_model(model, features, encoded, weights)
            else:
                model = original.convert_to_refit_full_template()
                model.set_contexts(str(Path(runtime['work_directory']) / f'selected_{index}'))
                model.fit(X=features, y=encoded, sample_weight=weights, time_limit=remaining,
                          num_cpus=int(runtime.get('num_cpus', 1)), num_gpus=0)
                model.save()
            fitted.append(model)
        return SelectedModel(self.engine, fitted, self.weights, self.transformer, self.labels,
                             getattr(self, 'classes_', None))

    def predict_proba(self, frame):
        features = self.transformer.transform(frame) if self.transformer is not None else frame
        values = []
        for model in self.models:
            probabilities = np.asarray(model.predict_proba(features))
            if probabilities.ndim == 1:
                probabilities = np.column_stack([1 - probabilities, probabilities])
            values.append(probabilities)
        return sum(weight * value for weight, value in zip(self.weights, values))

    def predict(self, frame):
        if hasattr(self, 'classes_'):
            return self.classes_[self.predict_proba(frame).argmax(axis=1)]
        features = self.transformer.transform(frame) if self.transformer is not None else frame
        return sum(weight * np.asarray(model.predict(features)) for weight, model in zip(self.weights, self.models))


def freeze(model, engine, classification):
    if engine == 'flaml':
        return SelectedModel(engine, [model.model], [1.], model.feature_transformer,
                             model.label_transformer, model.classes_ if classification else None)
    trainer = model._trainer
    from autogluon.core.models import WeightedEnsembleModel
    weights = {}

    def collect(name, weight):
        native = trainer.load_model(name)
        if isinstance(native, WeightedEnsembleModel):
            for source, part in native._get_model_weights().items():
                collect(source, weight * part)
        else:
            weights[name] = weights.get(name, 0.) + weight

    collect(model.model_best, 1.)
    return SelectedModel(engine, [trainer.load_model(name) for name in weights], list(weights.values()),
                         model._learner.feature_generator, model._learner.label_cleaner if classification else None,
                         model.class_labels if classification else None)


def refit_saved(state, plan, filename, defaults):
    import cloudpickle
    from .automl import _features
    from .features import create_features
    from .operations import _classification, _prediction, _requires_probabilities
    from .predictions import attach_probabilities
    from .supervised_features import row_weights
    frozen = cloudpickle.loads(Path(filename).read_bytes())
    if callable(getattr(frozen, 'refit_tables', None)):
        _, result = frozen.refit_tables(state, plan, defaults)
        return result
    preparation = state['preparation']
    targets, ids = preparation['data']['target_columns'], preparation['data'].get('id_columns', [])
    feature_targets = preparation.get('all_target_columns', targets)
    inherited = state['fit'].attrs.get('feature_recipes', [])
    recipes = [*inherited, *(feature.model_dump() for feature in plan.features)]
    created = [recipe['name'] for recipe in inherited]
    raw = [state[scope].drop(columns=created) for scope in ('fit', 'test')]
    fit, query = create_features(raw, recipes, forbidden=feature_targets, policy=preparation.get('validation', {}))
    columns = plan.columns or [column for column in fit if column not in set(feature_targets + ids)]
    columns = list(dict.fromkeys([*columns, *(feature.name for feature in plan.features)]))
    columns = [column for column in columns if column not in plan.drop_columns]
    x, query, _ = _features(fit[columns], query[columns], query[columns], feature_targets, ids)
    weights = row_weights(raw[0], plan.row_weights.model_dump()) if plan.row_weights else None
    model = frozen.refit(x, fit[targets[0]], preparation['runtime'], weights)
    probability = _classification(state, defaults) and _requires_probabilities(state, defaults)
    values = model.predict_proba(query) if probability else model.predict(query)
    classes = getattr(model, 'classes_', None)
    prediction = _prediction(values, state, defaults, labels=classes, probabilities=probability)
    if classes is not None:
        attach_probabilities(prediction, model.predict_proba(query), classes)
    return {'validation_predictions': prediction.copy(), 'test_predictions': prediction}

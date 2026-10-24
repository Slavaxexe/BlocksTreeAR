"""Fit-only calibration, thresholds and blending with a separate selection partition."""

from dataclasses import dataclass, field
import re
import time

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp
from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split


def _name(metric):
    value = re.sub(r'[^a-z0-9]', '', str(metric).lower())
    return {'rootmeansquarederror': 'rmse', 'meansquarederror': 'mse',
            'meanabsoluteerror': 'mae', 'r2score': 'r2', 'accuracyscore': 'accuracy',
            'f1macro': 'macrof1', 'auc': 'rocauc', 'crossentropy': 'logloss'}.get(value, value)


def _native(value):
    return value.item() if isinstance(value, np.generic) else value


def _values(values, classification, classes=None):
    array = np.asarray(values, dtype=float)
    if not np.isfinite(array).all():
        raise ValueError('Postprocessing predictions contain non-finite values')
    if not classification:
        if array.ndim == 2 and array.shape[1] == 1:
            array = array[:, 0]
        if array.ndim != 1:
            raise ValueError('Regression postprocessing expects one prediction per row')
        return array
    if array.ndim != 2 or classes is None or array.shape[1] != len(classes):
        raise ValueError('Classification postprocessing requires probabilities aligned with explicit classes')
    if (array < -1e-8).any() or (array > 1 + 1e-8).any() or (array.sum(axis=1) <= 0).any():
        raise ValueError('Classification postprocessing received invalid probabilities')
    array = np.clip(array, 1e-12, 1)
    return array / array.sum(axis=1, keepdims=True)


def _encoded(y, classes):
    mapping = {_native(label): index for index, label in enumerate(classes)}
    try:
        return np.asarray([mapping[_native(label)] for label in np.asarray(y).reshape(-1)], dtype=int)
    except KeyError as error:
        raise ValueError('Calibration target is absent from explicit probability class order') from error


def _positive(classes, positive_class):
    if len(classes) != 2:
        return None
    if positive_class is None:
        return 1
    matches = [i for i, label in enumerate(classes) if label == positive_class]
    if len(matches) != 1:
        raise ValueError('positive_class must identify exactly one supplied class')
    return matches[0]


def score_predictions(y, values, *, classification, metric_name, classes=None,
                      threshold=.5, positive_class=None):
    """Return the raw objective value, using supplied class order and no fitted state."""
    metric = _name(metric_name)
    array = _values(values, classification, classes)
    if len(y) != len(array) or not len(array):
        raise ValueError('Postprocessing score inputs have unequal or empty lengths')
    if classification:
        encoded = _encoded(y, classes)
        positive = _positive(classes, positive_class)
        if metric == 'logloss':
            return float(log_loss(encoded, array, labels=np.arange(len(classes))))
        if metric == 'rocauc':
            if len(classes) == 2:
                return float(roc_auc_score((encoded == positive).astype(int), array[:, positive]))
            return float(roc_auc_score(encoded, array, labels=np.arange(len(classes)), multi_class='ovr', average='macro'))
        guessed = (np.where(array[:, positive] >= threshold, positive, 1 - positive)
                   if len(classes) == 2 else array.argmax(axis=1))
        if metric == 'accuracy':
            return float(accuracy_score(encoded, guessed))
        if metric == 'macrof1':
            return float(f1_score(encoded, guessed, labels=np.arange(len(classes)), average='macro', zero_division=0))
        if metric == 'f1' and len(classes) == 2:
            return float(f1_score(encoded, guessed, pos_label=positive, zero_division=0))
        raise ValueError(f'Unsupported classification postprocessing metric: {metric_name}')
    actual = np.asarray(y, dtype=float).reshape(-1)
    if not np.isfinite(actual).all():
        raise ValueError('Regression calibration targets contain non-finite values')
    residual = array - actual
    if metric == 'rmse':
        return float(np.sqrt(np.square(residual).mean()))
    if metric == 'mse':
        return float(np.square(residual).mean())
    if metric == 'mae':
        return float(np.abs(residual).mean())
    if metric == 'mape':
        return float(np.mean(np.abs(residual) / np.maximum(np.abs(actual), np.finfo(float).eps)))
    if metric == 'r2':
        denominator = float(np.square(actual - actual.mean()).sum())
        if denominator <= 0:
            raise ValueError('R2 calibration selection requires a nonconstant target')
        return float(1 - np.square(residual).sum() / denominator)
    raise ValueError(f'Unsupported regression postprocessing metric: {metric_name}')


@dataclass
class PostprocessingRecipe:
    classification: bool
    classes: list | None = None
    blend_alpha: float = 1.0
    temperature: float = 1.0
    threshold: float = .5
    positive_class: object = None
    report: dict = field(default_factory=dict)

    def transform(self, tuned_values, baseline_values=None):
        values = _values(tuned_values, self.classification, self.classes)
        if self.blend_alpha < 1:
            if baseline_values is None:
                raise ValueError('Selected blend requires predictions from the separately trained baseline')
            baseline = _values(baseline_values, self.classification, self.classes)
            if baseline.shape != values.shape:
                raise ValueError('Baseline and tuned predictions have different shapes')
            values = self.blend_alpha * values + (1 - self.blend_alpha) * baseline
        if self.classification:
            logits = np.log(np.clip(values, 1e-12, 1)) / self.temperature
            values = np.exp(logits - logsumexp(logits, axis=1, keepdims=True))
        return values

    def predict_labels(self, probabilities):
        if not self.classification:
            raise ValueError('Regression postprocessing does not produce class labels')
        values = _values(probabilities, True, self.classes)
        positive = _positive(self.classes, self.positive_class)
        indexes = (np.where(values[:, positive] >= self.threshold, positive, 1 - positive)
                   if len(self.classes) == 2 else values.argmax(axis=1))
        return np.asarray(self.classes)[indexes]

    def parameters(self):
        return {'tuned_weight': float(self.blend_alpha), 'baseline_weight': float(1 - self.blend_alpha),
                'temperature': float(self.temperature), 'binary_threshold': float(self.threshold)
                if self.classification and len(self.classes) == 2 else None}


class _Deadline(Exception):
    pass


def fit_postprocessing(y_calibration, tuned_calibration, baseline_calibration=None, *, classification,
                       metric_name, classes=None, seed=42, deadline=None, positive_class=None):
    """Fit parameters on calibration A, choose a frozen recipe on disjoint B.

    Both supplied predictors must have been trained without ANY calibration rows.
    Selection scores are internal observations, not unbiased generalization estimates.
    No refit on these calibration rows is permitted after freezing the recipe.
    """
    classes = [_native(label) for label in classes] if classes is not None else None
    report = {'status': 'skipped', 'scope': 'Held-out calibration rows within fit only; no outer labels',
              'metric': _name(metric_name), 'calibration_rows': int(len(y_calibration)),
              'score_caveat': 'Partition B selects a recipe; its score is not an unbiased final estimate'}
    identity = PostprocessingRecipe(classification, classes, positive_class=positive_class, report=report)

    def has_time():
        return deadline is None or time.monotonic() < deadline - .02

    if not has_time():
        report['reason'] = 'No remaining postprocessing budget'
        return identity
    y = np.asarray(y_calibration).reshape(-1)
    tuned = _values(tuned_calibration, classification, classes)
    baseline = (_values(baseline_calibration, classification, classes)
                if baseline_calibration is not None else None)
    if len(y) != len(tuned) or (baseline is not None and baseline.shape != tuned.shape):
        raise ValueError('Calibration predictions do not align with targets or each other')
    if len(y) < 30:
        report['reason'] = 'Fewer than 30 independent calibration rows'
        return identity
    stratify = None
    if classification:
        encoded = _encoded(y, classes)
        counts = np.bincount(encoded, minlength=len(classes))
        if len(classes) < 2 or counts.min() < 2:
            report['reason'] = 'Every class needs at least two independent calibration rows'
            return identity
        stratify = encoded
    indexes = np.arange(len(y))
    try:
        a, b = train_test_split(indexes, test_size=.5, random_state=seed, stratify=stratify)
    except ValueError:
        report['reason'] = 'Cannot form independent calibration parameter-fit and selection partitions'
        return identity
    report.update(parameter_fit_rows=int(len(a)), selection_rows=int(len(b)),
                  positive_class=classes[_positive(classes, positive_class)]
                  if classification and len(classes) == 2 else None)
    minimize = _name(metric_name) in {'rmse', 'mse', 'mae', 'mape', 'logloss'}

    def score(recipe, positions):
        values = recipe.transform(tuned[positions], baseline[positions] if baseline is not None else None)
        value = score_predictions(y[positions], values, classification=classification,
                                  metric_name=metric_name, classes=classes,
                                  threshold=recipe.threshold, positive_class=positive_class)
        if not np.isfinite(value):
            raise ValueError('Calibration objective is non-finite')
        return value

    def utility(value):
        return -value if minimize else value

    def make(alpha=1., temperature=1., threshold=.5):
        return PostprocessingRecipe(classification, classes, float(alpha), float(temperature), float(threshold), positive_class)

    recipes = [identity]
    alpha_best = 1.
    try:
        best_a = utility(score(identity, a))
    except (ValueError, TypeError) as error:
        report['reason'] = str(error)[:350]
        return identity
    if baseline is not None:
        for alpha in (0., .25, .5, .75):
            if not has_time():
                break
            candidate = make(alpha)
            value = utility(score(candidate, a))
            if value > best_a + 1e-12:
                best_a, alpha_best = value, alpha
        if alpha_best != 1:
            recipes.append(make(alpha_best))
    report['blend_search'] = {'tuned_weight_grid': [0., .25, .5, .75, 1.] if baseline is not None else [1.],
                              'selected_on_partition_a': float(alpha_best)}
    if classification:
        threshold_metric = len(classes) == 2 and _name(metric_name) in {'accuracy', 'f1', 'macrof1'}
        for alpha in dict.fromkeys((1., alpha_best)):
            if not has_time():
                break
            base_recipe = make(alpha)
            probability_a = base_recipe.transform(tuned[a], baseline[a] if baseline is not None else None)
            logits = np.log(np.clip(probability_a, 1e-12, 1))
            truth_a = _encoded(y[a], classes)

            def temperature_loss(temperature):
                if not has_time():
                    raise _Deadline()
                calibrated = np.exp(logits / temperature - logsumexp(logits / temperature, axis=1, keepdims=True))
                return float(log_loss(truth_a, calibrated, labels=np.arange(len(classes))))

            try:
                fitted = minimize_scalar(temperature_loss, bounds=(.3, 3.), method='bounded',
                                         options={'maxiter': 20, 'xatol': .03})
                temperature = float(fitted.x) if fitted.success and np.isfinite(fitted.fun) else 1.
            except _Deadline:
                temperature = 1.
            if abs(temperature - 1) > 1e-6:
                recipes.append(make(alpha, temperature))
            if threshold_metric:
                for scale in dict.fromkeys((1., temperature)):
                    recipe = make(alpha, scale)
                    best_threshold, best_score = .5, utility(score(recipe, a))
                    for threshold in np.linspace(.05, .95, 37):
                        if not has_time():
                            break
                        recipe.threshold = float(threshold)
                        value = utility(score(recipe, a))
                        if value > best_score + 1e-12:
                            best_threshold, best_score = float(threshold), value
                    if abs(best_threshold - .5) > 1e-6:
                        recipes.append(make(alpha, scale, best_threshold))
    # All candidates above are fixed before partition B is inspected.
    if not has_time():
        report['reason'] = 'Budget exhausted before independent recipe selection'
        return identity
    selected = identity
    try:
        baseline_b = score(identity, b)
    except (ValueError, TypeError) as error:
        report['reason'] = str(error)[:350]
        return identity
    best_b = utility(baseline_b)
    candidates = []
    seen = set()
    for recipe in recipes:
        key = (recipe.blend_alpha, recipe.temperature, recipe.threshold)
        if key in seen:
            continue
        seen.add(key)
        value = score(recipe, b)
        candidates.append({**recipe.parameters(), 'selection_metric': float(value)})
        if utility(value) > best_b + 1e-9:
            best_b, selected = utility(value), recipe
    report.update(status='completed', baseline_selection_metric=float(baseline_b),
                  selected_selection_metric=float(score(selected, b)), selected=selected.parameters(),
                  candidates=candidates, parameter_fit_scope='Partition A only',
                  selection_scope='Partition B only; never used to optimize continuous parameters',
                  model_policy='Baseline and tuned models exclude all calibration rows and are not refitted afterwards')
    selected.report = report
    return selected

"""Combine saved predictions, preparing reusable held-out source predictions on demand."""
from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from .plans import EnsemblePlan
from .predictions import attach_probabilities, load_pair
from .storage import save
from .ensemble_data import subset, concatenate


def matrix(frame, classes):
    if classes is None:
        return frame.to_numpy(dtype=float)
    probability = frame.attrs.get('probabilities')
    if probability is None:
        values = frame.iloc[:, 0].to_numpy()
        return np.column_stack([values == label for label in classes]).astype(float)
    positions = {str(label): index for index, label in enumerate(probability['classes'])}
    return np.column_stack([probability['values'][:, positions[str(label)]] if str(label) in positions
                            else np.zeros(len(frame)) for label in classes])


def blend(pairs, state, defaults, weights, method):
    from .operations import _ensemble
    combined = {'preparation': state['preparation']}
    names = []
    classes = list(dict.fromkeys(label for pair in pairs
        for label in pair['validation'].attrs.get('probabilities', {}).get('classes', [])))
    for index, pair in enumerate(pairs):
        names.append([f'validation_{index}', f'test_{index}'])
        for scope in ('validation', 'test'):
            frame = pair[scope]
            if method == 'probabilities':
                probability = frame.attrs.get('probabilities')
                if probability is None:
                    raise ValueError('Probability blending needs saved probabilities for every source; use vote for labels')
                frame = pd.DataFrame(matrix(frame, np.asarray(classes)), columns=classes)
            combined[f'{scope}_{index}'] = frame
    for scope in ('validation', 'test'):
        if scope in state:
            combined[scope] = state[scope]
    result = _ensemble(combined, {'sources': names, 'method': method,
                                  **({'weights': weights} if weights else {})}, defaults)
    for frame in result.values():
        frame.attrs['meta_held_out'] = all(pair['validation'].attrs.get('meta_held_out', True) for pair in pairs)
    return result


def condition_mask(frame, conditions):
    result = np.ones(len(frame), dtype=bool)
    for condition in conditions:
        values = frame[condition.column]
        mask = (getattr(values, condition.operation)() if condition.operation in {'isna', 'notna'}
                else getattr(values, condition.operation)(condition.value))
        result &= mask.fillna(False).to_numpy(dtype=bool)
    return result


def conditional(pairs, state, defaults, plan, method):
    result = blend(pairs, state, defaults, plan.weights or None, method)
    covered = {scope: np.zeros(len(state[scope]), dtype=bool) for scope in ('validation', 'test')}
    for rule in plan.rules:
        rows = {scope: np.flatnonzero(condition_mask(state[scope], rule.conditions) & ~covered[scope])
                for scope in covered}
        if method == 'rank_average':
            # Ranks describe the complete query, even when a rule selects only some of its rows.
            prediction = blend(pairs, state, defaults, rule.weights, method)
            prediction = {key: subset(frame, rows[key.removesuffix('_predictions')])
                          for key, frame in prediction.items()}
        else:
            partial = [{scope: subset(pair[scope], rows[scope]) for scope in covered} for pair in pairs]
            prediction = blend(partial, state, defaults, rule.weights, method)
        for scope in covered:
            frame, update = result[scope + '_predictions'], prediction[scope + '_predictions']
            frame.iloc[rows[scope]] = update.to_numpy()
            if 'probabilities' in frame.attrs and 'probabilities' in update.attrs:
                classes = np.asarray(frame.attrs['probabilities']['classes'])
                frame.attrs['probabilities']['values'][rows[scope]] = matrix(update, classes)
            covered[scope][rows[scope]] = True
    return result


def combiner(plan, classification, runtime):
    from .operations import _estimator
    if plan.method == 'stacking':
        factory = plan.estimator or ('sklearn.linear_model.LogisticRegression' if classification else 'sklearn.linear_model.Ridge')
        parameters = plan.parameters or ({'max_iter': 500} if classification and plan.estimator is None else {})
    else:
        factory = plan.estimator or 'sklearn.tree.DecisionTreeClassifier'
        parameters = plan.parameters or ({'max_depth': 3, 'min_samples_leaf': 50} if plan.estimator is None else {})
    model = _estimator(factory, parameters, runtime)
    if not callable(getattr(model, 'fit', None)) or not callable(getattr(model, 'predict', None)):
        raise ValueError('Choose a sklearn-compatible combiner with fit and predict')
    if plan.method == 'gating' and not callable(getattr(model, 'predict_proba', None)):
        raise ValueError('A gate must classify source IDs and provide predict_proba')
    return model, factory


def learned(pairs, state, defaults, plan, classes):
    from .operations import _predict, _prediction, _preprocess
    if len(state['preparation']['data']['target_columns']) != 1:
        raise ValueError('Learned ensembles currently support one target')
    runtime = state['preparation']['runtime']
    # Only the combiner is fitted here; source models are never executed.
    model, factory = combiner(plan, classes is not None, runtime)
    target_name = state['preparation']['data']['target_columns'][0]
    meta = state['auxiliary']['ensemble_meta']
    encoder = LabelEncoder().fit(pd.concat([state['fit'][target_name], meta[target_name]])) if classes is not None else None
    target = encoder.transform(meta[target_name]) if encoder is not None else meta[target_name].to_numpy()
    features = _preprocess(state, {'feature_columns': plan.feature_columns}) if plan.method == 'gating' or plan.passthrough else None
    matrices = []
    for pair in pairs:
        if 'meta' not in pair or not pair['meta'].attrs.get('meta_held_out', False):
            raise ValueError('Learned ensembles need saved held-out meta predictions; no source retraining is performed')
        if len(pair['meta']) != len(meta):
            raise ValueError('Sources must share the same ensemble holdout')
        matrices.append(matrix(pair['meta'], classes))
    covered = np.ones(len(meta), dtype=bool)
    outer = {scope: [matrix(pair[scope], classes) for pair in pairs] for scope in ('validation', 'test')}
    if plan.method == 'stacking':
        train = np.column_stack(matrices)
        query = {scope: np.column_stack(outer[scope]) for scope in outer}
        if plan.passthrough:
            meta_features = features['transformer'].transform(meta[features['transformer'].feature_names_in_])
            train = np.column_stack([train, meta_features])
            query = {scope: np.column_stack([query[scope], features[scope + '_features']]) for scope in query}
        model.fit(train[covered], target[covered])
        result = _predict({**state, 'model': {'estimator': model, 'labels': encoder, 'classification': encoder is not None},
            **{scope + '_features': value for scope, value in query.items()}}, {}, defaults)
    else:
        if encoder is not None:
            losses = np.column_stack([-np.log(np.clip(value[np.arange(len(target)), target], 1e-12, 1)) for value in matrices])
        else:
            metric = ''.join(c for c in state['preparation']['metric']['name'].lower() if c.isalnum())
            errors = [value[:, 0] - target for value in matrices]
            losses = np.column_stack([np.abs(error) if metric in {'mae', 'meanabsoluteerror'} else error ** 2 for error in errors])
        best = losses.argmin(axis=1)
        meta_features = features['transformer'].transform(meta[features['transformer'].feature_names_in_])
        model.fit(meta_features, best)
        result = {}
        for scope in outer:
            weights = np.zeros((len(state[scope]), len(pairs)))
            weights[:, np.asarray(model.classes_, dtype=int)] = model.predict_proba(features[scope + '_features'])
            values = sum(weights[:, index, None] * value for index, value in enumerate(outer[scope]))
            if classes is not None:
                from .operations import _requires_probabilities
                probability = _requires_probabilities(state, defaults)
                prediction = _prediction(values if probability else classes[values.argmax(axis=1)],
                    state, defaults, labels=classes, probabilities=probability)
                attach_probabilities(prediction, values, classes)
            else:
                prediction = _prediction(values, state, defaults)
            result[scope + '_predictions'] = prediction
    import copy
    import cloudpickle
    from .storage import atomic_bytes
    transformer = copy.copy(features['transformer']) if features else None
    if transformer is not None:
        transformer.__dict__.pop('fit_source_', None)
    atomic_bytes(Path(runtime['work_directory']) / 'combiner.pkl',
                 cloudpickle.dumps({'model': model, 'labels': encoder, 'transformer': transformer,
                                    'plan': plan.model_dump()}))
    for frame in result.values():
        frame.attrs['meta_held_out'] = False
    save(Path(runtime['work_directory']) / 'ensemble_full_diagnostics.json', {
        'method': plan.method, 'source_nodes': plan.source_nodes, 'oof_rows': int(covered.sum()),
        'fit_rows': len(state['fit']), 'source_retraining': bool(runtime.get('prepared_meta_sources')),
        'prepared_meta_sources': runtime.get('prepared_meta_sources', []), 'meta_method': 'shared_holdout', 'estimator': factory,
        'meta_training_scope': 'Saved predictions on a shared inner holdout; outer labels unavailable',
        'gate_loss': ('per-row log loss' if encoder is not None else 'per-row absolute/squared error') if plan.method == 'gating' else None,
        'passthrough': plan.passthrough, 'feature_columns': features['transformer'].diagnostics_['selected_columns'] if features else []})
    return result


def run_saved_ensemble(state, value, defaults):
    from .operations import _classification, _requires_probabilities
    plan = value if isinstance(value, EnsemblePlan) else EnsemblePlan.model_validate(value)
    from .task_types import family
    if family(defaults.get('task_type')) == 'clustering':
        raise ValueError('Cluster IDs are not aligned between independently fitted models; select one clustering solution')
    if plan.method in {'stacking', 'gating'} and len(state['preparation']['data']['target_columns']) != 1:
        raise ValueError('Use mean/vote/conditional for multiple targets; learned stacking/gating requires one target')
    runtime = state['preparation']['runtime']
    root = Path(runtime['run_directory']).resolve()
    pairs = [load_pair(root, source, runtime['preparation_sha256'], defaults,
                      state['preparation']['metric']['name']) for source in plan.source_nodes]
    prepared_meta_sources = []
    need_meta = runtime.get('is_meta_fit') or plan.method in {'stacking', 'gating'} or plan.optimize_weights
    if need_meta and not runtime.get('is_smoke_check'):
        from .meta_predictions import ensure
        missing = sum('meta' not in pair for pair in pairs)
        for index, source in enumerate(plan.source_nodes):
            if 'meta' not in pairs[index]:
                remaining = max(0., runtime['deadline'] - time.monotonic())
                meta_deadline = time.monotonic() + remaining * .9 / missing
                ensure(root, source, runtime['preparation_sha256'], meta_deadline)
                missing -= 1
                prepared_meta_sources.append(source)
                pairs[index] = load_pair(root, source, runtime['preparation_sha256'], defaults,
                                         state['preparation']['metric']['name'])
    if prepared_meta_sources:
        save(Path(runtime['work_directory']) / 'meta_preparation.json', {'sources': prepared_meta_sources,
            'scope': 'Separate inner-train fit; cached meta predictions reused on later requests'})
    runtime['prepared_meta_sources'] = prepared_meta_sources
    if runtime.get('is_meta_fit'):
        pairs = [{'validation': pair['meta'], 'test': pair['meta']} for pair in pairs]
    for pair in pairs:
        if runtime.get('meta_rows'):
            if 'meta' not in pair:
                raise ValueError('Source has no saved ensemble holdout predictions; start a new run')
            valid = pair['validation']
            meta = pair['meta']
            if runtime.get('is_smoke_check'):
                valid = subset(valid, np.arange(runtime['outer_validation_rows']))
                meta = subset(meta, np.arange(runtime['meta_rows']))
            pair['validation'] = concatenate([valid, meta])
        for scope in ('validation', 'test'):
            rows = len(state[scope])
            if runtime.get('is_smoke_check'):
                pair[scope] = subset(pair[scope], np.arange(rows))
            elif len(pair[scope]) != rows:
                raise ValueError('Saved predictions do not match the current fixed row order/count')
    if plan.optimize_weights and not runtime.get('is_smoke_check'):
        from .weight_search import optimize
        weights = optimize(pairs, state, defaults, plan)
        plan = plan.model_copy(update={'weights': weights})
    classification = _classification(state, defaults)
    method = plan.combine or ('probabilities' if classification and all('probabilities' in pair['validation'].attrs for pair in pairs)
                              else 'vote' if classification and not _requires_probabilities(state, defaults) else 'mean')
    if plan.method in {'stacking', 'gating'}:
        if runtime.get('is_smoke_check'):
            combiner(plan, classification, runtime)
            return blend(pairs, state, defaults, plan.weights or None, method)
        target = state['preparation']['data']['target_columns'][0]
        classes = LabelEncoder().fit(pd.concat([state['fit'][target],
            state['auxiliary']['ensemble_meta'][target]])).classes_ if classification else None
        return learned(pairs, state, defaults, plan, classes)
    if plan.method == 'conditional':
        return conditional(pairs, state, defaults, plan, method)
    result = blend(pairs, state, defaults, plan.weights or None,
                   plan.method if plan.method in {'mean', 'vote', 'probabilities', 'rank_average'} else method)
    if plan.optimize_weights:
        for frame in result.values():
            frame.attrs['meta_held_out'] = False
    return result


def rank_average(pairs, state, defaults, weights=None):
    """Rank-normalized score averaging for ranking models across queries."""
    return blend(pairs, state, defaults, weights, method='rank_average')

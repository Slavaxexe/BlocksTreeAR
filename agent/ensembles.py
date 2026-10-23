"""Cached blending, conditional rules, and fit-only OOF stacking/gating."""
import io
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold, TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder

from .data import file_hash
from .plans import EnsemblePlan, ToolCall
from .predictions import attach_probabilities, load_pair, source_node
from .storage import atomic_bytes, fingerprint, save


def subset(frame, rows):
    result = frame.iloc[rows].reset_index(drop=True)
    if 'probabilities' in frame.attrs:
        probability = frame.attrs['probabilities']
        attach_probabilities(result, probability['values'][rows], probability['classes'])
    return result


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
    return _ensemble(combined, {'sources': names, 'method': method,
                               **({'weights': weights} if weights else {})}, defaults)


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


def folds(fit, target, classes, preparation, count, seed):
    policy = preparation.get('validation', {})
    strategy = str(policy.get('strategy', '')).lower()
    if policy.get('group_columns'):
        groups = pd.factorize(pd.MultiIndex.from_frame(fit[policy['group_columns']]))[0]
        return list(GroupKFold(count).split(fit, target, groups))
    if 'group' in strategy:
        raise ValueError('Group-aware OOF needs explicit group_columns')
    if any(word in strategy for word in ('time', 'temporal', 'chronological')):
        column = policy.get('split_column') or policy.get('time_column')
        if not column:
            raise ValueError('Time-aware OOF needs an explicit time/split column')
        periods = np.sort(fit[column].dropna().unique())
        return [(np.flatnonzero(fit[column].isin(periods[train])), np.flatnonzero(fit[column].isin(periods[valid])))
                for train, valid in TimeSeriesSplit(count).split(periods)]
    splitter = StratifiedKFold(count, shuffle=True, random_state=seed) if classes is not None else KFold(count, shuffle=True, random_state=seed)
    return list(splitter.split(fit, target))


def recipe(source, destination, defaults):
    """Compile immutable saved operations, freezing completed tuning where possible."""
    from .assembly import implementation_hash
    finish = json.loads((source / 'finish.json').read_text(encoding='utf-8'))
    if implementation_hash(source) != finish['code_sha256']:
        raise ValueError('Source recipe changed after its evaluation')
    manifest = json.loads((source / 'assembly.json').read_text(encoding='utf-8'))
    for block in manifest['blocks']:
        call = ToolCall.model_validate(block.get('operation'))
        if call.kind not in {'constant', 'features', 'preprocess', 'fit', 'predict', 'automl', 'ensemble'}:
            raise ValueError('OOF sources must be trainable recipes, not cached prediction combinations')
        parameters = dict(call.parameters)
        if call.kind == 'fit':
            key = fingerprint(call.outputs.get('model', 'model'))[:12]
            path = source / f'fit_{key}_full_diagnostics.json'
            facts = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
            parameters['parameters'] = {**parameters.get('parameters', {}), **facts.get('best_parameters', {})}
            parameters['search_space'] = {}
        elif call.kind == 'automl':
            path = source / f"{parameters['engine']}_full_diagnostics.json"
            facts = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
            parameters.update(search_space={}, num_trials=1)
            if parameters['engine'] == 'flaml' and facts.get('best_estimator'):
                name = facts['best_estimator']
                parameters.update(models=[name], model_parameters={name: facts['best_config']})
        call = ToolCall.model_validate({**call.model_dump(), 'parameters': parameters})
        code = ('from agent.operations import run_operation\n\ndef run(state):\n'
                f'    return run_operation(state, {call.model_dump()!r}, {defaults!r})\n')
        path = destination / 'modules' / (block['name'] + '.py')
        atomic_bytes(path, code.encode('utf-8'))
        block.update(operation=call.model_dump(), sha256=file_hash(path))
    atomic_bytes(destination / 'start.json', (source / 'start.json').read_bytes())
    save(destination / 'assembly.json', manifest)


def oof(root, source_id, state, defaults, plan, classes, deadline):
    from .assembly import execute
    runtime = state['preparation']['runtime']
    source, finish = source_node(root, source_id, runtime['preparation_sha256'])
    fit = state['fit'].reset_index(drop=True)
    target_name = state['preparation']['data']['target_columns'][0]
    target = fit[target_name].to_numpy()
    splits = folds(fit, target, classes, state['preparation'], plan.folds, plan.seed)
    key = fingerprint({'preparation': runtime['preparation_sha256'], 'source': finish['code_sha256'],
        'diagnostics': [file_hash(path) for path in sorted(source.glob('*_full_diagnostics.json'))],
        'folds': plan.folds, 'seed': plan.seed, 'classes': classes.tolist() if classes is not None else None})
    cache = root / 'prediction_cache' / (key + '.npz')
    values = np.zeros((len(fit), len(classes) if classes is not None else 1))
    covered = np.zeros(len(fit), dtype=bool)
    if cache.exists():
        with np.load(cache, allow_pickle=False) as saved:
            values, covered = saved['values'], saved['covered']
        if len(values) != len(fit):
            raise ValueError('OOF cache row count differs from fit')
    for number, (train, valid) in enumerate(splits):
        if covered[valid].all():
            continue
        remaining = deadline - time.monotonic()
        if remaining <= 1:
            raise TimeoutError('OOF budget exhausted; existing source submissions remain available')
        directory = Path(runtime['work_directory']) / 'oof' / source_id / f'fold_{number}'
        recipe(source, directory, defaults)
        preparation = {**state['preparation'], 'runtime': {**runtime, 'work_directory': str(directory),
            'deadline': time.monotonic() + remaining / (len(splits) - number), 'is_smoke_check': False}}
        prediction, _ = execute(directory, fit.iloc[train].reset_index(drop=True),
            fit.iloc[valid].drop(columns=target_name).reset_index(drop=True), state['test'], preparation,
            state.get('auxiliary'))
        values[valid] = matrix(prediction, classes)
        covered[valid] = True
        buffer = io.BytesIO()
        np.savez_compressed(buffer, values=values, covered=covered)
        atomic_bytes(cache, buffer.getvalue())
    return values, covered


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


def learned(pairs, state, defaults, plan, classes, root):
    from .operations import _predict, _prediction, _preprocess
    if len(state['preparation']['data']['target_columns']) != 1:
        raise ValueError('Learned ensembles currently support one target')
    runtime = state['preparation']['runtime']
    # Reject invalid constructor settings before paying for any OOF training.
    model, factory = combiner(plan, classes is not None, runtime)
    target_name = state['preparation']['data']['target_columns'][0]
    encoder = LabelEncoder().fit(state['fit'][target_name]) if classes is not None else None
    target = encoder.transform(state['fit'][target_name]) if encoder is not None else state['fit'][target_name].to_numpy()
    features = _preprocess(state, {'feature_columns': plan.feature_columns}) if plan.method == 'gating' or plan.passthrough else None
    matrices, masks = [], []
    started = time.monotonic()
    deadline = started + (runtime['deadline'] - started) * 0.85
    for index, source in enumerate(plan.source_nodes):
        limit = time.monotonic() + max(0., deadline - time.monotonic()) / (len(pairs) - index)
        values, covered = oof(root, source, state, defaults, plan, classes, limit)
        matrices.append(values)
        masks.append(covered)
    covered = np.logical_and.reduce(masks)
    if not covered.any():
        raise ValueError('No shared held-out fit rows for the learned ensemble')
    outer = {scope: [matrix(pair[scope], classes) for pair in pairs] for scope in ('validation', 'test')}
    if plan.method == 'stacking':
        train = np.column_stack(matrices)
        query = {scope: np.column_stack(outer[scope]) for scope in outer}
        if plan.passthrough:
            train = np.column_stack([train, features['fit_features']])
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
        model.fit(features['fit_features'][covered], best[covered])
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
    save(Path(runtime['work_directory']) / 'ensemble_full_diagnostics.json', {
        'method': plan.method, 'source_nodes': plan.source_nodes, 'oof_rows': int(covered.sum()),
        'fit_rows': len(target), 'folds': plan.folds, 'estimator': factory,
        'meta_training_scope': 'Held-out/OOF predictions inside fit only; outer labels unavailable',
        'gate_loss': ('per-row log loss' if encoder is not None else 'per-row absolute/squared error') if plan.method == 'gating' else None,
        'passthrough': plan.passthrough, 'feature_columns': features['transformer'].diagnostics_['selected_columns'] if features else []})
    return result


def run_saved_ensemble(state, value, defaults):
    from .operations import _classification, _requires_probabilities
    plan = value if isinstance(value, EnsemblePlan) else EnsemblePlan.model_validate(value)
    runtime = state['preparation']['runtime']
    root = Path(runtime['run_directory']).resolve()
    pairs = [load_pair(root, source, runtime['preparation_sha256'], defaults,
                      state['preparation']['metric']['name']) for source in plan.source_nodes]
    for pair in pairs:
        for scope in ('validation', 'test'):
            rows = len(state[scope])
            if runtime.get('is_smoke_check'):
                pair[scope] = subset(pair[scope], np.arange(rows))
            elif len(pair[scope]) != rows:
                raise ValueError('Saved predictions do not match the current fixed row order/count')
    classification = _classification(state, defaults)
    method = plan.combine or ('probabilities' if classification and all('probabilities' in pair['validation'].attrs for pair in pairs)
                              else 'vote' if classification and not _requires_probabilities(state, defaults) else 'mean')
    if plan.method in {'stacking', 'gating'}:
        if runtime.get('is_smoke_check'):
            combiner(plan, classification, runtime)
            return blend(pairs, state, defaults, plan.weights or None, method)
        classes = LabelEncoder().fit(state['fit'][state['preparation']['data']['target_columns'][0]]).classes_ if classification else None
        return learned(pairs, state, defaults, plan, classes, root)
    if plan.method == 'conditional':
        return conditional(pairs, state, defaults, plan, method)
    return blend(pairs, state, defaults, plan.weights or None,
                 plan.method if plan.method in {'mean', 'vote', 'probabilities'} else method)

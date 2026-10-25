"""Refit the selected solution on all labelled rows, with search decisions frozen."""
import argparse
import json
from pathlib import Path
import time

import cloudpickle
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .contracts import Preparation
from .context import preparation_view
from .data import file_hash, input_hashes, module
from .features import create_features
from .plans import AutoMLPlan, EnsemblePlan, ToolCall
from .predictions import attach_probabilities, source_node
from .storage import atomic_bytes, fingerprint, load, save


def clone_estimator(model):
    """Freeze selected configurations, preserving estimator-owned preprocessing."""
    from sklearn.base import clone
    from sklearn.ensemble import StackingClassifier, StackingRegressor
    if type(model).__module__.startswith('flaml.automl.time_series'):
        return clone(model)
    if isinstance(model, (StackingClassifier, StackingRegressor)):
        result = clone(model)
        result.set_params(estimators=[(name, clone_estimator(fitted))
            for (name, _), fitted in zip(model.estimators, model.estimators_)],
            final_estimator=clone_estimator(model.final_estimator_), n_jobs=1)
        return result
    if hasattr(model, 'estimator') and model.__class__.__module__.startswith(('flaml', 'agent.automl')):
        # FLAML wrappers encode categories for XGBoost/forests and prepare other inputs.
        # Use the library class so a local timed subclass cannot retain an expired deadline.
        factory = next(factory for factory in type(model).__mro__
                       if factory.__module__.startswith('flaml.automl.model'))
        return factory(**model.get_params())
    # FLAML's timed CatBoost subclass captures an old deadline; use the native class for refit.
    for factory in type(model).__mro__:
        if factory.__module__.startswith('catboost') and factory.__name__ in {'CatBoostClassifier', 'CatBoostRegressor'}:
            return factory(**model.get_params())
    return clone(model)


class FinalRefit:
    def __init__(self, root, seconds):
        self.root = root
        self.preparation = load(root / 'preparation.json', Preparation)
        self.reader = module(root / 'prepare.py', self.preparation.code_sha256)
        self.config = self.preparation.model_dump(mode='json')
        self.source = Path(self.preparation.task_directory)
        if input_hashes(self.source, self.preparation.data.files) != self.preparation.source_hashes:
            raise ValueError('Refit inputs changed after preparation')
        self.tables = self.reader.load_data(self.source, self.config)
        self.train, self.test = self.tables['train'], self.tables['test']
        extraction = json.loads((root / 'extraction.json').read_text(encoding='utf-8'))
        self.defaults = {key: extraction.get('problem' if key != 'prediction_type' else 'output', {}).get(key)
                         for key in ('task_type', 'class_labels', 'prediction_type')}
        self.deadline = time.monotonic() + seconds
        self.memo = {}
        self.metadata = preparation_view(self.preparation)
        self.metadata['runtime'] = {'deadline': self.deadline, 'time_budget_seconds': seconds,
            'run_directory': str(root / 'final'), 'preparation_sha256': file_hash(root / 'preparation.json'),
            'source_run_directory': str(root),
            'is_final_refit': True, 'is_smoke_check': False, 'num_cpus': 1, 'num_gpus': 0,
            'meta_rows': 0, 'fit_row_positions': np.arange(len(self.train))}

    def run_node(self, node_id):
        from .assembly import implementation_hash
        from .operations import run_operation
        if node_id in self.memo:
            return self.memo[node_id]
        if time.monotonic() >= self.deadline:
            raise TimeoutError('Final refit budget exhausted')
        source, finish = source_node(self.root, node_id, self.metadata['runtime']['preparation_sha256'])
        if implementation_hash(source) != finish['code_sha256']:
            raise ValueError('Refit source changed after evaluation')
        directory = self.root / 'final' / node_id
        directory.mkdir(parents=True, exist_ok=True)
        preparation = {**self.metadata, 'runtime': {**self.metadata['runtime'], 'work_directory': str(directory)}}
        state = {'fit': self.train, 'validation': self.test, 'test': self.test,
                 'preparation': preparation, 'auxiliary': {}}
        manifest = json.loads((source / 'assembly.json').read_text(encoding='utf-8'))
        for block in manifest['blocks']:
            call = ToolCall.model_validate(block['operation'])
            params = dict(call.parameters)
            if call.kind == 'fit':
                key = fingerprint(call.outputs.get('model', 'model'))[:12]
                facts = json.loads((source / f'fit_{key}_full_diagnostics.json').read_text(encoding='utf-8'))
                params['parameters'] = {**params.get('parameters', {}), **facts.get('best_parameters', {})}
                params['search_space'] = {}
                call = call.model_copy(update={'parameters': params})
            if call.kind == 'saved_ensemble':
                result = self.ensemble(source, EnsemblePlan.model_validate(params), state)
                state.update({call.outputs.get(key, key): value for key, value in result.items()})
            elif call.kind == 'automl':
                facts = json.loads((source / f"{params['engine']}_full_diagnostics.json").read_text(encoding='utf-8'))
                if facts.get('task_family') == 'multi_target':
                    mapped = {**state, **{key: state[value] for key, value in call.inputs.items()}}
                    result = self.multi_target(AutoMLPlan.model_validate(params), facts, mapped)
                    state.update({call.outputs.get(key, key): value for key, value in result.items()})
                    continue
                if facts.get('selected_model_file'):
                    from .selected_model import refit_saved
                    mapped = {**state, **{key: state[value] for key, value in call.inputs.items()}}
                    result = refit_saved(mapped, AutoMLPlan.model_validate(params), facts['selected_model_file'], self.defaults)
                    state.update({call.outputs.get(key, key): value for key, value in result.items()})
                elif params['engine'] == 'autogluon':
                    mapped = {**state, **{key: state[value] for key, value in call.inputs.items()}}
                    result = self.autogluon(AutoMLPlan.model_validate(params), facts, mapped)
                    state.update({call.outputs.get(key, key): value for key, value in result.items()})
                else:
                    mapped = {**state, **{key: state[value] for key, value in call.inputs.items()}}
                    result = self.flaml(AutoMLPlan.model_validate(params), facts, mapped)
                    state.update({call.outputs.get(key, key): value for key, value in result.items()})
            else:
                state.update(run_operation(state, call, self.defaults))
        result = {'validation': state['validation_predictions'], 'test': state['test_predictions']}
        self.memo[node_id] = result
        return result

    def multi_target(self, plan, facts, state):
        from .automl_tasks import head_state, join_heads, constant_head
        from .selected_model import refit_saved
        from .automl import _remaining
        results = []
        for index, entry in enumerate(facts['targets']):
            child, defaults = head_state(state, entry['target'], self.defaults, entry['selection_metric'])
            runtime = state['preparation']['runtime']
            directory = Path(runtime['work_directory']) / f'target_{index:03d}'
            directory.mkdir(parents=True, exist_ok=True)
            child['preparation']['runtime'] = {**runtime, 'work_directory': str(directory),
                'deadline': time.monotonic() + _remaining(runtime) / (len(facts['targets']) - index)}
            detail = entry['diagnostics']
            if detail.get('constant'):
                result = constant_head(child, defaults)
            elif detail.get('selected_model_file'):
                result = refit_saved(child, plan, detail['selected_model_file'], defaults)
            elif plan.engine == 'autogluon':
                result = self.autogluon(plan, detail, child, defaults)
            else:
                result = self.flaml(plan, detail, child, defaults)
            results.append(result)
        return join_heads(results, state, self.defaults)

    def flaml(self, plan, facts, state, defaults=None):
        from sklearn.base import clone
        from .automl import _features
        from .operations import _classification, _prediction, _requires_probabilities
        defaults = defaults or self.defaults
        snapshot = cloudpickle.loads(Path(facts['model_file']).read_bytes())
        core = clone_estimator(snapshot['model'])
        targets = state['preparation']['data']['target_columns']
        ids = state['preparation']['data'].get('id_columns', [])
        feature_targets = state['preparation'].get('all_target_columns', targets)
        fit, query, _ = create_features([state['fit'], state['test'], state['test']],
                                        plan.features, forbidden=feature_targets)
        columns = plan.columns or [column for column in fit if column not in set(feature_targets + ids)]
        columns = list(dict.fromkeys([*columns, *(feature.name for feature in plan.features)]))
        columns = [column for column in columns if column not in plan.drop_columns]
        fit, query, _ = _features(fit[columns], query[columns], query[columns], feature_targets, ids)
        transformer = snapshot['feature_transformer']
        fit, query = transformer.transform(fit), transformer.transform(query)
        labels = clone(snapshot['label_transformer']) if snapshot['label_transformer'] is not None else None
        target = state['fit'][targets[0]].to_numpy()
        encoded = labels.fit_transform(target) if labels is not None else target
        core.fit(fit, encoded)
        classification = _classification(state, defaults)
        probability = classification and _requires_probabilities(state, defaults)
        values = core.predict_proba(query) if probability else core.predict(query)
        if labels is not None and not probability:
            values = labels.inverse_transform(np.asarray(values, dtype=int))
        classes = labels.classes_ if labels is not None else getattr(core, 'classes_', None)
        frame = _prediction(values, state, defaults, labels=classes, probabilities=probability)
        if classification and callable(getattr(core, 'predict_proba', None)):
            attach_probabilities(frame, core.predict_proba(query), classes)
        destination = Path(state['preparation']['runtime']['work_directory']) / 'model.pkl'
        atomic_bytes(destination, cloudpickle.dumps({'model': core, 'feature_transformer': transformer,
                                                    'label_transformer': labels, 'columns': columns}))
        save(destination.with_suffix('.json'), {'training_rows': len(target), 'search': False})
        return {'validation_predictions': frame.copy(), 'test_predictions': frame}

    def autogluon(self, plan, facts, state, defaults=None):
        from autogluon.tabular import TabularPredictor
        from .automl import _features
        from .operations import _prediction, _requires_probabilities, _classification
        defaults = defaults or self.defaults
        model_directory = Path(facts['model_directory'])
        with np.load(model_directory.parent / 'training_rows.npz', allow_pickle=False) as saved:
            rows = saved['rows']
        inherited = state['fit'].attrs.get('feature_recipes', [])
        recipes = [*inherited, *(feature.model_dump() for feature in plan.features)]
        targets = state['preparation']['data']['target_columns']
        ids = state['preparation']['data'].get('id_columns', [])
        feature_targets = state['preparation'].get('all_target_columns', targets)
        train, test = state['fit'], state['test']
        base, full, query = create_features([train.iloc[rows], train, test], recipes, forbidden=feature_targets)
        excluded = set(feature_targets + ids)
        columns = plan.columns or [column for column in full if column not in excluded]
        columns = list(dict.fromkeys([*columns, *(feature.name for feature in plan.features)]))
        columns = [column for column in columns if column not in plan.drop_columns]
        _, full, query = _features(base[columns], full[columns], query[columns],
                                   feature_targets, ids)
        predictor = TabularPredictor.load(str(model_directory), verbosity=0)
        extra_rows = np.setdiff1d(np.arange(len(train)), rows)
        extra = full.iloc[extra_rows].copy()
        extra[predictor.label] = train.iloc[extra_rows][targets[0]].to_numpy()
        extra.index = np.arange(len(rows), len(rows) + len(extra))
        # Native refit reuses the selected models and frozen hyperparameters, without a new AutoML search.
        mapping = predictor.refit_full(model=facts['best_model'], set_best_to_refit_full=False,
            train_data_extra=extra, num_cpus=1, num_gpus=0, fit_strategy='sequential')
        model_name = mapping[facts['best_model']]
        probability = _requires_probabilities(state, defaults)
        if _classification(state, defaults):
            probabilities = predictor.predict_proba(query, model=model_name)
            values = probabilities.to_numpy() if probability else predictor.predict(query, model=model_name).to_numpy()
            frame = _prediction(values, state, defaults, labels=list(probabilities.columns), probabilities=probability)
            attach_probabilities(frame, probabilities.to_numpy(), list(probabilities.columns))
        else:
            frame = _prediction(predictor.predict(query, model=model_name).to_numpy(), state, defaults)
        save(Path(state['preparation']['runtime']['work_directory']) / 'autogluon_refit.json',
             {'model_directory': str(model_directory), 'model': model_name, 'rows': len(self.train)})
        return {'validation_predictions': frame.copy(), 'test_predictions': frame}

    def ensemble(self, source, plan, state):
        from .ensembles import blend, conditional, matrix
        from .operations import _predict, _prediction, _requires_probabilities
        pairs = [self.run_node(node_id) for node_id in plan.source_nodes]
        if plan.optimize_weights:
            weights = json.loads((source / 'weights_full_diagnostics.json').read_text(encoding='utf-8'))['weights']
            plan = plan.model_copy(update={'weights': weights, 'optimize_weights': False})
        from .operations import _classification
        classification = _classification(state, self.defaults)
        method = plan.combine or ('probabilities' if classification and all('probabilities' in pair['test'].attrs for pair in pairs)
                                  else 'vote' if classification else 'mean')
        if plan.method == 'conditional':
            return conditional(pairs, state, self.defaults, plan, method)
        if plan.method not in {'stacking', 'gating'}:
            return blend(pairs, state, self.defaults, plan.weights or None, plan.method)
        frozen = cloudpickle.loads((source / 'combiner.pkl').read_bytes())
        model, labels, transformer = frozen['model'], frozen['labels'], frozen['transformer']
        classes = labels.classes_ if labels is not None else None
        matrices = [matrix(pair['test'], classes) for pair in pairs]
        features = transformer.transform(self.test[transformer.feature_names_in_]) if transformer is not None else None
        if plan.method == 'stacking':
            query = np.column_stack(matrices)
            if plan.passthrough:
                query = np.column_stack([query, features])
            return _predict({**state, 'model': {'estimator': model, 'labels': labels, 'classification': classification},
                             'validation_features': query, 'test_features': query}, {}, self.defaults)
        weights = np.zeros((len(self.test), len(pairs)))
        weights[:, np.asarray(model.classes_, dtype=int)] = model.predict_proba(features)
        values = sum(weights[:, index, None] * value for index, value in enumerate(matrices))
        probability = classification and _requires_probabilities(state, self.defaults)
        frame = _prediction(values if probability or not classification else classes[values.argmax(axis=1)],
                            state, self.defaults, labels=classes, probabilities=probability)
        if classification:
            attach_probabilities(frame, values, classes)
        return {'validation_predictions': frame.copy(), 'test_predictions': frame}

    def publish(self, node_id):
        prediction = self.run_node(node_id)['test']
        if len(prediction) != len(self.test):
            raise ValueError('Final prediction count differs from test')
        destination = self.root / 'final' / 'submission.csv'
        self.reader.write_submission(self.test, prediction, self.tables, self.config, destination)
        if not destination.is_file() or not destination.stat().st_size:
            raise ValueError('Final refit produced no submission')
        atomic_bytes(self.source / self.preparation.output.filename, destination.read_bytes())
        save(self.root / 'final' / 'refit.json', {'status': 'completed', 'node_id': node_id,
             'training_rows': len(self.train), 'test_rows': len(self.test), 'refitted_sources': list(self.memo),
             'submission_file': 'final/submission.csv', 'validation_metric_recomputed': False})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-directory', type=Path, required=True)
    parser.add_argument('--node-id', required=True)
    parser.add_argument('--time-budget', type=float, required=True)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        FinalRefit(args.run_directory.resolve(), args.time_budget).publish(args.node_id)


if __name__ == '__main__':
    main()

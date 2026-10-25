"""Saved fitted choices and small factual summaries; no model search here."""
import json
from pathlib import Path

import cloudpickle

from .data import within
from .predictions import source_node


def frozen_available(facts):
    if facts.get('task_family') == 'multi_target':
        return all(item['diagnostics'].get('constant') or frozen_available(item['diagnostics'])
                   for item in facts['targets'])
    return bool(facts.get('selected_model_file'))


def refit_automl(state, plan, defaults):
    from .selected_model import refit_saved
    runtime = state['preparation']['runtime']
    root = Path(runtime.get('source_run_directory', runtime['run_directory']))
    source_id = plan.reuse_node or runtime.get('source_node_id')
    node, _ = source_node(root, source_id, runtime['preparation_sha256'])
    facts = json.loads((node / f'{plan.engine}_full_diagnostics.json').read_text(encoding='utf-8'))
    if not frozen_available(facts):
        raise ValueError('This source has no frozen choice for a meta refit; use a fixed-weight blend instead')
    if facts.get('task_family') == 'multi_target':
        from .finalize import FinalRefit
        refitter = FinalRefit.__new__(FinalRefit)
        refitter.defaults = defaults
        result = refitter.multi_target(plan, facts, state)
    else:
        filename = within(root, facts['selected_model_file'])
        result = refit_saved(state, plan, filename, defaults)
    result['automl_diagnostics'] = {**facts, **result.get('automl_diagnostics', {}), 'configuration': plan.model_dump(),
        'search_repeated': False, 'selection_reused': True, 'inherited_from': source_id,
        'fit_rows': len(state['fit']), 'inner_validation': 'fixed source choice; no new search',
        'qualification': 'Source model selection is reused; this is not a new independent model-selection estimate.'}
    if 'inner_best_loss' in result['automl_diagnostics']:
        result['automl_diagnostics']['source_inner_best_loss'] = result['automl_diagnostics'].pop('inner_best_loss')
    return result


def selected_file(root, node_id, preparation_sha256):
    node, _ = source_node(root, node_id, preparation_sha256)
    paths = {facts['selected_model_file'] for path in node.glob('*_full_diagnostics.json')
             if (facts := json.loads(path.read_text(encoding='utf-8'))).get('selected_model_file')}
    if len(paths) != 1:
        raise ValueError(f'{node_id}: one saved AutoML choice is required')
    return within(root, paths.pop())


def source_estimator(root, node_id, preparation_sha256):
    node, _ = source_node(root, node_id, preparation_sha256)
    manifests = [json.loads(path.read_text(encoding='utf-8')) for path in node.glob('training_*.json')]
    fits = [item for item in manifests if item['kind'] == 'fit']
    if len(fits) == 1:
        artifact = within(root, fits[0]['cache']) / 'result.pkl'
        model = cloudpickle.loads(artifact.read_bytes())['result']['model']['estimator']
    else:
        choice = cloudpickle.loads(selected_file(root, node_id, preparation_sha256).read_bytes())
        if getattr(choice, 'engine', None) != 'flaml' or len(choice.models) != 1:
            raise ValueError('Native parameter inheritance currently needs one ordinary model or FLAML winner')
        model = choice.models[0].model
    return model


def estimator_facts(model):
    options = model.get_params(deep=False) if hasattr(model, 'get_params') else {}
    # Nested Python estimators are not API facts and do not belong in prompts.
    serializable = {}
    for key, value in options.items():
        try:
            json.dumps(value, allow_nan=False)
        except (ValueError, TypeError):
            continue
        serializable[key] = value
    return {'native_estimator': type(model).__module__ + '.' + type(model).__name__,
            'native_parameters': serializable}


def comparison_facts(root, parent_id):
    if not parent_id:
        return {}
    parent = []
    for path in (root / 'nodes' / parent_id).glob('*_full_diagnostics.json'):
        facts = json.loads(path.read_text(encoding='utf-8'))
        parent.append({key: facts[key] for key in ('native_estimator', 'native_parameters',
            'best_estimator', 'best_config', 'inner_validation', 'selected_columns') if key in facts})
    return {'parent_id': parent_id, 'parent_execution_facts': [item for item in parent if item],
            'controlled_change_verified': False,
            'qualification': 'Check actual model parameters, preprocessing, features and rows on both sides. '
                             'Hypothesis text alone does not establish an isolated change.'}

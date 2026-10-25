"""Prepare honest ensemble-training predictions once, only when requested."""
from pathlib import Path
import json
import time

import numpy as np

from .data import file_hash, input_hashes, module, within
from .predictions import load_pair, save_pair, source_node
from .storage import load
from .contracts import Preparation
from .context import preparation_view
from .training_cache import locked


def available(root, node_id, memo=None):
    """A fitted combiner has no independent next-level holdout to reuse."""
    memo = {} if memo is None else memo
    if node_id in memo:
        return memo[node_id]
    blocks = json.loads((root / 'nodes' / node_id / 'assembly.json').read_text(encoding='utf-8'))['blocks']
    result = True
    for block in blocks:
        call = block.get('operation')
        if call is None:
            result = False
            break
        if call['kind'] == 'automl':
            from .model_evidence import frozen_available
            path = root / 'nodes' / node_id / f"{call['parameters']['engine']}_full_diagnostics.json"
            if not path.exists() or not frozen_available(json.loads(path.read_text(encoding='utf-8'))):
                result = False
                break
        if call['kind'] == 'saved_ensemble':
            parameters = call['parameters']
            if parameters['method'] in {'stacking', 'gating'} or parameters.get('optimize_weights'):
                result = False
                break
            if not all(available(root, source, memo) for source in parameters['source_nodes']):
                result = False
                break
    memo[node_id] = result
    return result


def ensure(root, node_id, preparation_sha256, deadline):
    """Never obtain meta predictions from a model fitted on meta labels."""
    from .assembly import execute, implementation_hash
    node, finish = source_node(root, node_id, preparation_sha256)
    directory = node / 'meta_fit'
    with locked(directory / 'training.lock', deadline):
        pair = load_pair(root, node_id, preparation_sha256)
        if 'meta' in pair:
            if not pair['meta'].attrs.get('meta_held_out', False):
                raise ValueError(f'{node_id}: meta predictions are not held out')
            return
        if not available(root, node_id):
            raise ValueError(f'{node_id}: no independent meta recipe for this learned ensemble')
        if time.monotonic() >= deadline:
            raise TimeoutError('No time left to prepare ensemble meta predictions')
        if implementation_hash(node) != finish['code_sha256']:
            raise ValueError('Meta source changed after evaluation')
        preparation = load(root / 'preparation.json', Preparation)
        if file_hash(root / 'preparation.json') != preparation_sha256:
            raise ValueError('Meta preparation changed')
        split_path = within(root, preparation.validation['ensemble_split_file'])
        if file_hash(split_path) != preparation.validation['ensemble_split_sha256']:
            raise ValueError('Meta split changed')
        source = Path(preparation.task_directory)
        if input_hashes(source, preparation.data.files) != preparation.source_hashes:
            raise ValueError('Meta input files changed')
        reader = module(root / 'prepare.py', preparation.code_sha256)
        tables = reader.load_data(source, preparation.model_dump(mode='json'))
        with np.load(split_path, allow_pickle=False) as split:
            train_rows, meta_rows = split['train'], split['meta']
        train = tables['train']
        query = train.iloc[meta_rows].drop(columns=preparation.data.target_columns).copy()
        metadata = preparation_view(preparation)
        metadata['runtime'] = {'deadline': deadline, 'time_budget_seconds': deadline - time.monotonic(),
            'num_cpus': 1, 'num_gpus': 0, 'is_smoke_check': False, 'is_meta_fit': True,
            'work_directory': str(directory), 'run_directory': str(root),
            'source_node_id': node_id,
            'preparation_sha256': preparation_sha256, 'fit_row_positions': train_rows,
            'meta_rows': 0, 'outer_validation_rows': len(query)}
        auxiliary = {key: value for key, value in tables.items() if key not in {'train', 'test'}}
        prediction, _ = execute(node, train.iloc[train_rows].copy(), query, query.copy(), metadata, auxiliary)
        if len(prediction) != len(meta_rows):
            raise ValueError('Meta prediction count differs from the shared holdout')
        prediction.attrs['meta_held_out'] = True
        # Save one extra prediction frame, not another copy of the source node/model files.
        save_pair(directory, {'validation': prediction}, preparation_sha256,
                  finish['code_sha256'], 'predictions.csv')

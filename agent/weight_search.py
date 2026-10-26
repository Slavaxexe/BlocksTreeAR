"""Fast blend-weight search on saved, held-out inner predictions only."""
from pathlib import Path
import json
import time

import numpy as np

from .data import file_hash, module
from .storage import save


def optimize(pairs, state, defaults, plan):
    from .ensembles import blend
    from .ensemble_data import subset
    if not all('meta' in pair and pair['meta'].attrs.get('meta_held_out', False) for pair in pairs):
        raise ValueError('Weight search needs meta_ready sources with honest held-out predictions')
    meta = state['auxiliary']['ensemble_meta']
    if any(len(pair['meta']) != len(meta) for pair in pairs):
        raise ValueError('Weight sources must use the same inner holdout')
    runtime = state['preparation']['runtime']
    root = Path(runtime['run_directory'])
    reader = module(root / 'prepare.py', file_hash(root / 'prepare.py'))
    inner_pairs = [{'validation': pair['meta'], 'test': subset(pair['meta'], np.array([], dtype=int))} for pair in pairs]
    inner = {**state, 'validation': meta.drop(columns=state['preparation']['data']['target_columns']),
             'test': state['test'].iloc[:0]}
    config = json.loads((root / 'preparation.json').read_text(encoding='utf-8'))
    direction = state['preparation']['metric']['direction']
    if direction not in {'minimize', 'maximize'}:
        raise ValueError('Weight search needs a known metric direction')
    initial = np.asarray(plan.weights or [1.] * len(pairs), dtype=float)
    initial /= initial.sum()
    rng = np.random.default_rng(42)
    weights, best = initial, None
    measured = []
    for number in range(plan.weight_trials):
        if time.monotonic() >= runtime['deadline']:
            break
        candidate = initial if number == 0 else (np.eye(len(pairs))[number - 1]
                    if number <= len(pairs) else rng.dirichlet(1. + weights * 8))
        prediction = blend(inner_pairs, inner, defaults, candidate.tolist(), plan.method)['validation_predictions']
        from .task_types import score_predictions
        value = score_predictions(reader, meta, prediction, config, defaults.get('task_type'))
        if not np.isfinite(value):
            continue
        measured.append(value)
        if best is None or (value < best if direction == 'minimize' else value > best):
            weights, best = candidate, value
    if best is None:
        raise ValueError('Weight search produced no finite inner-holdout score')
    save(Path(runtime['work_directory']) / 'weights_full_diagnostics.json', {'weights': weights.tolist(),
        'source_nodes': plan.source_nodes, 'metric': state['preparation']['metric']['name'],
        'inner_score': best, 'trials': len(measured), 'scope': 'shared inner holdout only'})
    return weights.tolist()

"""Budget-aware obligations: measured feature coverage rather than prompt promises."""
import hashlib
import json
from pathlib import Path
import numpy as np


def context_signature(context):
    digest = hashlib.sha256()
    for name in ('features_path', 'labels_path'):
        with Path(context[name]).open('rb') as stream:
            digest.update(hashlib.file_digest(stream, 'sha256').digest())
    with np.load(context['folds_path'], allow_pickle=False) as folds:
        for name in sorted(folds.files):
            array = folds[name]
            digest.update(name.encode());digest.update(str(array.dtype).encode());digest.update(array.tobytes())
    digest.update(json.dumps({key: context.get(key) for key in ('target', 'classes', 'metric', 'positive_index',
        'id_cols', 'time_column', 'entity_key', 'group_columns', 'seed')},
        sort_keys=True, ensure_ascii=False).encode())
    return digest.hexdigest()


def previous_winner(root, current, context):
    signature = context_signature(context)
    best = None
    folders = sorted((root/'artifacts').glob('classification_[0-9]*'), reverse=True)[:12]
    for folder in folders:
        if folder.resolve() == current.resolve():continue
        try:
            report = json.loads((folder/'report.json').read_text(encoding='utf-8'))
            old_context = json.loads((folder/'context.json').read_text(encoding='utf-8'))
            if any(old_context.get(key) != context.get(key) for key in ('target', 'classes', 'metric', 'positive_index')):continue
            if context_signature(old_context) != signature:continue
            record = next(row for row in report['experiments'] if row['id'] == report['best_single']['id'])
            if record['status'] != 'ok':continue
            better = best is None or (record['cv'] < best['reference_cv'] if context['metric'] == 'log_loss'
                                     else record['cv'] > best['reference_cv'])
            if better:
                best = {'signature': signature, 'reference_cv': record['cv'], 'spec': record['spec'],
                    'source_artifact': str(folder), 'source_experiment': record['id'],
                    'rule': 'Re-evaluate on current fixed folds; never import a historical score or OOF as a new result'}
        except (OSError, ValueError, KeyError, StopIteration):
            continue
    return signature, best


def required_trials(profile, general, blocks):
    """Generate applicable hypotheses by measured types; no dataset/class names."""
    trials = []
    cats = [row['feature'] for row in profile.get('categorical', []) if row['unique'] >= 2]
    if cats and 'oof_target_encoding' in blocks:
        trials.append({'id': 'categorical_encoding', 'title': 'Cross-fitted class-probability encoding',
            'blocks_add': ['oof_target_encoding'], 'evidence': ['Categorical inputs exist; compare source-only cross-fitted class probabilities.']})
    elif cats:
        trials.append({'id': 'categorical_encoding', 'title': 'Training-only frequency encoding',
            'preprocessing': {'frequency_columns': cats[:4]}, 'evidence': ['Categorical inputs exist; encode fit-only frequencies without future labels.']})
    if profile.get('constants') or profile.get('duplicate_columns'):
        trials.append({'id': 'feature_quality', 'title': 'Remove measured redundant columns',
            'preprocessing': {'drop_constants': True, 'drop_duplicates': True},
            'evidence': ['Training EDA found constants or duplicate columns; removal is tested, not assumed beneficial.']})
    else:
        overrides = [row['feature'] for row in general.get('potential_categorical_numeric', [])][:4]
        if overrides:
            trials.append({'id': 'feature_quality', 'title': 'Compare low-cardinality numeric features as categories',
                'preprocessing': {'categorical_columns': overrides},
                'evidence': ['EDA found low-cardinality numeric inputs; categorical semantics must be validated.']})
        elif profile.get('correlated_pairs'):
            trials.append({'id': 'feature_quality', 'title': 'Test redundant numeric correlations',
                'preprocessing': {'correlation_threshold': .995}, 'evidence': ['EDA measured strongly correlated numeric pairs.']})
    if any(abs(row.get('skew') or 0) > 3 for row in profile.get('numeric', [])):
        trials.append({'id': 'tail_or_missingness', 'title': 'Compare extreme values and outlier indicators',
            'preprocessing': {'clip_quantile': .01, 'outlier_flags': True},
            'evidence': ['Training numeric distributions are skewed; bounded clipping is one hypothesis, not row deletion.']})
    elif profile.get('high_missing'):
        trials.append({'id': 'tail_or_missingness', 'title': 'Compare missingness indicators',
            'preprocessing': {'missing_flags': True}, 'evidence': ['EDA measured substantial missingness.']})
    trials.append({'id': 'model_family', 'title': 'Evaluate another model family on the same folds',
        'model_alternative': True, 'preferred_model': 'catboost' if cats else 'extra_trees',
        'evidence': ['One model family does not cover alternative inductive biases.']})
    return trials


def satisfies(record, trial):
    if record.get('status') != 'ok':return False
    spec = record['spec']
    if trial.get('balance_trial'):
        return bool(spec['training'].get('class_weight_power', 0) > 0
            or spec['model']['params'].get('class_weight') or spec['model']['params'].get('auto_class_weights'))
    if trial.get('blocks_add'):
        return set(trial['blocks_add']).issubset(spec['blocks'])
    if trial.get('model_alternative'):
        return spec['model']['model_id'] not in {'lightgbm', 'prior'}
    policy = spec.get('preprocessing', {})
    for key, value in trial.get('preprocessing', {}).items():
        actual = policy.get(key)
        if isinstance(value, list):
            if not actual:return False
        elif actual != value:return False
    return True


def protocol_status(records, trials, deferred=None):
    deferred = deferred or {}
    result = []
    for trial in trials:
        matches = [row for row in records if satisfies(row, trial)]
        tagged = [row for row in records if row.get('protocol_id') == trial['id']
            or (trial.get('balance_trial') and row.get('source') == 'balance_monitor')]
        successful = bool(matches)
        result.append({'id': trial['id'], 'title': trial['title'],
            'status': 'completed' if successful else 'deferred_budget' if trial['id'] in deferred
                else 'attempt_failed' if tagged else 'pending',
            'experiments': [row['id'] for row in matches or tagged],
            'scores': [row['cv'] for row in matches], 'reason': deferred.get(trial['id'])})
    return result

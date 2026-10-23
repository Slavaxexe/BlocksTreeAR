"""Last-resort answer without the LLM: when the agent ended without submission.csv, infer the table roles from the
files themselves (target = train column absent from test, ids and output columns from the sample submission) and
predict with the shift-aware toolkit.gbdt. Nothing here depends on a particular dataset.
"""

import json
from pathlib import Path
import re
import sys
import time

import numpy as np
import pandas as pd

SKIP = {'artifacts', 'outputs', 'final', 'nodes', '__pycache__'}


def tables(directory: Path):
    """train / test / sample CSV files found anywhere under the task directory."""
    found = {}
    for path in sorted(directory.rglob('*.csv')):
        relative = path.relative_to(directory)
        if any(part in SKIP or part.startswith('.') for part in relative.parts[:-1]) or path.name == 'submission.csv':
            continue
        name = path.stem.casefold()
        role = 'sample' if 'sample' in name else 'train' if 'train' in name else 'test' if 'test' in name else None
        if role and role not in found:
            found[role] = path
    if 'train' not in found or 'test' not in found:
        raise FileNotFoundError(f'train/test CSV not found under {directory}: {sorted(found)}')
    read = lambda path: pd.read_csv(path, sep=None, engine='python') if path.stat().st_size < 1_000_000 else pd.read_csv(path)
    return {role: read(path) for role, path in found.items()}


def roles(train, test, sample, text):
    missing = [c for c in train.columns if c not in test.columns]
    target = next((c for c in (sample.columns if sample is not None else []) if c in missing), None) or \
        next((c for c in missing if c.casefold() in text.casefold()), None) or (missing[0] if missing else None)
    if target is None:
        raise ValueError('No train column is absent from test: target unknown')
    shared = [c for c in test.columns if c in train.columns]
    ids = [c for c in (sample.columns if sample is not None else []) if c in test.columns]
    ids = ids or [c for c in shared if test[c].is_unique and re.search(r'(^|_)id$', c.casefold())][:1]
    columns = list(sample.columns) if sample is not None else ids + [target]
    return target, ids, columns


def preparation(train, test, target, ids, text):
    from .worker import shift_split, target_summary
    from .contracts import DataBinding
    lowered = text.casefold()
    numeric = pd.api.types.is_numeric_dtype(train[target]) and train[target].nunique() > 20
    if numeric:
        name, direction = ('MAE', 'minimize') if re.search(r'\bmae\b|mean absolute', lowered) else ('RMSE', 'minimize')
    else:
        name, direction = ('Macro F1', 'maximize') if 'macro' in lowered else ('ROC AUC', 'maximize') \
            if re.search(r'\bauc\b|roc', lowered) else ('Accuracy', 'maximize')
    binding = DataBinding(files={}, target_columns=[target], id_columns=ids)
    rows = np.arange(len(train))
    _, _, shift = shift_split(train, test, binding, {'strategy': 'random'}, rows[: len(rows) * 4 // 5], rows[len(rows) * 4 // 5:])
    return {'data': {'target_columns': [target], 'id_columns': ids}, 'task_type': 'regression' if numeric else 'classification',
            'metric': {'name': name, 'definition': name, 'direction': direction}, 'prediction_scope': 'row',
            'profile': {'target': target_summary(train, [target])}, 'validation': {'shift': shift}, 'scorer': scorer(name, target)}


def scorer(name, target):
    from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, roc_auc_score
    def score(reference, predictions, config):
        truth, guess = reference[target].to_numpy(), predictions[target].to_numpy()
        if name == 'RMSE':
            return float(np.sqrt(np.mean((truth.astype(float) - guess.astype(float)) ** 2)))
        if name == 'MAE':
            return float(mean_absolute_error(truth, guess))
        if name == 'Macro F1':
            return float(f1_score(truth.astype(str), guess.astype(str), average='macro'))
        if name == 'ROC AUC':
            return float(roc_auc_score(truth, guess))
        return float(accuracy_score(truth.astype(str), guess.astype(str)))
    return score


def emergency(task_path: Path, log=print) -> Path:
    """Write submission.csv next to task.md from a controller-only model; constant predictions if that fails too."""
    from .toolkit import gbdt
    started, task_path = time.monotonic(), Path(task_path).resolve()
    directory, text = task_path.parent, task_path.read_text(encoding='utf-8', errors='replace')
    found = tables(directory)
    train, test, sample = found['train'], found['test'], found.get('sample')
    target, ids, columns = roles(train, test, sample, text)
    train = train[train[target].notna()].reset_index(drop=True)
    try:
        config = preparation(train, test, target, ids, text)
        values = gbdt(train, config)(test)[target].to_numpy()
        model = 'toolkit.gbdt'
    except Exception as error:  # still answer: most frequent class or median
        log(f'fallback model failed ({type(error).__name__}: {error}); writing a constant prediction')
        labels = train[target]
        constant = labels.median() if pd.api.types.is_numeric_dtype(labels) and labels.nunique() > 20 else labels.mode().iloc[0]
        values, model = np.repeat(constant, len(test)), 'constant'
    result = pd.DataFrame({c: test[c].to_numpy() for c in ids if c in test})
    for column in columns:
        if column not in result:
            result[column] = values if column == target else (sample[column].iloc[0] if sample is not None else values)
    destination = directory / 'submission.csv'
    temporary = destination.with_suffix('.csv.tmp')
    result[columns].to_csv(temporary, index=False)
    temporary.replace(destination)
    if len(pd.read_csv(destination)) != len(test):
        raise ValueError('Fallback submission row count differs from test')
    log(json.dumps({'fallback': model, 'target': target, 'ids': ids, 'columns': columns, 'rows': len(test),
                    'seconds': round(time.monotonic() - started, 1)}))
    return destination


if __name__ == '__main__':
    emergency(Path(sys.argv[1] if len(sys.argv) > 1 else 'task.md'))

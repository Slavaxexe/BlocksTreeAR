"""Reusable predictions bound to one preparation and immutable source nodes."""
import io
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from .data import file_hash, within
from .storage import atomic_bytes, save


def attach_probabilities(frame, values, classes):
    frame.attrs['probabilities'] = {'values': np.asarray(values, dtype=float),
                                  'classes': np.asarray(classes).tolist()}
    return frame


def save_pair(directory, frames, preparation_sha256, code_sha256, validation_file):
    record = {'preparation_sha256': preparation_sha256, 'code_sha256': code_sha256,
              'probabilities': False, 'meta_held_out': frames['validation'].attrs.get('meta_held_out', True), 'frames': {}}
    probabilities = {}
    classes = None
    for scope, frame in frames.items():
        name = validation_file if scope == 'validation' else scope + '_predictions.csv'
        path = within(directory, name)
        atomic_bytes(path, frame.to_csv(index=False).encode('utf-8'))
        record['frames'][scope] = {'file': name, 'sha256': file_hash(path), 'rows': len(frame),
            'dtypes': {str(column): str(dtype) for column, dtype in frame.dtypes.items()}}
        probability = frame.attrs.get('probabilities')
        if probability is not None:
            if classes is not None and classes != probability['classes']:
                raise ValueError('Validation/test probability class order differs')
            classes = probability['classes']
            probabilities[scope] = probability['values']
    if len(probabilities) == len(frames):
        buffer = io.BytesIO()
        np.savez_compressed(buffer, **probabilities)
        path = directory / 'probabilities.npz'
        atomic_bytes(path, buffer.getvalue())
        record.update(probabilities=True, classes=classes, probability_sha256=file_hash(path))
    save(directory / 'predictions.json', record)


def source_node(root, node_id, preparation_sha256):
    if not re.fullmatch(r'node_[0-9]{3}', node_id):
        raise ValueError('Source must name a saved node')
    directory = within(root / 'nodes', node_id)
    finish = json.loads((directory / 'finish.json').read_text(encoding='utf-8'))
    if finish['status'] != 'completed' or finish['preparation_sha256'] != preparation_sha256:
        raise ValueError(f'{node_id}: source did not complete on this preparation')
    return directory, finish


def load_pair(root, node_id, preparation_sha256, defaults=None, metric=None):
    directory, finish = source_node(root, node_id, preparation_sha256)
    header = directory / 'predictions.json'
    frames = {}
    if header.exists():
        record = json.loads(header.read_text(encoding='utf-8'))
        if (record['preparation_sha256'], record['code_sha256']) != (preparation_sha256, finish['code_sha256']):
            raise ValueError(f'{node_id}: prediction metadata differs from the executed node')
        for scope, specification in record['frames'].items():
            path = within(directory, specification['file'])
            if file_hash(path) != specification['sha256']:
                raise ValueError(f'{node_id}: saved predictions changed')
            frames[scope] = pd.read_csv(path, dtype=specification['dtypes'], keep_default_na=False)
        if record['probabilities']:
            path = directory / 'probabilities.npz'
            if file_hash(path) != record['probability_sha256']:
                raise ValueError(f'{node_id}: saved probabilities changed')
            with np.load(path, allow_pickle=False) as values:
                for scope, frame in frames.items():
                    attach_probabilities(frame, values[scope], record['classes'])
        for frame in frames.values():
            frame.attrs['meta_held_out'] = record.get('meta_held_out', False)
        return frames
    # Older runs saved label/value CSVs without an extra prediction manifest.
    labels = (defaults or {}).get('class_labels') or []
    name = ''.join(c for c in (metric or '').lower() if c.isalnum())
    dtype = str if labels and all(isinstance(label, str) for label in labels) and name not in {'auc', 'rocauc', 'logloss'} else None
    frames['validation'] = pd.read_csv(within(directory, finish['validation_file']), dtype=dtype, keep_default_na=False)
    frame = pd.read_csv(within(directory, finish['submission_file']), dtype=dtype, keep_default_na=False)
    frames['test'] = frame[list(frames['validation'].columns)]
    return frames


def prediction_catalog(root, rows, leader_id, cache, defaults=None, metric=None):
    candidates = []
    for row in rows:
        if row.result.status != 'completed' or row.result.value is None:
            continue
        try:
            directory = root / 'nodes' / row.node_id
            if not (directory / 'assembly.json').exists() or not (directory / 'finish.json').exists():
                continue
            key = (row.node_id, row.result.code_sha256)
            if key not in cache:
                frames = load_pair(root, row.node_id, row.result.preparation_sha256, defaults, metric)
                probabilities = all('probabilities' in frame.attrs for frame in frames.values())
                # Baker needs only a compact comparison; do not retain test/probability matrices here.
                validation = frames['validation'].copy()
                validation.attrs = {}
                cache[key] = {'validation': validation, 'probabilities': probabilities,
                              'meta_ready': 'meta' in frames and frames['meta'].attrs.get('meta_held_out', False)}
            candidates.append({'node_id': row.node_id, 'value': row.result.value,
                'probabilities': cache[key]['probabilities'],
                'oof_recipe': cache[key]['meta_ready'], 'meta_ready': cache[key]['meta_ready']})
        except (OSError, ValueError, KeyError):
            continue
    leader = next((row for row in rows if row.node_id == leader_id), None)
    if leader is not None:
        reference = cache.get((leader.node_id, getattr(leader.result, 'code_sha256', None)), {}).get('validation')
        if reference is not None:
            for candidate in candidates:
                row = next(row for row in rows if row.node_id == candidate['node_id'])
                current = cache[(row.node_id, row.result.code_sha256)]['validation']
                if current.shape == reference.shape:
                    if any(word in str((defaults or {}).get('task_type', '')).lower() for word in ('class', 'binary')):
                        candidate['prediction_disagreement_with_leader'] = float(np.mean(current.to_numpy() != reference.to_numpy()))
                    else:
                        a, b = current.to_numpy(dtype=float).ravel(), reference.to_numpy(dtype=float).ravel()
                        candidate['prediction_correlation_with_leader'] = float(np.corrcoef(a, b)[0, 1]) if a.std() and b.std() else None
    return candidates

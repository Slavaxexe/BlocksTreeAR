"""One shared inner holdout for reusable ensemble training predictions."""
import io

import numpy as np
from sklearn.model_selection import train_test_split

from .storage import atomic_bytes


def make_split(frame, targets, policy, classification=False):
    from .validation import structured_holdout
    rows = np.arange(len(frame))
    indices = structured_holdout(frame, policy, .2, 42)
    if indices is not None:
        return indices
    labels = frame[targets[0]] if len(targets) == 1 else None
    # Stratify discrete targets when each class can occur on both sides.
    discrete = labels is not None and classification
    stratify = labels if discrete and labels.value_counts().min() >= 2 and labels.nunique() < len(rows) * .2 else None
    train, meta = train_test_split(rows, test_size=.2, random_state=42, stratify=stratify)
    if discrete:
        rare = np.flatnonzero(labels.map(labels.value_counts()) < 2)
        train, meta = np.union1d(train, rare), np.setdiff1d(meta, rare)
    return np.sort(train), np.sort(meta)


def save_split(path, fit, frame, targets, policy, classification=False):
    train, meta = make_split(frame.iloc[fit].reset_index(drop=True), targets, policy, classification)
    if not len(train) or not len(meta):
        raise ValueError('The ensemble holdout needs nonempty training and held-out rows')
    buffer = io.BytesIO()
    np.savez(buffer, train=fit[train], meta=fit[meta])
    atomic_bytes(path, buffer.getvalue())


def subset(frame, rows):
    result = frame.iloc[rows].reset_index(drop=True)
    if 'probabilities' in frame.attrs:
        from .predictions import attach_probabilities
        values = frame.attrs['probabilities']
        attach_probabilities(result, values['values'][rows], values['classes'])
    return result


def concatenate(frames):
    import pandas as pd
    from .predictions import attach_probabilities
    pieces = []
    for frame in frames:
        piece = frame.copy(deep=False)
        piece.attrs = {}
        pieces.append(piece)
    result = pd.concat(pieces, ignore_index=True)
    if all('probabilities' in frame.attrs for frame in frames):
        classes = frames[0].attrs['probabilities']['classes']
        if any(frame.attrs['probabilities']['classes'] != classes for frame in frames):
            raise ValueError('Saved probability class orders differ')
        attach_probabilities(result, np.concatenate([frame.attrs['probabilities']['values'] for frame in frames]), classes)
    result.attrs['meta_held_out'] = all(frame.attrs.get('meta_held_out', True) for frame in frames)
    return result

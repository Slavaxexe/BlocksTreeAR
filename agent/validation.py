"""Shared metadata and structured holdouts; outer split indices remain unchanged."""
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


def metadata(policy):
    policy = dict(policy or {})
    column = policy.get('time_column') or policy.get('split_column') or policy.get('year_column')
    if column:
        # Preserve aliases: generated prepare.py may still read the original key.
        policy['time_column'] = column
    return policy


def strategy(policy):
    description = str(policy.get('strategy') or policy.get('method') or '').lower()
    if policy.get('group_columns') or 'group' in description:
        return 'group'
    if any(word in description for word in ('time', 'temporal', 'chronological')):
        return 'time'
    return 'ordered' if 'ordered' in description else 'random'


def structured_holdout(frame, policy, fraction, seed, kind='auto'):
    """Return group/time/row-order indices, or None for a random split."""
    policy = metadata(policy)
    kind = strategy(policy) if kind == 'auto' else kind
    rows = np.arange(len(frame))
    if kind == 'group':
        columns = policy.get('group_columns', [])
        if not columns:
            raise ValueError('Group holdout needs group_columns or an explicit group_by')
        groups = pd.factorize(pd.MultiIndex.from_frame(frame[columns]), sort=False)[0]
        return next(GroupShuffleSplit(n_splits=1, test_size=fraction, random_state=seed).split(rows, groups=groups))
    if kind == 'time':
        from .supervised_features import ordered_time
        column = policy.get('time_column')
        if not column:
            raise ValueError('Temporal holdout needs time_column (split_column/year_column aliases are accepted)')
        if column not in frame:
            raise ValueError(f'Temporal holdout column {column!r} is absent from the input table')
        times = ordered_time(frame[column])
        periods = np.unique(times)
        if len(periods) < 2:
            raise ValueError('Temporal holdout needs at least two distinct periods')
        cut = periods[max(1, int(len(periods) * (1 - fraction)))]
        return rows[times < cut], rows[times >= cut]
    if kind == 'ordered':
        cut = max(1, int(len(rows) * (1 - fraction)))
        return rows[:cut], rows[cut:]
    return None

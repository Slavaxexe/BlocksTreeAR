"""Bounded structural key proposals, without targets or semantic decisions."""
from itertools import combinations
import time

import numpy as np
import pandas as pd


def discover(train, query, excluded=(), seconds=2., max_checks=96):
    started = time.monotonic()
    names = [c for c in train if c in query and c not in excluded]
    unique = {c: train[c].nunique() for c in names}
    periods = []
    for column in names:
        if pd.api.types.is_bool_dtype(train[column]):
            continue
        if not (pd.api.types.is_numeric_dtype(train[column]) or pd.api.types.is_datetime64_any_dtype(train[column])):
            continue
        if 2 <= unique[column] <= 50 and not train[column].isna().any():
            source, future = train[column], query[column].dropna()
            separated = bool(len(future) and (future.min() > source.max() or future.max() < source.min()))
            periods.append((separated, unique[column], column))
    periods.sort(key=lambda value: (not value[0], value[1], value[2]))
    candidates, checked = [], 0
    for _, _, period in periods[:2]:
        pool = [c for c in names if c != period and 1 < unique[c] < len(train) * .95]
        # Include both discriminating measurements and categorical qualifiers.
        high = sorted(pool, key=lambda c: -unique[c])[:6]
        low = sorted(pool, key=lambda c: unique[c])[:6]
        pool = list(dict.fromkeys(high + low))
        queue, visited = list(combinations(pool, 2)), set()
        local = []
        for width in (2, 3, 4):
            for key in queue:
                if time.monotonic() - started >= seconds or checked >= max_checks:
                    break
                if not any(unique[c] >= max(2, np.sqrt(len(train))) for c in key):
                    continue
                checked += 1
                source_key = pd.MultiIndex.from_frame(train[list(key)])
                query_key = pd.MultiIndex.from_frame(query[list(key)])
                duplicate = train.duplicated([*key, period], keep=False)
                missing = train[list(key)].isna().any(axis=1)
                repeats = train.groupby(list(key), dropna=False)[period].transform('nunique') > 1
                row = {'period_candidate': period, 'columns': list(key),
                    'source_rows': len(train), 'query_rows': len(query),
                    'query_row_coverage': float(query_key.isin(source_key).mean()) if len(query) else 0.,
                    'query_key_coverage': float(query_key.unique().isin(source_key.unique()).mean()) if len(query) else 0.,
                    'within_period_ambiguity_fraction': float(duplicate.mean()),
                    'cross_period_repeat_row_fraction': float(repeats.mean()),
                    'missing_key_row_fraction': float(missing.mean())}
                local.append(row)
                visited.add(frozenset(key))
            ranked = sorted(local, key=lambda x: (x['query_row_coverage'] - x['within_period_ambiguity_fraction'],
                                                  x['cross_period_repeat_row_fraction']), reverse=True)
            queue = []
            for row in [item for item in ranked if len(item['columns']) == width][:6]:
                for column in pool:
                    key = tuple(dict.fromkeys([*row['columns'], column]))
                    if len(key) == width + 1 and frozenset(key) not in visited:
                        visited.add(frozenset(key))
                        queue.append(key)
            candidates.extend(ranked[:4])
            if time.monotonic() - started >= seconds or checked >= max_checks:
                break
    selected = {}
    for item in candidates:
        selected[(item['period_candidate'], tuple(item['columns']))] = item
    ranked = sorted(selected.values(), key=lambda x: x['query_row_coverage'] - x['within_period_ambiguity_fraction'], reverse=True)
    return {'candidates': ranked[:8], 'checks': checked, 'elapsed_seconds': time.monotonic() - started,
            'scope': 'Unlabelled source/query structure only; coverage is not strictly-past history coverage.',
            'qualification': 'Period candidates are numeric/date patterns, not confirmed chronology. '
                             'Keys are not proven entity IDs or invariant attributes. Verify with entities/feature analyses before adoption.'}

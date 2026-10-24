"""Measurements for proposed object keys; no assumptions about dataset/domain."""
import numpy as np
import pandas as pd

from .supervised_features import key_codes, ordered_time


def inspect_entities(fit, query, request, targets):
    candidates = request.key_candidates or [request.columns]
    time = request.split_column
    if time in targets:
        raise ValueError('Entity time cannot be a target')
    rows = []
    for columns in candidates:
        if not columns or set(columns) & set(targets):
            raise ValueError('Specify a nonempty key with no target columns')
        if not set(columns) <= set(fit) or not set(columns) <= set(query):
            raise ValueError('Entity keys must exist in both compared tables')
        source_keys, query_keys = key_codes(fit, query, columns)
        counts = pd.Series(source_keys).value_counts()
        row = {'key': columns, 'unique_keys': len(counts),
               'repeated_fit_row_fraction': float(np.mean(pd.Series(source_keys).map(counts) > 1)),
               'comparison_rows': len(query),
               'comparison_key_coverage': float(np.mean(np.isin(query_keys, counts.index))) if len(query) else None,
               'missing_key_row_fraction': float(fit[columns].isna().any(axis=1).mean())}
        if time:
            source_time, query_time = ordered_time(fit[time]), ordered_time(query[time])
            periods = pd.DataFrame({'key': source_keys, 'time': source_time})
            distinct = periods.groupby('key').time.nunique()
            earliest = periods.groupby('key').time.min()
            prior = pd.Series(query_keys).map(earliest).to_numpy() < query_time
            row.update(keys_in_multiple_periods=int((distinct > 1).sum()),
                duplicate_key_period_row_fraction=float(periods.duplicated(['key', 'time'], keep=False).mean()),
                comparison_strict_past_coverage=float(prior.mean()) if len(query) else None)
            row['within_period_ambiguous_rows'] = int(periods.duplicated(['key', 'time'], keep=False).sum())
            row['within_period_ambiguity_fraction'] = row['duplicate_key_period_row_fraction']
            row['source_rows'] = len(fit)
        rows.append(row)
    result = {'candidates': rows, 'comparison': request.compare,
              'definitions': {
                  'repeated_fit_row_fraction': 'Source rows whose key occurs more than once ACROSS ALL periods. Repetition is not within-period ambiguity and can enable history.',
                  'within_period_ambiguity_fraction': 'Source rows with duplicate (key, time), divided by source_rows. This is the measured within-period ambiguity.',
                  'comparison_strict_past_coverage': 'Fraction of comparison rows with a matching source key at a strictly earlier time. Not proof of real-world identity.'},
              'qualification': 'Candidate keys are unverified: matches and history coverage do not prove the same real-world object.'}
    if time:
        def summary(frame):
            values = ordered_time(frame[time])
            return {'rows': len(frame), 'periods': int(pd.Series(values).nunique()),
                    'min': float(values.min()) if len(values) else None,
                    'max': float(values.max()) if len(values) else None}
        result['time'] = {'column': time, 'fit': summary(fit), 'comparison': summary(query)}
    return result

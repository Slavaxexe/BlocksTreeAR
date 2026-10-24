"""Classification time/group splits and fold-local probability history features."""
import itertools
import re
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import (GroupKFold, GroupShuffleSplit, KFold, StratifiedKFold,
                                     StratifiedGroupKFold, train_test_split)


BLOCKS = {
    'base': 'Original columns; categories and medians fitted on this training partition only.',
    'missing': 'Missing indicators selected on fit and number of missing values per row.',
    'families': 'Row aggregates of repeated numeric feature families.',
    'log_numeric': 'Logarithms of fit-selected nonnegative skewed numeric columns.',
    'entity_history': 'Class frequencies and last class of the same inferred entity in strictly earlier periods.',
    'group_history': 'Smoothed class probabilities in categorical groups from strictly earlier periods.',
    'similar_history': 'Past class probabilities for category and fit-defined numeric size bucket.',
    'oof_target_encoding': 'Class-probability encoding cross-fitted entirely within outer-fit; never global label means.',
}


def period_values(series):
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors='coerce').astype(float)
    dates = pd.to_datetime(series, errors='coerce', utc=True)
    return pd.Series(np.where(dates.notna(), dates.astype('int64')/86400e9, np.nan), index=series.index)


def detect_time(train, test, target, ids):
    matches = []
    for name in train:
        if name in {target, *ids} or name not in test:
            continue
        numeric = pd.api.types.is_numeric_dtype(train[name])
        if not numeric and not re.search(r'date|time|period|year|month|дата|время|период|год', name, re.I):
            continue
        a, b = period_values(train[name]), period_values(test[name])
        if not a.notna().all() or not b.notna().all() or a.nunique() < 3:
            continue
        # Numeric IDs/counters with thousands of unique values require a time-like name.
        if numeric and a.nunique() > 400 and not re.search(r'date|time|period|year|month|дата|время|период|год', name, re.I):
            continue
        if b.min() > a.max():
            matches.append({'col': name, 'kind': 'numeric' if numeric else 'datetime',
                'n_train_periods': int(a.nunique()), 'train_max': float(a.max()), 'test_min': float(b.min())})
    return max(matches, key=lambda value: value['n_train_periods']) if matches else None


def key_values(frame, columns):
    # Hash each column separately to avoid delimiter collisions.
    return pd.util.hash_pandas_object(frame[columns].astype('string').fillna('__missing__'), index=False).to_numpy()


def discover_entity(train, test, columns, time_column, max_seconds=6):
    """Structural discovery only: no target or class labels enter key selection."""
    started = time.monotonic()
    candidates = []
    for name in columns:
        if name == time_column:
            continue
        count = train[name].nunique()
        if not 2 <= count <= max(3, len(train)*.8):
            continue
        if time_column:
            recurrence = train.groupby(name, dropna=False)[time_column].nunique()
            if len(recurrence) and (recurrence > 1).mean() > .3:
                candidates.append((name, count))
        elif re.search(r'entity|customer|client|account|device|patient|machine|user|building|store|shop|product|group|объект|клиент', name, re.I):
            candidates.append((name, count))
    candidates.sort(key=lambda value: -value[1])
    best, tried = None, 0
    current = []
    for _ in range(min(4, len(candidates))):
        step = None
        for name, _count in candidates[:18]:
            if name in current or time.monotonic()-started > max_seconds:
                continue
            columns = current+[name]
            groups = key_values(train, columns)
            other = key_values(test, columns)
            coverage = float(pd.Series(other).isin(set(groups)).mean())
            if time_column:
                periods = period_values(train[time_column])
                pairs = pd.DataFrame({'g': groups, 't': periods})
                counts = pairs.groupby(['g', 't']).size()
                repeat = float((pairs.groupby('g').t.nunique() > 1).mean())
                purity = 1./max(1., float(counts.mean()))
                value = coverage*repeat*np.sqrt(purity)
            else:
                repeat = float(1.-pd.Series(groups).nunique()/len(train))
                purity = 1.
                value = repeat
            info = {'key': columns, 'score': float(value), 'test_rows_with_history': coverage,
                    'within_period_uniqueness': purity, 'repeated_group_fraction': repeat,
                    'selection_scope': 'feature structure only; no target labels'}
            tried += 1
            if step is None or info['score'] > step['score']:
                step = info
        if step is None or step['score'] <= .05 or best and step['score'] <= best['score']*1.01:
            break
        best, current = step, step['key']
    if best:
        best['candidates_tried'] = tried
    return best


def design_folds(frame, y, time_column=None, entity_key=None, seed=42):
    n = len(frame)
    index = np.arange(n)
    groups = key_values(frame, entity_key) if entity_key else None
    periods = period_values(frame[time_column]).to_numpy() if time_column else None
    folds = []
    if time_column:
        if not np.isfinite(periods).all():
            raise ValueError('Chronological validation cannot use missing timestamps')
        unique = np.unique(periods)
        if len(unique) <= 30:
            valid_periods = unique[-2:]
            ranges = [(periods < value, periods == value) for value in valid_periods]
        else:
            boundaries = np.unique(np.quantile(periods, [.6, .8, 1.]))
            ranges = [(periods < low, (periods >= low) & (periods < high if i == 0 else periods <= high))
                      for i, (low, high) in enumerate(zip(boundaries[:-1], boundaries[1:]))]
        for fit_mask, valid_mask in ranges:
            tr, va = index[fit_mask], index[valid_mask]
            if len(tr) >= max(10, n*.25) and len(va) >= 2:
                folds.append({'train': tr, 'valid': va, 'label': f'past<{periods[va].min():g} -> next period'})
        if not folds:
            raise ValueError('Cannot build usable chronological folds; random fallback would violate temporal isolation')
        kind = 'time_rolling_origin'
    else:
        minimum = int(pd.Series(y).value_counts().min())
        if groups is not None and len(np.unique(groups)) >= 2:
            splits = min(4, len(np.unique(groups)))
            splitter = StratifiedGroupKFold(splits, shuffle=True, random_state=seed)
            outer = splitter.split(index, y, groups)
            kind = 'stratified_group_kfold'
        elif minimum >= 2:
            splitter = StratifiedKFold(min(4, minimum), shuffle=True, random_state=seed)
            outer = splitter.split(index, y)
            kind = 'stratified_kfold'
        else:
            outer = KFold(min(4, n//10), shuffle=True, random_state=seed).split(index)
            kind = 'kfold_rare_class_warning'
        folds = [{'train': tr, 'valid': va, 'label': f'fold{i}'} for i, (tr, va) in enumerate(outer)]
    for fold in folds:
        tr = fold['train']
        if time_column:
            values = np.unique(periods[tr])
            cutoff = values[-1] if len(values) <= 30 else np.quantile(periods[tr], .85)
            fit, early = tr[periods[tr] < cutoff], tr[periods[tr] >= cutoff]
        elif groups is not None and len(np.unique(groups[tr])) >= 2:
            a, b = next(GroupShuffleSplit(1, test_size=.15, random_state=seed).split(tr, groups=groups[tr]))
            fit, early = tr[a], tr[b]
        else:
            labels = np.asarray(y)[tr]
            counts = pd.Series(labels).value_counts()
            stratify = labels if counts.min() >= 2 and int(np.ceil(len(tr)*.15)) >= len(counts) else None
            fit, early = train_test_split(tr, test_size=.15, random_state=seed, stratify=stratify)
        if len(fit) < 5 or not len(early):
            fit, early = tr, np.array([], dtype=int)
        fold['fit'], fold['early_stop'] = np.asarray(fit), np.asarray(early)
    weights = np.array([len(fold['valid']) for fold in folds], float)
    return {'kind': kind, 'folds': folds, 'weights': (weights/weights.sum()).tolist(),
            'labels': [fold['label'] for fold in folds],
            'sizes': [{key: len(fold[key]) for key in ('train', 'fit', 'early_stop', 'valid')} for fold in folds]}


class Features:
    """Fit metadata and label history on exactly the declared source partition."""
    def __init__(self, frame, target, ids, n_classes, time_column=None, entity_key=None, group_columns=(), seed=42):
        self.frame, self.target, self.ids, self.n_classes = frame, target, ids, n_classes
        self.time_column, self.entity_key, self.group_columns, self.seed = time_column, entity_key, list(group_columns), seed

    def available(self):
        return {key: value for key, value in BLOCKS.items()
                if not (key == 'entity_history' and not (self.time_column and self.entity_key))
                and not (key in {'group_history', 'similar_history'} and not (self.time_column and self.group_columns))
                and not (key == 'oof_target_encoding' and (self.time_column or not self.group_columns))}

    def fit(self, source, y, blocks):
        self.source = np.asarray(source, dtype=int)
        self.y = np.asarray(y, dtype=int)  # ONLY labels indexed by source, never validation labels.
        if len(self.y) != len(self.source):
            raise ValueError('History labels must correspond only to source rows')
        self.blocks = list(dict.fromkeys(['base', *blocks]))
        if set(self.blocks)-set(self.available()):
            raise ValueError('Requested unavailable classification feature block')
        fit = self.frame.iloc[self.source]
        self.columns = [name for name in self.frame if name not in {self.target, *self.ids}]
        self.cats = [name for name in self.columns if not pd.api.types.is_numeric_dtype(fit[name]) and name != self.time_column]
        self.nums = [name for name in self.columns if name not in self.cats]
        self.categories = {name: list(pd.unique(fit[name].astype('string').fillna('__NA__'))) for name in self.cats}
        self.medians = {name: float(pd.to_numeric(fit[name], errors='coerce').replace([np.inf, -np.inf], np.nan).median())
                        for name in self.nums if name != self.time_column}
        self.missing = [name for name in self.columns if fit[name].isna().any()]
        self.logs = [name for name in self.nums if name != self.time_column
                     and pd.to_numeric(fit[name], errors='coerce').min() >= 0
                     and pd.to_numeric(fit[name], errors='coerce').skew() > 2]
        self.families = {}
        for name in self.nums:
            parts = name.split('_', 1)
            family = parts[1] if len(parts) == 2 else re.sub(r'\d+$', '', name)
            self.families.setdefault(family, []).append(name)
        self.families = {key: value for key, value in self.families.items() if len(value) >= 3}
        self.prior = (np.bincount(self.y, minlength=self.n_classes)+1)/(len(self.y)+self.n_classes)
        self.history_classes = np.argsort(-np.bincount(self.y, minlength=self.n_classes))[:32]
        numeric_key = [name for name in (self.entity_key or []) if name in self.nums and name != self.time_column]
        self.bucket_column = numeric_key[0] if numeric_key else next((name for name in self.nums if name != self.time_column and fit[name].nunique() > 20), None)
        if self.bucket_column:
            self.bucket_edges = np.unique(pd.to_numeric(fit[self.bucket_column], errors='coerce').dropna().quantile(np.linspace(0, 1, 11)))
        self.crossfit = {}
        if 'oof_target_encoding' in self.blocks:
            counts = pd.Series(self.y).value_counts()
            n = min(3, int(counts.min()))
            splitter = StratifiedKFold(n, shuffle=True, random_state=self.seed).split(self.source, self.y) if n >= 2 else KFold(min(3, len(self.source)), shuffle=True, random_state=self.seed).split(self.source)
            for a, b in splitter:
                for position in b:self.crossfit[int(self.source[position])] = a
        return self

    def _keys(self, rows, columns, similar=False):
        keys = self.frame.iloc[rows][columns].astype('string').fillna('__NA__').copy()
        if similar and self.bucket_column:
            values = pd.to_numeric(self.frame.iloc[rows][self.bucket_column], errors='coerce')
            keys['bucket'] = np.searchsorted(self.bucket_edges, values, side='right')
        return pd.util.hash_pandas_object(keys, index=False).to_numpy()

    def _history(self, rows, columns, prefix, similar=False):
        from ar_features import past_target_stats
        # This helper is unchanged regression code, used on binary indicators only.
        if self.time_column:
            union = np.unique(np.r_[self.source, rows])
            positions = pd.Index(union).get_indexer(rows)
            keys = self.frame.iloc[union][columns].reset_index(drop=True).astype('string').fillna('__NA__')
            if similar and self.bucket_column:
                keys['bucket'] = np.searchsorted(self.bucket_edges, pd.to_numeric(self.frame.iloc[union][self.bucket_column], errors='coerce'), side='right')
            periods = period_values(self.frame.iloc[union][self.time_column]).reset_index(drop=True)
            values = np.full(len(union), np.nan)
            source_positions = pd.Index(union).get_indexer(self.source)
            output = {}
            for label in self.history_classes:
                values[:] = np.nan
                values[source_positions] = (self.y == label).astype(float)
                history = past_target_stats(keys, periods, pd.Series(values), f'{prefix}_c{label}', full=prefix == 'entity')
                global_past = past_target_stats(pd.DataFrame({'all': np.zeros(len(union), dtype=int)}),
                    periods, pd.Series(values), 'global', full=False)['global_mean']
                probability = history[f'{prefix}_c{label}_mean'].fillna(global_past).fillna(1./self.n_classes)
                output[f'{prefix}_p{label}'] = probability.iloc[positions].to_numpy()
                if prefix == 'entity':
                    last = history[f'{prefix}_c{label}_last'].fillna(global_past).fillna(1./self.n_classes)
                    output[f'{prefix}_last_p{label}'] = last.iloc[positions].to_numpy()
                    output[f'{prefix}_trend_p{label}'] = (last-probability).iloc[positions].to_numpy()
                if label == self.history_classes[0]:
                    output[f'{prefix}_count'] = history[f'{prefix}_c{label}_n'].iloc[positions].to_numpy()
            return output
        source_keys, apply_keys = self._keys(self.source, columns), self._keys(rows, columns)
        result = {f'{prefix}_p{label}': np.zeros(len(rows)) for label in self.history_classes}
        partitions = {}
        for position, row in enumerate(rows):
            subset = self.crossfit.get(int(row))
            # Rows in one inner fold share the SAME index-array object. Hash a
            # constant-size identifier, not O(n_source) indices for every row.
            identifier = id(subset) if subset is not None else None
            partitions.setdefault(identifier, (subset, []))[1].append(position)
        for subset, positions in partitions.values():
            subset = subset if subset is not None else np.arange(len(self.source))
            labels = self.y[subset]
            prior = (np.bincount(labels, minlength=self.n_classes)+1)/(len(labels)+self.n_classes)
            data = pd.DataFrame({'key': source_keys[subset]})
            count = data.groupby('key').size()
            for label in self.history_classes:
                data['indicator'] = labels == label
                positive = data.groupby('key').indicator.sum()
                mapping = (positive+10*prior[label])/(count+10)
                result[f'{prefix}_p{label}'][positions] = pd.Series(apply_keys[positions]).map(mapping).fillna(prior[label]).to_numpy()
        return result

    def transform(self, rows):
        rows = np.asarray(rows, dtype=int)
        raw = self.frame.iloc[rows]
        output = pd.DataFrame(index=np.arange(len(rows)))
        for name in self.nums:
            values = period_values(raw[name]) if name == self.time_column else pd.to_numeric(raw[name], errors='coerce')
            median = self.medians.get(name, 0.)
            output[name] = values.replace([np.inf, -np.inf], np.nan).fillna(median if np.isfinite(median) else 0.).to_numpy()
        for name in self.cats:
            values = raw[name].astype('string').fillna('__NA__').to_numpy()
            output[name] = pd.Categorical(values, categories=self.categories[name])
            # Missing/unknown categories are a stable fit-derived sentinel for all engines.
            output[name] = output[name].cat.add_categories(['__UNKNOWN__']) if '__UNKNOWN__' not in output[name].cat.categories else output[name]
            output[name] = output[name].fillna('__UNKNOWN__')
        if 'missing' in self.blocks:
            output['n_missing'] = raw[self.columns].isna().sum(axis=1).to_numpy()
            for name in self.missing:output['isna__'+name] = raw[name].isna().to_numpy().astype(float)
        if 'log_numeric' in self.blocks:
            for name in self.logs:output['log__'+name] = np.log1p(output[name].clip(lower=0))
        if 'families' in self.blocks:
            for family, columns in self.families.items():
                for operation in ('mean', 'std', 'min', 'max'):
                    output[f'family__{family}__{operation}'] = getattr(output[columns], operation)(axis=1)
        if 'entity_history' in self.blocks:
            output = pd.concat([output, pd.DataFrame(self._history(rows, self.entity_key, 'entity'))], axis=1)
        if 'group_history' in self.blocks or 'oof_target_encoding' in self.blocks:
            for i, name in enumerate(self.group_columns[:4]):
                output = pd.concat([output, pd.DataFrame(self._history(rows, [name], f'group{i}'))], axis=1)
        if 'similar_history' in self.blocks:
            if self.bucket_column:
                output = pd.concat([output, pd.DataFrame(self._history(rows, [self.group_columns[0]], 'similar', True))], axis=1)
        return output.replace([np.inf, -np.inf], np.nan)

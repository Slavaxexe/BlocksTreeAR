"""Bounded tabular EDA and explicit, partition-local feature transformations."""
from typing import Literal
import hashlib
import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator


class Interaction(BaseModel):
    model_config = ConfigDict(extra='forbid')
    columns: list[str] = Field(min_length=2, max_length=2)
    operation: Literal['sum', 'difference', 'product', 'ratio', 'category_pair']


class FeaturePolicy(BaseModel):
    model_config = ConfigDict(extra='forbid')
    drop_columns: list[str] = Field(default_factory=list, max_length=32)
    drop_constants: bool = False
    drop_duplicates: bool = False
    drop_missing_above: float = Field(default=1., ge=.5, le=1.)
    correlation_threshold: float | None = Field(default=None, ge=.9, le=.999999)
    clip_quantile: float = Field(default=0., ge=0., le=.05)
    outlier_flags: bool = False
    missing_flags: bool = False
    categorical_columns: list[str] = Field(default_factory=list, max_length=24)
    rare_category_min_fraction: float = Field(default=0., ge=0., le=.05)
    frequency_columns: list[str] = Field(default_factory=list, max_length=12)
    absence_indicators: dict[str, list[str]] = Field(default_factory=dict)
    interactions: list[Interaction] = Field(default_factory=list, max_length=12)

    @model_validator(mode='after')
    def bounded_tokens(self):
        if len(self.absence_indicators) > 12 or any(len(values) > 12 for values in self.absence_indicators.values()):
            raise ValueError('Use at most 12 absence columns with at most 12 tokens each')
        return self


class FittedFeaturePolicy:
    def __init__(self, policy):
        self.policy = policy

    def fit(self, frame, raw=None):
        p = self.policy
        raw = frame if raw is None else raw
        referenced = set(p.drop_columns+p.categorical_columns+p.frequency_columns+list(p.absence_indicators))
        referenced.update(name for action in p.interactions for name in action.columns)
        unknown = referenced-set(frame)
        if unknown:raise ValueError(f'EDA policy references unavailable features: {sorted(unknown)}')
        self.input_columns = list(frame)
        removed = {name: 'explicit' for name in p.drop_columns}
        self.numeric = [name for name in frame if pd.api.types.is_numeric_dtype(frame[name]) and name not in p.categorical_columns]
        self.missing = {name: float(raw[name].isna().mean()) for name in frame if name in raw}
        for name in frame:
            if p.drop_constants and frame[name].nunique(dropna=False) <= 1:removed[name] = 'constant'
            if self.missing.get(name, 0.) > p.drop_missing_above:removed[name] = 'missing_fraction'
        if p.drop_duplicates:
            seen = {}
            for name in frame:
                if name in removed:continue
                signature = hashlib.sha256(pd.util.hash_pandas_object(frame[name], index=False).to_numpy().tobytes()).digest()
                if signature in seen and frame[name].equals(frame[seen[signature]]):removed[name] = 'duplicate_of:'+seen[signature]
                else:seen[signature] = name
        if p.correlation_threshold:
            numeric = [name for name in self.numeric if name not in removed][:80]
            subset = frame[numeric].sample(min(3000, len(frame)), random_state=42)
            correlations = subset.corr(method='spearman').abs()
            kept = []
            for name in numeric:
                if any(correlations.at[name, previous] >= p.correlation_threshold for previous in kept):
                    removed[name] = 'correlated_on_training_partition'
                else:kept.append(name)
        # A constant-only dataset still needs its measured prior baseline.
        if len(removed) == len(frame) and not p.interactions and not p.frequency_columns and len(frame.columns):
            if all(reason != 'explicit' for reason in removed.values()):removed.pop(frame.columns[0])
        self.removed = removed
        self.bounds = {}
        if p.clip_quantile or p.outlier_flags:
            quantile = p.clip_quantile or .01
            for name in self.numeric:
                values = pd.to_numeric(frame[name], errors='coerce').astype(float).replace([np.inf, -np.inf], np.nan)
                low, high = values.quantile([quantile, 1-quantile])
                if np.isfinite(low) and np.isfinite(high) and high > low:self.bounds[name] = (float(low), float(high))
        self.categories, self.frequent, self.category_markers = {}, {}, {}
        for name in frame:
            if name in p.categorical_columns or (p.rare_category_min_fraction and isinstance(frame[name].dtype, pd.CategoricalDtype)):
                values = frame[name].astype('string').fillna('__NA__')
                frequencies = values.value_counts(normalize=True)
                frequent = set(frequencies.index[frequencies >= p.rare_category_min_fraction])
                rare, unseen = '__EDA_RARE__', '__EDA_UNSEEN__'
                while rare in frequencies:rare += '_'
                while unseen in frequencies:unseen += '_'
                self.frequent[name] = frequent
                self.category_markers[name] = (rare, unseen)
                self.categories[name] = sorted(frequent)+[rare, unseen]
        self.frequencies = {name: frame[name].astype('string').fillna('__NA__').value_counts(normalize=True)
            for name in p.frequency_columns}
        self.pair_frequencies = {}
        for i, action in enumerate(p.interactions):
            if action.operation == 'category_pair':
                self.pair_frequencies[i] = self._pair(frame, action.columns).value_counts(normalize=True)
            elif any(name not in self.numeric for name in action.columns):
                raise ValueError('Arithmetic interactions require numeric columns not converted to categories')
        self.report = {'removed_columns': removed, 'clip_bounds': self.bounds,
            'categorical_overrides': p.categorical_columns, 'frequency_columns': p.frequency_columns,
            'interactions': [action.model_dump() for action in p.interactions],
            'scope': 'fit on this training partition only; rows and labels preserved'}
        return self

    @staticmethod
    def _pair(frame, columns):
        return pd.util.hash_pandas_object(frame[columns].astype('string').fillna('__NA__'), index=False)

    def transform(self, frame, raw=None):
        p = self.policy
        raw = frame if raw is None else raw
        output = frame.copy()
        def add(name, values):
            if name in output:raise ValueError(f'Generated EDA feature collides with an existing name: {name}')
            output[name] = np.asarray(values)
        if p.missing_flags:
            for name, fraction in self.missing.items():
                if fraction > 0 and name in raw:add('eda_missing__'+name, raw[name].isna().astype(float))
        for name, tokens in p.absence_indicators.items():
            add('eda_absence__'+name, frame[name].astype('string').isin(tokens).astype(float))
        for name, (low, high) in self.bounds.items():
            values = pd.to_numeric(output[name], errors='coerce')
            if p.outlier_flags:add('eda_outlier__'+name, ((values < low) | (values > high)).astype(float))
            if p.clip_quantile:output[name] = values.clip(low, high)
        for name, mapping in self.frequencies.items():
            add('eda_frequency__'+name, frame[name].astype('string').fillna('__NA__').map(mapping).fillna(0.).astype(float))
        for i, action in enumerate(p.interactions):
            left, right = action.columns
            if action.operation == 'category_pair':
                values = self._pair(frame, action.columns).map(self.pair_frequencies[i]).fillna(0.)
            else:
                a, b = pd.to_numeric(output[left]), pd.to_numeric(output[right])
                if action.operation == 'sum':values = a+b
                elif action.operation == 'difference':values = a-b
                elif action.operation == 'product':values = a*b
                else:values = a/b.replace(0, np.nan)
                values = values.replace([np.inf, -np.inf], np.nan).fillna(0.)
            add('eda_interaction__'+str(i), values)
        for name, categories in self.categories.items():
            values = frame[name].astype('string').fillna('__NA__')
            rare, unseen = self.category_markers[name]
            # All categories seen during fit are known, even if merged as rare.
            known = set(self.frequent[name])
            values = values.where(values.isin(known), rare if p.rare_category_min_fraction else unseen)
            output[name] = pd.Categorical(values, categories=categories)
        output = output.drop(columns=list(self.removed))
        if not len(output.columns):raise ValueError('EDA policy removed every model feature')
        return output


def extended_profile(train, test, columns, target=None, max_rows=5000):
    """Descriptions and hypotheses only; no global transformation is fitted here."""
    a = train.sample(min(max_rows, len(train)), random_state=42)
    b = test.sample(min(max_rows, len(test)), random_state=42)
    numeric, categorical, constants, high_missing = [], [], [], []
    for name in columns[:200]:
        values = a[name]
        missing = float(values.isna().mean())
        unique = int(values.nunique())
        if unique <= 1:constants.append(name)
        if missing >= .5:high_missing.append({'feature': name, 'fraction': missing})
        if pd.api.types.is_numeric_dtype(values):
            clean = pd.to_numeric(values, errors='coerce').astype(float).replace([np.inf, -np.inf], np.nan)
            quantiles = clean.quantile([.01, .25, .5, .75, .99])
            median = quantiles.loc[.5]
            numeric.append({'feature': name, 'unique': unique, 'missing': missing,
                'zero_fraction': float(values.eq(0).mean()), 'nonfinite': int((values.notna() & ~np.isfinite(values)).sum()),
                'q01': float(quantiles.loc[.01]) if np.isfinite(quantiles.loc[.01]) else None,
                'median': float(median) if np.isfinite(median) else None,
                'q99': float(quantiles.loc[.99]) if np.isfinite(quantiles.loc[.99]) else None,
                'skew': float(clean.skew()) if np.isfinite(clean.skew()) else None,
                'test_missing': float(b[name].isna().mean())})
        else:
            values = values.astype('string').fillna('__NA__');other = b[name].astype('string').fillna('__NA__')
            counts = values.value_counts(normalize=True)
            aa, bb = counts.align(other.value_counts(normalize=True), fill_value=0.)
            categorical.append({'feature': name, 'unique': unique, 'missing': missing,
                'top_values': {str(key): float(value) for key, value in counts.head(4).items()},
                'rare_mass_below_1pct': float(counts[counts < .01].sum()),
                'unseen_test_fraction': float((~other.isin(set(values))).mean()),
                'distribution_distance': float((aa-bb).abs().sum()/2),
                'mean_string_length': float(values.str.len().mean())})
    nums = [row['feature'] for row in numeric if row['unique'] > 1][:64]
    correlations, duplicates = [], []
    if nums:
        matrix = a[nums].replace([np.inf, -np.inf], np.nan).corr(method='spearman').abs()
        for i, name in enumerate(nums):
            for other in nums[:i]:
                value = matrix.at[name, other]
                if np.isfinite(value) and value >= .9:
                    correlations.append({'columns': [other, name], 'absolute_spearman': float(value)})
    hashes = {}
    for name in columns[:200]:
        signature = int(pd.util.hash_pandas_object(a[name], index=False).sum())
        for other in hashes.get(signature, []):
            if a[name].equals(a[other]):duplicates.append([other, name])
        hashes.setdefault(signature, []).append(name)
    separation = []
    if target and a[target].nunique() >= 2:
        for name in nums:
            values = pd.to_numeric(a[name], errors='coerce').replace([np.inf, -np.inf], np.nan)
            frame = pd.DataFrame({'x': values, 'label': a[target]})
            groups = frame.groupby('label', observed=True).x.agg(['mean', 'count'])
            variance = float(((values-values.mean())**2).sum())
            association = float((((groups['mean']-values.mean())**2)*groups['count']).sum()/variance) if variance > 0 else 0.
            separation.append({'feature': name, 'between_class_variance_fraction': association})
    return {'sample_rows': len(a), 'features_analyzed': min(200, len(columns)),
        'omitted_features': max(0, len(columns)-200), 'constants': constants, 'duplicate_columns': duplicates[:20],
        'high_missing': high_missing[:20], 'numeric': numeric[:80], 'categorical': categorical[:40],
        'correlated_pairs': sorted(correlations, key=lambda row: -row['absolute_spearman'])[:24],
        'class_separation': sorted(separation, key=lambda row: -row['between_class_variance_fraction'])[:16],
        'scope': 'sampled EDA hypotheses; target association only on supplied training partition; never an instruction to delete labels/rows'}


def compact_profile(profile):
    if not profile:return None
    result = dict(profile)
    result['numeric'] = sorted(profile['numeric'], key=lambda row:
        -(abs(row['skew'] or 0)+5*row['missing']+row['nonfinite']))[:16]
    result['categorical'] = sorted(profile['categorical'], key=lambda row:
        -(row['unseen_test_fraction']+row['rare_mass_below_1pct']))[:12]
    result['omitted_numeric_profiles'] = max(0, len(profile['numeric'])-16)
    result['omitted_categorical_profiles'] = max(0, len(profile['categorical'])-12)
    return result


def confusion_evidence(frame, labels, classes, pairs):
    """Describe separability of confused classes using supplied FIT labels only."""
    from scipy.stats import ks_2samp
    names = {str(label): i for i, label in enumerate(classes)}
    result = []
    for pair in pairs[:3]:
        left, right = names[pair['true']], names[pair['predicted']]
        a, b = frame.loc[labels == left], frame.loc[labels == right]
        if min(len(a), len(b)) < 2:continue
        comparisons = []
        for name in frame.columns[:80]:
            if pd.api.types.is_numeric_dtype(frame[name]):
                x = pd.to_numeric(a[name]).replace([np.inf, -np.inf], np.nan).dropna()
                y = pd.to_numeric(b[name]).replace([np.inf, -np.inf], np.nan).dropna()
                if not len(x) or not len(y):continue
                comparisons.append({'feature': name, 'distance': float(ks_2samp(x, y).statistic),
                    'median_true_class': float(x.median()), 'median_confused_class': float(y.median()),
                    'zero_fraction_true': float(x.eq(0).mean()), 'zero_fraction_confused': float(y.eq(0).mean())})
            else:
                x, y = a[name].value_counts(normalize=True), b[name].value_counts(normalize=True)
                x, y = x.align(y, fill_value=0.)
                comparisons.append({'feature': name, 'distance': float((x-y).abs().sum()/2)})
        result.append({'true': pair['true'], 'confused_with': pair['predicted'],
            'fit_support': [len(a), len(b)], 'features': sorted(comparisons, key=lambda row: -row['distance'])[:5]})
    return {'pairs': result, 'scope': 'diagnostic sample of first outer-fold FIT only; no validation or test labels'}

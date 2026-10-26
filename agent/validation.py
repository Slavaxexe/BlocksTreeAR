"""Shared metadata and structured holdouts; outer split indices remain unchanged."""
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split


SPLIT_CATALOG = {
    'random': 'Seeded row holdout.',
    'stratified': 'Preserve strata proportions; stratify_columns defaults to targets. Optional stratify_bins for numeric targets.',
    'group': 'Keep complete groups on one side; composite group_columns are supported.',
    'stratified_group': 'Choose a grouped holdout balancing strata and requested size; size is approximate.',
    'time': 'Earlier-to-later holdout by time_column; choose latest validation_periods, cutoff, validation_rows or fraction of periods. gap_periods excludes adjacent earlier periods.',
    'group_time': 'Time holdout, then exclude from fit every group occurring in validation. Can be infeasible for fully repeated panels.',
    'ordered': 'Hold out the last validation_rows or fraction of rows in supplied order. gap_rows excludes adjacent fit rows.',
}


def query_range_evidence(train, query, excluded=()):
    """Measure separated numeric/date ranges without deciding what any column means."""
    result = []
    for column in train:
        if column in excluded or column not in query:
            continue
        a, b = train[column], query[column]
        numeric = pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b)
        dated = pd.api.types.is_datetime64_any_dtype(a) and pd.api.types.is_datetime64_any_dtype(b)
        if pd.api.types.is_bool_dtype(a) or not (numeric or dated):
            continue
        if numeric:
            a, b = a.replace([np.inf, -np.inf], np.nan), b.replace([np.inf, -np.inf], np.nan)
        else:
            a, b = pd.to_datetime(a, utc=True), pd.to_datetime(b, utc=True)
        a, b = a.dropna(), b.dropna()
        if not len(a) or not len(b):
            continue
        side = 'above_train' if b.min() > a.max() else 'below_train' if b.max() < a.min() else None
        if side:
            value = float if numeric else str
            result.append({'column': column, 'query_range': side, 'train_min': value(a.min()),
                'train_max': value(a.max()), 'query_min': value(b.min()), 'query_max': value(b.max()),
                'train_distinct': int(a.nunique()), 'query_distinct': int(b.nunique()),
                'qualification': 'Measured range separation, not proof of chronology; interpret using task semantics'})
    return result


def metadata(policy):
    policy = dict(policy or {})
    column = policy.get('time_column') or policy.get('split_column') or policy.get('year_column')
    if column:
        # Preserve aliases: generated prepare.py may still read the original key.
        policy['time_column'] = column
    return policy


def strategy(policy):
    description = str(policy.get('strategy') or policy.get('method') or '').lower()
    timed = any(word in description for word in ('time', 'temporal', 'chronological'))
    if policy.get('group_columns') or 'group' in description:
        return 'group_time' if timed else 'stratified_group' if 'stratif' in description else 'group'
    if timed:
        return 'time'
    if 'stratif' in description:
        return 'stratified'
    return 'ordered' if 'ordered' in description else 'random'


def split_plan(policy, targets=()):
    from .plans import SplitPlan
    policy = metadata(policy.get('split_plan', policy))
    description = str(policy.get('strategy') or policy.get('method') or 'random').lower().replace('-', '_')
    aliases = {'ordered_tail': 'ordered', 'time_based': 'time', 'temporal': 'time', 'chronological': 'time',
               'stratified_random': 'stratified', 'random_stratified': 'stratified',
               'grouped': 'group', 'group_shuffle': 'group', 'group_holdout': 'group',
               'time_group': 'group_time', 'temporal_group': 'group_time',
               'random_holdout': 'random', 'random_split': 'random', 'holdout': 'random'}
    kind = aliases.get(description, description)
    if kind not in SPLIT_CATALOG:
        raise ValueError(f'Unsupported split strategy: {description}. Choose from {list(SPLIT_CATALOG)}')
    settings = {key: value for key, value in policy.items() if key in SplitPlan.model_fields}
    settings['strategy'] = kind
    settings['fraction'] = policy.get('fraction', policy.get('validation_fraction', policy.get('holdout_fraction', .2)))
    if 'test_size' in policy:
        settings['validation_rows' if type(policy['test_size']) is int else 'fraction'] = policy['test_size']
    if 'rows' in policy:
        settings['validation_rows'] = policy['rows']
    if settings['strategy'] in {'stratified', 'stratified_group'} and not settings.get('stratify_columns'):
        settings['stratify_columns'] = list(targets)
    return SplitPlan.model_validate(settings)


def split_rows(frame, plan, targets=()):
    """Execute a declared plan. Every returned number is a position in the original frame."""
    from .plans import SplitPlan
    from .supervised_features import ordered_time
    plan = plan if isinstance(plan, SplitPlan) else split_plan(plan, targets)
    rows = np.arange(len(frame), dtype=np.int64)
    count = plan.validation_rows or int(np.ceil(len(rows) * plan.fraction))
    if not 0 < count < len(rows):
        raise ValueError('Validation size must leave nonempty fit and validation')
    columns = [*plan.group_columns, *plan.stratify_columns, *([plan.time_column] if plan.time_column else [])]
    if set(columns) - set(frame):
        raise ValueError(f'Split columns are absent: {sorted(set(columns) - set(frame))}')
    if set([*plan.group_columns, plan.time_column]) & set(targets):
        raise ValueError('Targets cannot define split time or isolation groups; use stratify_columns')

    if plan.strategy in {'time', 'group_time'}:
        times = ordered_time(frame[plan.time_column])
        periods = np.unique(times)
        if len(periods) < 2:
            raise ValueError('Temporal holdout needs at least two distinct periods')
        if plan.cutoff is not None:
            cut = ordered_time(pd.Series([plan.cutoff]))[0]
        elif plan.validation_periods is not None:
            if plan.validation_periods >= len(periods):
                raise ValueError('validation_periods must leave an earlier training period')
            cut = periods[-plan.validation_periods]
        elif plan.validation_rows is not None:
            cut = np.sort(times)[-count]
        else:
            cut = periods[max(1, int(len(periods) * (1 - plan.fraction)))]
        earlier = periods[periods < cut]
        if len(earlier) <= plan.gap_periods:
            raise ValueError('Time cutoff/gap leaves no training periods')
        fit = rows[times <= earlier[-1 - plan.gap_periods]]
        valid = rows[times >= cut]
        if plan.strategy == 'group_time':
            groups = pd.factorize(pd.MultiIndex.from_frame(frame[plan.group_columns]), sort=False)[0]
            fit = fit[~np.isin(groups[fit], groups[valid])]
    elif plan.strategy == 'ordered':
        fit, valid = rows[:max(0, len(rows) - count - plan.gap_rows)], rows[-count:]
    elif plan.strategy == 'group':
        groups = pd.factorize(pd.MultiIndex.from_frame(frame[plan.group_columns]), sort=False)[0]
        fit, valid = next(GroupShuffleSplit(n_splits=1, test_size=count / len(rows), random_state=plan.seed).split(rows, groups=groups))
    elif plan.strategy in {'stratified', 'stratified_group'}:
        if not plan.stratify_columns:
            raise ValueError('Stratified split needs stratify_columns or task targets')
        strata = frame[plan.stratify_columns]
        if plan.stratify_bins is not None:
            if len(plan.stratify_columns) != 1:
                raise ValueError('stratify_bins requires one numeric stratification column')
            strata = pd.qcut(strata.iloc[:, 0], plan.stratify_bins, duplicates='drop').to_frame()
        labels = pd.factorize(pd.MultiIndex.from_frame(strata), sort=False)[0]
        counts = pd.Series(labels).value_counts()
        fixed = rows[pd.Series(labels).map(counts).to_numpy() < 2]
        if len(fixed) and plan.rare_class_policy == 'error':
            raise ValueError('Singleton strata cannot occur on both sides; choose train_only, bins or another strategy')
        if plan.strategy == 'stratified':
            available = np.setdiff1d(rows, fixed)
            fit, valid = train_test_split(available, test_size=count, random_state=plan.seed, stratify=labels[available])
            fit = np.union1d(fit, fixed)
        else:
            groups = pd.factorize(pd.MultiIndex.from_frame(frame[plan.group_columns]), sort=False)[0]
            label_groups = pd.DataFrame({'label': labels, 'group': groups}).drop_duplicates().groupby('label').size()
            rare = label_groups.index[label_groups < 2]
            if len(rare) and plan.rare_class_policy == 'error':
                raise ValueError('Some strata occur in only one group; choose train_only or another strategy')
            fixed = np.union1d(fixed, rows[np.isin(labels, rare)])
            fixed = rows[np.isin(groups, groups[fixed])]
            available = np.setdiff1d(rows, fixed)
            if len(available) <= count:
                raise ValueError('Too few rows remain for stratified group validation')
            target_rates = np.bincount(labels[available]) / len(available)
            candidates = GroupShuffleSplit(n_splits=20, test_size=count / len(available), random_state=plan.seed)
            def loss(indices):
                valid_rows = available[indices[1]]
                rates = np.bincount(labels[valid_rows], minlength=len(target_rates)) / len(valid_rows)
                return abs(len(valid_rows) / len(rows) - count / len(rows)) + abs(rates - target_rates).sum()
            chosen = min(candidates.split(available, groups=groups[available]), key=loss)
            fit, valid = np.union1d(available[chosen[0]], fixed), available[chosen[1]]
    else:
        fit, valid = train_test_split(rows, test_size=count, random_state=plan.seed)
    if not len(fit) or not len(valid):
        raise ValueError('Split constraints leave an empty fit or validation; change the plan')
    return np.sort(fit), np.sort(valid)


def structured_holdout(frame, policy, fraction, seed, kind='auto'):
    """Return group/time/row-order indices, or None for a random split."""
    policy = metadata(policy)
    kind = strategy(policy) if kind == 'auto' else kind
    if kind in {'group', 'time', 'ordered', 'group_time', 'stratified_group'} or kind == 'stratified' and policy.get('stratify_columns'):
        # Inner splits use their own size/cutoff, never absolute boundaries of the outer validation.
        settings = {key: policy[key] for key in ('time_column', 'group_columns', 'stratify_columns',
            'stratify_bins', 'rare_class_policy', 'gap_periods', 'gap_rows') if key in policy}
        if kind not in {'group', 'stratified_group', 'group_time'}:
            settings.pop('group_columns', None)
        if kind not in {'stratified', 'stratified_group'}:
            settings.pop('stratify_columns', None)
            settings.pop('stratify_bins', None)
        if kind not in {'time', 'group_time'}:
            settings.pop('gap_periods', None)
        if kind != 'ordered':
            settings.pop('gap_rows', None)
        return split_rows(frame, {**settings, 'strategy': kind, 'fraction': fraction, 'seed': seed})
    return None

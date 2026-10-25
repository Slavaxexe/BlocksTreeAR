"""Feature recipes: learn statistics on fit, then transform all frames identically."""
import numpy as np
import pandas as pd

from .plans import FeatureSpec


SUPERVISED_GUIDE = (
    'history: columns=single/composite key; parameters={time_column,value_column,'
    'aggregation:last/mean/sum/std/min/max/count/trend/age,window:null or number of distinct periods,'
    'lag:1,min_history:1,max_age:null,fallback:missing/global,class_label:null}. '
    'value_column may be a declared target ONLY here: every row uses fit values with strictly earlier time, '
    'excluding the same period and future; duplicate key/time values are aggregated. '
    'target_encoding: columns=key, parameters={target,smoothing:10,folds:5,seed:42,validation:auto/random/group/time,'
    'group_column:null,time_column:null,class_label:null}. Training rows are encoded out of fold; '
    'queries use fit only; temporal encoding uses strictly earlier data. Use class_label for categorical targets. '
    'AutoML supports these recipes with holdout and bag_folds=stack_levels=0; '
    'feature statistics are rebuilt on inner train, then the selected model is refit on all fit. '
    'FLAML internal ensemble must be false; saved ensembles remain available. Time/group inputs cannot be targets.')


def create_features(frames, recipes, *, forbidden=(), policy=None):
    from .validation import metadata
    policy = metadata(policy)
    frames = [frame.reset_index(drop=True).copy() for frame in frames]
    recipes = [value if isinstance(value, FeatureSpec) else FeatureSpec.model_validate(value) for value in recipes]
    inherited = frames[0].attrs.get('feature_recipes', [])
    for value in recipes:
        spec = value if isinstance(value, FeatureSpec) else FeatureSpec.model_validate(value)
        columns, op, parameters = spec.columns, spec.operation, spec.parameters
        if set(columns) & set(forbidden) or spec.name in forbidden:
            raise ValueError('Feature recipes cannot read or overwrite targets')
        if spec.name in frames[0]:
            raise ValueError(f'Feature {spec.name!r} already exists; choose a new name')
        if not set(columns) <= set(frames[0]):
            raise ValueError(f'Unknown feature inputs: {sorted(set(columns) - set(frames[0]))}')
        fit = frames[0]
        if op in {'history', 'target_encoding'}:
            from .supervised_features import history, target_encoding
            protected = {parameters.get('time_column'), parameters.get('group_column')} & set(forbidden)
            if protected:
                raise ValueError('Target cannot define a time or grouping column')
            if op == 'target_encoding' and parameters['target'] not in forbidden:
                raise ValueError('Target encoding must name a declared task target')
            results = (history(frames, columns, parameters) if op == 'history' else
                       target_encoding(frames, columns, parameters, policy or {}))
            for frame, result in zip(frames, results):
                frame[spec.name] = np.where(np.isfinite(result), result, np.nan)
            continue
        if op == 'expression':
            from .feature_expression import FeatureExpression
            expression = FeatureExpression(parameters['expression'], columns).fit(fit)
        elif op == 'frequency':
            frequencies = fit[columns[0]].astype('string').fillna('__missing__').value_counts(normalize=True)
        elif op == 'clip':
            bounds = pd.to_numeric(fit[columns[0]], errors='coerce').quantile(
                [parameters.get('lower', 0.01), parameters.get('upper', 0.99)]).to_numpy()
        elif op in {'group_mean', 'group_std', 'group_min', 'group_max', 'group_nunique', 'group_count', 'group_ratio', 'group_diff'}:
            group_key = fit[columns[0]].astype('string').fillna('__missing__')
            if op == 'group_count':
                stat_map = group_key.value_counts()
                fallback = 0
            elif op == 'group_nunique':
                if len(columns) != 2:
                    raise ValueError('group_nunique needs [group_column, target_column]')
                val_key = fit[columns[1]].astype('string').fillna('__missing__')
                stat_map = val_key.groupby(group_key).nunique()
                fallback = 1
            else:
                if len(columns) != 2:
                    raise ValueError(f'{op} needs [group_column, numeric_column]')
                values = pd.to_numeric(fit[columns[1]], errors='coerce')
                grouped = values.groupby(group_key)
                if op in {'group_mean', 'group_ratio', 'group_diff'}:
                    stat_map = grouped.mean()
                    fallback = values.mean()
                elif op == 'group_std':
                    stat_map = grouped.std().fillna(0)
                    fallback = 0.0 if np.isnan(values.std()) else values.std()
                elif op == 'group_min':
                    stat_map = grouped.min()
                    fallback = values.min()
                elif op == 'group_max':
                    stat_map = grouped.max()
                    fallback = values.max()
        for frame in frames:
            if op == 'expression':
                result = expression.transform(frame)
            elif op == 'missing':
                result = frame[columns].isna().any(axis=1).astype(int)
            elif op == 'frequency':
                result = frame[columns[0]].astype('string').fillna('__missing__').map(frequencies).fillna(0)
            elif op in {'group_mean', 'group_std', 'group_min', 'group_max', 'group_nunique', 'group_count', 'group_ratio', 'group_diff'}:
                grp = frame[columns[0]].astype('string').fillna('__missing__')
                mapped = grp.map(stat_map).fillna(fallback)
                if op == 'group_diff':
                    num = pd.to_numeric(frame[columns[1]], errors='coerce')
                    result = num - mapped
                elif op == 'group_ratio':
                    num = pd.to_numeric(frame[columns[1]], errors='coerce')
                    result = num / mapped.replace(0, np.nan)
                else:
                    result = mapped
            elif op == 'datetime':
                part = parameters.get('part', 'month')
                result = getattr(pd.to_datetime(frame[columns[0]], errors='coerce').dt, part)
            else:
                numbers = frame[columns].apply(pd.to_numeric, errors='coerce')
                if op in {'sum', 'mean', 'product'}:
                    result = numbers.agg({'sum': 'sum', 'mean': 'mean', 'product': 'prod'}[op], axis=1)
                elif op in {'difference', 'ratio'}:
                    if len(columns) != 2:
                        raise ValueError(f'{op} needs exactly two columns')
                    a, b = numbers.iloc[:, 0], numbers.iloc[:, 1]
                    result = a - b if op == 'difference' else a / b.replace(0, np.nan)
                elif op == 'signed_log':
                    x = numbers.iloc[:, 0]
                    result = np.sign(x) * np.log1p(np.abs(x))
                else:  # clip
                    result = numbers.iloc[:, 0].clip(*bounds)
            frame[spec.name] = result.replace([np.inf, -np.inf], np.nan).to_numpy()
    for frame in frames:
        frame.attrs['feature_recipes'] = inherited + [recipe.model_dump() for recipe in recipes]
    return frames

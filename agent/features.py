"""Feature recipes: learn statistics on fit, then transform all frames identically."""
import numpy as np
import pandas as pd

from .plans import FeatureSpec


def create_features(frames, recipes, *, forbidden=()):
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
        if op == 'frequency':
            frequencies = fit[columns[0]].astype('string').fillna('__missing__').value_counts(normalize=True)
        elif op == 'clip':
            bounds = pd.to_numeric(fit[columns[0]], errors='coerce').quantile(
                [parameters.get('lower', 0.01), parameters.get('upper', 0.99)]).to_numpy()
        elif op == 'group_mean':
            if len(columns) != 2:
                raise ValueError('group_mean needs [group_column, numeric_column]')
            values = pd.to_numeric(fit[columns[1]], errors='coerce')
            averages = values.groupby(fit[columns[0]].astype('string').fillna('__missing__')).mean()
            fallback = values.mean()
        for frame in frames:
            if op == 'missing':
                result = frame[columns].isna().any(axis=1).astype(int)
            elif op == 'frequency':
                result = frame[columns[0]].astype('string').fillna('__missing__').map(frequencies).fillna(0)
            elif op == 'group_mean':
                result = frame[columns[0]].astype('string').fillna('__missing__').map(averages).fillna(fallback)
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

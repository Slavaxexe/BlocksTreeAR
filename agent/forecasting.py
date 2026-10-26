"""Recursive forecasts on ordered observations, with independently keyed series."""
import copy
import time

import numpy as np
import pandas as pd

from .eda import _matrix
from .supervised_features import ordered_time


def forecast(state, parameters, defaults):
    from .operations import _estimator, _prediction, _preprocess
    from .task_types import fit_targets
    binding = state['preparation']['data']
    targets = binding['target_columns']
    time_column = parameters.get('time_column') or binding.get('time_column')
    series = parameters.get('series_columns', binding.get('series_columns', []))
    if not targets or not time_column:
        raise ValueError('Forecast requires numeric target_columns and time_column')
    lags = sorted(set(parameters.get('lags', [1])))
    windows = sorted(set(parameters.get('rolling_windows', [])))
    if not lags and not windows:
        raise ValueError('Choose at least one lag or rolling window for recursive forecasting')
    columns = parameters.get('feature_columns') or [column for column in state['fit']
        if column not in targets + binding.get('id_columns', [])]
    if set(columns) & set(targets):
        raise ValueError('Known future features cannot include targets')
    history, last_time = {}, {}
    keep = max([*lags, *windows])
    runtime = state['preparation'].get('runtime', {})

    def key_for(row):
        return tuple(None if pd.isna(row[column]) else row[column] for column in series)

    def design(frame, memory):
        rows = []
        for _, row in frame.iterrows():
            key = key_for(row)
            past = memory.get(key, [])
            values = {column: row[column] for column in columns}
            if time_column in values:
                values[time_column] = float(row[time_column])
            for index, target in enumerate(targets):
                for lag in lags:
                    values[f'__forecast_{index}_lag_{lag}'] = past[-lag][index] if len(past) >= lag else np.nan
                for window in windows:
                    values[f'__forecast_{index}_mean_{window}'] = (
                        float(np.mean([point[index] for point in past[-window:]])) if past else np.nan)
            rows.append(values)
        return pd.DataFrame(rows, index=frame.index)

    def ordered(frame):
        if frame.duplicated([*series, time_column]).any():
            raise ValueError('Forecast requires unique series/time rows; aggregate duplicate observations first')
        times = ordered_time(frame[time_column])
        return times, np.argsort(times, kind='stable')

    fit = state['fit'].reset_index(drop=True)
    y = fit[targets].to_numpy(dtype=float)
    if not np.isfinite(y).all():
        raise ValueError('Forecast training targets must be finite numeric values')
    times, order = ordered(fit)
    sorted_fit = fit.iloc[order]
    groups = sorted_fit.groupby(series if series else lambda index: 0, sort=False, dropna=False)
    training = fit[columns].copy()
    if time_column in training:
        training[time_column] = times
    # Unique series/time rows make these shifts strictly historical; never use the current target.
    for index, target in enumerate(targets):
        for lag in lags:
            training[f'__forecast_{index}_lag_{lag}'] = groups[target].shift(lag)
        for window in windows:
            training[f'__forecast_{index}_mean_{window}'] = groups[target].transform(
                lambda values: values.shift(1).rolling(window, min_periods=1).mean())
    for _, group in groups:
        key = key_for(group.iloc[-1])
        history[key] = list(group[targets].to_numpy(dtype=float)[-keep:])
        last_time[key] = times[group.index[-1]]
    input_columns = list(training)
    training[targets] = fit[targets]
    # Fit preprocessing once; recursion only transforms new rows.
    staged = {**state, 'fit': training,
              'validation': training.drop(columns=targets).iloc[:1],
              'test': training.drop(columns=targets).iloc[:1]}
    prepared = _preprocess(staged, {'feature_columns': input_columns})
    model = _estimator(parameters.get('estimator', 'sklearn.ensemble.HistGradientBoostingRegressor'),
                       parameters.get('parameters', {}), runtime)
    model = fit_targets(model, prepared['fit_features'], y[:, 0] if len(targets) == 1 else y,
                        None, [], False, runtime)
    transformer = prepared['transformer']
    result = {}
    for scope in ('validation', 'test'):
        query = state[scope].reset_index(drop=True)
        query_times, order = ordered(query)
        query[time_column] = query_times
        memory = copy.deepcopy(history)
        predictions = np.empty((len(query), len(targets)))
        # Build time batches once, instead of scanning the whole query for every timestamp.
        batches = pd.Series(query_times).groupby(query_times, sort=True).groups
        for timestamp, positions in batches.items():
            if time.monotonic() >= runtime.get('deadline', float('inf')):
                raise TimeoutError('Forecast node budget exhausted')
            positions = np.asarray(positions)
            batch = query.iloc[positions]
            if any(timestamp <= last_time.get(key_for(row), -np.inf) for _, row in batch.iterrows()):
                raise ValueError('Forecast queries must follow the observed fit history for each series')
            x = transformer.transform(_matrix(design(batch, memory), input_columns))
            values = np.asarray(model.predict(x)).reshape(len(batch), len(targets))
            predictions[positions] = values
            for (_, row), value in zip(batch.iterrows(), values):
                key = key_for(row)
                memory.setdefault(key, []).append(value)
                memory[key] = memory[key][-keep:]
        result[scope + '_predictions'] = _prediction(predictions, state, defaults)
    return result

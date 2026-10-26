"""Target-derived features with explicit training boundaries and no query-label reads."""
from collections import deque
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold

from .plans import HistorySettings, TargetEncodingSettings


def ordered_time(values):
    """Numeric periods keep their units; timestamps become days."""
    numeric = pd.to_numeric(values, errors='coerce')
    if numeric.notna().all() and not pd.api.types.is_datetime64_any_dtype(values):
        result = numeric.to_numpy(dtype=float)
    else:
        dates = pd.to_datetime(values, errors='raise', utc=True)
        if dates.isna().any():
            raise ValueError('Ordered features require non-missing times')
        result = dates.to_numpy(dtype='datetime64[ns]').astype('int64').astype(float) / (86400 * 1e9)
    if not np.isfinite(result).all():
        raise ValueError('Ordered features require finite times')
    return result


def key_codes(fit, query, columns):
    # Factorize actual tuples; concatenated strings can confuse different composite keys.
    keys = pd.MultiIndex.from_frame(pd.concat([fit[columns], query[columns]], ignore_index=True))
    codes = pd.factorize(keys, sort=False)[0]
    return codes[:len(fit)], codes[len(fit):]


def numeric_values(fit, column, class_label):
    if class_label is not None:
        return fit[column].eq(class_label).where(fit[column].notna()).to_numpy(dtype=float, na_value=np.nan)
    return pd.to_numeric(fit[column], errors='raise').to_numpy(dtype=float, na_value=np.nan)


def _range_extreme(values, start, end, maximum):
    queue, cursor = deque(), 0
    result = np.full(len(end), np.nan)
    for index in np.lexsort((start, end)):
        while cursor < end[index]:
            while queue and (values[queue[-1]] <= values[cursor] if maximum else values[queue[-1]] >= values[cursor]):
                queue.pop()
            queue.append(cursor)
            cursor += 1
        while queue and queue[0] < start[index]:
            queue.popleft()
        if queue:
            result[index] = values[queue[0]]
    return result


def _past(source, query_keys, query_time, settings):
    """Aggregate distinct source periods, then look strictly to their left."""
    table = source.groupby(['key', 'time'], sort=True).value.agg(['count', 'sum', 'min', 'max'])
    square = source.assign(square=source.value ** 2).groupby(['key', 'time']).square.sum()
    table['square'] = square
    result = np.full(len(query_time), np.nan)
    query_groups = pd.Series(np.arange(len(query_keys))).groupby(query_keys).groups
    for key, group in table.groupby(level=0, sort=False):
        positions = query_groups.get(key)
        if positions is None:
            continue
        positions = np.asarray(positions)
        times = group.index.get_level_values('time').to_numpy(dtype=float)
        end = np.searchsorted(times, query_time[positions], side='left') - (settings.lag - 1)
        end = np.maximum(end, 0)
        start = np.zeros(len(end), dtype=int) if settings.window is None else np.maximum(0, end - settings.window)
        if settings.max_age is not None:
            start = np.maximum(start, np.searchsorted(times, query_time[positions] - settings.max_age, side='left'))
        start = np.minimum(start, end)

        def interval(values):
            prefix = np.r_[0., np.cumsum(values)]
            return prefix[end] - prefix[start]

        counts, sums = group['count'].to_numpy(), group['sum'].to_numpy()
        count, total = interval(counts), interval(sums)
        usable = (end > start) & (count >= settings.min_history)
        last = np.maximum(0, end - 1)
        with np.errstate(divide='ignore', invalid='ignore'):
            mean = total / count
            operation = settings.aggregation
            if operation == 'last':
                values = sums[last] / counts[last]
            elif operation == 'mean':
                values = mean
            elif operation == 'sum':
                values = total
            elif operation == 'count':
                values = count
            elif operation == 'age':
                values = query_time[positions] - times[last]
            elif operation == 'std':
                values = np.sqrt(np.maximum(0., interval(group.square.to_numpy()) / count - mean ** 2))
            elif operation == 'trend':
                t = times - times[0]
                st, stt, sty = interval(t * counts), interval(t * t * counts), interval(t * sums)
                values = (sty - st * mean) / (stt - st * st / count)
            else:
                array = group[operation].to_numpy()
                values = _range_extreme(array, start, end, operation == 'max')
        result[positions] = np.where(usable, values, np.nan)
    return result


def history(frames, columns, parameters):
    settings = HistorySettings.model_validate(parameters)
    fit, query = frames[0], pd.concat(frames, ignore_index=True)
    source_keys, query_keys = key_codes(fit, query, columns)
    source_time, query_time = ordered_time(fit[settings.time_column]), ordered_time(query[settings.time_column])
    values = numeric_values(fit, settings.value_column, settings.class_label)
    source = pd.DataFrame({'key': source_keys, 'time': source_time, 'value': values}).dropna(subset=['value'])
    source = source.loc[np.isfinite(source.value)]
    result = _past(source, query_keys, query_time, settings)
    if settings.fallback == 'global':
        global_values = _past(source.assign(key=0), np.zeros(len(query), dtype=int), query_time, settings)
        result = np.where(np.isfinite(result), result, global_values)
    return np.split(result, np.cumsum([len(frame) for frame in frames])[:-1])


def _encode(fit, query, columns, values, smoothing):
    source_keys, query_keys = key_codes(fit, query, columns)
    statistics = pd.DataFrame({'key': source_keys, 'value': values}).groupby('key').value.agg(['sum', 'count'])
    prior = float(np.nanmean(values))
    mapping = (statistics['sum'] + smoothing * prior) / (statistics['count'] + smoothing)
    return pd.Series(query_keys).map(mapping).fillna(prior).to_numpy()


def target_encoding(frames, columns, parameters, policy):
    from .validation import metadata, strategy as validation_strategy
    policy = metadata(policy)
    settings = TargetEncodingSettings.model_validate(parameters)
    fit = frames[0]
    values = numeric_values(fit, settings.target, settings.class_label)
    strategy = settings.validation
    if strategy == 'auto':
        strategy = validation_strategy(policy)
    if strategy == 'ordered':
        column = '__encoding_position__'
        while any(column in frame for frame in frames):
            column += '_'
        frames = [frame.assign(**{column: np.arange(len(frame)) + (len(fit) if index else 0)})
                  for index, frame in enumerate(frames)]
        settings = settings.model_copy(update={'time_column': column})
        strategy = 'time'
    if strategy == 'time':
        time_column = settings.time_column or policy.get('time_column') or policy.get('split_column')
        if not time_column:
            raise ValueError('Temporal target encoding needs time_column')
        arguments = {'time_column': time_column, 'value_column': settings.target,
                     'class_label': settings.class_label}
        means = history(frames, columns, {**arguments, 'aggregation': 'mean'})
        counts = history(frames, columns, {**arguments, 'aggregation': 'count'})
        # A global past-only prior: the full-fit mean would reveal future target values.
        global_frames = [frame.assign(__encoding_global_key__=0) for frame in frames]
        priors = history(global_frames, ['__encoding_global_key__'], {**arguments, 'aggregation': 'mean'})
        result = []
        for mean, count, prior in zip(means, counts, priors):
            count = np.nan_to_num(count)
            smoothed = (np.nan_to_num(mean) * count + settings.smoothing * prior) / (count + settings.smoothing)
            result.append(np.where(count > 0, smoothed, prior))
        return result
    if len(fit) < 2:
        raise ValueError('Cross-fitted target encoding needs at least two training rows')
    if strategy == 'group':
        groups = policy.get('group_columns') or ([settings.group_column] if settings.group_column else [])
        if settings.group_column:
            groups = [settings.group_column]
        if not groups:
            raise ValueError('Grouped target encoding needs group_column or validation.group_columns')
        codes, _ = key_codes(fit, fit.iloc[:0], groups)
        folds = min(settings.folds, len(np.unique(codes)))
        if folds < 2:
            raise ValueError('Grouped target encoding needs at least two groups')
        splits = GroupKFold(n_splits=folds).split(fit, values, codes)
    else:
        splits = KFold(n_splits=min(settings.folds, len(fit)), shuffle=True, random_state=settings.seed).split(fit)
    training = np.full(len(fit), np.nan)
    for train, valid in splits:
        training[valid] = _encode(fit.iloc[train], fit.iloc[valid], columns, values[train], settings.smoothing)
    return [training, *(_encode(fit, frame, columns, values, settings.smoothing) for frame in frames[1:])]


def row_weights(frame, parameters):
    from .plans import RowWeights
    settings = RowWeights.model_validate(parameters)
    times = ordered_time(frame[settings.time_column])
    # Normalize to mean 1; no validation/test dates or targets influence training weights.
    weights = np.exp2(np.maximum(-1000., (times - times.max()) / settings.half_life))
    return weights / weights.mean()

"""Task adapters around the installed AutoML engines; outer evaluation stays shared."""
from pathlib import Path
import time

import numpy as np
import pandas as pd


def feature_tables(state, plan):
    from .features import create_features
    from .automl import _features
    preparation = state['preparation']
    targets = preparation.get('all_target_columns', preparation['data']['target_columns'])
    ids = preparation['data'].get('id_columns', [])
    inherited = state['fit'].attrs.get('feature_recipes', [])
    recipes = [*inherited, *(feature.model_dump() for feature in plan.features)]
    created = [recipe['name'] for recipe in inherited]
    raw = [state[scope].drop(columns=created) for scope in ('fit', 'validation', 'test')]
    frames = create_features(raw, recipes, forbidden=targets, policy=preparation.get('validation', {}))
    excluded = {*targets, *ids, *preparation['data'].get('query_columns', [])}
    columns = plan.columns or [column for column in frames[0] if column not in excluded]
    columns = list(dict.fromkeys([*columns, *(feature.name for feature in plan.features)]))
    columns = [column for column in columns if column not in plan.drop_columns]
    if set(columns) & set(targets):
        raise ValueError('AutoML inputs cannot include target columns')
    return frames, _features(*(frame[columns] for frame in frames), targets, ids), columns


class RankingChoice:
    """A frozen ranker, including its own preprocessing and query-aware refit."""
    def __init__(self, model, transformer, query_columns):
        self.model, self.transformer, self.query_columns = model, transformer, query_columns

    def refit_tables(self, state, plan, defaults):
        from .finalize import clone_estimator
        from .automl import _remaining
        from .task_types import group_codes
        (fit, _, _), (x, valid_x, test_x), _ = feature_tables(state, plan)
        groups = group_codes(fit, self.query_columns)
        order = np.argsort(groups, kind='stable')
        features = self.transformer.transform(x.copy())
        target = state['preparation']['data']['target_columns'][0]
        model = clone_estimator(self.model)
        model.fit(features.iloc[order], fit[target].to_numpy()[order],
                  groups=groups[order], budget=_remaining(state['preparation']['runtime']))
        fitted = RankingChoice(model, self.transformer, self.query_columns)
        return fitted, fitted.predict_pair(valid_x, test_x, target)

    def predict_pair(self, valid_x, test_x, target):
        return {scope + '_predictions': pd.DataFrame({target: self.model.predict(self.transformer.transform(frame.copy()))})
                for scope, frame in [('validation', valid_x), ('test', test_x)]}


def run_ranking(state, plan, defaults):
    import cloudpickle
    from .automl import _fit_flaml, _remaining
    from .eda import _inner_split
    from .plans import AnalysisRequest
    from .task_types import group_codes, ranking_score
    from .storage import atomic_bytes
    if plan.engine != 'flaml':
        raise ValueError('Grouped ranking is supported by FLAML; AutoGluon Tabular has no ranking task')
    if plan.ensemble or plan.row_weights:
        raise ValueError('Grouped FLAML ranking uses a frozen ranker; use separate saved ensembles and unweighted queries')
    models = plan.models or ['lgbm', 'xgboost']
    if not set(models) <= {'lgbm', 'xgboost', 'xgb_limitdepth'}:
        raise ValueError('FLAML ranking models: lgbm, xgboost, xgb_limitdepth')
    plan = plan.model_copy(update={'models': models})
    preparation = state['preparation']
    if len(preparation['data']['target_columns']) != 1:
        raise ValueError('Ranking requires exactly one relevance target')
    target = preparation['data']['target_columns'][0]
    keys = preparation['data'].get('query_columns', []) or ([plan.group_by] if plan.group_by else [])
    group_codes(state['fit'], keys)
    name = ''.join(char for char in preparation['metric']['name'].lower() if char.isalnum() or char == '@')
    if not name.startswith(('ndcg', 'map', 'meanaverageprecision')):
        raise ValueError('FLAML ranking adapter supports the shared NDCG/MAP primary metrics')
    policy = {'strategy': 'group', 'group_columns': keys}
    request = AnalysisRequest(name='ranking', kind='task_predictive', validation_fraction=plan.holdout_fraction)
    train, valid = _inner_split(state['fit'], None, request, policy, False)
    hidden = state['fit'].iloc[valid].drop(columns=preparation['data']['target_columns'])
    inner = {**state, 'fit': state['fit'].iloc[train], 'validation': hidden, 'test': hidden}
    (fit, validation, _), (x, valid_x, _), columns = feature_tables(inner, plan)
    groups = group_codes(fit, keys)
    order = np.argsort(groups, kind='stable')
    valid_groups = group_codes(validation, keys)
    tuning = (valid_x, state['fit'][target].iloc[valid].reset_index(drop=True))
    if plan.validation == 'cv' and not preparation['runtime'].get('is_smoke_check'):
        if plan.features or state['fit'].attrs.get('feature_recipes'):
            raise ValueError('Feature-aware ranking needs holdout so recipes are rebuilt within its training rows')
        (fit, _, _), (x, _, _), columns = feature_tables(state, plan)
        groups = group_codes(fit, keys)
        order = np.argsort(groups, kind='stable')
        tuning, valid_groups = None, None
    def metric(X_val, y_val, estimator, labels=None, X_train=None, y_train=None,
               weight_val=None, weight_train=None, config=None, groups_val=None, groups_train=None):
        reference = pd.DataFrame({'query': groups_val, target: np.asarray(y_val)})
        score = ranking_score(reference, estimator.predict(X_val), target, ['query'], name,
                              preparation['metric'].get('parameters', {}))
        return 1. - score, {'primary_metric': score}
    runtime = preparation['runtime']
    selection = {**runtime, 'deadline': time.monotonic() + _remaining(runtime) * .6}
    model, diagnostics = _fit_flaml(x.iloc[order], fit[target].iloc[order], metric, False, selection, 42, plan,
                                   tuning,
                                   task='rank', groups=groups[order], groups_val=valid_groups)
    choice = RankingChoice(model.model, model.feature_transformer, keys)
    choice, output = choice.refit_tables(state, plan, defaults)
    filename = Path(runtime.get('work_directory', '.')) / 'selected_model.pkl'
    atomic_bytes(filename, cloudpickle.dumps(choice))
    output['automl_diagnostics'] = {**diagnostics, 'engine': 'flaml', 'task_family': 'ranking',
        'selected_model_file': str(filename), 'query_columns': keys, 'feature_count': len(columns),
        'inner_fit_rows': len(train) if tuning is not None else None,
        'inner_validation_rows': len(valid) if tuning is not None else None,
        'cv_rows': len(fit) if tuning is None else None,
        'inner_validation': 'group_cv' if tuning is None else 'complete query holdout',
        'folds': plan.folds if tuning is None else 1, 'refitted_on_fit': True}
    return output


def head_state(state, target, defaults, metric=None):
    from .operations import _requires_probabilities
    from .task_types import family
    preparation = state['preparation']
    targets = preparation['data']['target_columns']
    kind = family(defaults.get('task_type'), targets, state['fit'])
    mapping = preparation['output'].get('class_columns', {}).get(target, {})
    probability = kind == 'classification' and _requires_probabilities(state, defaults)
    output = {**preparation['output'], 'columns': list(mapping) if probability and mapping else [target]}
    child = {**state,
        'preparation': {**preparation, 'data': {**preparation['data'], 'target_columns': [target]},
                        'all_target_columns': preparation.get('all_target_columns', targets),
                        'prediction_metric': preparation.get('prediction_metric', preparation['metric']),
                        'output': output, 'metric': {**preparation['metric'], **({'name': metric} if metric else {})}}}
    child_defaults = {**defaults, 'task_type': 'classification' if kind == 'classification' else
                      'forecasting' if kind == 'forecasting' else 'regression', 'class_labels': None}
    return child, child_defaults


def join_heads(results, state, defaults):
    from .operations import _requires_probabilities
    output = {}
    for scope in ('validation', 'test'):
        frames = []
        for target, result in zip(state['preparation']['data']['target_columns'], results):
            frame = result[scope + '_predictions'].copy()
            mapping = state['preparation']['output'].get('class_columns', {}).get(target, {})
            if mapping and _requires_probabilities(state, defaults):
                frame = frame.rename(columns=mapping)
            frame.attrs = {}
            frames.append(frame.reset_index(drop=True))
        output[scope + '_predictions'] = pd.concat(frames, axis=1)
    return output


def constant_head(state, defaults):
    from .operations import _requires_probabilities
    target = state['preparation']['data']['target_columns'][0]
    counts = state['fit'][target].value_counts(normalize=True)
    columns = state['preparation']['output']['columns']
    probability = _requires_probabilities(state, defaults)
    result = {}
    for scope in ('validation', 'test'):
        if probability and columns == [target]:
            values = {target: float(counts.get(1, 0.))}
        elif probability:
            values = {column: float(sum(value for label, value in counts.items() if str(label) == column)) for column in columns}
        else:
            values = {target: state['fit'][target].mode().iloc[0]}
        result[scope + '_predictions'] = pd.DataFrame({column: [value] * len(state[scope]) for column, value in values.items()})
    return result


def run_targets(state, plan, defaults):
    from .automl import _remaining, run_automl
    from .task_types import family
    from .storage import save
    targets = state['preparation']['data']['target_columns']
    runtime = state['preparation']['runtime']
    unknown = set(plan.target_metrics) - set(targets)
    if unknown:
        raise ValueError(f'target_metrics names unknown targets: {sorted(unknown)}')
    results, diagnostics = [], []
    for index, target in enumerate(targets):
        metric = plan.target_metrics.get(target)
        kind = family(defaults.get('task_type'), targets, state['fit'])
        name = ''.join(char for char in state['preparation']['metric']['name'].lower() if char.isalnum())
        if metric is None and kind == 'classification':
            if any(word in str(defaults.get('task_type')).lower().replace('-', '_') for word in ('multilabel', 'multi_label')) and 'f1' in name:
                metric = 'F1'
            elif name in {'hammingloss', 'subsetaccuracy', 'exactmatch'}:
                metric = 'Accuracy'
            elif 'f1' in name and name not in {'f1', 'macrof1', 'f1macro'}:
                metric = 'Macro F1'
        child, child_defaults = head_state(state, target, defaults, metric)
        directory = Path(runtime.get('work_directory', '.')) / f'target_{index:03d}'
        directory.mkdir(parents=True, exist_ok=True)
        seconds = _remaining(runtime) / (len(targets) - index)
        child['preparation']['runtime'] = {**runtime, 'deadline': time.monotonic() + seconds,
                                          'work_directory': str(directory)}
        if kind == 'classification' and child['fit'][target].nunique() == 1:
            result = constant_head(child, child_defaults)
            facts = {'constant': True}
        else:
            result = run_automl(child, plan.engine, config=plan.model_copy(update={'target_metrics': {}}).model_dump(), **child_defaults)
            facts = result['automl_diagnostics']
        results.append(result)
        diagnostics.append({'target': target, 'selection_metric': child['preparation']['metric']['name'], 'diagnostics': facts})
    output = join_heads(results, state, defaults)
    output['automl_diagnostics'] = {'engine': plan.engine, 'task_family': 'multi_target', 'targets': diagnostics,
        'outer_metric': state['preparation']['metric']['name'], 'fit_rows': len(state['fit']),
        'target_dependency': 'independent models; other target columns are never model inputs',
        'budget': 'one shared node deadline, remaining time divided between unfinished targets'}
    phase = 'smoke' if runtime.get('is_smoke_check') else 'full'
    save(Path(runtime.get('work_directory', '.')) / f'{plan.engine}_{phase}_diagnostics.json', output['automl_diagnostics'])
    return output


def forecast_tables(state, plan, *, build_features=True):
    from .features import create_features
    binding = state['preparation']['data']
    time_column = plan.time_column or binding.get('time_column')
    series = plan.series_columns or binding.get('series_columns', [])
    if not time_column:
        raise ValueError('Native forecasting requires time_column in the plan or data binding')
    inherited = state['fit'].attrs.get('feature_recipes', [])
    recipes = [*inherited, *(feature.model_dump() for feature in plan.features)]
    created = [recipe['name'] for recipe in inherited]
    raw = [state[scope].drop(columns=created) for scope in ('fit', 'validation', 'test')]
    targets = state['preparation'].get('all_target_columns', binding['target_columns'])
    frames = (create_features(raw, recipes, forbidden=targets, policy=state['preparation'].get('validation', {}))
              if build_features else raw)
    excluded = {*targets, *binding.get('id_columns', []), *series, time_column}
    columns = plan.columns or [column for column in frames[0] if column not in excluded]
    columns = list(dict.fromkeys([time_column, *columns, *(feature.name for feature in plan.features)]))
    columns = [column for column in columns if column not in plan.drop_columns and column not in series]
    if time_column not in columns or set(columns) & set(targets):
        raise ValueError('Forecast inputs need the time column and must exclude targets')
    converted = []
    for frame in frames:
        frame = frame.reset_index(drop=True).copy()
        values = frame[time_column]
        if pd.api.types.is_numeric_dtype(values):
            if values.isna().any() or not np.equal(values, values.astype(np.int64)).all():
                raise ValueError('Native forecast numeric periods must be integer codes; supply datetimes for fractional times')
            offset = pd.tseries.frequencies.to_offset(plan.frequency or 'D')
            frame[time_column] = [pd.Timestamp('2000-01-01') + int(value) * offset for value in values]
        else:
            frame[time_column] = pd.to_datetime(values, utc=True).dt.tz_localize(None)
        converted.append(frame)
    return converted, columns, time_column, series


def series_groups(frame, columns):
    if not columns:
        return [((), frame)]
    return [(key if isinstance(key, tuple) else (key,), group)
            for key, group in frame.groupby(columns, sort=False, dropna=False)]


def series_key(key):
    return tuple(None if pd.isna(value) else value for value in key)


class ForecastChoice:
    """Frozen native FLAML series models; refit never repeats the AutoML search."""
    def __init__(self, choices, time_column, series_columns, columns, target, horizon, fallback):
        self.choices, self.time_column, self.series_columns = choices, time_column, series_columns
        self.columns, self.target, self.horizon, self.fallback = columns, target, horizon, fallback

    def refit_tables(self, state, plan, defaults):
        import copy
        from .automl import _remaining
        from .finalize import clone_estimator
        from flaml.automl.time_series.ts_data import TimeSeriesDataset
        (fit, valid, test), _, _, _ = forecast_tables(state, plan)
        fitted = {}
        groups = series_groups(fit, self.series_columns)
        for index, (key, group) in enumerate(groups):
            key = series_key(key)
            template = self.choices.get(key, next(iter(self.choices.values())))
            group = group.sort_values(self.time_column)
            transformer = copy.deepcopy(template['transformer'])
            x, y = transformer.transform(group[self.columns].copy(), group[self.target].copy())
            data = pd.concat([x.reset_index(drop=True), y.reset_index(drop=True)], axis=1)
            model = clone_estimator(template['model'])
            model.fit(TimeSeriesDataset(data, self.time_column, [self.target]), None,
                      budget=_remaining(state['preparation']['runtime']) / (len(groups) - index), period=self.horizon)
            fitted[key] = {'model': model, 'transformer': transformer,
                           'end': group[self.time_column].max(), 'frequency': template['frequency']}
        choice = ForecastChoice(fitted, self.time_column, self.series_columns, self.columns,
                                self.target, self.horizon, float(fit[self.target].mean()))
        return choice, choice.predict_pair(valid, test)

    def predict_pair(self, validation, test):
        output = {}
        known = pd.concat([validation, test], ignore_index=True)
        for scope, frame in [('validation', validation), ('test', test)]:
            values = np.full(len(frame), self.fallback, dtype=float)
            frame = frame.reset_index(drop=True)
            for key, query in series_groups(frame, self.series_columns):
                key = series_key(key)
                choice = self.choices.get(key)
                if choice is None:
                    continue  # New series have no fitted history; retain the explicit mean fallback.
                if (query[self.time_column] <= choice['end']).any():
                    raise ValueError('Native forecast queries must follow the fit history')
                grid = pd.date_range(choice['end'], query[self.time_column].max(), freq=choice['frequency'])[1:]
                if not query[self.time_column].isin(grid).all():
                    raise ValueError('Forecast query times do not lie on the declared/inferred calendar grid')
                supplied = known
                for column, value in zip(self.series_columns, key):
                    supplied = supplied[supplied[column].isna() if value is None else supplied[column].eq(value)]
                supplied = supplied[self.columns].drop_duplicates(self.time_column)
                bridge = pd.DataFrame({self.time_column: grid}).merge(supplied, on=self.time_column, how='left', indicator=True)
                if len(self.columns) > 1 and bridge['_merge'].eq('left_only').any():
                    raise ValueError('Missing future exogenous rows between fit and the forecast horizon')
                bridge = bridge.drop(columns='_merge')
                transformed = choice['transformer'].transform(bridge.copy())
                predicted = np.asarray(choice['model'].predict(transformed)).reshape(-1)
                lookup = pd.Series(predicted, index=grid)
                values[query.index] = lookup.loc[query[self.time_column]].to_numpy()
            output[scope + '_predictions'] = pd.DataFrame({self.target: values})
        return output


def run_forecast(state, plan, defaults):
    import cloudpickle
    from flaml import AutoML
    from .automl import _METRICS, _remaining
    from .auto_settings import model_settings
    from .storage import atomic_bytes
    if plan.engine != 'flaml':
        raise ValueError('AutoGluon TimeSeries is not installed or included in the provided requirements; use FLAML native forecasting')
    if plan.row_weights or plan.ensemble or plan.bag_folds or plan.stack_levels:
        raise ValueError('FLAML native forecasting uses individual series predictors without tabular weights/bagging/stacking')
    (fit, validation, test), columns, time_column, series = forecast_tables(state, plan, build_features=False)
    target = state['preparation']['data']['target_columns'][0]
    runtime = state['preparation']['runtime']
    name = ''.join(char for char in state['preparation']['metric']['name'].lower() if char.isalnum())
    if name not in _METRICS or _METRICS[name][0] != 'regression':
        raise ValueError('Native forecast requires a supported regression selection metric')
    models = plan.models or ['lgbm', 'arima', 'naive']
    allowed = {'lgbm', 'xgboost', 'xgb_limitdepth', 'rf', 'extra_tree', 'catboost', 'lassolars',
               'arima', 'sarimax', 'holt-winters', 'naive', 'snaive', 'avg', 'savg'}
    if not set(models) <= allowed:
        raise ValueError('Choose installed FLAML native forecast models; prophet/tft need unavailable dependencies')
    plan = plan.model_copy(update={'models': models})
    models, parameters = model_settings(plan, False)
    groups = series_groups(fit, series)
    horizon = plan.horizon or max([len(group) for frame in (validation, test) for _, group in series_groups(frame, series)] or [1])
    choices, details = {}, []
    selection_runtime = {**runtime, 'deadline': time.monotonic() + _remaining(runtime) * .6}
    for index, (key, group) in enumerate(groups):
        group = group.sort_values(time_column)
        if group[time_column].duplicated().any():
            raise ValueError('Native forecast requires unique series/time rows')
        frequency = plan.frequency or pd.infer_freq(group[time_column])
        if not frequency:
            raise ValueError('Native forecast needs a regular calendar; provide frequency and regularized data')
        seconds = _remaining(selection_runtime) / (len(groups) - index)
        directory = Path(runtime.get('work_directory', '.')) / f'series_{index:03d}'
        directory.mkdir(parents=True, exist_ok=True)
        tuning = {}
        training = group
        if plan.validation == 'holdout':
            if len(group) <= horizon + 2:
                raise ValueError('Forecast series needs at least horizon+3 observations for an inner holdout')
            positions = group.index.to_numpy()
            hidden = state['fit'].iloc[positions[-horizon:]].drop(
                columns=state['preparation'].get('all_target_columns', state['preparation']['data']['target_columns']))
            inner = {**state, 'fit': state['fit'].iloc[positions[:-horizon]], 'validation': hidden, 'test': hidden}
            (training, held_out, _), _, _, _ = forecast_tables(inner, plan)
            training = training.sort_values(time_column)
            held_out = held_out.sort_values(time_column)
            reference = group[target].iloc[-horizon:]
            tuning = {'X_val': held_out[columns].copy(), 'y_val': reference.copy()}
        elif plan.features or state['fit'].attrs.get('feature_recipes'):
            raise ValueError('Feature-aware native forecasting needs holdout; recipes are rebuilt from inner training rows')
        model = AutoML()
        model.fit(X_train=training[columns].copy(), y_train=training[target].copy(),
                  task='ts_forecast', time_col=time_column, period=horizon, metric=_METRICS[name][1],
                  time_budget=seconds, estimator_list=models, custom_hp=parameters,
                  max_iter=plan.num_trials, eval_method=plan.validation, n_splits=plan.folds,
                  retrain_full=False, n_jobs=int(runtime.get('num_cpus', 1)), verbose=0,
                  mlflow_logging=False, log_file_name=str(directory / 'search.jsonl'), **tuning)
        choices[series_key(key)] = {'model': model.model, 'transformer': model.feature_transformer,
            'frequency': frequency, 'end': group[time_column].max()}
        details.append({'series': [str(value) for value in key], 'best_estimator': model.best_estimator,
                        'best_config': model.best_config, 'inner_fit_rows': len(training),
                        'inner_validation_rows': horizon if tuning else None})
    choice = ForecastChoice(choices, time_column, series, columns, target, horizon, float(fit[target].mean()))
    choice, output = choice.refit_tables(state, plan, defaults)
    filename = Path(runtime.get('work_directory', '.')) / 'selected_model.pkl'
    atomic_bytes(filename, cloudpickle.dumps(choice))
    output['automl_diagnostics'] = {'engine': 'flaml', 'task_family': 'forecasting', 'series': details,
        'selected_model_file': str(filename), 'horizon': horizon, 'time_column': time_column,
        'series_columns': series, 'feature_count': len(columns), 'refitted_on_fit': True,
        'cold_start': 'mean of fit targets for previously unseen series',
        'future_labels_used': False}
    return output

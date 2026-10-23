"""Controller-written nodes: a strong generic GBDT and AutoML references instead of a constant LLM baseline.

Their sources are a few lines calling toolkit.py, so checks, execution, repair, Baker parents and the final
refit treat them like any generated node, and Qwen sees short reference code it can extend.
"""

from .toolkit import metric_names, task_kind, temporal_column

TOOLKIT = task_kind.__module__  # alfa_agent.toolkit in the package, alfa_toolkit in the flat platform ZIP

GBDT = '''"""Controller baseline: shift-aware LightGBM; loss / class weights chosen by the task scorer."""

from {toolkit} import gbdt


def fit_model(fit, preparation):
    return gbdt(fit, preparation)
'''

AUTOML = '''"""Controller AutoML reference ({engine}) under a fixed time limit."""

from {toolkit} import automl


def fit_model(fit, preparation):
    return automl(fit, preparation, {engine!r}, {limit!r})
'''

TOOLKIT_GUIDE = {
    'import': f'from {TOOLKIT} import gbdt, target_stats, automl',
    'gbdt': 'gbdt(fit, preparation, *, transform=None, params=None, candidates=None, tune_classes=True) -> predictor. '
            'Shift-aware LightGBM for one target: categories with a fit vocabulary, dates as calendar parts, inner '
            'split along the temporal column, early stopping, loss (L2/Huber/log) or class weights (plain/balanced/'
            'sqrt) chosen by the task scorer, per-class probability multipliers, final retrain on all fit rows. '
            'transform(frame, fit, training) -> frame adds features (keep row order; training=True only when frame '
            'is fit itself). params override LightGBM parameters (num_leaves, learning_rate, min_child_samples, '
            'colsample_bytree, reg_lambda, ...). candidates: [(name, extra_params, option)], option None|"log"|'
            '"balanced"|"sqrt_balanced"; they are tried IN ADDITION to the defaults, and a Huber candidate without '
            'alpha gets a data-derived alpha.',
    'target_stats': 'target_stats(reference, frame, keys, target, *, training=False, time=None, folds=5, prefix=None) '
                    '-> DataFrame aligned to frame.index: mean target (per class for labels), count and latest earlier '
                    'value for rows sharing `keys` (a repeated entity). training=True: each row sees only earlier '
                    'periods of `time` or other folds, so it is leakage-safe inside transform.',
    'automl': "automl(fit, preparation, engine, time_limit) -> predictor; engine 'autogluon'|'lightautoml'|'flaml'.",
    'example': 'def transform(frame, fit, training):\n'
               '    frame = frame.copy()\n'
               '    frame["ratio"] = frame["a"] / (frame["b"] + 1)\n'
               '    return frame.join(target_stats(fit, frame, ["key1", "key2"], TARGET, training=training, time=TIME))\n'
               'def fit_model(fit, preparation):\n'
               '    return gbdt(fit, preparation, transform=transform, params={"num_leaves": 63})',
    'when_not': 'Write your own model code for multi-target, sequence/batch or other outputs gbdt rejects, '
                'or when the hypothesis needs another learner; you may still blend it with gbdt predictions.',
}

def template(preparation: dict, threads: int) -> tuple[str, dict] | None:
    """Source and plan of the controller GBDT node, or None to keep the LLM baseline."""
    kind = task_kind(preparation)
    if kind is None:
        return None
    task, proba = kind['task'], kind['proba']
    temporal = temporal_column(preparation)
    choices = 'L2, Huber and log-target' if task == 'regression' else 'unweighted, balanced and sqrt-balanced class weights'
    return GBDT.format(toolkit=TOOLKIT), {
        'rationale': 'Controller baseline: a robust gradient-boosting model gives the search a strong measured leader '
                     'before any LLM-generated experiment.',
        'expected_effect': 'Much better than a constant prediction; later nodes improve features, loss and ensembling.',
        'pipeline': {
            'data': {'features': 'all train/test columns except targets and ids; dates as calendar parts; high-cardinality text dropped'},
            'preprocessing': {'numeric': 'as is, NaN kept', 'categorical': 'pandas category with fit vocabulary'},
            'model': {'family': 'LightGBM via toolkit.gbdt', 'task': task, 'probability_output': proba,
                      'candidates': choices, 'num_leaves': 31, 'learning_rate': .05},
            'training': {'inner_split': f'latest 15% of {temporal}' if temporal else 'stratified/random 15% of fit',
                         'early_stopping': 100, 'selection': 'task scorer on the inner split',
                         'post_processing': 'per-class probability multipliers tuned by the scorer' if task == 'classification' and not proba else None,
                         'final': 'retrain on all fit rows with 1.15x best iterations'}}}


def automl(preparation: dict, threads: int, time_limit: float) -> list[tuple[str, dict]]:
    """AutoGluon, LightAutoML and FLAML reference nodes under the GBDT node's applicability rules."""
    kind = task_kind(preparation)
    if kind is None:
        return []
    names = metric_names(preparation)
    return [(AUTOML.format(toolkit=TOOLKIT, engine=engine, limit=int(time_limit)), {
        'rationale': f'Controller AutoML reference ({engine}): a strong library search the agent must beat or blend with.',
        'expected_effect': 'Competitive measured score from automated model search and ensembling under a time limit.',
        'pipeline': {'data': {'features': 'all columns except targets and ids'},
                     'preprocessing': {'automl': f'{engine} internal preprocessing'},
                     'model': {'family': f'{engine} via toolkit.automl', 'task': kind['task'], 'probability_output': kind['proba'],
                               'metric': names.get(engine),
                               'preset': 'best_quality without dynamic stacking' if engine == 'autogluon' else 'default'},
                     'training': {'time_limit_seconds': int(time_limit), 'threads': threads}}})
            for engine in ('autogluon', 'lightautoml', 'flaml')]

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
    'purpose': 'Routine helpers so generated code does not re-implement encoding, validation and metric tuning. '
               'They contain no model choice: pick the learner the hypothesis needs (CatBoost, XGBoost, LightGBM, '
               'sklearn, torch, AutoML, blends) and use only the helpers that help.',
    'import': f'from {TOOLKIT} import Features, labels, split, inner_score, class_scale, output, target_stats, gbdt, automl',
    'Features': 'features = Features(fit, preparation, transform=None, kind="category"|"string"|"codes"): features.train '
                'is the encoded fit, features(frame) encodes a prediction batch identically; features.categorical lists '
                'categorical columns. "string" suits CatBoost (cat_features=features.categorical), "category" LightGBM/'
                'XGBoost(enable_categorical=True)/HistGradientBoosting, "codes" sklearn/torch. Dates -> calendar parts; '
                'targets/ids/free text dropped. transform(frame, fit, training) -> frame adds features first.',
    'labels': 'target, y, classes = labels(fit, preparation): classification labels as 0..K-1 codes, classes[code] maps back.',
    'split': 'train_rows, valid_rows = split(fit, preparation): inner holdout positions following the measured shift '
             '(latest periods of the temporal column, else stratified). Use for early stopping, weights, blending.',
    'inner_score': 'inner_score(values, fit, valid_rows, preparation) -> task metric on inner rows, higher is better.',
    'class_scale': 'scale = class_scale(probability, classes, fit, valid_rows, preparation): per-class multipliers; '
                   'predict classes[argmax(probability * scale)] (helps macro/averaged label metrics).',
    'output': 'return output(frame, preparation, values) -> DataFrame of ids + target, for predict(frame).',
    'target_stats': 'target_stats(reference, frame, keys, target, *, training=False, time=None) -> ONLY the new columns '
                    '(leakage-safe mean/count/last target of rows sharing keys); join them: frame = frame.join(target_stats('
                    'fit, frame, keys, target, training=training)). transform must return the whole frame with added columns.',
    'gbdt': 'gbdt(fit, preparation, transform=None, params=None, candidates=None) -> predictor: a ready LightGBM-only '
            'pipeline (loss/class weights chosen by the scorer). Classification predictors expose .proba(frame) and '
            '.classes for blending. It cannot train other learners.',
    'automl': "automl(fit, preparation, engine, time_limit, *, transform=None) -> predictor; engine 'autogluon'|'lightautoml'|"
              "'flaml'. transform adds features first (e.g. target_stats). Only these arguments exist.",
    'example': 'features = Features(fit, preparation, kind="string"); target, y, classes = labels(fit, preparation)\n'
               'train, valid = split(fit, preparation)\n'
               'model = CatBoostClassifier(iterations=3000, learning_rate=0.05, thread_count=4, verbose=0, '
               'allow_writing_files=False, auto_class_weights="SqrtBalanced")\n'
               'model.fit(features.train.iloc[train], y[train], cat_features=features.categorical, '
               'eval_set=(features.train.iloc[valid], y[valid]), early_stopping_rounds=100)\n'
               'scale = class_scale(model.predict_proba(features.train.iloc[valid]), classes, fit, valid, preparation)\n'
               '# then refit on all rows with model.get_best_iteration() and\n'
               'def predict(frame): return output(frame, preparation, classes[(final.predict_proba(features(frame)) * scale).argmax(1)])',
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


def automl(preparation: dict, threads: int, time_limit: float, long_limit: float | None = None) -> list[tuple[str, dict]]:
    """AutoGluon, LightAutoML and FLAML reference nodes under the GBDT node's applicability rules."""
    kind = task_kind(preparation)
    if kind is None:
        return []
    names = metric_names(preparation)
    limits = {engine: int(long_limit if engine == 'autogluon' and long_limit else time_limit) for engine in ('autogluon', 'lightautoml', 'flaml')}
    return [(AUTOML.format(toolkit=TOOLKIT, engine=engine, limit=limits[engine]), {
        'rationale': f'Controller AutoML reference ({engine}): a strong library search the agent must beat or blend with.',
        'expected_effect': 'Competitive measured score from automated model search and ensembling under a time limit.',
        'pipeline': {'data': {'features': 'all columns except targets and ids'},
                     'preprocessing': {'automl': f'{engine} internal preprocessing'},
                     'model': {'family': f'{engine} via toolkit.automl', 'task': kind['task'], 'probability_output': kind['proba'],
                               'metric': names.get(engine),
                               'preset': 'best_quality without dynamic stacking' if engine == 'autogluon' else 'default'},
                     'training': {'time_limit_seconds': limits[engine], 'threads': threads,
                                  'validation': 'AutoGluon tunes on the shift-aware inner split' if engine == 'autogluon' else 'library default'}}})
            for engine in ('autogluon', 'lightautoml', 'flaml')]

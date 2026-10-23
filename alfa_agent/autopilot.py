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
            '"balanced"|"sqrt_balanced".',
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

# Verified recipes for the generated code; the libraries stay ordinary tools Qwen may combine or extend.
AUTOML_GUIDE = {
    'when': 'AutoML is a tool, not a ban: run it on raw or engineered features, blend it with other models inside one '
            'fit_model, or start a node from an AutoML parent. Give each library at most ~45% of experiment_timeout '
            'and a tempfile.TemporaryDirectory kept alive by the returned predictor (e.g. predict.workspace = workspace). '
            'Libraries validate with random internal CV, so the outer shift-aware validation still decides.',
    'autogluon': "from autogluon.tabular import TabularPredictor; TabularPredictor(label=target, problem_type="
                 "'regression'|'binary'|'multiclass', eval_metric='root_mean_squared_error'|'f1_macro'|..., path=workspace.name, "
                 "verbosity=0).fit(train_df_with_target, time_limit=T, presets='best_quality', num_cpus=threads, num_gpus=0, "
                 "dynamic_stacking=False); predict(df)/predict_proba(df)[label]. Drop id columns before fit/predict. "
                 "Tiny smoke data (<2000 rows): presets='medium_quality', time_limit ~8. FastAI is not installed (skipped).",
    'lightautoml': "from lightautoml.automl.presets.tabular_presets import TabularAutoML; from lightautoml.tasks import Task; "
                   "TabularAutoML(task=Task('reg'|'binary'|'multiclass'), timeout=T, cpu_limit=threads, "
                   "general_params={'use_algos': [['lgb', 'lgb_tuned', 'cb_tuned']]}, reader_params={'n_jobs': threads, "
                   "'cv': 5, 'random_state': 0}).fit_predict(train_df, roles={'target': target, 'drop': id_columns}, "
                   "verbose=0); predict(df).data is a numpy array (multiclass columns follow reader.class_mapping: "
                   "original label -> index; None means labels are already 0..K-1). Its linear model fails on unseen "
                   "categories; stratified CV needs >= 5 rows per class (oversample rarer classes).",
    'flaml': "from flaml import AutoML; m = AutoML(); m.fit(X_train=X, y_train=y, task='classification'|'regression', "
             "metric='rmse'|'macro_f1'|'mae'|'roc_auc'|'log_loss'|'accuracy'|'r2'|'auto', time_budget=T, n_jobs=threads, "
             "verbose=0, seed=0, log_file_name=''); m.predict(X)/m.predict_proba(X). Convert string columns to pandas "
             "Categorical with a vocabulary fixed on fit so prediction batches encode identically.",
    'outputs': 'Return identifier columns as well as targets; class labels with the original dtype.',
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

"""Offline API coverage and reviewed guidance for the organizers' pinned packages.

Keep this module in the ZIP root. Guidance describes libraries, never a dataset.
The builder extracts signatures and docstrings from the installed distributions.
"""

# Discover public functions/classes in these modules; inspect each class's public
# methods too. Torch's huge top-level namespace is intentionally selected below.
MODULES = {
    'autogluon.tabular': ['autogluon.tabular', 'autogluon.common.space', 'autogluon.core.metrics'],
    'lightautoml': [
        'lightautoml.automl.presets.tabular_presets', 'lightautoml.tasks',
        'lightautoml.automl.base', 'lightautoml.automl.blend',
        'lightautoml.reader.base', 'lightautoml.dataset.np_pd_dataset',
        'lightautoml.pipelines.features.lgb_pipeline', 'lightautoml.pipelines.features.linear_pipeline',
        'lightautoml.pipelines.ml.base', 'lightautoml.pipelines.ml.nested_ml_pipe',
        'lightautoml.ml_algo.boost_lgbm', 'lightautoml.ml_algo.boost_cb',
        'lightautoml.ml_algo.linear_sklearn',
    ],
    'flaml': ['flaml', 'flaml.tune', 'flaml.automl.model'],
    'catboost': ['catboost', 'catboost.utils'],
    'lightgbm': ['lightgbm', 'lightgbm.callback'],
    'xgboost': ['xgboost', 'xgboost.callback'],
    'optuna': ['optuna', 'optuna.study', 'optuna.trial', 'optuna.samplers', 'optuna.pruners',
               'optuna.distributions', 'optuna.search_space', 'optuna.storages'],
    'torch': ['torch.nn', 'torch.nn.functional', 'torch.optim', 'torch.optim.lr_scheduler',
              'torch.utils.data'],
    'scikit-learn': [
        'sklearn.pipeline', 'sklearn.compose', 'sklearn.preprocessing', 'sklearn.impute',
        'sklearn.model_selection', 'sklearn.metrics', 'sklearn.linear_model', 'sklearn.ensemble',
        'sklearn.tree', 'sklearn.svm', 'sklearn.neighbors', 'sklearn.naive_bayes',
        'sklearn.multiclass', 'sklearn.multioutput', 'sklearn.calibration',
        'sklearn.discriminant_analysis', 'sklearn.cluster', 'sklearn.decomposition',
        'sklearn.feature_selection', 'sklearn.feature_extraction.text',
        'sklearn.kernel_approximation', 'sklearn.dummy', 'sklearn.inspection',
        'sklearn.neural_network', 'sklearn.mixture', 'sklearn.gaussian_process',
    ],
    'imbalanced-learn': ['imblearn.pipeline', 'imblearn.over_sampling', 'imblearn.under_sampling',
                        'imblearn.combine', 'imblearn.ensemble'],
    'category-encoders': ['category_encoders'],
    'feature-engine': ['feature_engine.encoding', 'feature_engine.imputation',
                      'feature_engine.selection', 'feature_engine.creation',
                      'feature_engine.transformation', 'feature_engine.outliers',
                      'feature_engine.discretisation'],
    'statsmodels': ['statsmodels.api', 'statsmodels.formula.api', 'statsmodels.tsa.api'],
    'scipy': ['scipy.optimize', 'scipy.sparse', 'scipy.special', 'scipy.stats', 'scipy.linalg'],
    'pandas': [], 'numpy': [], 'polars': [], 'pyarrow': [],
    'joblib': ['joblib'], 'threadpoolctl': ['threadpoolctl'],
    'matplotlib': [], 'seaborn': ['seaborn'], 'tqdm': ['tqdm'],
    'psutil': [], 'cloudpickle': [],
    'pydantic': [], 'python-dotenv': ['dotenv'], 'PyYAML': [], 'orjson': [],
    'jsonschema': [], 'httpx': [], 'requests': [], 'aiohttp': [], 'tenacity': [],
    'openai': [], 'setuptools': [], 'langchain': [], 'langgraph': [],
}

SYMBOLS = {
    'torch': [
        'torch.tensor', 'torch.as_tensor', 'torch.from_numpy', 'torch.zeros', 'torch.ones',
        'torch.arange', 'torch.stack', 'torch.cat', 'torch.softmax', 'torch.log_softmax',
        'torch.argmax', 'torch.no_grad', 'torch.inference_mode', 'torch.manual_seed',
        'torch.set_num_threads', 'torch.save', 'torch.load', 'torch.device', 'torch.Tensor',
    ],
    'pandas': ['pandas.DataFrame', 'pandas.Series', 'pandas.Categorical', 'pandas.CategoricalDtype',
               'pandas.read_csv', 'pandas.read_parquet', 'pandas.read_json', 'pandas.concat',
               'pandas.merge', 'pandas.get_dummies', 'pandas.to_numeric', 'pandas.to_datetime',
               'pandas.cut', 'pandas.qcut', 'pandas.factorize', 'pandas.isna'],
    'numpy': ['numpy.array', 'numpy.asarray', 'numpy.zeros', 'numpy.ones', 'numpy.full',
              'numpy.arange', 'numpy.linspace', 'numpy.concatenate', 'numpy.stack', 'numpy.hstack',
              'numpy.where', 'numpy.select', 'numpy.clip', 'numpy.nan_to_num', 'numpy.isfinite',
              'numpy.unique', 'numpy.bincount', 'numpy.argsort', 'numpy.argmax', 'numpy.mean',
              'numpy.nanmean', 'numpy.median', 'numpy.quantile', 'numpy.log1p', 'numpy.expm1',
              'numpy.savez', 'numpy.load', 'numpy.random.default_rng', 'numpy.random.Generator',
              'numpy.linalg.lstsq', 'numpy.linalg.solve'],
    'polars': ['polars.DataFrame', 'polars.LazyFrame', 'polars.Series', 'polars.Expr',
               'polars.read_csv', 'polars.scan_csv', 'polars.read_parquet', 'polars.scan_parquet',
               'polars.from_pandas', 'polars.concat', 'polars.col', 'polars.lit', 'polars.when'],
    'pyarrow': ['pyarrow.Table', 'pyarrow.RecordBatch', 'pyarrow.array', 'pyarrow.table',
                'pyarrow.csv.read_csv', 'pyarrow.parquet.read_table', 'pyarrow.parquet.write_table'],
    'matplotlib': ['matplotlib.pyplot.subplots', 'matplotlib.pyplot.figure', 'matplotlib.pyplot.plot',
                   'matplotlib.pyplot.scatter', 'matplotlib.pyplot.savefig', 'matplotlib.pyplot.close'],
    'psutil': ['psutil.Process', 'psutil.virtual_memory', 'psutil.cpu_count'],
    'cloudpickle': ['cloudpickle.dump', 'cloudpickle.dumps', 'cloudpickle.load', 'cloudpickle.loads'],
    'pydantic': ['pydantic.BaseModel', 'pydantic.Field', 'pydantic.TypeAdapter',
                 'pydantic.field_validator', 'pydantic.model_validator', 'pydantic.create_model'],
    'PyYAML': ['yaml.safe_load', 'yaml.safe_dump'],
    'orjson': ['orjson.loads', 'orjson.dumps'],
    'jsonschema': ['jsonschema.validate', 'jsonschema.Draft202012Validator'],
    'httpx': ['httpx.Client', 'httpx.AsyncClient', 'httpx.Timeout', 'httpx.Limits'],
    'requests': ['requests.Session', 'requests.get', 'requests.post'],
    'aiohttp': ['aiohttp.ClientSession', 'aiohttp.ClientTimeout', 'aiohttp.TCPConnector'],
    'tenacity': ['tenacity.retry', 'tenacity.Retrying', 'tenacity.AsyncRetrying',
                 'tenacity.stop_after_attempt', 'tenacity.wait_exponential'],
    'openai': ['openai.OpenAI', 'openai.AsyncOpenAI'],
    'setuptools': ['setuptools.find_packages', 'setuptools.find_namespace_packages'],
    'langchain': ['langchain.chains.base.Chain'],
    'langgraph': ['langgraph.graph.StateGraph', 'langgraph.graph.MessageGraph',
                  'langgraph.graph.state.CompiledStateGraph'],
}

# Constructor kwargs forwarded to these bases are documented separately. This is
# source-backed delegation, not a claim that every MRO parameter is accepted.
FORWARDED_TO = {
    'xgboost.XGBClassifier': ['xgboost.sklearn.XGBModel'],
    'xgboost.XGBRegressor': ['xgboost.sklearn.XGBModel'],
    'xgboost.XGBRanker': ['xgboost.sklearn.XGBModel'],
    'xgboost.XGBRFClassifier': ['xgboost.XGBClassifier', 'xgboost.sklearn.XGBModel'],
    'xgboost.XGBRFRegressor': ['xgboost.XGBRegressor', 'xgboost.sklearn.XGBModel'],
}

# Only named dictionaries used by AutoGluon's actual kwargs validation are read.
VALIDATED_KWARGS = {
    'autogluon.tabular.TabularPredictor.fit': [
        ('_validate_fit_kwargs', 'fit_kwargs_default'),
        ('_fit_extra_kwargs_dict', 'return'),
    ],
}

PRIMARY = {
    'autogluon.tabular': ['autogluon.tabular.TabularPredictor'],
    'lightautoml': ['lightautoml.automl.presets.tabular_presets.TabularAutoML', 'lightautoml.tasks.Task'],
    'flaml': ['flaml.AutoML'],
    'catboost': ['catboost.CatBoostClassifier', 'catboost.CatBoostRegressor', 'catboost.Pool'],
    'lightgbm': ['lightgbm.LGBMClassifier', 'lightgbm.LGBMRegressor'],
    'xgboost': ['xgboost.XGBClassifier', 'xgboost.XGBRegressor'],
    'optuna': ['optuna.create_study', 'optuna.study.Study', 'optuna.trial.Trial'],
    'torch': ['torch.nn.Module', 'torch.optim.AdamW', 'torch.nn.CrossEntropyLoss'],
}

ALIASES = {
    'autogluon.tabular': ['autogluon', 'TabularPredictor'],
    'lightautoml': ['lightautoml', 'TabularAutoML', 'LightAutoML'],
    'flaml': ['flaml'], 'catboost': ['catboost'], 'lightgbm': ['lightgbm', 'LGBMClassifier', 'LGBMRegressor'],
    'xgboost': ['xgboost', 'XGBClassifier', 'XGBRegressor'], 'optuna': ['optuna', 'TPE'],
    'torch': ['torch', 'pytorch'], 'scikit-learn': ['sklearn', 'scikit-learn'],
    'imbalanced-learn': ['imblearn', 'imbalanced-learn', 'SMOTE'],
    'category-encoders': ['category_encoders'], 'feature-engine': ['feature_engine'],
    'statsmodels': ['statsmodels'], 'numpy': ['numpy'], 'pandas': ['pandas'],
    'scipy': ['scipy'], 'polars': ['polars'], 'pyarrow': ['pyarrow'],
}

GUIDANCE = {
    'torch': {
        'sources': ['https://github.com/pytorch/pytorch/blob/v2.5.1/torch/nn/modules/module.py'],
        'notes': ['Training uses train(), zero_grad(), loss.backward() and optimizer.step(); inference uses eval() with no_grad()/inference_mode().',
                  'Save feature/label mappings and architecture with weights. Clone/deepcopy a best state_dict so later updates do not overwrite its tensors.',
                  'Honor CPU/GPU and shared time limits; check long loops within batches. Prediction must preserve every input row in order.'],
    },
    'catboost': {
        'sources': ['https://catboost.ai/docs/en/concepts/python-reference_catboostclassifier_fit'],
        'notes': ['Use thread_count for CPU limits. time_left is not supported by the installed constructor, fit or train API; use supported callbacks/early stopping and budgeted iteration counts.',
                  'Categorical input values must be strings/integers with missing values handled. Keep bootstrap_type compatible with subsample/bagging_temperature.'],
    },
    'autogluon.tabular.TabularPredictor': {
        'sources': ['https://auto.gluon.ai/1.3.1/api/autogluon.tabular.TabularPredictor.html',
                    'https://auto.gluon.ai/1.3.1/api/autogluon.tabular.TabularPredictor.fit.html'],
        'notes': [
            'Constructor takes label and eval_metric. fit takes ONE train_data table containing the label column.',
            'Pass time_limit at the TOP LEVEL of predictor.fit. time_budget and seed are not fit parameters.',
            'Pass hyperparameters separately from hyperparameter_tune_kwargs. For an HPO dict, scheduler and searcher are required in 1.3.1, e.g. scheduler=local, searcher=random, num_trials=5. This dict does not contain the model hyperparameters.',
            'Model names include GBM, CAT, XGB, NN_TORCH. A plain list of values inside one model config is not automatically a search space; use documented space objects or a list of complete model configurations.',
            'Use a unique writable path for each predictor. Keep its files until predict finishes; deleting the directory can break lazy model loading.',
            'Bagging/stacking and HPO multiply fit cost. Use fit-only internal validation, the supplied time_limit and CPU/GPU limits. Exclude models that need missing extras or external downloads.',
            'eval_metric=f1_macro is AutoGluon naming; FLAML uses macro_f1. Metric names are library-specific.',
        ],
        'example': "predictor = TabularPredictor(label=target, eval_metric='f1_macro', path=model_dir)\npredictor.fit(train_data=fit_table, time_limit=seconds, num_cpus=cpus, num_gpus=0, hyperparameters={'GBM': {}})\npredictions = predictor.predict(features)",
    },
    'lightautoml.automl.presets.tabular_presets.TabularAutoML': {
        'sources': ['https://lightautoml.readthedocs.io/en/latest/pages/modules/generated/lightautoml.automl.presets.tabular_presets.TabularAutoML.html'],
        'notes': [
            'Public entry points are TabularAutoML and Task from lightautoml.tasks; do not invent MultiClassTabularAutoML or lightautoml.data imports.',
            'Task names include binary, multiclass and reg, not sklearn-style classification/regression strings.',
            'Constructor timeout is the search budget; cpu_limit and gpu_ids control resources. Training is fit_predict with roles specifying the target column.',
            'fit_predict returns OOF predictions as a LightAutoML dataset. predict returns a dataset too: read .data. Multiclass output is a probability matrix, not final labels.',
            'Encode string classification targets explicitly if necessary and retain the exact inverse label mapping for probability columns. Do not assume another estimator uses the same class order.',
            'Internal folds/valid_data must come from fit. Do not use the fixed outer validation labels. Avoid text/image presets requiring missing extras or pretrained downloads.',
        ],
        'example': "from lightautoml.tasks import Task\nautoml = TabularAutoML(task=Task('multiclass'), timeout=seconds, cpu_limit=cpus, gpu_ids=None)\noof = automl.fit_predict(fit_table, roles={'target': target}, verbose=0)\nprobabilities = automl.predict(features).data",
    },
    'flaml.AutoML': {
        'sources': ['https://microsoft.github.io/FLAML/docs/reference/automl/automl/'],
        'notes': [
            'AutoML.fit settings include metric, time_budget, task, estimator_list, n_jobs and log_file_name. Use metric=macro_f1 for multiclass macro F1.',
            'eval_metric and log_file are not substitutes for metric and log_file_name: extra fit kwargs may be forwarded to the selected estimator and cause errors there.',
            'Built-in estimator names include lgbm, xgboost, catboost, rf and extra_tree. xgb is not the registered learner name.',
            'Use X_train/y_train or dataframe/label. Fit-only X_val/y_val or internal CV are allowed; the fixed outer validation labels are unavailable.',
            'Keep estimator_list compatible with the task and installed extras. The time budget belongs to the entire AutoML search, not separately to every learner.',
            'Custom metric functions return (loss_to_minimize, metrics_to_log). Convert a score to a loss explicitly. Per-estimator fit options belong in fit_kwargs_by_estimator, not a single incompatible eval_metric forwarded to every learner.',
        ],
        'example': "automl = AutoML()\nautoml.fit(X_train=X, y_train=y, task='classification', metric='macro_f1', time_budget=seconds, n_jobs=cpus, estimator_list=['lgbm', 'xgboost', 'catboost'], verbose=0)\npredictions = automl.predict(X_test)",
    },
    'catboost.CatBoostClassifier': {
        'sources': ['https://catboost.ai/docs/en/concepts/python-reference_catboostclassifier',
                    'https://catboost.ai/docs/en/concepts/python-reference_catboostclassifier_fit',
                    'https://catboost.ai/docs/en/references/training-parameters/common'],
        'notes': [
            'auto_class_weights accepts Balanced/SqrtBalanced; class_weights accepts explicit numeric weights, not the string Balanced. Do not combine both weighting mechanisms.',
            'time_left and train_subsample are not supported constructor or fit kwargs. Use bounded iterations, early stopping on a fit-only holdout, or a supported CPU after_iteration callback for cooperative timing.',
            'CPU after_iteration callback objects return False to stop. Recompute remaining time before each fit; early stopping alone is not a wall-clock limit.',
            'subsample requires a compatible bootstrap_type such as Bernoulli. Bayesian bootstrap uses bagging_temperature; do not mix incompatible sampling settings.',
            'Categorical values must be strings or integers with missing values handled explicitly. pandas Categorical values unseen in its categories may become NaN.',
            'predict can return an (n, 1) label array. For a single target use np.asarray(result).reshape(-1). predict_proba columns follow model.classes_.',
            'Use thread_count for CPU threads, random_seed for seeding, and cat_features for categorical column names/indices. Refit costs additional time after choosing iterations.',
        ],
        'example': "model = CatBoostClassifier(iterations=iterations, auto_class_weights='Balanced', thread_count=cpus, random_seed=42, verbose=False)\nmodel.fit(X_fit, y_fit, cat_features=cat_columns)\nlabels = np.asarray(model.predict(X_test)).reshape(-1)",
    },
    'catboost.CatBoostRegressor': {
        'sources': ['https://catboost.ai/docs/en/concepts/python-reference_catboostregressor'],
        'notes': ['Use CatBoostRegressor for sklearn-style iterations/learning_rate/cat_features kwargs; generic catboost.CatBoost expects a params dictionary and is not an interchangeable class.',
                  'Use the CatBoost API and budget/categorical restrictions described for CatBoostClassifier, but do not apply classification-only weights or label conversion to regression.',
                  'For ordinary regression predict produces numeric values. Choose loss_function to match the task; validation metrics and training objectives need not be the same.'],
    },
    'lightgbm.LGBMClassifier': {
        'sources': ['https://lightgbm.readthedocs.io/en/v4.6.0/pythonapi/lightgbm.LGBMClassifier.html',
                    'https://lightgbm.readthedocs.io/en/v4.6.0/Parameters.html'],
        'notes': [
            'class_weight accepts a dict, balanced or None. balanced_subsample is not supported here.',
            'is_unbalance/scale_pos_weight apply to binary and multiclassova, not ordinary multiclass. For ordinary multiclass use class_weight or sample_weight.',
            'subsample/bagging_fraction only activates with subsample_freq/bagging_freq > 0. Both frequency and fraction must be configured to test bagging.',
            'feature_fraction aliases colsample_bytree. Avoid setting conflicting aliases: an explicitly supplied primary name can override another setting silently.',
            'Use n_jobs for CPU limits. eval_metric and callbacks belong to fit. Early stopping requires eval_set; reserve time for the final fit after parameter search.',
            'For built-in objectives predict returns labels and predict_proba returns probabilities ordered by classes_. Custom objectives change prediction semantics.',
        ],
        'example': "model = LGBMClassifier(class_weight='balanced', subsample=0.8, subsample_freq=1, n_jobs=cpus, verbosity=-1)\nmodel.fit(X_fit, y_fit)\nlabels = model.predict(X_test)",
    },
    'lightgbm.LGBMRegressor': {
        'sources': ['https://lightgbm.readthedocs.io/en/v4.6.0/pythonapi/lightgbm.LGBMRegressor.html'],
        'notes': ['The bagging/alias/callback/time-budget rules for LGBMClassifier also apply. Classification class_weight and class-balancing arguments are not regression strategies.',
                  'Keep RMSE/MSE/MAE objective and metric semantics distinct. Select the minimum when minimizing a loss, and initialize a minimum search with +infinity or no incumbent.'],
    },
    'lightgbm.LGBMClassifier.fit': {
        'sources': ['https://lightgbm.readthedocs.io/en/v4.6.0/pythonapi/lightgbm.LGBMClassifier.html'],
        'notes': [
            'sklearn custom objective receives (y_true, y_pred), NOT (preds, Dataset) from the native train API. Multiclass predictions and returned gradient/Hessian are (n_samples, n_classes).',
            'Return the derivative d(loss)/d(raw_score), not its negative. For softmax cross entropy this is p - one_hot. A wrong sign can run successfully while increasing the loss.',
            'Return 2D multiclass gradient/Hessian arrays. LightGBM flattens them in Fortran order; pre-flattening in C order scrambles sample/class correspondence.',
            'For custom multiclass objectives, raw scores require the correct softmax/argmax and inverse label mapping. A completed smoke check does not prove derivative correctness.',
        ],
    },
    'xgboost.XGBClassifier': {
        'sources': ['https://github.com/dmlc/xgboost/blob/v2.1.4/python-package/xgboost/sklearn.py'],
        'notes': [
            'In XGBoost 2.1 sklearn classification targets must be integer indices 0..K-1. Store an inverse mapping for original labels and align probability columns before ensembling.',
            'eval_metric, early_stopping_rounds and callbacks are constructor parameters in 2.1, not fit kwargs. eval_set and sample_weight are passed to fit.',
            'The classifier constructor forwards kwargs to XGBModel. Native booster parameters may also be accepted; absence from the short subclass signature alone is not a rejection.',
            'scale_pos_weight is for binary imbalance. For multiclass use sample_weight as needed. Use n_jobs for CPU limits.',
            'Use enable_categorical with a compatible tree method and stable pandas category handling when using categorical features. Do not mix arbitrary per-table category codes.',
        ],
        'example': "labels, y_int = np.unique(y, return_inverse=True)\nmodel = XGBClassifier(n_jobs=cpus, tree_method='hist', eval_metric='mlogloss')\nmodel.fit(X_fit, y_int)\npredictions = labels[model.predict(X_test).astype(int)]",
    },
    'xgboost.XGBRegressor': {
        'sources': ['https://github.com/dmlc/xgboost/blob/v2.1.4/python-package/xgboost/sklearn.py'],
        'notes': ['Constructor kwargs delegate to XGBModel. In 2.1 configure eval_metric, early_stopping_rounds and callbacks in the constructor; fit receives eval_set.',
                  'Regression targets remain numeric; do not integer-encode continuous target values.'],
    },
    'optuna.study.Study.optimize': {
        'sources': ['https://optuna.readthedocs.io/en/v4.2.1/reference/generated/optuna.study.Study.html#optuna.study.Study.optimize'],
        'notes': [
            'timeout is checked between trials and does not interrupt a currently running trial. A trial with several folds must budget/check each fold and use model-level stopping when available.',
            'n_trials is an upper bound, not evidence of completed trials. Record COMPLETE/PRUNED/FAIL counts and chosen parameters if they will be discussed later.',
            'If all trials failed or were pruned, best_trial/best_params is unavailable. Inspect completed trials before selecting a best result; keep a valid budget-aware fallback.',
            'Use direction=maximize for scores and minimize for losses. Selection code and initial incumbent must use the same direction.',
            'n_jobs controls concurrent Optuna trials, separately from each estimator CPU count. Their product can oversubscribe a shared machine.',
        ],
    },
    'torch.nn.Module': {
        'sources': ['https://github.com/pytorch/pytorch/blob/v2.5.1/torch/nn/modules/module.py'],
        'notes': [
            'Separate model.train() for fitting from model.eval() plus no_grad()/inference_mode() for prediction. Dropout/batch normalization otherwise change inference behavior.',
            'state_dict contains parameters/buffers, not architecture or preprocessing. Preserve the architecture, feature order and label mapping with weights.',
            'A best_state = model.state_dict() assignment retains tensor references; clone tensors/deepcopy to preserve a true best checkpoint.',
            'Check a shared deadline within minibatches for long epochs. Use supplied CPU limits and cuda only if a GPU is actually available and allowed.',
        ],
    },
    'torch.nn.CrossEntropyLoss': {
        'sources': ['https://github.com/pytorch/pytorch/blob/v2.5.1/torch/nn/modules/loss.py'],
        'notes': ['For class-index targets pass raw logits of shape (N, C) and integer torch.long labels of shape (N,) in 0..C-1; do not apply softmax before the loss.',
                  'Class weight is a C-element tensor with the same class order and compatible dtype/device as logits. Preserve inverse label mapping for final predictions.'],
    },
    'torch.nn.BCEWithLogitsLoss': {
        'sources': ['https://github.com/pytorch/pytorch/blob/v2.5.1/torch/nn/modules/loss.py'],
        'notes': ['Pass raw logits and compatible floating targets. Apply sigmoid for probabilities at inference; do not apply sigmoid before BCEWithLogitsLoss.',
                  'BCEWithLogitsLoss supports binary/multilabel setups, not an automatic replacement for single-label multiclass CrossEntropyLoss.'],
    },
    'torch.optim.AdamW': {
        'sources': ['https://github.com/pytorch/pytorch/blob/v2.5.1/torch/optim/adamw.py'],
        'notes': ['Initialize the optimizer with model.parameters(), clear gradients, call loss.backward(), then optimizer.step(). Accumulation is deliberate only when gradients are not cleared each step.',
                  'AdamW uses decoupled weight decay. Its weight_decay is not interchangeable with adding a loss penalty under every optimizer.',
                  'Step most learning-rate schedulers after optimizer.step(); ReduceLROnPlateau instead takes the monitored metric. Preserve the chosen metric direction.'],
    },
    'torch.utils.data.DataLoader': {
        'sources': ['https://github.com/pytorch/pytorch/blob/v2.5.1/torch/utils/data/dataloader.py'],
        'notes': ['num_workers=0 is a simple portable choice for generated CPU code, especially with Windows spawn and nested worker processes.',
                  'Shuffle only training batches; prediction order must match validation/test. Use drop_last=False for inference so the last rows are not lost.'],
    },
    'statsmodels.api.OLS': {
        'sources': ['https://www.statsmodels.org/v0.14.0/generated/statsmodels.regression.linear_model.OLS.html'],
        'notes': ['OLS does not automatically add an intercept to the array/DataFrame interface; add a constant consistently to fit and prediction exog if required. Formula interfaces handle intercepts separately.',
                  'model.fit() returns a results object. Keep preprocessing, exog column order, categorical design information and the fitted results together for prediction.',
                  'Use only fit labels to estimate or select a statistical model. Training summaries, p-values and in-sample R-squared do not measure outer-validation prediction quality.'],
    },
    'sklearn.preprocessing.TargetEncoder': {
        'sources': ['https://scikit-learn.org/1.5/modules/generated/sklearn.preprocessing.TargetEncoder.html'],
        'notes': ['fit_transform uses internal cross-fitting; fit(X, y).transform(X) is not equivalent and can leak each training row target into its own feature.',
                  'For a custom target encoder, encode each fit row with a mapping trained without that row/fold. Averaging fold training mappings and applying the result to all fit rows is not OOF encoding.'],
    },
    'sklearn.model_selection': {
        'sources': ['https://scikit-learn.org/1.5/modules/cross_validation.html'],
        'notes': ['Internal selection, threshold tuning and calibration must use only fit rows/labels. Fixed outer validation/test labels are not inputs to a solution block.',
                  'The same outer split is shared by every node. Internal stratification/group/time rules should follow the task and data; do not recreate or alter the outer split.',
                  'A tiny smoke subset may omit rare classes. Its class count and training duration are not facts about full execution.'],
    },
    'imblearn.pipeline.Pipeline': {
        'sources': ['https://imbalanced-learn.org/stable/references/generated/imblearn.pipeline.Pipeline.html'],
        'notes': ['Place resampling inside the fit/CV training pipeline. Resampling the full dataset before making folds leaks information across folds.',
                  'SMOTE needs enough minority neighbors; small smoke/fold subsets may require lower k_neighbors or another sampler. Never resample fixed validation/test.'],
    },
    'sklearn.preprocessing.OneHotEncoder': {
        'sources': ['https://scikit-learn.org/1.5/modules/generated/sklearn.preprocessing.OneHotEncoder.html'],
        'notes': ['In sklearn 1.5 use sparse_output, not the removed sparse parameter. Learn categories on fit and handle unseen values explicitly.',
                  'Sparse matrices may be required to keep memory bounded; some downstream estimators need dense arrays. Match the block interface and resource budget.'],
    },
    'pandas.DataFrame': {
        'sources': ['https://pandas.pydata.org/pandas-docs/version/2.2/reference/api/pandas.DataFrame.html'],
        'notes': ['Pandas aligns Series/DataFrame values by index during assignment. If feature transformations reset or reorder indices, keep targets and IDs explicitly aligned.',
                  'Use fit-derived categorical vocabularies consistently. Casting validation values to a fixed CategoricalDtype can turn unseen values into NaN.',
                  'Avoid target or ID columns entering feature matrices unless the task explicitly treats an ID as a legitimate feature. Preserve output row order.'],
    },
}

# Guidance versions are pinned to the organizers' requirements. Live signatures
# remain authoritative if the platform installs a different version.
REVIEWED_VERSIONS = {
    'autogluon.tabular': '1.3.1', 'lightautoml': '0.4.2', 'flaml': '2.3.6',
    'catboost': '1.2.8', 'lightgbm': '4.6.0', 'xgboost': '2.1.4',
    'optuna': '4.2.1', 'torch': '2.5.1', 'scikit-learn': '1.5.2',
    'imbalanced-learn': '0.12.4', 'pandas': '2.2.3', 'statsmodels': '0.14.0',
}

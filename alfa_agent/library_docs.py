"""Short, version-specific notes for the permitted libraries, selected per node by what its plan or error mentions.

Each note is the API shape and the pitfalls generated code actually hit, kept to a few hundred characters so the
implementer gets the one or two it needs instead of every library in its context.
"""

import re

DOCS = {
    'lightgbm': "LightGBM 4.6 sklearn API: LGBMClassifier/LGBMRegressor(n_estimators, learning_rate, num_leaves, "
                "min_child_samples, subsample+subsample_freq, colsample_bytree, reg_lambda, class_weight, objective="
                "'huber'|'quantile'|'tweedie'|..., n_jobs=4, verbose=-1). Early stopping only via fit(..., eval_set=[(X, y)], "
                "callbacks=[lightgbm.early_stopping(100, verbose=False)]); best_iteration_. Categoricals: pandas category "
                "dtype with categories fixed on fit (unseen -> NaN). predict_proba columns follow model.classes_.",
    'catboost': "CatBoost 1.2.8: CatBoostClassifier/Regressor(iterations, learning_rate, depth, l2_leaf_reg, loss_function, "
                "auto_class_weights='Balanced'|'SqrtBalanced', thread_count=4, random_seed=0, verbose=0, "
                "allow_writing_files=False). fit(X, y, cat_features=[ALL non-numeric columns incl. derived strings], "
                "eval_set=(Xv, yv), early_stopping_rounds=100). Categorical values must be str without NaN: "
                ".astype(str). Classifier.predict returns a 2-D array: use .ravel(); probabilities via predict_proba.",
    'xgboost': "XGBoost 2.1.4: XGBClassifier/XGBRegressor(n_estimators, learning_rate, max_depth, subsample, "
               "colsample_bytree, reg_lambda, tree_method='hist', n_jobs=4, early_stopping_rounds=100, eval_metric=...) — "
               "early_stopping_rounds and eval_metric go to the CONSTRUCTOR, not fit; fit(X, y, eval_set=[(Xv, yv)], "
               "verbose=False). Labels must be 0..K-1 (encode strings yourself). Categoricals: enable_categorical=True "
               "with pandas category dtype fixed on fit, or numeric codes.",
    'sklearn': "scikit-learn 1.5.2: OneHotEncoder(handle_unknown='ignore', sparse_output=False); OrdinalEncoder("
               "handle_unknown='use_encoded_value', unknown_value=-1); HistGradientBoostingClassifier/Regressor("
               "categorical_features='from_dtype', class_weight='balanced', early_stopping=True) handles NaN natively; "
               "RandomForest/ExtraTrees need imputed numeric input (SimpleImputer) and n_jobs=4. Fit every transformer "
               "on fit only; sklearn.metrics f1_score(average='macro') etc. SVC/KNN scale badly above ~20k rows.",
    'autogluon': "AutoGluon 1.3.1: from autogluon.tabular import TabularPredictor; TabularPredictor(label=target, "
                 "problem_type='regression'|'binary'|'multiclass', eval_metric='root_mean_squared_error'|'f1_macro'|..., "
                 "path=tempdir, verbosity=0).fit(df_with_target, time_limit=T, presets='best_quality', num_cpus=4, "
                 "num_gpus=0, dynamic_stacking=False, sample_weight='balance_weight' for imbalanced labels, tuning_data="
                 "optional own holdout); predict(df) / predict_proba(df)[label]. Drop id columns. Keep the tempdir alive "
                 "with the predictor. FastAI is not installed. Toolkit: automl(fit, preparation, 'autogluon', T).",
    'lightautoml': "LightAutoML 0.4.2: from lightautoml.automl.presets.tabular_presets import TabularAutoML; "
                   "from lightautoml.tasks import Task; TabularAutoML(task=Task('reg'|'binary'|'multiclass'), timeout=T, "
                   "cpu_limit=4, general_params={'use_algos': [['lgb', 'lgb_tuned', 'cb_tuned']]}, reader_params={'n_jobs': 4, "
                   "'cv': 5, 'random_state': 0}).fit_predict(df, roles={'target': target, 'drop': ids}, verbose=0); "
                   "predict(df).data is numpy; multiclass columns follow reader.class_mapping (label -> index, None if "
                   "already 0..K-1). Linear model breaks on unseen categories; >= 5 rows per class for CV.",
    'flaml': "FLAML 2.3.6: from flaml import AutoML; m = AutoML(); m.fit(X_train=X, y_train=y, task='classification'|"
             "'regression', metric='rmse'|'macro_f1'|'mae'|'roc_auc'|'log_loss'|'accuracy'|'r2'|'auto', time_budget=T, "
             "n_jobs=4, verbose=0, seed=0, log_file_name='', estimator_list=['lgbm', 'xgboost', 'catboost', 'rf', "
             "'extra_tree']); predict / predict_proba. Give string columns a pandas category dtype fixed on fit.",
    'torch': "PyTorch 2.5.1 (CPU): torch.set_num_threads(4); torch.manual_seed(0). Standardize numeric features with "
             "fit statistics, embed or one-hot categoricals with a fit vocabulary (index 0 for unseen). Save the best "
             "epoch with copy.deepcopy(model.state_dict()) (state_dict().copy() keeps live tensors). Use an inner split "
             "for early stopping, model.eval() and torch.no_grad() in predict. On tables neural nets usually trail "
             "gradient boosting; use them mainly as a blend member.",
    'optuna': "Optuna 4.2.1: optuna.logging.set_verbosity(optuna.logging.WARNING); study = optuna.create_study("
              "direction=..., sampler=optuna.samplers.TPESampler(seed=0)); study.optimize(objective, n_trials=N, "
              "timeout=T, n_jobs=1). Objective scores an INNER split of fit (preparation['scorer']); keep the total "
              "inside experiment_timeout and refit the best parameters on all fit rows.",
    'imblearn': "imbalanced-learn 0.12.4: SMOTE/RandomOverSampler(random_state=0).fit_resample(X, y) needs numeric, "
                "imputed X; resample only the inner TRAIN part, never validation/test. On large tables prefer class "
                "weights (cheaper, usually as good for macro metrics).",
    'category_encoders': "category_encoders 2.6.4: TargetEncoder/CatBoostEncoder(cols=[...]).fit(X_fit, y_fit) then "
                         "transform(X) for prediction; fitting on the same rows it transforms leaks the target — for "
                         "entity statistics prefer toolkit.target_stats(training=True).",
}

ALIASES = {  # plan/error wording → note
    'lightgbm': 'lightgbm', 'lgbm': 'lightgbm', 'catboost': 'catboost', 'xgboost': 'xgboost', 'xgb': 'xgboost',
    'sklearn': 'sklearn', 'scikit': 'sklearn', 'randomforest': 'sklearn', 'random forest': 'sklearn',
    'extratrees': 'sklearn', 'histgradient': 'sklearn', 'logisticregression': 'sklearn', 'ridge': 'sklearn',
    'autogluon': 'autogluon', 'lightautoml': 'lightautoml', 'lama': 'lightautoml', 'flaml': 'flaml',
    'torch': 'torch', 'pytorch': 'torch', 'neural': 'torch', 'mlp': 'torch', 'optuna': 'optuna',
    'smote': 'imblearn', 'imblearn': 'imblearn', 'imbalanced-learn': 'imblearn', 'oversampl': 'imblearn',
    'category_encoders': 'category_encoders', 'targetencoder': 'category_encoders', 'target encoder': 'category_encoders',
}


def library_notes(text: str, limit: int = 2) -> dict[str, str]:
    """Notes for the first `limit` libraries mentioned in text (plan, error, previous imports)."""
    text = text.casefold()
    found = sorted((match.start(), library) for alias, library in ALIASES.items()
                   for match in [re.search(re.escape(alias), text)] if match)
    names = list(dict.fromkeys(library for _, library in found))[:limit]
    return {name: DOCS[name] for name in names}

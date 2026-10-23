"""Classification preparation, Qwen selection and the common training/refit stage.

The imported upstream preparation/model-selection implementations currently describe
regression. Classification therefore has its own explicit boundary, without changing
their source or pretending their regression losses are classification losses.
"""
import asyncio
import json
import math
from pathlib import Path
import re
import time

import numpy as np
import pandas as pd
from pydantic import Field, ValidationInfo, model_validator
from sklearn.model_selection import train_test_split

from agents import JSONAgents
from config import Settings
from contracts import Contract
from data import file_hash, installed_libraries
from storage import atomic_bytes, save
from training_component import run_job, run_training
from training_contracts import SelectedModel, TrainingInput


CLASSIFICATION_LIBRARIES = {
    'logistic_regression': 'scikit-learn', 'sgd': 'scikit-learn',
    'random_forest': 'scikit-learn', 'extra_trees': 'scikit-learn',
    'hist_gradient_boosting': 'scikit-learn', 'mlp': 'scikit-learn',
    'knn': 'scikit-learn', 'lightgbm': 'lightgbm', 'catboost': 'catboost',
    'xgboost': 'xgboost', 'flaml': 'flaml', 'autogluon': 'autogluon.tabular',
}


class ClassificationSelection(Contract):
    selected_models: list[SelectedModel] = Field(min_length=1, max_length=5)
    rationale: str

    @model_validator(mode='after')
    def permitted_classifiers(self, info: ValidationInfo):
        allowed = set((info.context or {}).get('allowed_families', CLASSIFICATION_LIBRARIES))
        families = [model.family for model in self.selected_models]
        if len(set(families)) != len(families):
            raise ValueError('Select distinct classifier families')
        if set(families) - allowed:
            raise ValueError('Classifier family is unavailable or absent from requirements.txt')
        if any(model.target_transform != 'none' for model in self.selected_models):
            raise ValueError('Classification requires target_transform=none')
        n_classes = (info.context or {}).get('n_classes')
        if n_classes:
            _freeze_selection_losses(self, n_classes)
        return self


def _read_table(path, ids):
    path = Path(path)
    if path.suffix.lower() == '.parquet':
        result = pd.read_parquet(path)
        for column in ids:
            if column in result:
                result[column] = result[column].astype('string')
        return result
    return pd.read_csv(path, dtype={column: str for column in ids})


def _profile(frame, target, ids, test):
    """Measure only outer-fit values; test contributes its public shape/schema."""
    features = [column for column in frame if column not in {target, *ids}]
    columns = []
    for name in features:
        series = frame[name]
        item = {'name': name, 'dtype': str(series.dtype), 'missing_ratio': float(series.isna().mean()),
                'unique_values': int(series.nunique(dropna=True))}
        if pd.api.types.is_numeric_dtype(series):
            values = pd.to_numeric(series, errors='coerce').replace([np.inf, -np.inf], np.nan).dropna()
            if len(values):
                item['numeric_summary'] = {key: float(value) for key, value in
                    values.quantile([0., .25, .5, .75, 1.]).rename(
                        index={0.: 'min', .25: 'q25', .5: 'median', .75: 'q75', 1.: 'max'}).items()}
        columns.append(item)
    counts = frame[target].value_counts(dropna=False)
    return {'profile_scope': 'outer_fit_only', 'rows': len(frame), 'feature_count': len(features),
            'columns': columns, 'class_counts': {str(label): int(count) for label, count in counts.items()},
            'number_of_classes': len(counts),
            'majority_minority_ratio': float(counts.max()/counts.min()),
            'test_rows': len(test), 'test_feature_columns': [name for name in test if name not in ids],
            'preprocessing_policy': 'Imputation, scaling and encoding are fitted inside training folds only; '
                                    'outer validation labels never enter EDA or model selection.'}


def _available_classifiers(requirements_text, installed):
    allowed = {match.group(1).lower() for line in requirements_text.splitlines()
               if (match := re.match(r'^\s*([A-Za-z0-9_.-]+)(?:\[[^\]]+\])?==', line))}
    return {family: {'distribution': distribution, 'installed_version': installed[distribution]}
            for family, distribution in CLASSIFICATION_LIBRARIES.items()
            if distribution.lower() in allowed and distribution in installed}


def classification_losses(n_classes):
    """Native classifier losses; this registry never contains regression losses."""
    return {
        'logistic_regression': {'fixed_loss': 'logistic_cross_entropy', 'native_options': [{}]},
        'sgd': {'native_options': [{'loss': 'log_loss'}, {'loss': 'modified_huber'}]},
        'random_forest': {'native_options': [{'criterion': 'gini'}, {'criterion': 'entropy'}, {'criterion': 'log_loss'}]},
        'extra_trees': {'native_options': [{'criterion': 'gini'}, {'criterion': 'entropy'}, {'criterion': 'log_loss'}]},
        'hist_gradient_boosting': {'native_options': [{'loss': 'log_loss'}]},
        'lightgbm': {'native_options': [{'objective': 'binary' if n_classes == 2 else 'multiclass'}]},
        'catboost': {'native_options': [{'loss_function': 'Logloss' if n_classes == 2 else 'MultiClass'}]},
        'xgboost': {'native_options': [{'objective': 'binary:logistic' if n_classes == 2 else 'multi:softprob'}]},
        'mlp': {'fixed_loss': 'logistic_cross_entropy', 'native_options': [{}]},
        'knn': {'fixed_loss': 'neighbor_vote', 'native_options': [{}]},
        'flaml': {'fixed_loss': 'classifier_engine_native_cross_entropy', 'native_options': [{}]},
        'autogluon': {'fixed_loss': 'classifier_engine_native_cross_entropy', 'native_options': [{}]},
    }


def _probe_worker(job, budget):
    """Short measured baselines on a subset strictly within frozen outer fit."""
    from sklearn.compose import ColumnTransformer, make_column_selector
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
    from threadpoolctl import threadpool_limits
    from classification_templates import plain_definition
    from classification_backend import metric_score
    started = time.monotonic()
    frame = pd.read_parquet(job['train']).iloc[job['fit']].reset_index(drop=True)
    if len(frame) > 10000:
        _, positions = train_test_split(np.arange(len(frame)), test_size=10000, random_state=43,
                                        stratify=frame[job['target']])
        frame = frame.iloc[positions].reset_index(drop=True)
    encoder = LabelEncoder().fit(frame[job['target']])
    y = encoder.transform(frame[job['target']])
    positive = int(encoder.transform([job['positive_label']])[0]) if job.get('positive_label') is not None else 1
    x = frame.drop(columns=[job['target'], *job['id_columns']]).replace([np.inf, -np.inf], np.nan)
    for column in x.select_dtypes(include=['object', 'string', 'category']):
        x[column] = x[column].astype(object).map(lambda value: str(value) if pd.notna(value) else np.nan)
    fit, valid = train_test_split(np.arange(len(frame)), test_size=.25, random_state=44, stratify=y)
    outcomes = []
    for family in ('lightgbm', 'logistic_regression', 'extra_trees'):
        if family not in job['available'] or time.monotonic()-started > budget*.75:
            continue
        one_started = time.monotonic()
        try:
            estimator, parameters = plain_definition(family, 43)
            parameters.update({'lightgbm': {'n_estimators': 75, 'num_leaves': 15},
                               'logistic_regression': {'max_iter': 150, 'tol': .003},
                               'extra_trees': {'n_estimators': 60, 'min_samples_leaf': 2}}[family])
            preparation = ColumnTransformer([
                ('numeric', Pipeline([('impute', SimpleImputer(strategy='median', keep_empty_features=True)),
                                      ('scale', StandardScaler())]), make_column_selector(dtype_include=np.number)),
                ('categorical', Pipeline([('impute', SimpleImputer(strategy='constant', fill_value='__MISSING__')),
                                          ('encode', OneHotEncoder(handle_unknown='ignore'))]),
                 make_column_selector(dtype_exclude=np.number))], sparse_threshold=.3)
            model = Pipeline([('preprocess', preparation), ('model', estimator(**parameters))])
            with threadpool_limits(limits=2):
                model.fit(x.iloc[fit], y[fit])
                prediction = model.predict(x.iloc[valid])
                probabilities = model.predict_proba(x.iloc[valid])
            outcomes.append({'family': family, 'status': 'completed',
                             'score': metric_score(y[valid], prediction, probabilities, job['metric'], len(encoder.classes_), positive),
                             'parameters': parameters, 'fit_rows': len(fit), 'validation_rows': len(valid),
                             'duration_seconds': time.monotonic()-one_started})
        except Exception as error:
            outcomes.append({'family': family, 'status': 'failed', 'error': str(error)[-1200:],
                             'duration_seconds': time.monotonic()-one_started})
    return {'status': 'completed', 'scope': 'inner_holdout_within_outer_fit_only', 'sample_rows': len(frame),
            'primary_metric': job['metric'], 'results': outcomes, 'duration_seconds': time.monotonic()-started,
            'limitation': 'Small inner probes are hints for model selection, not comparable to final outer scores.'}


def _freeze_selection_losses(selection, n_classes):
    registry = classification_losses(n_classes)
    models = []
    for model in selection.selected_models:
        item = model.model_dump(mode='json')
        options = registry[model.family]['native_options']
        loss_keys = set().union(*(set(option) for option in options))
        selected = {key: item['parameters'][key] for key in loss_keys if key in item['parameters']}
        compatible = [option for option in options if all(option.get(key) == value for key, value in selected.items())]
        if not compatible:
            raise ValueError('Selected native loss is not supported for this classification task: '+model.family)
        native = compatible[0]
        item['parameters'].update(native)
        item['selection_metadata'].update(loss_native=native, locked_parameters=list(native),
            loss_description=registry[model.family].get('fixed_loss', 'native_classifier_loss'))
        models.append(item)
    return models


async def run_classification_pipeline(task_dir, output, *, target, metric, id_columns=None,
                                      submission=None, time_budget=4800., trial_timeout=180., max_trials=12):
    if any(not math.isfinite(value) or value <= 0 for value in (time_budget, trial_timeout)):
        raise ValueError('Time limits must be positive and finite')
    started = time.monotonic()
    deadline = started+time_budget
    output, task_dir = Path(output).resolve(), Path(task_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    summary = {'status': 'running', 'stage': 'preparation', 'task_type': 'classification',
               'time_budget_seconds': time_budget, 'warnings': []}

    def checkpoint(**updates):
        summary.update(updates, elapsed_seconds=time.monotonic()-started)
        save(output/'pipeline_report.json', summary)

    try:
        checkpoint()
        ids = list(id_columns or [])
        handoff = output/'handoff'
        handoff.mkdir()
        sources = {role: next((task_dir/(role+extension) for extension in ('.csv', '.parquet')
                               if (task_dir/(role+extension)).is_file()), None)
                   for role in ('train', 'test', 'sample_submission')}
        if sources['train'] is None or sources['test'] is None:
            raise ValueError('Classification task directory must contain train.csv/test.csv or train.parquet/test.parquet')
        train, test = (_read_table(sources[role], ids) for role in ('train', 'test'))
        if target not in train or target in test:
            raise ValueError('Train must contain the target and test must not contain it')
        if not train.columns.is_unique or not test.columns.is_unique or not len(test):
            raise ValueError('Input columns must be unique and hidden test must be nonempty')
        if any(column not in train or column not in test for column in ids):
            raise ValueError('Every declared identifier must occur in train and test')
        if set(train.columns)-{target} != set(test.columns):
            raise ValueError('Train/test feature and identifier columns must match')
        before = len(train)
        train = train.loc[train[target].notna()].reset_index(drop=True)
        if before != len(train):
            summary['warnings'].append(f'Dropped {before-len(train)} rows with missing classification target')
        if len(train) < 30:
            raise ValueError('Need at least 30 labeled rows')
        if pd.api.types.is_numeric_dtype(train[target]) and not np.isfinite(train[target].to_numpy()).all():
            raise ValueError('Numeric classification labels must be finite')
        counts = train[target].value_counts()
        if len(counts) < 2 or counts.min() < 2:
            raise ValueError('Stratified classification validation requires at least two classes and two rows per class')
        # The split is made before EDA/selection and remains fixed for every candidate.
        fit, valid = train_test_split(np.arange(len(train)), test_size=.2, random_state=42,
                                    stratify=train[target].to_numpy())
        if set(train.iloc[fit][target].unique()) != set(counts.index) or set(train.iloc[valid][target].unique()) != set(counts.index):
            raise ValueError('Every class must occur in both stratified outer subsets')
        save(handoff/'split.json', {'fit': fit.tolist(), 'valid': valid.tolist()})
        files = {}
        for role, frame in (('train', train), ('test', test)):
            files[role] = str(handoff/(role+'.parquet'))
            frame.to_parquet(files[role], index=False)
        files['sample_submission'] = None
        if sources['sample_submission']:
            sample = _read_table(sources['sample_submission'], ids)
            files['sample_submission'] = str(handoff/'sample_submission.parquet')
            sample.to_parquet(files['sample_submission'], index=False)
        eda = _profile(train.iloc[fit], target, ids, test)
        save(handoff/'eda.json', eda)
        checkpoint(preparation={'status': 'completed', 'train_rows': len(train), 'outer_fit_rows': len(fit),
                                'outer_validation_rows': len(valid), 'test_rows': len(test),
                                'class_count': len(counts), 'dropped_missing_target_rows': before-len(train),
                                'profile_scope': 'outer_fit_only', 'split': 'stratified_random_seed42'},
                   stage='model_selection')
        selection_output = output/'model_selection'
        selection_output.mkdir()
        root = Path(__file__).resolve().parent
        requirements = (root/'requirements.txt').read_text(encoding='utf-8-sig')
        available = _available_classifiers(requirements, installed_libraries())
        if not available:
            raise ValueError('No classification libraries are both installed and permitted by requirements.txt')
        probe_dir = selection_output/'probe'
        probe_dir.mkdir()
        save(probe_dir/'job.json', {'train': files['train'], 'fit': fit.tolist(), 'target': target,
            'id_columns': ids, 'metric': metric, 'available': list(available), 'output_dir': str(probe_dir),
            'positive_label': (submission or {}).get('positive_label')})
        try:
            probe = await run_job(probe_dir/'job.json', probe_dir, deadline,
                                  min(90., time_budget*.1), module='classification_pipeline')
        except Exception as error:
            probe = {'status': 'failed', 'scope': 'inner_holdout_within_outer_fit_only',
                     'error': str(error)[-1500:], 'results': []}
            summary['warnings'].append('Classification inner probe failed; no score fabricated: '+str(error)[-1000:])
        save(selection_output/'probe.json', probe)
        selection_budget = min(180., time_budget*.2, deadline-time.monotonic())
        selection_started = time.monotonic()
        selection_source = 'qwen'
        agents = None
        selection_input = {'task_type': 'classification', 'primary_metric': metric, 'eda': eda,
                           'available_classifiers': available, 'inner_probe': probe,
                           'classification_loss_registry': classification_losses(len(counts)),
                           'remaining_seconds': max(0., deadline-time.monotonic()),
                           'trial_timeout_seconds': trial_timeout, 'max_trials': max_trials}
        save(selection_output/'stage_input.json', selection_input)
        try:
            agents = JSONAgents(Settings.from_env(), min(deadline, time.monotonic()+selection_budget), max_calls=2)
            agents.root.log_directory = selection_output
            prompt = ('Choose classifier families for a TABULAR CLASSIFICATION task using this measured '
                'outer-fit-only profile. The upstream primary metric is immutable. Recommend 3-5 diverse '
                'justified families, or fewer if the budget makes that necessary. You select model families '
                'and initial parameters; another Qwen will tune them and a deterministic evaluator will rank '
                'them by the supplied primary metric. Select only available_classifiers. Do not install '
                'dependencies or generate Python code. All models are CLASSIFIERS, never regressors. '
                'target_transform must be none. Keep parameters small and native: ordinary classifiers '
                'use sklearn-compatible APIs; CatBoost uses iterations/thread_count/auto_class_weights, '
                'LightGBM uses n_estimators/num_leaves/class_weight. Avoid hard-coding a binary objective '
                'for multiclass data. The backend owns label encoding, probabilities, CPU limits, preprocessing, '
                'stratified folds and final all-row refit. AutoML can be selected when its additional cost '
                'is justified; FLAML task and AutoGluon problem type are owned by the backend. '
                'For imbalanced data consider macro F1 and class_weight, but do not claim a score without '
                'measuring it. Neither outer validation labels nor hidden test labels are available. '
                'Measured inner_probe results are small-sample hints, not final outer scores. Choose a loss '
                'from classification_loss_registry for each classifier and put its native options directly '
                'in parameters. These loss parameters are frozen during later tuning. The primary metric '
                'remains fixed even when training loss differs.\n'+
                json.dumps(selection_input, ensure_ascii=False))
            selected = await asyncio.wait_for(agents.ask(ClassificationSelection, prompt,
                'classification_model_selection', validation={'allowed_families': list(available),
                                                              'n_classes': len(counts)}), selection_budget)
        except Exception as error:
            selection_source = 'explicit_rules_fallback'
            reason = str(error) or f'Qwen selection exceeded {selection_budget:.1f}s'
            summary['warnings'].append('Classification model selection used explicit fallback: '+reason[-1200:])
            fallback = [family for family in ('lightgbm', 'catboost', 'logistic_regression') if family in available]
            if not fallback:
                fallback = [next(iter(available))]
            selected = ClassificationSelection.model_validate({'selected_models': [
                {'family': family, 'rationale': 'Unmeasured baseline selected by explicit fallback rules after Qwen failure',
                 'selection_metadata': {'source': selection_source}} for family in fallback],
                'rationale': 'No score is assumed; the common training stage will compare these permitted classifiers.'},
                context={'allowed_families': list(available)})
        selected_models = _freeze_selection_losses(selected, len(counts))
        save(selection_output/'selection.json', {**selected.model_dump(mode='json'), 'selected_models': selected_models})
        selection_report = {'status': 'completed', 'source': selection_source,
            'duration_seconds': time.monotonic()-selection_started,
            'selected_models': selected_models, 'probe': probe,
            'llm_calls': [call.model_dump(mode='json') for call in agents.calls] if agents else []}
        save(selection_output/'result.json', selection_report)
        save(selection_output/'stage_output.json', {'task_type': 'classification', 'primary_metric': metric,
             **selection_report, 'rationale': selected.rationale})
        submission_spec = dict(submission or {})
        submission_spec.setdefault('filename', 'submission.csv')
        submission_spec.setdefault('prediction_column', target)
        raw = {'schema_version': 1, 'task_type': 'classification', 'files': files,
               'target': target, 'id_columns': ids, 'primary_metric': metric,
               'selected_models': selected_models,
               'eda': eda, 'outer_split': str(handoff/'split.json'), 'submission': submission_spec}
        contract = TrainingInput.model_validate(raw)
        input_path = handoff/'input.json'
        save(input_path, contract)
        save(handoff/'provenance.json', {'source_hashes': {role: file_hash(path) for role, path in sources.items() if path},
            'profile_scope': 'outer_fit_only', 'split_sha256': file_hash(handoff/'split.json'),
            'selection_source': selection_source, 'primary_metric': metric,
            'hidden_test_labels_used': False})
        checkpoint(stage='training', model_selection=selection_report, training_input=str(input_path))
        remaining = deadline-time.monotonic()
        if remaining <= 0:
            raise TimeoutError('No budget remains after classification preparation/model selection')
        report = await run_training(input_path, output/'training', time_budget=remaining,
                                    trial_timeout=trial_timeout, max_trials=max_trials)
        result_submission = report.get('submission')
        if result_submission and Path(result_submission).is_file():
            destination = output/Path(result_submission).name
            atomic_bytes(destination, Path(result_submission).read_bytes())
            summary['submission'] = str(destination)
        checkpoint(status=report['status'], stage='complete', primary_metric=report['primary_metric'],
                   winner=report.get('winner'), search_stop_reason=report.get('search_stop_reason'),
                   full_training_rows=report.get('final_refit', {}).get('full_training_rows'),
                   training_report=str(output/'training/report.json'), error=report.get('error'))
    except Exception as error:
        checkpoint(status='failed', stage='complete', error=str(error)[-2000:])
    return summary


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Isolated classification model-selection probe worker')
    parser.add_argument('--job', type=Path, required=True)
    parser.add_argument('--time-budget', type=float, required=True)
    args = parser.parse_args()
    job = json.loads(args.job.read_text(encoding='utf-8'))
    try:
        result = _probe_worker(job, args.time_budget)
    except Exception as error:
        result = {'status': 'failed', 'error': str(error)[-2000:], 'error_type': type(error).__name__}
    save(Path(job['output_dir'])/'result.json', result)
    raise SystemExit(0 if result['status'] == 'completed' else 2)

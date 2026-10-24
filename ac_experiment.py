"""Classifier experiments and all-row refits, isolated by hard process deadlines."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator
from sklearn.metrics import classification_report, confusion_matrix

from ac_data import Features, period_values
from ac_custom import apply_code, check_code
from ac_models import fit_predict, labels_from_probabilities, score, validate_spec
from ac_diagnostics import fit_class_weights
from ac_eda import FeaturePolicy, FittedFeaturePolicy


class ModelSpec(BaseModel):
    model_config = ConfigDict(extra='forbid')
    model_id: str = 'lightgbm'
    loss: str = 'cross_entropy'
    params: dict = Field(default_factory=dict)


class TrainingSpec(BaseModel):
    model_config = ConfigDict(extra='forbid')
    sample_weight: str = 'none'
    recent_decay: float = Field(default=.8, ge=.2, le=1.)
    drop_patterns: list[str] = Field(default_factory=list)
    threshold: float = Field(default=.5, ge=.05, le=.95)
    class_weight_power: float = Field(default=0., ge=0., le=1.)
    class_weight_clip: float = Field(default=20., ge=1., le=100.)
    max_boost_rounds: int = Field(default=2000, ge=20, le=2000)

    @model_validator(mode='after')
    def valid_weight(self):
        if self.sample_weight not in {'none', 'recent'}:
            raise ValueError('sample_weight must be none or recent')
        return self


class ExperimentSpec(BaseModel):
    model_config = ConfigDict(extra='forbid')
    blocks: list[str] = Field(default_factory=lambda: ['base'])
    model: ModelSpec = Field(default_factory=ModelSpec)
    training: TrainingSpec = Field(default_factory=TrainingSpec)
    feature_code: str | None = Field(default=None, max_length=12000)
    preprocessing: FeaturePolicy = Field(default_factory=FeaturePolicy)

    @model_validator(mode='after')
    def native_classifier(self):
        errors = validate_spec(self.model.model_id, self.model.loss, self.model.params)
        if self.model.model_id == 'prior' and (self.feature_code or self.preprocessing != FeaturePolicy()
                or set(self.blocks)-{'base'} or self.training.class_weight_power or self.training.drop_patterns):
            errors.append('The emergency prior baseline uses class counts only; no feature policies or weighting apply')
        if self.training.class_weight_power and (self.model.params.get('class_weight') is not None
                or self.model.params.get('auto_class_weights') is not None):
            errors.append('Use either soft sample weights or native class weights, never both')
        if self.feature_code:
            errors += check_code(self.feature_code)
        if errors:
            raise ValueError('; '.join(errors))
        return self

    def key(self):
        value = self.model_dump()
        value['blocks'] = sorted(set(['base', *self.blocks]))
        return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()[:16]

    def summary(self):
        return f'{self.model.model_id}/{self.model.loss} [{"+".join(self.blocks)}] {self.model.params}' + (f' soft_weights={self.training.class_weight_power:g}' if self.training.class_weight_power else '') + (f' rounds<={self.training.max_boost_rounds}' if self.training.max_boost_rounds < 2000 else '') + (' + EDA policy' if self.preprocessing != FeaturePolicy() else '') + (' + Qwen function' if self.feature_code else '')


def worker(spec_path, context_path, output):
    spec = ExperimentSpec.model_validate_json(Path(spec_path).read_text(encoding='utf-8'))
    context = json.loads(Path(context_path).read_text(encoding='utf-8'))
    output = Path(output)
    frame = pd.read_parquet(context['features_path'])
    y = np.load(context['labels_path'], allow_pickle=False)
    started = time.monotonic()
    count = context['n_train']
    classes = context['classes']
    positive = context['positive_index']
    preparation_reports = []
    def progress(stage, completed=0, total=0):
        (output/'progress.json').write_text(json.dumps({'stage': stage, 'completed_folds': completed,
            'total_folds': total, 'elapsed_s': time.monotonic()-started}), encoding='utf-8')

    def prepared(source, apply):
        if spec.model.model_id == 'prior':
            return pd.DataFrame({'prior': np.zeros(len(source))}), [pd.DataFrame({'prior': np.zeros(len(rows))}) for rows in apply]
        generator = Features(frame, context['target'], context['id_cols'], len(classes),
            context['time_column'], context['entity_key'], context['group_columns'], context['seed'])
        generator.fit(source, y[source], spec.blocks)
        train = generator.transform(source)
        other = [generator.transform(rows) for rows in apply]
        if spec.preprocessing != FeaturePolicy():
            preparation = FittedFeaturePolicy(spec.preprocessing).fit(train, frame.iloc[source])
            train = preparation.transform(train, frame.iloc[source])
            other = [preparation.transform(values, frame.iloc[rows]) for values, rows in zip(other, apply)]
            if len(preparation_reports) < 10:preparation_reports.append(preparation.report)
        if spec.feature_code:
            train = apply_code(spec.feature_code, train)
            other = [apply_code(spec.feature_code, values) for values in other]
        if spec.training.drop_patterns:
            keep = [name for name in train if not any(pattern in name for pattern in spec.training.drop_patterns)]
            if not keep:
                raise ValueError('drop_patterns removed all feature columns')
            train, other = train[keep], [values[keep] for values in other]
        if any(list(values) != list(train) for values in other):
            raise ValueError('Generated feature schema differs between fit and predictions')
        return train, other

    def weights(source):
        class_weights = fit_class_weights(y[source], spec.training.class_weight_power, spec.training.class_weight_clip)
        if spec.training.sample_weight == 'recent':
            if not context['time_column']:
                raise ValueError('Recent sample weights require an actual temporal column')
            values = period_values(frame.iloc[source][context['time_column']]).to_numpy()
            # Scale rank, not nanoseconds/calendar-year distances.
            ranks = np.searchsorted(np.unique(values), values)
            recent = spec.training.recent_decay**(ranks.max()-ranks)
            return recent if class_weights is None else recent*class_weights
        return class_weights

    kwargs = dict(model_id=spec.model.model_id, loss=spec.model.loss, params=spec.model.params,
        n_classes=len(classes), n_jobs=context['n_jobs'], seed=context['seed'],
        max_rounds=spec.training.max_boost_rounds)
    if context['mode'] == 'cv':
        from ar_validation import load_folds
        folds = load_folds(Path(context['folds_path']))
        oof = np.full((count, len(classes)), np.nan)
        fold_scores, train_scores, iterations = [], [], []
        importance = []
        metadata = []
        for fold_index, fold in enumerate(folds, 1):
            progress('preparing_fold', fold_index-1, len(folds))
            tr, fit, es, valid = (fold[key] for key in ('train', 'fit', 'early_stop', 'valid'))
            print(f'Fold {fold_index}/{len(folds)}: preparing fit={len(fit)}, train={len(tr)}, valid={len(valid)}', flush=True)
            print('Fit class counts: '+str(np.bincount(y[fit], minlength=len(classes)).tolist()), flush=True)
            # Early stopping sees no outer labels. Then refit on ALL outer-train
            # rows with the measured iteration count and freshly fitted history.
            if len(es) and spec.model.model_id in {'lightgbm', 'catboost', 'xgboost'}:
                xf, (xe,) = prepared(fit, [es])
                _, iteration, _ = fit_predict(x=xf, y=y[fit], early_x=xe, early_y=y[es], predict=[],
                    sample_weight=weights(fit), **kwargs)
            else:
                iteration = min(250, spec.training.max_boost_rounds) if spec.model.model_id in {'lightgbm', 'catboost', 'xgboost'} else None
            train_sample = tr[np.random.default_rng(42).choice(len(tr), min(10000, len(tr)), replace=False)]
            xt, (xv, xs) = prepared(tr, [valid, train_sample])
            (prediction, prediction_train), fitted_iterations, imp = fit_predict(x=xt, y=y[tr],
                early_x=None, early_y=None, predict=[xv, xs], n_rounds=iteration, sample_weight=weights(tr), **kwargs)
            oof[valid] = prediction
            progress('fold_completed', fold_index, len(folds))
            print(f'Fold {fold_index}/{len(folds)}: classifier finished in {time.monotonic()-started:.1f}s cumulative', flush=True)
            try:
                fold_scores.append(score(context['metric'], y[valid], prediction, spec.training.threshold, positive))
            except ValueError:
                if context['metric'] != 'roc_auc':raise
                fold_scores.append(None)
            try:
                train_scores.append(score(context['metric'], y[train_sample], prediction_train, spec.training.threshold, positive))
            except ValueError:
                if context['metric'] != 'roc_auc':raise
            iterations.append(fitted_iterations)
            metadata.append({'source_rows': len(tr), 'valid_rows': len(valid), 'columns': xt.shape[1],
                             'class_counts_fit': np.bincount(y[fit], minlength=len(classes)).tolist(),
                             'class_counts_validation': np.bincount(y[valid], minlength=len(classes)).tolist(),
                             'history_source': 'outer_train_only; strict earlier time for temporal features'})
            if imp is not None:importance.append(imp/max(float(imp.sum()), 1e-12))
        mask = np.isfinite(oof).all(axis=1)
        np.save(output/'oof.npy', oof, allow_pickle=False)
        pooled = score(context['metric'], y[:count][mask], oof[mask], spec.training.threshold, positive)
        predicted = labels_from_probabilities(oof[mask], spec.training.threshold, positive)
        evidence = pd.DataFrame({'row_index': np.where(mask)[0], 'true_label': y[:count][mask], 'prediction': predicted})
        for i in range(len(classes)):evidence[f'probability_class_{i}'] = oof[mask, i]
        evidence.to_csv(output/'validation_predictions.csv', index=False)
        finite_folds = [value for value in fold_scores if value is not None]
        result = {'status': 'ok', 'cv': pooled, 'folds': fold_scores, 'cv_std': float(np.std(finite_folds)) if finite_folds else None,
            'train_cv': float(np.mean(train_scores)) if train_scores else None, 'best_iters': iterations, 'validation_rows': int(mask.sum()),
            'score_scope': 'OOF on fixed outer folds; adaptively inspected during research', 'fold_details': metadata,
            'classification_report': classification_report(y[:count][mask], predicted, labels=np.arange(len(classes)),
                target_names=[str(value) for value in classes], zero_division=0, output_dict=True),
            'confusion_matrix': confusion_matrix(y[:count][mask], predicted, labels=np.arange(len(classes))).tolist()}
        if importance:
            imp = pd.concat(importance, axis=1).mean(axis=1).sort_values(ascending=False)
            result['top_features'] = [[str(name), float(value)] for name, value in imp.head(15).items()]
    else:
        progress('full_data_refit')
        source = np.arange(count)
        xt, (test,) = prepared(source, [np.arange(count, len(frame))])
        (prediction,), iteration, _ = fit_predict(x=xt, y=y, early_x=None, early_y=None, predict=[test],
            n_rounds=context.get('n_rounds') or 300, sample_weight=weights(source), **kwargs)
        np.save(output/'test_pred.npy', prediction, allow_pickle=False)
        result = {'status': 'ok', 'n_rounds': iteration, 'full_training_rows': count, 'score': None,
                  'prediction_rows': len(prediction), 'score_scope': 'no score for full-data refit'}
    result['seconds'] = time.monotonic()-started
    result['preprocessing_fits'] = preparation_reports
    (output/'result.json').write_text(json.dumps(result, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def run_in_subprocess(spec, context, output, timeout):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output/'spec.json').write_text(spec.model_dump_json(indent=2), encoding='utf-8')
    (output/'ctx.json').write_text(json.dumps(context), encoding='utf-8')
    started = time.monotonic()
    env = {key: value for key, value in os.environ.items() if key.upper() in
           {'PATH', 'SYSTEMROOT', 'WINDIR', 'TEMP', 'TMP', 'USERPROFILE', 'LOCALAPPDATA'}}
    env.update(PYTHONIOENCODING='utf-8', OMP_NUM_THREADS=str(context['n_jobs']),
               OPENBLAS_NUM_THREADS=str(context['n_jobs']), MKL_NUM_THREADS=str(context['n_jobs']))
    with (output/'execution.log').open('wb') as log:
        process = subprocess.Popen([sys.executable, '-u', str(Path(__file__).resolve()),
            str(output/'spec.json'), str(output/'ctx.json'), str(output)], env=env,
            stdout=log, stderr=subprocess.STDOUT, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        try:
            process.wait(timeout=max(.1, timeout))
        except subprocess.TimeoutExpired:
            import psutil
            try:
                for child in reversed(psutil.Process(process.pid).children(recursive=True)):
                    try:child.kill()
                    except psutil.Error:pass
            except psutil.Error:pass
            process.kill();process.wait()
            try:details = json.loads((output/'progress.json').read_text(encoding='utf-8'))
            except (OSError, ValueError):details = {}
            return {'status': 'timeout', 'error': 'Classification experiment deadline exceeded; partial folds are not scored',
                'seconds': time.monotonic()-started, 'timeout_scope': context.get('timeout_scope', 'experiment'),
                'retryable': True, 'progress': details}
    if process.returncode or not (output/'result.json').is_file():
        return {'status': 'failed', 'error': (output/'execution.log').read_text(encoding='utf-8', errors='replace')[-1800:],
                'seconds': time.monotonic()-started}
    result = json.loads((output/'result.json').read_text(encoding='utf-8'))
    result['worker_seconds'] = result.get('seconds')
    result['seconds'] = time.monotonic()-started
    return result


if __name__ == '__main__':
    try:
        worker(*sys.argv[1:4])
    except subprocess.TimeoutExpired:
        (Path(sys.argv[3])/'result.json').write_text(json.dumps({'status': 'timeout',
            'error': 'Generated feature function exceeded its time limit', 'timeout_scope': 'feature_function',
            'retryable': True, 'seconds': 0.}), encoding='utf-8')

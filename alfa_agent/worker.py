"""Common runtime: measured preparation and one generated experiment in a child process."""

import argparse
import csv
import io
import json
import os
from pathlib import Path
import pickle
import re
import sys
import time
import tempfile
import warnings

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .contracts import Finalization, Finish, Preparation, Start
from .data import file_hash, input_hashes, module, profile, within
from .storage import atomic_bytes, load, save

# Two executors share the platform's 8 CPUs; the controller passes these through the child environment.
THREADS = int(os.environ.get('ALFA_THREADS', '2'))
SHIFT_CHECK = os.environ.get('ALFA_SHIFT_CHECK') == '1'


def validate_submission(path: Path, output, expected_rows: int, *, test=None, id_columns=()):
    if not path.is_file() or not path.stat().st_size:
        raise ValueError('Submission writer produced no nonempty output file')
    if (output.format or '').upper() not in {'CSV', 'TSV'}:
        return
    with path.open(encoding='utf-8-sig', newline='') as stream:
        rows = csv.reader(stream, delimiter='\t' if output.format.upper() == 'TSV' else ',', strict=True)
        header = next(rows, [])
        if not header or (output.columns and header != output.columns):
            raise ValueError('Submission columns or column order differ from preparation.output.columns')
        identifiers = {header.index(column): column for column in id_columns if column in header}
        if test is not None and (len(test) != expected_rows or any(column not in test for column in identifiers.values())):
            raise ValueError('Submission identifier reference does not match the test table')
        count = 0
        for row in rows:
            if len(row) != len(header):
                raise ValueError('Submission contains a row with the wrong number of columns')
            for index, value in enumerate(row):
                if not value.strip():
                    raise ValueError(f'Submission contains an empty value in {header[index]}')
                if index in identifiers and test is not None:
                    column = identifiers[index]
                    if count >= len(test) or value != str(test[column].iloc[count]):
                        raise ValueError(f'Submission identifier/order differs from test in {column} at row {count + 1}')
            count += 1
        if count != expected_rows:
            raise ValueError(f'Submission row count {count} differs from test row count {expected_rows}')


def ordinal(series):
    """Numeric or parseable date values; None for other columns."""
    if pd.api.types.is_bool_dtype(series):
        return None
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(float)
    if not (pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)):
        return None
    present = series.dropna().astype(str)
    if not len(present) or present.str.fullmatch(r'[\d\s:/.T+-]{6,}').mean() < .95:
        return None
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        parsed = pd.to_datetime(series, errors='coerce', format='mixed')
    if parsed.notna().sum() < .95 * len(present):
        return None
    return pd.Series(parsed.to_numpy().astype('datetime64[ns]').astype(np.int64), index=series.index).where(parsed.notna()).astype(float)


def adversarial(train, test, columns):
    """Out-of-fold probability that a train row looks like test; AUC near 0.5 means no measurable shift."""
    import lightgbm
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    frame = pd.concat([train[columns], test[columns]], ignore_index=True)
    for column in columns:
        values = ordinal(frame[column])
        frame[column] = values if values is not None else frame[column].astype(str).astype('category').cat.codes
    labels = np.r_[np.zeros(len(train)), np.ones(len(test))]
    probability, gains = np.zeros(len(frame)), np.zeros(len(columns))
    for fit, valid in StratifiedKFold(3, shuffle=True, random_state=0).split(frame, labels):
        model = lightgbm.LGBMClassifier(n_estimators=100, learning_rate=.1, subsample=.8, subsample_freq=1,
                                        colsample_bytree=.8, importance_type='gain', n_jobs=THREADS, verbose=-1)
        model.fit(frame.iloc[fit], labels[fit])
        probability[valid] = model.predict_proba(frame.iloc[valid])[:, 1]
        gains += model.feature_importances_
    top = [columns[i] for i in np.argsort(-gains)[:5]]
    return float(roc_auc_score(labels, probability)), probability[:len(train)], top


def shift_split(train, test, data, validation, fit, valid):
    """Measure train/test shift; make validation resemble test when the generated split clearly does not.

    A test period strictly after all train values requires a temporal holdout. Otherwise strong covariate
    shift (adversarial AUC >= 0.8) selects the most test-like train rows, unless a group/time split was chosen.
    """
    excluded = set(data.target_columns) | set(data.id_columns)
    columns = [c for c in test.columns if c in train.columns and c not in excluded]
    report, temporal = {}, []
    for column in columns:
        before, after = ordinal(train[column]), ordinal(test[column])
        if before is None or after is None or before.nunique() < 2 or not after.notna().any():
            continue
        # Period codes and timestamps are whole numbers; an out-of-range continuous feature is covariate shift.
        whole = np.all(np.mod(before.dropna().to_numpy(), 1) == 0) and np.all(np.mod(after.dropna().to_numpy(), 1) == 0)
        if whole and after.min() > before.max():
            temporal.append((before.nunique(), column, before))
    if temporal:
        _, column, values = max(temporal, key=lambda item: item[0])
        report.update(temporal_column=column, train_range=[float(values.min()), float(values.max())],
                      test_range=[float(ordinal(test[column]).min()), float(ordinal(test[column]).max())])
        if not values.iloc[fit].max() <= values.iloc[valid].min():
            positions = np.flatnonzero(values.notna().to_numpy())
            threshold = np.quantile(values.iloc[positions], .8, method='higher')
            # A repeated earliest period can occupy the 80th percentile. Keep whole periods
            # together, but move the boundary forward so the training side is never empty.
            periods = np.sort(values.iloc[positions].unique())
            if threshold <= periods[0]:
                threshold = periods[1]  # temporal candidates have at least two distinct periods
            later = values.iloc[positions].to_numpy() >= threshold
            fit, valid = positions[~later], positions[later]
            report.update(override='temporal_holdout', reason=f'All test values of {column} are later than train; '
                          f'validation is the latest train period(s) {column} >= {threshold:g}.')
        return fit, valid, report
    try:
        auc, probability, top = adversarial(train, test, columns)
    except Exception as error:  # Diagnostics must not block a task the generated split already supports.
        return fit, valid, {'adversarial_error': f'{type(error).__name__}: {error}'[:300]}
    report.update(adversarial_auc=round(auc, 4), shifted_features=top)
    described = json.dumps(validation, default=str).casefold()
    if auc >= .8 and not any(word in described for word in ('group', 'time', 'temporal', 'date', 'chrono', 'order')):
        later = np.zeros(len(train), bool)
        later[np.argsort(-probability, kind='stable')[:max(1, round(.2 * len(train)))]] = True
        fit, valid = np.flatnonzero(~later), np.flatnonzero(later)
        report.update(override='adversarial_holdout', reason='Train and test differ strongly; validation is the '
                      '20% of train rows a classifier finds most test-like.')
    return fit, valid, report


def test_overlap(train, test, data):
    """Share of test rows whose exact value occurs in train, for high-cardinality columns.

    Near 1 for a many-valued attribute hints at repeated entities (the same object seen in train),
    whose earlier targets can be strong leakage-safe features.
    """
    excluded = set(data.target_columns) | set(data.id_columns)
    shares = {}
    for column in test.columns:
        if column in excluded or column not in train or train[column].nunique() < 50:
            continue
        try:
            shares[str(column)] = round(float(test[column].isin(set(train[column].dropna())).mean()), 3)
        except TypeError:
            continue
    return dict(sorted(shares.items(), key=lambda item: -item[1])[:15])


def target_summary(train, targets):
    result = {}
    for column in targets:
        series = train[column]
        summary = {'distinct': int(series.nunique()), 'missing': int(series.isna().sum())}
        if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
            values = series.dropna().astype(float)
            summary.update(mean=float(values.mean()), std=float(values.std()), skew=float(values.skew()),
                           quantiles={str(q): float(values.quantile(q)) for q in (0, .01, .25, .5, .75, .99, 1)})
        if summary['distinct'] <= 50:
            summary['counts'] = {str(k): int(v) for k, v in series.value_counts().items()}
        result[column] = json.loads(json.dumps(summary).replace('NaN', 'null'))
    return result


def validate_split(fit, valid, rows):
    for indices in (fit, valid):
        if indices.ndim != 1 or not len(indices) or indices.dtype.kind not in 'iu':
            raise ValueError('Split must contain nonempty one-dimensional integer row positions')
        if len(np.unique(indices)) != len(indices) or indices.min() < 0 or indices.max() >= rows:
            raise ValueError('Split contains duplicate or out-of-range rows')
    if np.intersect1d(fit, valid).size:
        raise ValueError('Fit and validation rows overlap')


def prepare(directory: Path):
    path = directory / 'preparation.json'
    preparation = load(path, Preparation)
    reader = module(directory / 'prepare.py', preparation.code_sha256)
    config = preparation.model_dump(mode='json')
    source = Path(preparation.task_directory)
    hashes = input_hashes(source, preparation.data.files)
    tables = reader.load_data(source, config)
    measured = profile(tables)
    train, test = tables['train'], tables['test']
    if any(column not in train for column in preparation.data.target_columns):
        raise ValueError('Prepared target columns are absent from the loaded training table')
    fit, valid = (np.asarray(indices) for indices in reader.make_split(train, config))
    validate_split(fit, valid, len(train))
    issues, shift = [], {}
    if SHIFT_CHECK and preparation.prediction_scope == 'row':
        fit, valid, shift = shift_split(train, test, preparation.data, preparation.validation, fit, valid)
        validate_split(fit, valid, len(train))
        if shift.get('override'):
            issues.append('Validation split replaced: ' + shift['reason'])
        measured['target'] = target_summary(train, preparation.data.target_columns)
        measured['test_overlap'] = test_overlap(train, test, preparation.data)
    buffer = io.BytesIO()
    np.savez(buffer, fit=fit.astype(np.int64), valid=valid.astype(np.int64))
    atomic_bytes(directory / 'split.npz', buffer.getvalue())
    validation = {**preparation.validation, 'split_file': 'split.npz', 'fit_rows': len(fit),
                  'validation_rows': len(valid), 'split_sha256': file_hash(directory / 'split.npz'),
                  **({'shift': shift} if shift else {})}
    metric_ready = preparation.metric.status == 'ready' and callable(getattr(reader, 'score', None))
    metric = preparation.metric.model_copy(update={'status': 'ready' if metric_ready else 'needs_implementation'})
    issues += [] if metric_ready else ['Metric implementation is unavailable; predictions can still be generated']
    prepared = preparation.model_copy(update={'status': 'ready' if metric_ready else 'partial',
        'profile': measured, 'source_hashes': hashes, 'validation': validation, 'metric': metric, 'issues': issues})
    save(path, prepared)


def prediction_frame(predictor, features):
    before = features.copy(deep=True)
    result = predictor(features)
    if not features.equals(before):
        raise ValueError('Predictor mutated its input frame; use a copy for preprocessing')
    if not isinstance(result, pd.DataFrame) or len(result) != len(features):
        raise ValueError('Predictor must return a DataFrame with one row per input row')
    if result.isna().any().any() or any(not np.isfinite(result[c]).all() for c in result.select_dtypes('number')):
        raise ValueError('Predictions contain missing or non-finite values')
    return result.reset_index(drop=True)


def check_predictor(predictor, features, predictions):
    """Probe the SAME fitted predictor, never retrain a stochastic model for this check."""
    indices = np.unique(np.linspace(0, len(features) - 1, min(17, len(features)), dtype=int))[::-1]
    for positions in (indices, indices[:1]):
        actual = prediction_frame(predictor, features.iloc[positions].copy())
        expected = predictions.iloc[positions].reset_index(drop=True)
        try:
            pd.testing.assert_frame_equal(actual, expected, check_dtype=False, check_exact=False, rtol=1e-6, atol=1e-8)
        except AssertionError:
            raise ValueError('Frozen predictor changed predictions for the same rows in a reordered subset; '
                             'check batch-fitted encoders/statistics, category codes and inference randomness') from None


def diagnostics(reference, predictions, preparation):
    """Auxiliary measurements only; never replace the task's scorer with these metrics."""
    targets = preparation.data.target_columns
    if len(targets) != 1 or targets[0] not in predictions:
        return {}
    target = targets[0]
    kind = (preparation.task_type or '').casefold()
    if 'classif' in kind or 'классиф' in kind:
        from sklearn.metrics import precision_recall_fscore_support
        from sklearn.utils.multiclass import type_of_target
        truth, predicted = reference[target], predictions[target]
        if type_of_target(predicted) not in {'binary', 'multiclass'}:
            return {}  # A probability column is not a vector of predicted class labels.
        if pd.api.types.is_numeric_dtype(truth):
            truth, predicted = truth.astype(float), pd.to_numeric(predicted).astype(float)
        else:
            truth, predicted = truth.astype(str), predicted.astype(str)
        labels = sorted(set(truth) | set(predicted))
        precision, recall, f1, support = precision_recall_fscore_support(truth, predicted, labels=labels, zero_division=0)
        return {'classification': {str(label): {'precision': float(p), 'recall': float(r), 'f1': float(f), 'support': int(s)}
                for label, p, r, f, s in zip(labels, precision, recall, f1, support)}}
    if 'regress' in kind or 'регресс' in kind:
        residual = reference[target].to_numpy(dtype=float) - predictions[target].to_numpy(dtype=float)
        return {'regression': {'mae': float(np.mean(np.abs(residual))), 'rmse': float(np.sqrt(np.mean(residual ** 2)))}}
    return {}


def with_scorer(config, reader, preparation):
    """In-process callable for inner-split selection inside fit_model; it never sees outer validation rows."""
    return {**config, 'scorer': reader.score} if preparation.metric.status == 'ready' else config


def predict(preparation: Preparation, root: Path, fit_model, *, small=False):
    source = Path(preparation.task_directory)
    if input_hashes(source, preparation.data.files) != preparation.source_hashes:
        raise ValueError('Input files changed after preparation')
    split_path = within(root, preparation.validation['split_file'])
    if file_hash(split_path) != preparation.validation['split_sha256']:
        raise ValueError('Validation split changed after preparation')
    reader = module(root / 'prepare.py', preparation.code_sha256)
    config = preparation.model_dump(mode='json')
    tables = reader.load_data(source, config)
    train, test = tables['train'], tables['test']
    with np.load(split_path, allow_pickle=False) as split:
        fit, valid = split['fit'], split['valid']
    if small:
        fit, valid = fit[:200], valid[:50]
        tables = {key: frame.iloc[:50].copy() if key != 'train' and len(frame) == len(test) else frame
                  for key, frame in tables.items()}
        test = tables['test']
    reference = train.iloc[valid].reset_index(drop=True)
    validation_input = reference.drop(columns=preparation.data.target_columns)
    with threadpool_limits(limits=THREADS):
        # A fresh 0..n-1 index: generated code mixes label and positional indexing.
        predictor = fit_model(train.iloc[fit].reset_index(drop=True), with_scorer(config, reader, preparation))
        if not callable(predictor):
            raise ValueError('fit_model must return a fitted callable: predict(features) -> DataFrame')
        valid_predictions = prediction_frame(predictor, validation_input.copy())
        test_predictions = prediction_frame(predictor, test.copy())
        if preparation.prediction_scope == 'row':
            check_predictor(predictor, validation_input, valid_predictions)
            check_predictor(predictor, test, test_predictions)
    return reader, config, tables, reference, valid_predictions.reset_index(drop=True), test_predictions.reset_index(drop=True)


def smoke(directory: Path, preparation_path: Path):
    preparation = load(preparation_path, Preparation)
    program = module(directory / 'experiment.py', file_hash(directory / 'experiment.py'))
    reader, config, tables, _, _, predictions = predict(preparation, preparation_path.parent, program.fit_model, small=True)
    with tempfile.TemporaryDirectory() as temporary:
        destination = within(Path(temporary), preparation.output.filename)
        destination.parent.mkdir(parents=True, exist_ok=True)
        reader.write_submission(tables['test'], predictions, tables, config, destination)
        validate_submission(destination, preparation.output, len(tables['test']),
                            test=tables['test'], id_columns=preparation.data.id_columns)
    atomic_bytes(directory / 'smoke.json', json.dumps({'status': 'passed',
        'prediction_rows': len(predictions), 'prediction_columns': list(predictions.columns),
        'checks': ['prediction_frames', 'prediction_row_counts', 'finite_predictions', 'input_immutability', 'submission_format'] +
                  (['frozen_predictor_subset_order'] if preparation.prediction_scope == 'row' else []),
        'prediction_scope': preparation.prediction_scope}).encode())


def evaluate(start_path: Path, fit_model, *, start_sha256=None, code_sha256=None) -> Finish:
    start = load(start_path, Start)
    directory = start_path.parent.resolve()
    started = time.monotonic()
    preparation_path = within(directory.parent.parent, str((directory / start.preparation).resolve()))
    preparation = load(preparation_path, Preparation)
    start_sha256 = start_sha256 or file_hash(start_path)
    code_sha256 = code_sha256 or file_hash(within(directory, start.implementation))
    common = dict(node_id=start.node_id, metric=preparation.metric,
                  preparation_sha256=start.preparation_sha256, start_sha256=start_sha256, code_sha256=code_sha256)
    try:
        if file_hash(start_path) != start_sha256:
            raise ValueError('Saved plan changed before execution')
        if file_hash(preparation_path) != start.preparation_sha256:
            raise ValueError('Preparation changed after start')
        if file_hash(within(directory, start.implementation)) != code_sha256:
            raise ValueError('Experiment source changed after its check')
        reader, config, tables, reference, valid_predictions, test_predictions = predict(
            preparation, preparation_path.parent, fit_model)
        validation_file = within(directory, start.outputs['validation'])
        atomic_bytes(validation_file, valid_predictions.to_csv(index=False).encode())
        # Exact frames (dtypes preserved) let the controller blend nodes without re-reading CSV text.
        atomic_bytes(directory / 'predictions.pkl', pickle.dumps({'validation': valid_predictions, 'test': test_predictions}))
        submission_file = within(directory, start.outputs['submission'])
        submission_file.parent.mkdir(parents=True, exist_ok=True)
        reader.write_submission(tables['test'], test_predictions, tables, config, submission_file)
        validate_submission(submission_file, preparation.output, len(tables['test']),
                            test=tables['test'], id_columns=preparation.data.id_columns)
        value, evaluation, error = None, 'unavailable', None
        if preparation.metric.status == 'ready':
            try:
                value = float(reader.score(reference, valid_predictions, config))
                if not np.isfinite(value):
                    raise ValueError('Metric returned a non-finite value')
                evaluation = 'completed'
            except Exception as exception:
                value, evaluation, error = None, 'failed', f'Metric: {type(exception).__name__}: {exception}'[:2000]
        result = Finish(status='completed', value=value, evaluation_status=evaluation,
                        duration_seconds=time.monotonic() - started,
                        validation_file=validation_file.relative_to(directory).as_posix(),
                        submission_file=submission_file.relative_to(directory).as_posix(), error=error, **common)
        try:
            measured = diagnostics(reference, valid_predictions, preparation)
            json.dumps(measured, allow_nan=False)
            result = result.model_copy(update={'diagnostics': measured})
        except (ValueError, TypeError):  # Optional diagnostics must not invalidate a custom task's score.
            pass
    except Exception as exception:
        result = Finish(status='failed', duration_seconds=time.monotonic() - started,
                        error=f'{type(exception).__name__}: {exception}'[:2000], **common)
    save(directory / 'finish.json', result)
    return result


def refit(start_path: Path, member: bool = False):
    """Train the source on all labeled rows; a blend member keeps its raw test frame instead of a submission."""
    start, started = load(start_path, Start), time.monotonic()
    node, root = start_path.parent, start_path.parent.parent.parent
    preparation = load(root / 'preparation.json', Preparation)
    finish = load(node / 'finish.json', Finish)
    result = Finalization(status='running', node_id=start.node_id, preparation_sha256=start.preparation_sha256,
                          code_sha256=finish.code_sha256, start_sha256=finish.start_sha256)
    try:
        if file_hash(start_path) != result.start_sha256 or file_hash(root / 'preparation.json') != result.preparation_sha256:
            raise ValueError('Saved plan or preparation changed before final refit')
        source = Path(preparation.task_directory)
        if input_hashes(source, preparation.data.files) != preparation.source_hashes:
            raise ValueError('Input files changed before final refit')
        reader = module(root / 'prepare.py', preparation.code_sha256)
        program = module(node / 'experiment.py', result.code_sha256)
        config = preparation.model_dump(mode='json')
        tables = reader.load_data(source, config)
        train, test = tables['train'], tables['test']
        config['runtime'] = {'stage': 'final_refit', 'fit_rows': len(train)}
        with threadpool_limits(limits=THREADS):
            predictor = program.fit_model(train.copy(), with_scorer(config, reader, preparation))
            predictions = prediction_frame(predictor, test.copy())
            if preparation.prediction_scope == 'row':
                check_predictor(predictor, test, predictions)
        if member:
            atomic_bytes(node / 'refit_predictions.pkl', pickle.dumps(predictions))
            return
        destination = within(root / 'final', preparation.output.filename)
        destination.parent.mkdir(parents=True, exist_ok=True)
        reader.write_submission(test, predictions, tables, config, destination)
        validate_submission(destination, preparation.output, len(test), test=test, id_columns=preparation.data.id_columns)
        result = result.model_copy(update={'status': 'completed', 'fit_input_rows': len(train),
                                           'submission_file': destination.relative_to(root).as_posix()})
    except Exception as error:
        if member:  # finalization.json belongs to the single-winner refit
            raise
        result = result.model_copy(update={'status': 'failed', 'error': f'{type(error).__name__}: {error}'[-2000:]})
    result = result.model_copy(update={'duration_seconds': time.monotonic() - started})
    save(root / 'finalization.json', result)
    if result.status != 'completed':
        raise ValueError(result.error)


def combine(frames, weights, classification, *, id_columns=(), reference=None, prediction_type=None):
    """Combine aligned predictions, preserving IDs; ties in votes favor the first member.

    Explicit probability/score mode averages even integer 0/1. Explicit labels vote
    even for floating class codes. Without IDs, the worker's positional row contract
    applies: model frames must have equal indexes as well as equal lengths.
    """
    if not frames or len(frames) != len(weights):
        raise ValueError('Blend needs one weight per prediction frame')
    weights = np.asarray(weights, dtype=float)
    if not np.isfinite(weights).all() or (weights <= 0).any() or not np.isfinite(weights.sum()):
        raise ValueError('Blend weights must be finite and positive')
    if prediction_type not in (None, 'numeric', 'labels'):
        raise ValueError('Unknown blend prediction type')
    first = frames[0]
    if not isinstance(first, pd.DataFrame) or not first.columns.is_unique or not len(first.columns):
        raise ValueError('Blend expects DataFrames with unique columns')
    for frame in frames:
        if (not isinstance(frame, pd.DataFrame) or not frame.columns.equals(first.columns)
                or not frame.index.equals(first.index)):
            raise ValueError('Blend columns, row counts or row order differ')
        if frame.isna().any().any():
            raise ValueError('Blend predictions contain missing values')
    if reference is not None and len(reference) != len(first):
        raise ValueError('Blend row count differs from reference')
    result = first.copy()
    for column in result.columns:
        series = [frame[column] for frame in frames]
        if column in id_columns:
            expected = reference[column] if reference is not None and column in reference else series[0]
            if any(not s.reset_index(drop=True).equals(expected.reset_index(drop=True)) for s in series):
                raise ValueError(f'Blend identifier/order differs in {column}')
            continue
        numeric = all(pd.api.types.is_numeric_dtype(s.dtype) and not pd.api.types.is_bool_dtype(s.dtype) for s in series)
        if numeric and any(not np.isfinite(s.to_numpy(dtype=float)).all() for s in series):
            raise ValueError(f'Blend predictions are not finite in {column}')
        # Legacy direct callers can infer labels; production always supplies an explicit mode.
        whole = numeric and all(np.all(np.mod(s.to_numpy(dtype=float), 1) == 0) for s in series)
        vote = prediction_type == 'labels' or (prediction_type is None and (not numeric or classification and whole))
        if not vote:
            if not numeric:
                # Identical non-prediction metadata is preserved, never voted or averaged.
                if not all(s.equals(series[0]) for s in series):
                    raise ValueError(f'Non-numeric blend column differs: {column}')
                continue
            result[column] = sum((w / weights.sum()) * s.to_numpy(dtype=float) for w, s in zip(weights, series))
            continue
        values = [s.to_numpy() for s in series]
        # O(rows * members**2), O(rows) temporary memory; independent of label cardinality.
        best = np.full(len(first), -np.inf)
        chosen = np.empty(len(first), dtype=object)
        for candidate in values:
            tally = sum(w * (candidate == other) for w, other in zip(weights, values))
            better = tally > best
            chosen[better], best[better] = candidate[better], tally[better]
        result[column] = pd.Series(chosen, index=first.index).infer_objects()
    return result


def blend_prediction_type(root, preparation):
    """Use output semantics, never the accidental presence of integer probabilities."""
    if preparation.prediction_scope != 'row':
        raise ValueError('Automatic blend requires row-wise predictions')
    extraction = json.loads(within(root, preparation.extraction).read_text())
    kind = (extraction.get('output', {}).get('prediction_type') or '').casefold()
    if re.search(r'probab|proba|вероятн|logit|score', kind):
        return 'numeric'
    if re.search(r'label|class|метк|класс', kind):
        return 'labels'
    task = (preparation.task_type or '').casefold()
    if re.search(r'regress|регресс', task):
        return 'numeric'
    metric = (preparation.metric.name or '').casefold()
    if re.search(r'auc|roc|log.?loss|cross.?entropy|brier', metric):
        return 'numeric'
    if re.search(r'accuracy|f1|precision|recall|kappa|matthews|точност', metric):
        return 'labels'
    raise ValueError('Prediction semantics are ambiguous; retain the best individual model')


def blend(root: Path, limit: int):
    """Greedy forward selection with replacement (Caruana) over scored nodes on the shared validation rows."""
    preparation = load(root / 'preparation.json', Preparation)
    reader = module(root / 'prepare.py', preparation.code_sha256)
    config = preparation.model_dump(mode='json')
    train = reader.load_data(Path(preparation.task_directory), config)['train']
    with np.load(within(root, preparation.validation['split_file']), allow_pickle=False) as split:
        reference = train.iloc[split['valid']].reset_index(drop=True)
    prediction_type = blend_prediction_type(root, preparation)
    sign = -1. if preparation.metric.direction == 'minimize' else 1.
    classification = 'classif' in (preparation.task_type or '').casefold() or 'классиф' in (preparation.task_type or '').casefold()
    delta = json.loads((root / 'delta.json').read_text())
    scored = sorted((n for n in delta['nodes'] if n['result'].get('value') is not None and n['result']['status'] == 'completed'
                     and (root / 'nodes' / n['node_id'] / 'predictions.pkl').exists()), key=lambda n: -sign * n['result']['value'])[:10]
    frames = {n['node_id']: pickle.loads((root / 'nodes' / n['node_id'] / 'predictions.pkl').read_bytes())['validation'] for n in scored}
    score = lambda counts: sign * float(reader.score(reference, combine([frames[k] for k in counts], list(counts.values()),
                                                                        classification, id_columns=preparation.data.id_columns,
                                                                        reference=reference, prediction_type=prediction_type), config))
    counts, best = {}, -np.inf
    for _ in range(15):
        trials = {}
        for node in frames:
            if node not in counts and len(counts) >= limit:
                continue
            trial = {**counts, node: counts.get(node, 0) + 1}
            try:
                trials[node] = score(trial)
            except Exception:  # an incompatible frame is simply not a member
                continue
        if not trials or max(trials.values()) <= best + 1e-12:
            break
        node = max(trials, key=trials.get)
        counts, best = {**counts, node: counts.get(node, 0) + 1}, trials[node]
    single = sign * scored[0]['result']['value'] if scored else None
    selected = len(counts) > 1 and single is not None and best > single + 1e-3 * abs(single)
    plan = {'weights': counts, 'value': sign * best if counts else None, 'best_single': scored[0]['node_id'] if scored else None,
            'best_single_value': sign * single if single is not None else None, 'selected': selected,
            'rule': 'Caruana greedy selection with replacement; blend replaces the single winner when it improves validation by >0.1%'}
    atomic_bytes(root / 'blend.json', json.dumps(plan, indent=1).encode())


def blend_final(root: Path):
    preparation = load(root / 'preparation.json', Preparation)
    reader = module(root / 'prepare.py', preparation.code_sha256)
    config = preparation.model_dump(mode='json')
    tables = reader.load_data(Path(preparation.task_directory), config)
    plan = json.loads((root / 'blend.json').read_text())
    frames, sources = [], {}
    for node, weight in plan['weights'].items():
        refit_path = root / 'nodes' / node / 'refit_predictions.pkl'
        frames.append(pickle.loads(refit_path.read_bytes()) if refit_path.exists() else
                      pickle.loads((root / 'nodes' / node / 'predictions.pkl').read_bytes())['test'])
        sources[node] = 'refit_all_train' if refit_path.exists() else 'fit_only'
    kind = (preparation.task_type or '').casefold()
    predictions = combine(frames, list(plan['weights'].values()), 'classif' in kind or 'классиф' in kind,
                          id_columns=preparation.data.id_columns, reference=tables['test'],
                          prediction_type=blend_prediction_type(root, preparation))
    destination = within(root / 'final', preparation.output.filename)
    destination.parent.mkdir(parents=True, exist_ok=True)
    reader.write_submission(tables['test'], predictions, tables, config, destination)
    validate_submission(destination, preparation.output, len(tables['test']),
                        test=tables['test'], id_columns=preparation.data.id_columns)
    plan.update(sources=sources, submission_file=destination.relative_to(root).as_posix())
    atomic_bytes(root / 'blend.json', json.dumps(plan, indent=1).encode())


def main(fit_model=None):
    parser = argparse.ArgumentParser()
    if fit_model is None:
        stage = parser.add_mutually_exclusive_group(required=True)
        stage.add_argument('--prepare', type=Path)
        stage.add_argument('--check', type=Path)
        stage.add_argument('--refit', type=Path)
        stage.add_argument('--refit-member', type=Path)
        stage.add_argument('--blend', type=Path)
        stage.add_argument('--blend-final', type=Path)
        stage.add_argument('--api', type=Path)
        parser.add_argument('--preparation', type=Path)
        parser.add_argument('--members', type=int, default=3)
        args = parser.parse_args()
        if args.prepare:
            prepare(args.prepare.resolve())
        elif args.refit:
            refit(args.refit.resolve())
        elif args.refit_member:
            refit(args.refit_member.resolve(), member=True)
        elif args.blend:
            blend(args.blend.resolve(), args.members)
        elif args.blend_final:
            blend_final(args.blend_final.resolve())
        elif args.api:
            from .library_docs import api_reference
            node = args.api.resolve()
            code = (node / 'experiment.py').read_text() if (node / 'experiment.py').exists() else ''
            error = (node / 'last_error.txt').read_text() if (node / 'last_error.txt').exists() else ''
            atomic_bytes(node / 'api_reference.json', json.dumps(api_reference(code, error) if code else {}).encode())
        else:
            smoke(args.check.resolve(), args.preparation.resolve())
    else:
        parser.add_argument('--start', type=Path, required=True)
        parser.add_argument('--start-sha256')
        parser.add_argument('--code-sha256')
        args = parser.parse_args()
        result = evaluate(args.start.resolve(), fit_model, start_sha256=args.start_sha256, code_sha256=args.code_sha256)
        if result.status != 'completed':
            print(result.error, file=sys.stderr)
            raise SystemExit(1)


if __name__ == '__main__':
    main()

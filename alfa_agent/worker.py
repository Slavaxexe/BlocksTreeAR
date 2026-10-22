"""Common runtime: measured preparation and one generated experiment in a child process."""

import argparse
import csv
import io
import json
from pathlib import Path
import sys
import time
import tempfile

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .contracts import Finalization, Finish, Preparation, Start
from .data import file_hash, input_hashes, module, profile, within
from .storage import atomic_bytes, load, save


def validate_submission(path: Path, output, expected_rows: int):
    if not path.is_file() or not path.stat().st_size:
        raise ValueError('Submission writer produced no nonempty output file')
    if (output.format or '').upper() not in {'CSV', 'TSV'}:
        return
    with path.open(encoding='utf-8-sig', newline='') as stream:
        rows = csv.reader(stream, delimiter='\t' if output.format.upper() == 'TSV' else ',', strict=True)
        header = next(rows, [])
        if not header or (output.columns and header != output.columns):
            raise ValueError('Submission columns or column order differ from preparation.output.columns')
        count = 0
        for row in rows:
            if len(row) != len(header):
                raise ValueError('Submission contains a row with the wrong number of columns')
            count += 1
        if count != expected_rows:
            raise ValueError(f'Submission row count {count} differs from test row count {expected_rows}')


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
    for indices in (fit, valid):
        if indices.ndim != 1 or not len(indices) or indices.dtype.kind not in 'iu':
            raise ValueError('Split must contain nonempty one-dimensional integer row positions')
        if len(np.unique(indices)) != len(indices) or indices.min() < 0 or indices.max() >= len(train):
            raise ValueError('Split contains duplicate or out-of-range rows')
    if np.intersect1d(fit, valid).size:
        raise ValueError('Fit and validation rows overlap')
    buffer = io.BytesIO()
    np.savez(buffer, fit=fit.astype(np.int64), valid=valid.astype(np.int64))
    atomic_bytes(directory / 'split.npz', buffer.getvalue())
    validation = {**preparation.validation, 'split_file': 'split.npz', 'fit_rows': len(fit),
                  'validation_rows': len(valid), 'split_sha256': file_hash(directory / 'split.npz')}
    metric_ready = preparation.metric.status == 'ready' and callable(getattr(reader, 'score', None))
    metric = preparation.metric.model_copy(update={'status': 'ready' if metric_ready else 'needs_implementation'})
    issues = [] if metric_ready else ['Metric implementation is unavailable; predictions can still be generated']
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
    with threadpool_limits(limits=2):
        predictor = fit_model(train.iloc[fit].copy(), config)
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
        validate_submission(destination, preparation.output, len(tables['test']))
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
        submission_file = within(directory, start.outputs['submission'])
        submission_file.parent.mkdir(parents=True, exist_ok=True)
        reader.write_submission(tables['test'], test_predictions, tables, config, submission_file)
        validate_submission(submission_file, preparation.output, len(tables['test']))
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


def refit(start_path: Path):
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
        with threadpool_limits(limits=2):
            predictor = program.fit_model(train.copy(), config)
            predictions = prediction_frame(predictor, test.copy())
            if preparation.prediction_scope == 'row':
                check_predictor(predictor, test, predictions)
        destination = within(root / 'final', preparation.output.filename)
        destination.parent.mkdir(parents=True, exist_ok=True)
        reader.write_submission(test, predictions, tables, config, destination)
        validate_submission(destination, preparation.output, len(test))
        result = result.model_copy(update={'status': 'completed', 'fit_input_rows': len(train),
                                           'submission_file': destination.relative_to(root).as_posix()})
    except Exception as error:
        result = result.model_copy(update={'status': 'failed', 'error': f'{type(error).__name__}: {error}'[-2000:]})
    result = result.model_copy(update={'duration_seconds': time.monotonic() - started})
    save(root / 'finalization.json', result)
    if result.status != 'completed':
        raise ValueError(result.error)


def main(fit_model=None):
    parser = argparse.ArgumentParser()
    if fit_model is None:
        stage = parser.add_mutually_exclusive_group(required=True)
        stage.add_argument('--prepare', type=Path)
        stage.add_argument('--check', type=Path)
        stage.add_argument('--refit', type=Path)
        parser.add_argument('--preparation', type=Path)
        args = parser.parse_args()
        if args.prepare:
            prepare(args.prepare.resolve())
        elif args.refit:
            refit(args.refit.resolve())
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

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

from .contracts import Finish, Preparation, Start
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


def predict(preparation: Preparation, root: Path, fit_predict, *, small=False):
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
        valid_predictions, test_predictions = fit_predict(train.iloc[fit].copy(), validation_input.copy(), test.copy(), config)
    if any(not isinstance(frame, pd.DataFrame) for frame in (valid_predictions, test_predictions)):
        raise ValueError('fit_predict must return two DataFrames')
    if len(valid_predictions) != len(valid) or len(test_predictions) != len(test):
        raise ValueError('Prediction row count differs from validation or test')
    return reader, config, tables, reference, valid_predictions.reset_index(drop=True), test_predictions.reset_index(drop=True)


def smoke(directory: Path, preparation_path: Path):
    preparation = load(preparation_path, Preparation)
    program = module(directory / 'experiment.py', file_hash(directory / 'experiment.py'))
    reader, config, tables, _, _, predictions = predict(preparation, preparation_path.parent, program.fit_predict, small=True)
    with tempfile.TemporaryDirectory() as temporary:
        destination = within(Path(temporary), preparation.output.filename)
        destination.parent.mkdir(parents=True, exist_ok=True)
        reader.write_submission(tables['test'], predictions, tables, config, destination)
        validate_submission(destination, preparation.output, len(tables['test']))
    atomic_bytes(directory / 'smoke.json', json.dumps({'status': 'passed',
        'prediction_rows': len(predictions), 'prediction_columns': list(predictions.columns),
        'checks': ['prediction_frames', 'prediction_row_counts', 'submission_format']}).encode())


def evaluate(start_path: Path, fit_predict, *, start_sha256=None, code_sha256=None) -> Finish:
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
            preparation, preparation_path.parent, fit_predict)
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
    except Exception as exception:
        result = Finish(status='failed', duration_seconds=time.monotonic() - started,
                        error=f'{type(exception).__name__}: {exception}'[:2000], **common)
    save(directory / 'finish.json', result)
    return result


def main(fit_predict=None):
    parser = argparse.ArgumentParser()
    if fit_predict is None:
        stage = parser.add_mutually_exclusive_group(required=True)
        stage.add_argument('--prepare', type=Path)
        stage.add_argument('--check', type=Path)
        parser.add_argument('--preparation', type=Path)
        args = parser.parse_args()
        if args.prepare:
            prepare(args.prepare.resolve())
        else:
            smoke(args.check.resolve(), args.preparation.resolve())
    else:
        parser.add_argument('--start', type=Path, required=True)
        parser.add_argument('--start-sha256')
        parser.add_argument('--code-sha256')
        args = parser.parse_args()
        result = evaluate(args.start.resolve(), fit_predict, start_sha256=args.start_sha256, code_sha256=args.code_sha256)
        if result.status != 'completed':
            print(result.error, file=sys.stderr)
            raise SystemExit(1)


if __name__ == '__main__':
    main()

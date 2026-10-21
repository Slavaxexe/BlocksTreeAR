"""Common runtime: measured preparation and one generated experiment in a child process."""

import argparse
import io
from pathlib import Path
import sys
import time
import tempfile

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .contracts import Extraction, Finish, Preparation, Start
from .assembly import implementation_hash
from .context import preparation_view, requirements
from .data import file_hash, input_hashes, module, profile, within
from .storage import atomic_bytes, load, save


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
    group_columns = config['validation'].get('group_columns', [])
    if group_columns:
        fit_groups = train.iloc[fit][group_columns].drop_duplicates()
        valid_groups = train.iloc[valid][group_columns].drop_duplicates()
        if not fit_groups.merge(valid_groups, on=group_columns, how='inner').empty:
            raise ValueError(f'Fit and validation contain the same declared groups: {group_columns}')
    buffer = io.BytesIO()
    np.savez(buffer, fit=fit.astype(np.int64), valid=valid.astype(np.int64))
    atomic_bytes(directory / 'split.npz', buffer.getvalue())
    validation = {**config['validation'], 'split_file': 'split.npz', 'fit_rows': len(fit),
                  'validation_rows': len(valid), 'split_sha256': file_hash(directory / 'split.npz')}
    metric_ready = preparation.metric.status == 'ready' and callable(getattr(reader, 'score', None))
    metric = preparation.metric.model_copy(update={'status': 'ready' if metric_ready else 'needs_implementation'})
    issues = [] if metric_ready else ['Metric implementation is unavailable; predictions can still be generated']
    prepared = preparation.model_copy(update={'status': 'ready' if metric_ready else 'partial',
        'profile': measured, 'source_hashes': hashes, 'validation': validation, 'metric': metric, 'issues': issues})
    save(path, prepared)


def predict(preparation: Preparation, root: Path, fit_predict, *, small=False, time_budget=240, facts_path=None):
    started = time.monotonic()
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
        # Keep rare classes in a small classification fit; never draw outer validation labels.
        extraction = load(root / 'extraction.json', Extraction)
        task_type = (extraction.problem.task_type or '').lower()
        if 'class' in task_type and len(preparation.data.target_columns) == 1:
            target = train.iloc[fit][preparation.data.target_columns[0]]
            # Inner CV/holdout needs multiple examples, not a single representative.
            representatives = np.flatnonzero(target.groupby(target, dropna=False).cumcount().to_numpy() < 3)
            positions = np.unique(np.concatenate((representatives, np.arange(min(200, len(fit))))))
            fit = fit[positions]
        else:
            fit = fit[:200]
        valid = valid[:50]
        tables = {key: frame.iloc[:50].copy() if key != 'train' and len(frame) == len(test) else frame
                  for key, frame in tables.items()}
        test = tables['test']
    reference = train.iloc[valid].reset_index(drop=True)
    validation_input = reference.drop(columns=preparation.data.target_columns)
    if facts_path is not None:
        # These are measured input facts, separate for smoke and full execution.
        save(facts_path, {'code_sha256': implementation_hash(facts_path.parent),
             'fit_rows': len(fit), 'validation_rows': len(valid), 'test_rows': len(test),
             'fit_target_unique_counts': {column: int(train[column].iloc[fit].nunique())
                                          for column in preparation.data.target_columns}})
    metadata = preparation_view(preparation)
    metadata['requirements'] = requirements(load(root / 'extraction.json', Extraction), prepared=True)
    # Leave time for prediction, scoring and saving the final files after AutoML search.
    training_budget = max(1., (time_budget or 1200.) * 0.8)
    metadata['runtime'] = {'time_budget_seconds': training_budget, 'deadline': started + training_budget,
                           'num_cpus': 2, 'num_gpus': 0, 'is_smoke_check': small,
                           'work_directory': str(Path.cwd())}
    with threadpool_limits(limits=2):
        auxiliary = {role: frame.copy() for role, frame in tables.items() if role not in {'train', 'test'}}
        valid_predictions, test_predictions = fit_predict(train.iloc[fit].copy(), validation_input.copy(), test.copy(),
                                                         metadata, auxiliary)
    if any(not isinstance(frame, pd.DataFrame) for frame in (valid_predictions, test_predictions)):
        raise ValueError('fit_predict must return two DataFrames')
    if len(valid_predictions) != len(valid) or len(test_predictions) != len(test):
        raise ValueError('Prediction row count differs from validation or test')
    return reader, config, tables, reference, valid_predictions.reset_index(drop=True), test_predictions.reset_index(drop=True)


def smoke(directory: Path, preparation_path: Path, time_budget=45):
    preparation = load(preparation_path, Preparation)
    program = module(directory / 'experiment.py', file_hash(directory / 'experiment.py'))
    reader, config, tables, _, _, predictions = predict(
        preparation, preparation_path.parent, program.fit_predict, small=True, time_budget=time_budget,
        facts_path=directory / 'smoke_inputs.json')
    with tempfile.TemporaryDirectory() as temporary:
        destination = within(Path(temporary), preparation.output.filename)
        destination.parent.mkdir(parents=True, exist_ok=True)
        reader.write_submission(tables['test'], predictions, tables, config, destination)
        if not destination.is_file() or not destination.stat().st_size:
            raise ValueError('Smoke test produced no nonempty output file')


def evaluate(start_path: Path, fit_predict, *, start_sha256=None, code_sha256=None, time_budget=None) -> Finish:
    start = load(start_path, Start)
    directory = start_path.parent.resolve()
    started = time.monotonic()
    preparation_path = within(directory.parent.parent, str((directory / start.preparation).resolve()))
    preparation = load(preparation_path, Preparation)
    start_sha256 = start_sha256 or file_hash(start_path)
    code_sha256 = code_sha256 or implementation_hash(directory)
    common = dict(node_id=start.node_id, metric=preparation.metric,
                  preparation_sha256=start.preparation_sha256, start_sha256=start_sha256, code_sha256=code_sha256)
    try:
        if file_hash(start_path) != start_sha256:
            raise ValueError('Saved plan changed before execution')
        if file_hash(preparation_path) != start.preparation_sha256:
            raise ValueError('Preparation changed after start')
        if implementation_hash(directory) != code_sha256:
            raise ValueError('Assembled source changed after its check')
        reader, config, tables, reference, valid_predictions, test_predictions = predict(
            preparation, preparation_path.parent, fit_predict, time_budget=time_budget or start.timeout_seconds,
            facts_path=directory / 'execution_inputs.json')
        validation_file = within(directory, start.outputs['validation'])
        atomic_bytes(validation_file, valid_predictions.to_csv(index=False).encode())
        submission_file = within(directory, start.outputs['submission'])
        submission_file.parent.mkdir(parents=True, exist_ok=True)
        reader.write_submission(tables['test'], test_predictions, tables, config, submission_file)
        if not submission_file.is_file() or not submission_file.stat().st_size:
            raise ValueError('Submission writer produced no nonempty output file')
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
                        failed_block=getattr(exception, 'block', None),
                        error=f'{type(exception).__name__}: {exception}'[:2000], **common)
    save(directory / 'finish.json', result)
    return result


def main(fit_predict=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--time-budget', type=float)
    if fit_predict is None:
        stage = parser.add_mutually_exclusive_group(required=True)
        stage.add_argument('--prepare', type=Path)
        stage.add_argument('--check', type=Path)
        parser.add_argument('--preparation', type=Path)
        args = parser.parse_args()
        if args.prepare:
            prepare(args.prepare.resolve())
        else:
            smoke(args.check.resolve(), args.preparation.resolve(), time_budget=args.time_budget or 45)
    else:
        parser.add_argument('--start', type=Path, required=True)
        parser.add_argument('--start-sha256')
        parser.add_argument('--code-sha256')
        args = parser.parse_args()
        result = evaluate(args.start.resolve(), fit_predict, start_sha256=args.start_sha256,
                          code_sha256=args.code_sha256, time_budget=args.time_budget)
        if result.status != 'completed':
            print(result.error, file=sys.stderr)
            raise SystemExit(1)


if __name__ == '__main__':
    main()

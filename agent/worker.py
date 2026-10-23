"""Common runtime: measured preparation and one generated experiment in a child process."""

import argparse
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
from .assembly import implementation_hash
from .context import preparation_view
from .data import file_hash, input_hashes, module, profile, within
from .storage import atomic_bytes, load, save


def baseline(directory: Path):
    """Publish a cheap predictor before EDA or any model-selection calls."""
    from .operations import _constant
    preparation = load(directory / 'preparation.json', Preparation)
    reader = module(directory / 'prepare.py', preparation.code_sha256)
    config = preparation.model_dump(mode='json')
    tables = reader.load_data(Path(preparation.task_directory), config)
    extraction = json.loads((directory / 'extraction.json').read_text(encoding='utf-8'))
    defaults = {key: extraction.get('problem' if key != 'prediction_type' else 'output', {}).get(key)
                for key in ('task_type', 'class_labels', 'prediction_type')}
    state = {'fit': tables['train'], 'validation': tables['test'], 'test': tables['test'], 'preparation': config}
    predictions = _constant(state, defaults)['test_predictions']
    destination = directory / 'baseline' / preparation.output.filename
    destination.parent.mkdir(parents=True, exist_ok=True)
    reader.write_submission(tables['test'], predictions, tables, config, destination)
    atomic_bytes(Path(preparation.task_directory) / preparation.output.filename, destination.read_bytes())
    save(directory / 'baseline' / 'baseline.json', {'method': 'constant', 'fit_rows': len(tables['train']),
         'test_rows': len(predictions), 'submission_file': destination.relative_to(directory).as_posix()})


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
    from .ensemble_data import save_split
    extraction_path = directory / preparation.extraction
    extraction = json.loads(extraction_path.read_text(encoding='utf-8')) if extraction_path.exists() else {}
    task_type = (extraction.get('problem', {}).get('task_type') or '').lower()
    classification = any(word in task_type for word in ('class', 'binary'))
    if not task_type:
        from .automl import _METRICS
        name = ''.join(char for char in (preparation.metric.name or '').lower() if char.isalnum())
        family = _METRICS.get(name, (None,))[0]
        classification = family == 'classification' or (family is None and
            any(not pd.api.types.is_numeric_dtype(train[column]) for column in preparation.data.target_columns))
    validation = {key: value for key, value in config['validation'].items()
                  if not key.startswith('ensemble_') and key not in {'model_fit_rows', 'meta_rows'}}
    validation.update(split_file='split.npz', fit_rows=len(fit), validation_rows=len(valid),
                      split_sha256=file_hash(directory / 'split.npz'))
    ensemble_issue = None
    try:
        save_split(directory / 'ensemble_split.npz', fit, train, preparation.data.target_columns,
                   config['validation'], classification)
        with np.load(directory / 'ensemble_split.npz', allow_pickle=False) as inner:
            validation.update(model_fit_rows=len(inner['train']), meta_rows=len(inner['meta']))
        validation.update(ensemble_split_file='ensemble_split.npz',
                          ensemble_split_sha256=file_hash(directory / 'ensemble_split.npz'))
    except (ValueError, KeyError) as error:
        # A task without a usable inner split can still train standalone models and blend them.
        ensemble_issue = 'Learned ensembles unavailable: ' + str(error)
    metric_ready = preparation.metric.status == 'ready' and callable(getattr(reader, 'score', None))
    metric = preparation.metric.model_copy(update={'status': 'ready' if metric_ready else 'needs_implementation'})
    issues = [] if metric_ready else ['Metric implementation is unavailable; predictions can still be generated']
    if ensemble_issue:
        issues.append(ensemble_issue)
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
    meta = np.array([], dtype=int)
    if preparation.validation.get('ensemble_split_file'):
        inner_path = within(root, preparation.validation['ensemble_split_file'])
        if file_hash(inner_path) != preparation.validation['ensemble_split_sha256']:
            raise ValueError('Ensemble holdout changed after preparation')
        with np.load(inner_path, allow_pickle=False) as inner:
            fit, meta = inner['train'], inner['meta']
    if small:
        meta = meta[:50]
        fit, valid = fit[:200], valid[:50]
        tables = {key: frame.iloc[:50].copy() if key != 'train' and len(frame) == len(test) else frame
                  for key, frame in tables.items()}
        test = tables['test']
    reference = train.iloc[valid].reset_index(drop=True)
    meta_frame = train.iloc[meta].reset_index(drop=True)
    validation_input = pd.concat([reference, meta_frame], ignore_index=True).drop(columns=preparation.data.target_columns)
    if facts_path is not None:
        # These are measured input facts, separate for smoke and full execution.
        save(facts_path, {'code_sha256': implementation_hash(facts_path.parent),
             'fit_rows': len(fit), 'validation_rows': len(valid), 'test_rows': len(test),
             'fit_target_unique_counts': {column: int(train[column].iloc[fit].nunique())
                                          for column in preparation.data.target_columns}})
    metadata = preparation_view(preparation)
    # Leave time for prediction, scoring and saving the final files after AutoML search.
    training_budget = max(1., time_budget * 0.8)
    metadata['runtime'] = {'time_budget_seconds': training_budget, 'deadline': started + training_budget,
                           'num_cpus': 1, 'num_gpus': 0, 'is_smoke_check': small,
                           'work_directory': str(Path.cwd()), 'run_directory': str(root.resolve()),
                           'preparation_sha256': file_hash(root / 'preparation.json'),
                           'meta_rows': len(meta), 'outer_validation_rows': len(valid), 'fit_row_positions': fit}
    with threadpool_limits(limits=1):
        auxiliary = {role: frame.copy() for role, frame in tables.items() if role not in {'train', 'test'}}
        auxiliary['ensemble_meta'] = meta_frame
        valid_predictions, test_predictions = fit_predict(train.iloc[fit].copy(), validation_input.copy(), test.copy(),
                                                         metadata, auxiliary)
    if any(not isinstance(frame, pd.DataFrame) for frame in (valid_predictions, test_predictions)):
        raise ValueError('fit_predict must return two DataFrames')
    if len(valid_predictions) != len(valid) + len(meta) or len(test_predictions) != len(test):
        raise ValueError('Prediction row count differs from validation or test')
    from .ensemble_data import subset
    held_out = valid_predictions.attrs.get('meta_held_out', True)
    meta_predictions = subset(valid_predictions, np.arange(len(valid), len(valid) + len(meta)))
    valid_predictions = subset(valid_predictions, np.arange(len(valid)))
    valid_predictions.attrs.update(meta_predictions=meta_predictions, meta_held_out=held_out)
    return reader, config, tables, reference, valid_predictions, test_predictions.reset_index(drop=True)


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
        from .predictions import save_pair
        frames = {'validation': valid_predictions, 'test': test_predictions}
        if len(valid_predictions.attrs['meta_predictions']):
            frames['meta'] = valid_predictions.attrs['meta_predictions']
        save_pair(directory, frames,
                  start.preparation_sha256, code_sha256, start.outputs['validation'])
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
                value, evaluation = None, 'failed'
                error = f'Metric: {type(exception).__name__}: {exception}'[:1500]
                schema = {'rows': len(valid_predictions), 'columns': [str(name) for name in valid_predictions.columns],
                          'dtypes': {str(name): str(dtype) for name, dtype in valid_predictions.dtypes.items()}}
                error += '; prediction schema: ' + str(schema)
                failed_block = next(block.name for block in reversed(start.assembly.blocks)
                                    if 'validation_predictions' in block.provides)
                save(directory / 'block_error.json', {'stage': 'metric', 'block': failed_block,
                     'error': error, 'prediction_schema': schema})
        result = Finish(status='failed' if evaluation == 'failed' else 'completed',
                        failed_block=failed_block if evaluation == 'failed' else None,
                        value=value, evaluation_status=evaluation,
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
        stage.add_argument('--baseline', type=Path)
        stage.add_argument('--check', type=Path)
        parser.add_argument('--preparation', type=Path)
        args = parser.parse_args()
        if args.baseline:
            baseline(args.baseline.resolve())
        elif args.prepare:
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

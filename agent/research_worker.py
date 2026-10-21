"""Measured data profiles and bounded research programs, without holdout labels."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .contracts import Preparation
from .context import research_profile_view
from .data import file_hash, input_hashes, module, within
from .storage import canonical, load, save


def describe(frame):
    columns = {}
    for name in frame:
        series = frame[name]
        item = {'dtype': str(series.dtype), 'missing': int(series.isna().sum()),
                'unique': int(series.nunique())}
        if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
            finite = series.replace([np.inf, -np.inf], np.nan).dropna()
            item['quantiles'] = {str(q): float(value) for q, value in
                                 finite.quantile([0, .25, .5, .75, 1]).items()} if len(finite) else {}
            item['nonfinite'] = int(series.isin([np.inf, -np.inf]).sum())
        else:
            item['frequent_values'] = {str(k): int(v) for k, v in series.value_counts().head(10).items()}
        columns[str(name)] = item
    return {'rows': len(frame), 'columns': columns}


def read_tables(root):
    preparation = load(root / 'preparation.json', Preparation)
    source = Path(preparation.task_directory)
    reader = module(root / 'prepare.py', preparation.code_sha256)
    if preparation.source_hashes and input_hashes(source, preparation.data.files) != preparation.source_hashes:
        raise ValueError('Research inputs changed after preparation')
    return preparation, reader.load_data(source, preparation.model_dump(mode='json'))


def inspect(root):
    preparation, tables = read_tables(root)
    train, test = tables['train'], tables['test']
    targets, ids = preparation.data.target_columns, preparation.data.id_columns
    features = [c for c in train if c not in targets + ids]
    result = {'tables': {role: describe(frame) for role, frame in tables.items()},
              'target_counts': {name: {str(k): int(v) for k, v in train[name].value_counts().items()}
                                for name in targets if train[name].nunique() <= 100},
              'duplicate_feature_rows': int(train[features].duplicated().sum()) if features else 0,
              'train_only_columns': sorted(set(train) - set(test)),
              'test_only_columns': sorted(set(test) - set(train)),
              'target_columns': targets, 'id_columns': ids,
              'scope': 'Marginal profiles may summarize targets; structural probes receive no row-level targets'}
    save(root / 'research' / 'profile.json', result)
    save(root / 'preparation.json', preparation.model_copy(update={
        'source_hashes': input_hashes(Path(preparation.task_directory), preparation.data.files)}))


def execute(root, node):
    start = json.loads((node / 'start.json').read_text(encoding='utf-8'))
    if file_hash(root / 'preparation.json') != start['preparation_sha256']:
        raise ValueError('Research preparation changed after planning')
    preparation, tables = read_tables(root)
    targets = preparation.data.target_columns
    train = tables['train']
    if start['phase'] == 'structure':
        fit = train.drop(columns=targets)
        validation = fit.iloc[:0].copy()
    else:
        split_path = within(root, preparation.validation['split_file'])
        if file_hash(split_path) != preparation.validation['split_sha256']:
            raise ValueError('Research split changed after preparation')
        with np.load(split_path, allow_pickle=False) as split:
            fit = train.iloc[split['fit']].copy()
            validation = train.iloc[split['valid']].drop(columns=targets).copy()
    state = {'fit': fit, 'validation': validation, 'test': tables['test'].drop(columns=targets, errors='ignore'),
             'auxiliary': {role: frame for role, frame in tables.items() if role not in {'train', 'test'}},
             'preparation': preparation.model_dump(mode='json'),
             'profile': research_profile_view(json.loads((root / 'research' / 'profile.json').read_text(encoding='utf-8')))}
    # Holdout labels and the full labelled train are never passed to the generated program.
    program = module(node / 'probe.py', start['code_sha256'])
    with threadpool_limits(limits=1):
        measurements = program.run(state)
    if not isinstance(measurements, dict) or not measurements:
        raise ValueError('Research run(state) must return a nonempty measurement dict')
    if len(canonical(measurements).encode('utf-8')) > 64000:
        raise ValueError('Research output exceeds 64 KB; return aggregates instead of raw rows')
    save(node / 'measurements.json', {'values': measurements, 'phase': start['phase'],
         'fit_rows': len(fit), 'validation_rows': len(validation), 'test_rows': len(tables['test']),
         'code_sha256': start['code_sha256'], 'preparation_sha256': start['preparation_sha256']})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--node', type=Path)
    args = parser.parse_args()
    if args.node:
        execute(args.root.resolve(), args.node.resolve())
    else:
        inspect(args.root.resolve())


if __name__ == '__main__':
    main()

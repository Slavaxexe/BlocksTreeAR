"""Read-only adapter from margo preparation artifacts to our training contract."""
from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from storage import atomic_bytes, save
from training_contracts import TrainingInput, ValidationConstraints


_METRICS = {
    'rmse': 'rmse', 'root_mean_squared_error': 'rmse',
    'mse': 'mse', 'mean_squared_error': 'mse',
    'mae': 'mae', 'mean_absolute_error': 'mae',
    'r2': 'r2', 'r2_score': 'r2',
    'mape': 'mape', 'mean_absolute_percentage_error': 'mape',
}


def _metric(value):
    try:
        return _METRICS[str(value).strip().lower()]
    except KeyError as error:
        raise ValueError(f'Unsupported upstream primary metric: {value!r}') from error


def _json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _schemas(source_root):
    package = source_root / 'data_preparation'
    if not (package / 'schemas.py').is_file():
        raise ValueError('source_root must contain the margo data_preparation package')
    existing = sys.modules.get('data_preparation')
    if existing is not None and Path(existing.__file__).resolve().parent != package:
        raise ValueError('A different data_preparation package is already imported')
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    return importlib.import_module('data_preparation.schemas')


def _envelope(path, artifact_type, schemas):
    envelope = schemas.ArtifactEnvelope.model_validate(_json(path))
    if envelope.schema_version != schemas.SCHEMA_VERSION:
        raise ValueError(f'Unsupported margo schema version in {path.name}')
    if envelope.artifact_type.value != artifact_type or envelope.status.value != 'success' or envelope.errors:
        raise ValueError(f'Invalid or unsuccessful {artifact_type}: {path}')
    return schemas.validate_payload(envelope.artifact_type, envelope.payload).model_dump(mode='json')


def _resolve(value, preparation_dir, source_root, *, prepared=False):
    if not value:
        raise ValueError('Missing artifact file path')
    candidate = Path(value).expanduser()
    if candidate.is_absolute():
        if candidate.is_file():
            return candidate.resolve()
        raise ValueError(f'Artifact file does not exist: {candidate}')
    candidates = [source_root / candidate, preparation_dir / candidate,
                  preparation_dir.parent / candidate, Path.cwd() / candidate]
    if prepared:
        candidates.append(preparation_dir / candidate.name)
    for item in candidates:
        if item.is_file():
            return item.resolve()
    raise ValueError(f'Cannot resolve artifact file: {value}')


def _check_split(indices, clean):
    if not isinstance(indices, dict) or set(indices) != {'train', 'val'}:
        raise ValueError('margo must provide train/val split_indices')
    original = {int(row): position for position, row in enumerate(clean.index)}
    combined = []
    output = {}
    for source_name, target_name in [('train', 'fit'), ('val', 'valid')]:
        rows = indices[source_name]
        if not isinstance(rows, list) or not rows or any(type(row) is not int for row in rows):
            raise ValueError('Split indices must be nonempty lists of integer source row labels')
        if len(set(rows)) != len(rows) or any(row not in original for row in rows):
            raise ValueError('Split has duplicate, removed, or out-of-bounds source rows')
        combined.extend(rows)
        output[target_name] = [original[row] for row in rows]
    if len(set(combined)) != len(combined) or set(combined) != set(original):
        raise ValueError('margo split must be exhaustive and non-overlapping')
    return output


def _constraints(selection, manifest, train, split):
    declared_group = manifest.get('group_column')
    declared_time = manifest.get('time_column')
    if declared_group and declared_time:
        raise ValueError('Combined group/time constraints require an explicit supported upstream strategy')
    default = ({'strategy': 'group', 'group_columns': [declared_group]} if declared_group else
               {'strategy': 'time', 'time_column': declared_time} if declared_time else {})
    decision = ValidationConstraints.model_validate(selection.get('validation_constraints', default))
    if declared_group and (decision.strategy != 'group' or declared_group not in decision.group_columns):
        raise ValueError('Selection cannot remove declared upstream group constraints')
    if declared_time and (decision.strategy != 'time' or declared_time != decision.time_column):
        raise ValueError('Selection cannot remove declared upstream time constraints')
    fit, valid = train.iloc[split['fit']], train.iloc[split['valid']]
    if decision.strategy == 'group':
        if any(column not in train for column in decision.group_columns):
            raise ValueError('Declared group column is missing')
        if train[decision.group_columns].isna().any().any():
            raise ValueError('Group columns contain missing values')
        left = set(fit[decision.group_columns].itertuples(index=False, name=None))
        right = set(valid[decision.group_columns].itertuples(index=False, name=None))
        if left & right:
            raise ValueError('margo split leaks groups between training and validation')
    if decision.strategy == 'time':
        if decision.time_column not in train:
            raise ValueError('Declared time column is missing')
        stamps = pd.to_datetime(train[decision.time_column], errors='coerce', utc=True)
        if stamps.isna().any() or stamps.iloc[split['fit']].max() >= stamps.iloc[split['valid']].min():
            raise ValueError('margo split violates strictly earlier training timestamps')
    return decision


def build_training_handoff(preparation_dir, selection_path, source_root, output_dir) -> Path:
    """Validate margo's output and freeze a replayable recipe with original rows/IDs.

    No pickle is deserialized and no upstream source or artifact is modified.
    Model families must be supplied by the separate model-selection stage.
    """
    preparation_dir = Path(preparation_dir).resolve()
    source_root = Path(source_root).resolve()
    selection_path = Path(selection_path).resolve()
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise ValueError(f'Handoff output directory must be new: {output_dir}')
    schemas = _schemas(source_root)
    manifest = _envelope(preparation_dir / 'input_manifest.json', 'input_manifest', schemas)
    output = _envelope(preparation_dir / 'output_manifest.json', 'output_manifest', schemas)
    profile = _envelope(preparation_dir / 'data_profile.json', 'data_profile', schemas)
    report = _envelope(preparation_dir / 'preparation_report.json', 'preparation_report', schemas)
    if manifest['task_type'] != 'regression':
        raise ValueError('Only upstream regression preparation is supported')
    if output['target_columns'] != manifest['target_columns'] or output['id_columns'] != manifest['id_columns']:
        raise ValueError('Upstream input/output target or identifier declarations differ')
    target, ids = manifest['target_columns'][0], manifest['id_columns']
    selection = _json(selection_path)
    if not isinstance(selection, dict) or not selection.get('selected_models'):
        raise ValueError('The separate selection JSON must provide selected_models')
    primary_metric = _metric(manifest['metric_name'])
    if 'primary_metric' in selection and _metric(selection['primary_metric']) != primary_metric:
        raise ValueError('Selection primary_metric differs from authoritative upstream metric')
    files = {}
    for reference in manifest['files']:
        if reference['role'] in {'train', 'test', 'sample_submission'}:
            if reference['role'] in files:
                raise ValueError(f'Duplicate upstream file role: {reference["role"]}')
            files[reference['role']] = _resolve(reference['path'], preparation_dir, source_root)
    if not {'train', 'test'} <= files.keys():
        raise ValueError('Upstream manifest must identify raw train and test sources')
    raw = {role: pd.read_csv(path, dtype={column: str for column in ids}) for role, path in files.items()}
    train, test = raw['train'], raw['test']
    if target not in train or target in test:
        raise ValueError('Target must occur in train and must not occur in test')
    if any(column not in train or column not in test for column in ids):
        raise ValueError('Declared identifiers must occur in both raw train and test')
    numeric = pd.to_numeric(train[target], errors='coerce')
    if (train[target].notna() & numeric.isna()).any():
        raise ValueError('Raw target contains nonnumeric values')
    clean = train.loc[train[target].notna() & ~np.isinf(numeric)].copy()
    clean[target] = numeric.loc[clean.index]
    split = _check_split(output.get('split_indices'), clean)
    constraints = _constraints(selection, manifest, clean, split)
    artifact_paths = {}
    for name, positions in [('train', split['fit']), ('val', split['valid']), ('test', None)]:
        reference = output.get(f'{name}_features')
        if not reference:
            raise ValueError(f'Missing prepared {name} features')
        path = _resolve(reference['path'], preparation_dir, source_root, prepared=True)
        features = pd.read_parquet(path)
        artifact_paths[f'{name}_features'] = path
        expected_rows = len(test) if positions is None else len(positions)
        if (len(features) != expected_rows or len(features) != reference['n_rows']
                or list(features.columns) != output['feature_names']
                or list(features.columns) != reference['columns']
                or len(features.columns) != reference['n_columns']):
            raise ValueError(f'Prepared {name} feature schema or row count differs from manifest/split')
        if positions is not None:
            target_ref = output.get(f'{name}_target')
            if not target_ref:
                raise ValueError(f'Missing prepared {name} target')
            target_path = _resolve(target_ref['path'], preparation_dir, source_root, prepared=True)
            actual = pd.read_parquet(target_path)
            artifact_paths[f'{name}_target'] = target_path
            if (list(actual.columns) != [target] or target_ref['columns'] != [target]
                    or target_ref['n_columns'] != 1 or target_ref['n_rows'] != expected_rows
                    or len(actual) != expected_rows
                    or not np.array_equal(pd.to_numeric(actual[target], errors='coerce').to_numpy(),
                                          clean.iloc[positions][target].to_numpy())):
                raise ValueError(f'Prepared {name} target order differs from raw source split')
        id_ref = output.get(f'{name}_ids')
        if ids:
            if not id_ref or not id_ref.get('path'):
                raise ValueError(f'Missing prepared {name} identifiers')
            id_path = _resolve(id_ref['path'], preparation_dir, source_root, prepared=True)
            prepared_ids = pd.read_parquet(id_path)
            artifact_paths[f'{name}_ids'] = id_path
            if (len(prepared_ids) != expected_rows or list(prepared_ids.columns) != ids
                    or id_ref['n_rows'] != expected_rows or id_ref['columns'] != ids
                    or id_ref['n_columns'] != len(ids)):
                raise ValueError(f'Prepared {name} identifier schema/row count differs')
            expected_ids = (test if positions is None else clean.iloc[positions])[ids].reset_index(drop=True)
            for column in ids:
                expected = expected_ids[column]
                actual = prepared_ids[column].reset_index(drop=True)
                same_strings = expected.astype(str).equals(actual.astype(str))
                numeric_expected = pd.to_numeric(expected, errors='coerce')
                numeric_actual = pd.to_numeric(actual, errors='coerce')
                same_numeric = (numeric_expected.notna().all() and numeric_actual.notna().all()
                                and np.array_equal(numeric_expected.to_numpy(), numeric_actual.to_numpy()))
                if not (same_strings or same_numeric):
                    raise ValueError(f'Prepared {name} identifier order differs from raw source split')
    recipe_path = _resolve(output['recipe_path'], preparation_dir, source_root, prepared=True)
    recipe = schemas.FeaturePlanPayload.model_validate(_json(recipe_path)).model_dump(mode='json')
    if target in output['feature_names'] or set(ids) & set(output['feature_names']):
        raise ValueError('Prepared features must exclude target and identifiers')
    if any(item['column'] == target for item in recipe.get('column_transforms', [])) or any(
            target in item['source_columns'] for item in recipe.get('new_features', [])):
        raise ValueError('Preparation recipe must not use target as a feature source')
    source_hashes = {str(path.relative_to(source_root)).replace('\\', '/'): _sha(path)
                     for path in sorted((source_root / 'data_preparation').glob('*.py'))}
    output_dir.mkdir(parents=True)
    frozen_files = {}
    for role, frame in [('train', clean.reset_index(drop=True)), ('test', test),
                        ('sample_submission', raw.get('sample_submission'))]:
        if frame is not None:
            path = output_dir / f'{role}.parquet'
            atomic_bytes(path, frame.to_parquet(index=False))
            frozen_files[role] = str(path)
    save(output_dir / 'recipe.json', recipe)
    save(output_dir / 'split.json', split)
    contract = TrainingInput.model_validate({
        'files': frozen_files, 'target': target, 'id_columns': ids, 'primary_metric': primary_metric,
        'selected_models': selection['selected_models'],
        'submission': selection.get('submission', {}),
        'validation_constraints': constraints.model_dump(mode='json'),
        'eda': {'upstream_data_profile': profile, 'upstream_preparation_report': report,
                'upstream_model_selection': selection.get('selection_report', {}),
                'prepared_feature_names': output['feature_names'],
                'handoff_note': 'Preserve upstream outer split; replay the fixed recipe inside each training fold.'},
        'preparation': {'provider': 'margo', 'source_root': str(source_root),
                        'recipe': str(output_dir / 'recipe.json'), 'split': str(output_dir / 'split.json'),
                        'source_hashes': source_hashes, 'feature_names': output['feature_names']},
    })
    observed = {**files, **artifact_paths, 'recipe': recipe_path, 'selection': selection_path}
    for name in ['input_manifest', 'output_manifest', 'data_profile', 'preparation_report']:
        observed[name] = preparation_dir / f'{name}.json'
    save(output_dir / 'provenance.json', {
        'provider': 'margo', 'source_root': str(source_root), 'preparation_dir': str(preparation_dir),
        'artifacts': {name: {'path': str(path), 'sha256': _sha(path)} for name, path in observed.items()},
        'source_hashes': source_hashes, 'raw_train_rows': len(train), 'cleaned_train_rows': len(clean),
        'dropped_source_rows': train.index.difference(clean.index).tolist(),
        'split_source_rows': output['split_indices'], 'primary_metric': primary_metric,
        'limitations': ['Upstream data_profile was computed on full train before its split.',
                        'Original identifiers are restored from raw CSV as strings, separately from prepared ID files.',
                        'Selected models are supplied by a separate selection report; margo does not select models.'],
    })
    save(output_dir / 'input.json', contract)
    return output_dir / 'input.json'

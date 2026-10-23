"""Connect the untouched family/loss selection stage to preparation and training."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from margo_handoff import _check_split, _envelope, _json, _metric, _resolve, _schemas
from storage import atomic_bytes, save


MODEL_FAMILIES = {
    'ridge': 'ridge', 'elasticnet': 'elasticnet', 'huber_linear': 'huber_linear',
    'sgd_linear': 'sgd', 'mlp': 'mlp', 'knn': 'knn',
    'lightgbm': 'lightgbm', 'catboost': 'catboost', 'xgboost': 'xgboost',
    'hist_gb': 'hist_gradient_boosting', 'random_forest': 'random_forest',
    'extra_trees': 'extra_trees',
}


def build_stage_input(preparation_dir, margo_root, output_dir, *, probe_seconds=90,
                      probe_max_rows=12000, probe_folds=2):
    """Export only raw outer-fit rows; outer-validation labels never enter probes."""
    preparation_dir = Path(preparation_dir).resolve()
    margo_root = Path(margo_root).resolve()
    output_dir = Path(output_dir).resolve()
    schemas = _schemas(margo_root)
    manifest = _envelope(preparation_dir/'input_manifest.json', 'input_manifest', schemas)
    prepared = _envelope(preparation_dir/'output_manifest.json', 'output_manifest', schemas)
    profile = _envelope(preparation_dir/'data_profile.json', 'data_profile', schemas)
    report = _envelope(preparation_dir/'preparation_report.json', 'preparation_report', schemas)
    if manifest['task_type'] != 'regression' or len(manifest['target_columns']) != 1:
        raise ValueError('Model selection requires a single-target regression preparation')
    if prepared['target_columns'] != manifest['target_columns'] or prepared['id_columns'] != manifest['id_columns']:
        raise ValueError('Preparation input/output targets or identifiers differ')
    metric = _metric(manifest['metric_name'])
    target, ids = manifest['target_columns'][0], manifest['id_columns']
    train_refs = [item for item in manifest['files'] if item['role'] == 'train']
    if len(train_refs) != 1:
        raise ValueError('Preparation must identify exactly one raw train file')
    source = _resolve(train_refs[0]['path'], preparation_dir, margo_root)
    train = pd.read_csv(source, dtype={column: str for column in ids})
    if target not in train or any(column not in train for column in ids):
        raise ValueError('Raw training table is missing declared target or identifiers')
    numeric = pd.to_numeric(train[target], errors='coerce')
    if (train[target].notna() & numeric.isna()).any():
        raise ValueError('Raw regression target contains nonnumeric values')
    clean = train.loc[train[target].notna() & ~np.isinf(numeric)].copy()
    clean[target] = numeric.loc[clean.index]
    split = _check_split(prepared.get('split_indices'), clean)
    outer_fit = clean.iloc[split['fit']].copy()
    recipe_path = _resolve(prepared['recipe_path'], preparation_dir, margo_root, prepared=True)
    recipe = schemas.FeaturePlanPayload.model_validate(_json(recipe_path)).model_dump(mode='json')
    if any(item['column'] == target for item in recipe.get('column_transforms', [])) or any(
            target in item['source_columns'] for item in recipe.get('new_features', [])):
        raise ValueError('Preparation feature recipe must not use target as a feature source')
    if len(outer_fit) < max(4, int(probe_folds)+1):
        raise ValueError('Not enough outer-fit rows for the requested selection probe folds')
    if float(probe_seconds) <= 0 or int(probe_max_rows) < max(4, int(probe_folds)+1) or int(probe_folds) < 2:
        raise ValueError('Selection probe budget, row cap and fold count must be positive and usable')
    output_dir.mkdir(parents=True, exist_ok=True)
    fit_path = output_dir/'selection_fit.csv'
    atomic_bytes(fit_path, outer_fit.to_csv(index=False).encode('utf-8'))
    feature_frame = outer_fit.drop(columns=[target, *ids])
    categorical = [str(column) for column in feature_frame if (
        pd.api.types.is_object_dtype(feature_frame[column]) or
        isinstance(feature_frame[column].dtype, pd.CategoricalDtype) or
        pd.api.types.is_bool_dtype(feature_frame[column]))]
    # The external selector accepts one id_col; exclude additional identifiers explicitly.
    stage_input = {
        'task': {'task_type': 'regression', 'target': target, 'id_col': ids[0] if ids else None,
                 'target_metric': metric, 'metric_params': {},
                 'task_description': manifest.get('task_description', ''),
                 'primary_metric_authority': 'margo.input_manifest.metric_name'},
        'paths': {'train': str(fit_path), 'test': None},
        'hints': {'categorical_cols': categorical, 'drop_cols': ids[1:],
                  'group_col': manifest.get('group_column'), 'time_col': manifest.get('time_column')},
        'budget': {'probe_seconds': float(probe_seconds), 'probe_max_rows': int(probe_max_rows),
                   'probe_folds': int(probe_folds)},
        'eda': {'upstream_data_profile': profile, 'upstream_preparation_report': report,
                'prepared_feature_names': prepared['feature_names'], 'preparation_recipe': recipe,
                'selection_boundary': {
                    'train_rows': len(clean), 'outer_fit_rows': len(outer_fit),
                    'outer_validation_rows': len(split['valid']),
                    'raw_source_rows_used': outer_fit.index.tolist(),
                    'selection_and_probe_use_only_outer_fit': True,
                    'probe_feature_representation': 'raw outer-fit features; training replays the margo recipe',
                    'limitations': ['The upstream EDA/feature plan was computed before the outer split; '
                                    'it is not fully holdout-blind even though selection probes use outer-fit rows only.']}}
    }
    save(output_dir/'stage_input.json', stage_input)
    save(output_dir/'selection_input_provenance.json', {
        'preparation_dir': str(preparation_dir), 'margo_root': str(margo_root),
        'raw_train': str(source), 'primary_metric': metric, 'target': target, 'id_columns': ids,
        'outer_fit_rows': len(outer_fit), 'excluded_outer_validation_rows': len(split['valid']),
        'raw_source_rows_used': outer_fit.index.tolist(),
    })
    return stage_input


def build_training_selection(stage_output, *, primary_metric, target, id_columns, submission=None):
    """Keep each selected family/loss/target transform intact, without invented models."""
    metric = _metric(primary_metric)
    if stage_output.get('metric_source') != 'task_description':
        raise ValueError('Model selection must preserve the task metric; metric fallback is forbidden')
    if stage_output.get('target_metric') != metric:
        raise ValueError('Model-selection output metric differs from the authoritative task metric')
    task = stage_output.get('task', {})
    if task.get('task_type') != 'regression' or task.get('target') != target or task.get('target_metric') != metric:
        raise ValueError('Model-selection task type, target or metric changed during the stage')
    candidates = stage_output.get('candidates')
    if not isinstance(candidates, list) or not 1 <= len(candidates) <= 12:
        raise ValueError('Model selection must produce between one and twelve candidates')
    selected = []
    for candidate in sorted(candidates, key=lambda item: item['priority']):
        model_id = candidate['model_id']
        if model_id not in MODEL_FAMILIES:
            raise ValueError(f'Unsupported selected family: {model_id!r}; no candidate was silently dropped')
        transform = candidate['target_transform']
        if transform not in {'none', 'log1p'}:
            raise ValueError(f'Unsupported selected target transform: {transform!r}')
        parameters = candidate['loss_native']
        if not isinstance(parameters, dict):
            raise ValueError('loss_native must be a native estimator-parameter object')
        selected.append({
            'family': MODEL_FAMILIES[model_id], 'rationale': candidate['rationale'],
            'parameters': dict(parameters), 'target_transform': transform,
            'selection_metadata': {key: value for key, value in candidate.items() if key not in {'rationale'}} |
                                  {'locked_parameters': sorted(parameters), 'loss_native': dict(parameters)},
        })
    return {
        'primary_metric': metric, 'id_columns': list(id_columns), 'selected_models': selected,
        'submission': dict(submission or {}),
        'selection_report': {key: stage_output[key] for key in (
            'stage', 'version', 'target_metric', 'metric_source', 'metric_statistic', 'loss_policy',
            'diagnostics', 'probe', 'candidates', 'rejected', 'reasoning', 'automl_hint', 'source',
            'validation_errors', 'warnings', 'hints', 'timings') if key in stage_output},
    }

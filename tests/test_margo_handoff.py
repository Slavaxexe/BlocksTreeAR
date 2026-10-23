"""Boundary checks using small prepared artifacts; no Qwen or model training."""
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from agent.margo_handoff import build_training_handoff
from agent.training_contracts import TrainingInput


class _Payload:
    def __init__(self, data):
        self.data = data

    def model_dump(self, **kwargs):
        return self.data


class _Envelope:
    @staticmethod
    def model_validate(data):
        return SimpleNamespace(**{**data, 'artifact_type': SimpleNamespace(value=data['artifact_type']),
                                  'status': SimpleNamespace(value=data['status'])})


class MargoHandoffTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / 'margo'
        (self.source / 'data_preparation').mkdir(parents=True)
        (self.source / 'data_preparation' / 'schemas.py').write_text('# fixture source\n', encoding='utf-8')
        self.prepared = self.root / 'prepared'
        self.prepared.mkdir()
        train = pd.DataFrame({'id': ['0001', '0002', '0003', '0004', '0005', '0006'],
                              'x': [1., 2., 3., 4., 5., 6.],
                              'y': [10., None, 30., float('inf'), 50., 60.]})
        test = pd.DataFrame({'id': ['0007', '0008'], 'x': [7., 8.]})
        train.to_csv(self.source / 'train.csv', index=False)
        test.to_csv(self.source / 'test.csv', index=False)
        pd.DataFrame({'id': ['0008', '0007'], 'prediction': [0., 0.]}).to_csv(
            self.source / 'sample_submission.csv', index=False)
        manifest = {'task_type': 'regression', 'metric_name': 'root_mean_squared_error',
                    'target_columns': ['y'], 'id_columns': ['id'],
                    'files': [{'role': role, 'path': f'{role}.csv'}
                              for role in ['train', 'test', 'sample_submission']]}
        self._envelope('input_manifest', manifest)
        output = {'target_columns': ['y'], 'id_columns': ['id'], 'feature_names': ['x'],
                  'split_indices': {'train': [4, 0, 2], 'val': [5]},
                  'recipe_path': 'recipe.json'}
        for name, rows in [('train', [4, 0, 2]), ('val', [5]), ('test', None)]:
            frame = test if rows is None else train.iloc[rows]
            for kind, columns in [('features', ['x']), ('target', ['y']), ('ids', ['id'])]:
                if rows is None and kind == 'target':
                    continue
                data = frame[columns].reset_index(drop=True).copy()
                if kind == 'ids':
                    # margo's CSV loader parses these as numbers. Restore from raw CSV later.
                    data['id'] = pd.to_numeric(data['id'])
                path = self.prepared / f'{name}_{kind}.parquet'
                data.to_parquet(path, index=False)
                output[f'{name}_{kind}'] = {'path': path.name, 'format': 'parquet',
                                           'n_rows': len(data), 'n_columns': len(columns), 'columns': columns}
        self._envelope('output_manifest', output)
        self._envelope('data_profile', {'train_summary': {'n_rows': len(train)}})
        self._envelope('preparation_report', {'train_val_split': {'method': 'random'}})
        self._save(self.prepared / 'recipe.json', {'column_transforms': [], 'new_features': [],
                                                  'columns_to_drop': ['id']})
        self.selection = self.root / 'selection.json'
        self._save(self.selection, {'selected_models': [{'family': 'ridge', 'parameters': {'alpha': 1.}}],
                                    'primary_metric': 'rmse', 'submission': {'prediction_column': 'prediction'}})
        # Only the external artifact schemas are replaced. Our strict training contract,
        # parquet files, row mapping, metric authority and split checks execute normally.
        schemas = SimpleNamespace(SCHEMA_VERSION='1.0.0', ArtifactEnvelope=_Envelope,
                                  validate_payload=lambda kind, value: _Payload(value),
                                  FeaturePlanPayload=SimpleNamespace(model_validate=_Payload))
        patcher = patch('agent.margo_handoff._schemas', return_value=schemas)
        patcher.start()
        self.addCleanup(patcher.stop)

    @staticmethod
    def _save(path, value):
        path.write_text(json.dumps(value), encoding='utf-8')

    def _envelope(self, name, payload):
        self._save(self.prepared / f'{name}.json', {'schema_version': '1.0.0', 'artifact_type': name,
                                                  'status': 'success', 'payload': payload, 'errors': []})

    def test_dropped_row_mapping_original_ids_and_authoritative_metric(self):
        path = build_training_handoff(self.prepared, self.selection, self.source, self.root / 'handoff')
        contract = TrainingInput.model_validate_json(path.read_text(encoding='utf-8'))
        self.assertEqual(contract.primary_metric, 'rmse')
        self.assertEqual(pd.read_parquet(contract.files.train)['id'].tolist(), ['0001', '0003', '0005', '0006'])
        self.assertEqual(pd.read_parquet(contract.files.sample_submission)['id'].tolist(), ['0008', '0007'])
        split = json.loads(Path(contract.preparation.split).read_text(encoding='utf-8'))
        self.assertEqual(split, {'fit': [2, 0, 1], 'valid': [3]})
        self.assertEqual(contract.selected_models[0].family, 'ridge')
        provenance = json.loads(path.with_name('provenance.json').read_text(encoding='utf-8'))
        self.assertEqual(provenance['dropped_source_rows'], [1, 3])
        self.assertIn('data_preparation/schemas.py', contract.preparation.source_hashes)
        self.assertFalse((self.source / 'input.json').exists())

    def test_rejects_selection_metric_override(self):
        selection = json.loads(self.selection.read_text(encoding='utf-8'))
        selection['primary_metric'] = 'mae'
        self._save(self.selection, selection)
        with self.assertRaisesRegex(ValueError, 'authoritative upstream metric'):
            build_training_handoff(self.prepared, self.selection, self.source, self.root / 'bad_metric')

    def test_rejects_overlapping_outer_split(self):
        path = self.prepared / 'output_manifest.json'
        envelope = json.loads(path.read_text(encoding='utf-8'))
        envelope['payload']['split_indices']['val'] = [2, 5]
        self._save(path, envelope)
        with self.assertRaisesRegex(ValueError, 'non-overlapping'):
            build_training_handoff(self.prepared, self.selection, self.source, self.root / 'bad_split')


if __name__ == '__main__':
    unittest.main()

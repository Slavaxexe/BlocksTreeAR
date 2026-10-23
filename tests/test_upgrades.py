"""Composed features, native categoricals and honest saved-prediction weight search."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from agent.features import create_features
from agent.operations import run_operation
from agent.predictions import load_pair, save_pair
from tests import test_ensembles, test_operations


class UpgradeTests(unittest.TestCase):
    def test_expression_statistics_are_fit_only_and_targets_are_unavailable(self):
        fit = pd.DataFrame({'group': ['a', 'a', 'b'], 'value': [2., 4., 12.], 'target': [0, 1, 0]})
        query = pd.DataFrame({'group': ['a', 'new'], 'value': [30., 50.]})
        recipe = {'name': 'center', 'operation': 'expression', 'columns': ['group', 'value'],
                  'parameters': {'expression': 'where(x1 > 0, x1 - group_mean(x0, x1), 0)'}}
        _, transformed = create_features([fit, query], [recipe], forbidden=['target'])
        np.testing.assert_allclose(transformed.center, [27., 44.])
        recipe['parameters']['expression'] = "where(col('value').isna(), 50, col('value').fillna(0))"
        filled = create_features([fit.assign(value=[np.nan, 4., 12.])], [recipe], forbidden=['target'])[0]
        np.testing.assert_allclose(filled.center, [50., 4., 12.])
        recipe['parameters']['expression'] = 'col("target")'
        with self.assertRaises(ValueError):
            create_features([fit], [recipe], forbidden=['target'])
        recipe['parameters']['expression'] = '__import__("os").getcwd()'
        with self.assertRaises(ValueError):
            create_features([fit], [recipe], forbidden=['target'])

    def test_native_boosting_handles_unseen_categories(self):
        choices = [('catboost', 'catboost.CatBoostClassifier', {'iterations': 3, 'verbose': False, 'allow_writing_files': False}),
                   ('lightgbm', 'lightgbm.LGBMClassifier', {'n_estimators': 3, 'verbosity': -1}),
                   ('xgboost', 'xgboost.XGBClassifier', {'n_estimators': 3})]
        for backend, estimator, parameters in choices:
            with tempfile.TemporaryDirectory() as temporary:
                state = test_operations.OperationTests().state(True)
                state['preparation']['runtime']['work_directory'] = temporary
                state.update(run_operation(state, {'kind': 'preprocess', 'parameters': {
                    'encoding': 'native', 'native_backend': backend}}, {'task_type': 'classification'}))
                self.assertIsInstance(state['fit_features'], pd.DataFrame)
                state.update(run_operation(state, {'kind': 'fit', 'parameters': {
                    'estimator': estimator, 'parameters': parameters}}, {'task_type': 'classification'}))
                result = run_operation(state, {'kind': 'predict'}, {'task_type': 'classification'})
                self.assertEqual(len(result['test_predictions']), 12)

    def test_weight_search_uses_saved_meta_predictions_without_retraining(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            state, defaults, sha = test_ensembles.EnsembleTests().setup_run(root, False)
            (root / 'prepare.py').write_text('import numpy as np\ndef score(reference,predictions,config):\n    return float(np.mean((reference.y-predictions.y)**2))\n', encoding='utf-8')
            for index in (0, 1):
                node = root / 'nodes' / f'node_{index:03d}'
                frames = load_pair(root, node.name, sha)
                frames['meta']['y'] = state['auxiliary']['ensemble_meta'].y if index == 0 else 100000.
                finish = json.loads((node / 'finish.json').read_text())
                save_pair(node, frames, sha, finish['code_sha256'], 'validation_predictions.csv')
            call = {'kind': 'saved_ensemble', 'parameters': {'source_nodes': ['node_000', 'node_001'],
                    'method': 'mean', 'optimize_weights': True, 'weight_trials': 5}}
            with patch('agent.operations._fit', side_effect=AssertionError('Source was retrained')):
                result = run_operation(state, call, defaults)
            facts = json.loads((root / 'nodes/node_002/weights_full_diagnostics.json').read_text())
            np.testing.assert_allclose(facts['weights'], [1., 0.])
            self.assertFalse(result['validation_predictions'].attrs['meta_held_out'])

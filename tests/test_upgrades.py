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
    def test_eda_build_executes_without_replanning_and_changes_preserve_settings(self):
        from agent.builds import register_eda, compile_build, store
        from agent.contracts import Idea, Assembly, AssemblyChanges, BlockEdit
        from agent.plans import FeatureSpec
        from agent.eda import analyze_features
        state = test_operations.OperationTests().state()
        facts = analyze_features(state, {'name': 'sum', 'kind': 'feature', 'columns': ['a'],
            'features': [{'name': 'sum_ab', 'operation': 'sum', 'columns': ['a', 'b']}],
            'sample_rows': None}, task_type='regression')
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ready = register_eda(root, 'eda_' + 'a' * 16, facts['candidate']['recipe_configuration'], {})
            idea = Idea(purpose='improve', parent_id='node_000', change='Test the measured recipe', expected_effect='Measure')
            proposal = compile_build(ready, None, idea)
            for block in proposal.assembly.blocks:
                state.update(run_operation(state, block.operation, {'task_type': 'regression'}))
            self.assertEqual(len(state['test_predictions']), 12)
            self.assertEqual(state['model']['estimator'].n_estimators, 64)
            parent = store(root, 'node_001', proposal.assembly, ready['model'])
            extra = FeatureSpec(name='delta', operation='difference', columns=['a', 'b'])
            changed = compile_build(parent, AssemblyChanges(edits=[BlockEdit(block='features', features=[extra])]), idea)
            calls = {block.name: block.operation for block in changed.assembly.blocks}
            self.assertEqual(calls['fit'], proposal.assembly.blocks[2].operation)
            self.assertEqual(calls['preprocess'].parameters['scaling'], 'standard')
            self.assertEqual(calls['preprocess'].parameters['feature_columns'], ['a', 'sum_ab', 'delta'])
            self.assertEqual(parent['assembly']['blocks'][1]['operation']['parameters']['feature_columns'], ['a', 'sum_ab'])
            tuned = compile_build(parent, AssemblyChanges(edits=[BlockEdit(block='fit', parameters={
                'parameters': {'min_samples_leaf': 5}})]), idea)
            for block in tuned.assembly.blocks:
                state.update(run_operation(state, block.operation, {'task_type': 'regression'}))
            self.assertEqual(state['model']['estimator'].min_samples_leaf, 5)
            self.assertEqual(state['model']['estimator'].n_estimators, 64)

    def test_finalist_splits_keep_time_groups_and_original_positions(self):
        from agent.final_validation import extra_splits
        frame = pd.DataFrame({'period': np.repeat([1, 2, 3, 4, 5], 8),
                              'person': np.tile(np.arange(8), 5)})
        available = np.arange(32)
        timed = extra_splits(frame, available, {'strategy': 'time', 'time_column': 'period'}, [], 2)
        self.assertEqual(len(timed), 2)
        for train, valid in timed:
            self.assertLess(frame.iloc[train].period.max(), frame.iloc[valid].period.min())
            self.assertTrue(set(train) | set(valid) <= set(available))
            self.assertFalse(set(train) & set(valid))
        grouped = extra_splits(frame, available, {'strategy': 'group', 'group_columns': ['person']}, [], 2)
        for train, valid in grouped:
            self.assertFalse(set(frame.iloc[train].person) & set(frame.iloc[valid].person))

    def test_finalist_promotion_needs_matched_consistent_results(self):
        from agent.final_validation import choose
        candidates = [{'node_id': 'a', 'value': 40.}, {'node_id': 'b', 'value': 41.}]
        self.assertEqual(choose('a', candidates, {'a': [40., 40.], 'b': [30., 31.]}, 'minimize', 2), 'b')
        self.assertEqual(choose('a', candidates, {'a': [40., 40.], 'b': [20., 41.]}, 'minimize', 2), 'a')
        self.assertEqual(choose('a', candidates, {'a': [40., None], 'b': [20., 21.]}, 'minimize', 2), 'a')
        self.assertEqual(choose('a', candidates, {'a': [40.], 'b': [20.]}, 'minimize', 2), 'a')

    def test_unquoted_col_and_history_defaults_match_executor_meaning(self):
        from agent.feature_expression import FeatureExpression
        from agent.feature_trials import recipe_keys
        frame = pd.DataFrame({'period': [1, 2], 'built': [1990, 2000]})
        expression = FeatureExpression('col(period) - col(built)', ['period', 'built'])
        np.testing.assert_array_equal(expression.fit(frame).transform(frame), [-1989, -1998])
        with self.assertRaises(ValueError):
            FeatureExpression('col(secret)', ['period']).fit(frame)
        recipe = {'name': 'past', 'operation': 'history', 'columns': ['entity'],
                  'parameters': {'time_column': 'period', 'value_column': 'target'}}
        explicit = {**recipe, 'parameters': {**recipe['parameters'], 'window': None,
                    'max_age': None, 'lag': 1, 'min_history': 1, 'fallback': 'missing'}}
        self.assertEqual(recipe_keys([recipe]), recipe_keys([explicit]))
        explicit['parameters']['lag'] = 2
        self.assertNotEqual(recipe_keys([recipe]), recipe_keys([explicit]))

    def test_key_proposals_exclude_labels_and_report_ambiguity(self):
        from agent.entity_candidates import discover
        train = pd.DataFrame({'period': np.repeat([1, 2], 30), 'size': list(range(30)) * 2,
                              'kind': ['a'] * 60, 'target': list(range(60))})
        train['region'] = np.tile(['a', 'b'], 30)
        query = train.iloc[:10].assign(period=3)
        result = discover(train, query, excluded=['target'], seconds=2)
        self.assertTrue(result['candidates'])
        self.assertTrue(all('target' not in row['columns'] for row in result['candidates']))
        best = result['candidates'][0]
        self.assertEqual(best['query_row_coverage'], 1)
        self.assertEqual(best['within_period_ambiguity_fraction'], 0)

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

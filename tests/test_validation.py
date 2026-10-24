"""EDA, ensembles and AutoML must interpret the same split metadata."""
import unittest

import numpy as np
import pandas as pd

from agent.eda import _inner_split, analyze_features
from agent.ensemble_data import make_split
from agent.plans import AnalysisRequest, ToolCall
from agent.validation import metadata, query_range_evidence, split_plan, split_rows


class ValidationTests(unittest.TestCase):
    def test_preparation_runs_without_a_generated_split_function(self):
        import json
        import tempfile
        from pathlib import Path
        from agent.data import file_hash
        from agent.storage import save
        from agent.worker import prepare
        from tests.test_research import fixture
        with tempfile.TemporaryDirectory() as temporary:
            _, root, _ = fixture(Path(temporary))
            path = root/'prepare.py'
            path.write_text(path.read_text(encoding='utf-8').replace(
                'def make_split(train, config):\n    return np.arange(6), np.arange(6, 8)\n', ''), encoding='utf-8')
            config = json.loads((root/'preparation.json').read_text(encoding='utf-8'))
            config['code_sha256'] = file_hash(path)
            save(root/'preparation.json', config)
            prepare(root)
            with np.load(root/'split.npz') as split:
                np.testing.assert_array_equal(split['valid'], [6, 7])

    def test_builtin_time_split_keeps_original_positions_on_unsorted_nonrange_index(self):
        frame = pd.DataFrame({'period': [3, 1, 2, 3, 1, 2]}, index=[80, 10, 70, 40, 50, 90])
        fit, valid = split_rows(frame, {'strategy': 'time', 'time_column': 'period', 'validation_periods': 1})
        np.testing.assert_array_equal(fit, [1, 2, 4, 5])
        np.testing.assert_array_equal(valid, [0, 3])
        self.assertLess(frame.iloc[fit].period.max(), frame.iloc[valid].period.min())

    def test_time_cutoff_gap_and_group_isolation(self):
        frame = pd.DataFrame({'period': ['2024-01-03', '2024-01-01', '2024-01-02'] * 2})
        fit, valid = split_rows(frame, {'strategy': 'time', 'time_column': 'period',
                                      'cutoff': '2024-01-03', 'gap_periods': 1})
        np.testing.assert_array_equal(fit, [1, 4])
        np.testing.assert_array_equal(valid, [0, 3])
        panel = pd.DataFrame({'period': [1, 1, 1, 2], 'object': ['old', 'stay', 'new', 'new']})
        fit, valid = split_rows(panel, {'strategy': 'group_time', 'time_column': 'period',
                                      'group_columns': ['object'], 'validation_periods': 1})
        self.assertFalse(set(panel.iloc[fit].object) & set(panel.iloc[valid].object))

    def test_random_stratified_and_grouped_plans_are_reproducible(self):
        frame = pd.DataFrame({'object': np.repeat(np.arange(30), 4), 'label': np.repeat(np.arange(30) % 2, 4)})
        for kind in ['random', 'stratified', 'group', 'stratified_group']:
            policy = {'strategy': kind, 'fraction': .25, 'seed': 17}
            if 'group' in kind:
                policy['group_columns'] = ['object']
            plan = split_plan(policy, ['label'])
            fit, valid = split_rows(frame, plan, ['label'])
            again = split_rows(frame, plan, ['label'])
            np.testing.assert_array_equal(valid, again[1])
            self.assertFalse(set(fit) & set(valid))
            if 'group' in kind:
                self.assertFalse(set(frame.iloc[fit].object) & set(frame.iloc[valid].object))
            if 'stratif' in kind:
                self.assertEqual(set(frame.iloc[valid].label), {0, 1})
        fit, valid = split_rows(frame, {'strategy': 'ordered', 'rows': 8, 'gap_rows': 2})
        np.testing.assert_array_equal(valid, np.arange(112, 120))
        self.assertEqual(fit[-1], 109)

    def test_range_separation_does_not_force_qwen_to_choose_time(self):
        import tempfile
        from pathlib import Path
        from agent.contracts import PreparationDraft
        from agent.storage import save
        from agent.worker import prepare
        from tests.test_research import fixture
        measured = query_range_evidence(pd.DataFrame({'measurement': [1, 2, 3]}),
                                       pd.DataFrame({'measurement': [4, 5]}))
        self.assertEqual(measured[0]['query_range'], 'above_train')
        with tempfile.TemporaryDirectory() as temporary:
            task, root, draft = fixture(Path(temporary))
            choice = {**draft, 'validation': {'strategy': 'random', 'deployment_scenario': 'exchangeable',
                'deployment_reason': 'A shifted measurement range does not define observation time'}}
            accepted = PreparationDraft.model_validate(choice, context={'require_deployment_assessment': True})
            self.assertEqual(accepted.validation['strategy'], 'random')
            choice['validation'] = {**choice['validation'], 'deployment_scenario': 'future_period', 'time_column': 'measurement'}
            with self.assertRaisesRegex(ValueError, 'non-random'):
                PreparationDraft.model_validate(choice)
            import json
            saved = json.loads((root/'preparation.json').read_text(encoding='utf-8'))
            saved['validation'] = {'strategy': 'time', 'time_column': 'feature', 'deployment_scenario': 'future_period'}
            save(root/'preparation.json', saved)
            frame = pd.read_csv(task.parent/'train.csv')
            frame['feature'] = frame['feature'].iloc[::-1].to_numpy()
            frame.to_csv(task.parent/'train.csv', index=False)
            # The old generated function returns the wrong rows for this descending table.
            # The trusted executor ignores it and constructs correct original positions.
            prepare(root)
            with np.load(root/'split.npz') as split:
                self.assertLess(frame.iloc[split['fit']].feature.max(), frame.iloc[split['valid']].feature.min())

    def test_year_alias_produces_the_same_temporal_holdout(self):
        frame = pd.DataFrame({'period': [1, 1, 2, 2, 3, 3], 'y': range(6)})
        policy = {'strategy': 'time_based', 'year_column': 'period'}
        normalized = metadata(policy)
        self.assertEqual(normalized['time_column'], normalized['year_column'])
        self.assertNotIn('time_column', policy)
        request = AnalysisRequest(name='check', kind='predictive', validation_fraction=.2)
        actual = _inner_split(frame, frame.y, request, policy, False)
        reference = make_split(frame, ['y'], policy)
        for indices, expected in zip(actual, reference):
            np.testing.assert_array_equal(indices, expected)
        self.assertLess(frame.iloc[actual[0]].period.max(), frame.iloc[actual[1]].period.min())
        ToolCall(kind='automl', parameters={'engine': 'flaml', 'split_column': 'period'})

    def test_group_summary_accepts_redundant_target_in_columns(self):
        state = {'fit': pd.DataFrame({'group': ['a', 'a', 'b'], 'x': [1, 2, 3], 'y': [4, 6, 8]}),
                 'preparation': {'data': {'target_columns': ['y']}}}
        result = analyze_features(state, {'name': 'summary', 'kind': 'groups',
            'group_by': 'group', 'columns': ['x', 'y'], 'target': 'y'}, task_type='regression')
        self.assertTrue(result['target_available'])
        self.assertNotIn('y', result['features'])

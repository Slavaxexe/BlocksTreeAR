"""EDA, ensembles and AutoML must interpret the same split metadata."""
import unittest

import numpy as np
import pandas as pd

from agent.eda import _inner_split, analyze_features
from agent.ensemble_data import make_split
from agent.plans import AnalysisRequest, ToolCall
from agent.validation import metadata


class ValidationTests(unittest.TestCase):
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

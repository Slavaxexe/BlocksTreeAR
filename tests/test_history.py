"""Leakage boundaries for reusable supervised feature tools; no LLM calls."""
import unittest

import numpy as np
import pandas as pd

from agent.eda import analyze_features
from agent.features import create_features
from agent.operations import run_operation
from agent.supervised_features import row_weights
from tests import test_operations


class HistoryTests(unittest.TestCase):
    def test_automl_recipes_are_selected_on_inner_train_then_refitted(self):
        from tempfile import TemporaryDirectory
        import time
        from agent.automl import run_automl
        from agent.plans import AutoMLPlan
        from agent.selected_model import refit_saved
        for classification in (False, True):
            for engine in ('flaml', 'autogluon'):
                with self.subTest(engine=engine, classification=classification), TemporaryDirectory() as temporary:
                    state = test_operations.OperationTests().state(classification)
                    state['fit']['period'] = np.repeat([1, 2, 3], 60)
                    for scope in ('validation', 'test'):
                        state[scope]['period'] = 4
                    state['preparation']['validation'] = {'strategy': 'time_based', 'year_column': 'period'}
                    state['preparation']['runtime'].update(work_directory=temporary, deadline=time.monotonic() + 45)
                    recipe = {'name': 'past', 'operation': 'history', 'columns': ['group'], 'parameters': {
                        'time_column': 'period', 'value_column': 'target', 'class_label': 'yes' if classification else None}}
                    config = {'engine': engine, 'num_trials': 1, 'features': [recipe],
                        'models': ['xgboost'] if engine == 'flaml' else ['GBM'],
                        'model_parameters': {'xgboost': {'n_estimators': 3}} if engine == 'flaml' else {'GBM': {'num_boost_round': 3}},
                        'row_weights': {'time_column': 'period', 'half_life': 1}}
                    defaults = {'task_type': 'classification' if classification else 'regression'}
                    result = run_automl(state, engine=engine, config=config, **defaults)
                    facts = result['automl_diagnostics']
                    self.assertEqual((facts['inner_fit_rows'], facts['inner_validation_rows'], facts['full_fit_rows']), (120, 60, 180))
                    self.assertTrue(facts['refitted_on_fit'])
                    replay = refit_saved(state, AutoMLPlan.model_validate(config), facts['selected_model_file'], defaults)
                    self.assertEqual(len(replay['test_predictions']), 12)

    def test_history_excludes_self_same_period_and_future_labels(self):
        fit = pd.DataFrame({'entity': ['a', 'a', 'a', 'a', 'b'], 'period': [1, 2, 2, 3, 1],
                            'y': [10., 20., 40., 90., 100.]})
        query = pd.DataFrame({'entity': ['a', 'a', 'b', 'new'], 'period': [3, 4, 2, 4],
                              'y': ['query labels must never be read'] * 4})
        recipe = {'name': 'past', 'operation': 'history', 'columns': ['entity'],
                  'parameters': {'time_column': 'period', 'value_column': 'y'}}
        train, test = create_features([fit, query], [recipe], forbidden=['y'])
        np.testing.assert_allclose(train.past, [np.nan, 10, 10, 30, np.nan], equal_nan=True)
        np.testing.assert_allclose(test.past, [30, 90, 100, np.nan], equal_nan=True)
        changed = fit.copy()
        changed.loc[changed.period == 3, 'y'] = 9999.
        new_train, new_query = create_features([changed, query], [recipe], forbidden=['y'])
        np.testing.assert_allclose(new_train.past, train.past, equal_nan=True)
        self.assertEqual(new_query.past.iloc[0], 30)
        recipe['parameters'].update(aggregation='mean', window=1)
        self.assertEqual(create_features([fit, query], [recipe], forbidden=['y'])[1].past.iloc[0], 30)
        for operation, expected in [('min', 20), ('max', 40), ('count', 2)]:
            recipe['parameters']['aggregation'] = operation
            self.assertEqual(create_features([fit, query], [recipe], forbidden=['y'])[1].past.iloc[0], expected)

    def test_target_encoding_is_cross_fitted_and_group_aware(self):
        fit = pd.DataFrame({'key': ['a'] * 4 + ['b'] * 4, 'person': [1] * 4 + [2] * 4,
                            'y': np.arange(8, dtype=float)})
        recipe = {'name': 'encoded', 'operation': 'target_encoding', 'columns': ['key'],
                  'parameters': {'target': 'y', 'folds': 2}}
        baseline = create_features([fit], [recipe], forbidden=['y'])[0]
        changed = fit.copy()
        changed.loc[0, 'y'] = 9999.
        encoded = create_features([changed], [recipe], forbidden=['y'])[0]
        self.assertEqual(baseline.encoded.iloc[0], encoded.encoded.iloc[0])
        grouped = create_features([fit], [recipe], forbidden=['y'],
            policy={'group_columns': ['person']})[0]
        np.testing.assert_allclose(grouped.encoded, [5.5] * 4 + [1.5] * 4)
        recipe['parameters'].update(validation='time', time_column='person')
        temporal = create_features([fit], [recipe], forbidden=['y'])[0]
        self.assertTrue(temporal.encoded.iloc[:4].isna().all())
        np.testing.assert_allclose(temporal.encoded.iloc[4:], 1.5)

    def test_entity_probe_runs_without_targets_and_ignores_sample_limit(self):
        fit = pd.DataFrame({'kind': ['a', 'a', 'b'], 'area': [1, 1, 2], 'period': [1, 2, 1]})
        query = pd.DataFrame({'kind': ['a', 'a', 'new'], 'area': [1, 1, 3], 'period': [2, 3, 3]})
        state = {'fit': fit, 'test': query, 'preparation': {'data': {'target_columns': ['y']}}}
        result = analyze_features(state, {'name': 'keys', 'kind': 'entities',
            'key_candidates': [['kind', 'area']], 'split_column': 'period'}, task_type='regression')
        self.assertEqual(result['candidates'][0]['comparison_strict_past_coverage'], 2 / 3)
        self.assertFalse(result['target_available'])

    def test_weighted_gamma_and_seed_average_use_the_ordinary_route(self):
        state = test_operations.OperationTests().state()
        state['fit']['target'] = np.exp(state['fit'].a)
        state['fit']['period'] = np.resize([1, 2, 3], len(state['fit']))
        for scope in ('validation', 'test'):
            state[scope]['period'] = 4
        weights = row_weights(state['fit'], {'time_column': 'period', 'half_life': 1})
        self.assertAlmostEqual(weights[2] / weights[0], 4)
        dates = pd.DataFrame({'date': pd.Series(pd.date_range('2024-01-01', periods=3)).astype('datetime64[us]')})
        dated = row_weights(dates, {'time_column': 'date', 'half_life': 1})
        self.assertAlmostEqual(dated[2] / dated[0], 4)
        state.update(run_operation(state, {'kind': 'preprocess'}))
        state.update(run_operation(state, {'kind': 'fit', 'parameters': {
            'estimator': 'lightgbm.LGBMRegressor', 'parameters': {
                'objective': 'gamma', 'n_estimators': 3, 'verbosity': -1},
            'row_weights': {'time_column': 'period', 'half_life': 1}, 'seeds': [7, 11]}}))
        predictions = run_operation(state, {'kind': 'predict'})['test_predictions']
        self.assertEqual(len(state['model']['estimator'].models), 2)
        self.assertTrue(np.isfinite(predictions.to_numpy()).all())

    def test_feature_research_rebuilds_history_on_its_inner_train(self):
        state = test_operations.OperationTests().state()
        state['fit']['period'] = np.repeat([1, 2, 3], 60)
        state['preparation']['validation'] = {'strategy': 'temporal', 'time_column': 'period'}
        result = analyze_features(state, {'name': 'past_value', 'kind': 'feature',
            'columns': ['a'], 'baseline_columns': ['a'], 'estimator': 'linear',
            'features': [{'name': 'past', 'operation': 'history', 'columns': ['group'],
                'parameters': {'value_column': 'target', 'time_column': 'period'}}]}, task_type='regression')
        slices = result['candidate']['history_slices']['past']
        self.assertEqual(slices['coverage'], 1.)
        self.assertEqual(slices['with_history']['rows'], 60)


if __name__ == '__main__':
    unittest.main()

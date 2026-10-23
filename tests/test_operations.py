"""Feature measurements and trusted experiments, without external API calls."""
import time
import unittest

import numpy as np
import pandas as pd

from agent.eda import CATALOG, analyze_features, analysis_key
from agent.features import create_features
from agent.operations import run_operation
from agent.plans import AutoMLPlan, ToolCall
from agent.contracts import NodePlan


class OperationTests(unittest.TestCase):
    def state(self, classification=False):
        rng = np.random.default_rng(42)
        fit = pd.DataFrame({'id': range(180), 'a': rng.normal(size=180), 'b': rng.normal(size=180),
                            'group': np.resize(['one', 'two', 'three'], 180)})
        fit['target'] = np.where(fit.a > 0, 'yes', 'no') if classification else fit.a * 3 + fit.b
        query = fit.drop(columns='target').iloc[:12].copy()
        return {'fit': fit, 'validation': query.copy(), 'test': query.copy(),
                'preparation': {'data': {'target_columns': ['target'], 'id_columns': ['id']},
                    'metric': {'name': 'Macro F1' if classification else 'RMSE'}, 'validation': {'strategy': 'random'},
                    'output': {'columns': ['id', 'target']},
                    'runtime': {'num_cpus': 1, 'deadline': time.monotonic() + 30, 'is_smoke_check': False}}}

    def test_analysis_catalog_and_measurement_identity(self):
        state = self.state()
        for kind in CATALOG:
            request = {'name': 'check', 'kind': kind, 'columns': ['a', 'b'], 'group_by': 'group',
                       'baseline_columns': ['a', 'b', 'group'], 'repeats': 1}
            if kind == 'feature':
                request['features'] = [{'name': 'sum_ab', 'operation': 'sum', 'columns': ['a', 'b']}]
            result = analyze_features(state, request, task_type='regression')
            self.assertEqual(result['scope'], 'fit', kind)
            self.assertEqual(result['rows_used'], 180, kind)
            self.assertNotEqual(result.get('status'), 'failed', kind)
        a = {'name': 'first', 'kind': 'quality', 'columns': ['a', 'b']}
        self.assertEqual(analysis_key(a), analysis_key({**a, 'name': 'renamed', 'columns': ['b', 'a']}))
        test = analyze_features(state, {**a, 'dataset': 'test'}, task_type='regression')
        self.assertEqual((test['scope'], test['rows_used']), ('test', 12))
        self.assertNotEqual(analysis_key(a), analysis_key({**a, 'dataset': 'test'}))
        with self.assertRaises(ValueError):
            analyze_features(state, {'name': 'leak', 'kind': 'predictive', 'dataset': 'test'})
        measured = analyze_features(state, {'name': 'without_denominator', 'kind': 'feature',
            'baseline_columns': ['a'], 'features': [{'name': 'ratio', 'operation': 'ratio', 'columns': ['a', 'b']}]},
            task_type='regression')
        self.assertEqual(measured['candidate']['feature_count'], 2)

    def test_recipes_learn_from_fit_and_reject_targets(self):
        state = self.state()
        state['test']['group'] = 'unseen'
        recipe = [{'name': 'frequency', 'operation': 'frequency', 'columns': ['group']}]
        frames = create_features([state['fit'], state['test']], recipe, forbidden=['target'])
        self.assertTrue((frames[1].frequency == 0).all())
        with self.assertRaises(ValueError):
            create_features([state['fit']], [{'name': 'leak', 'operation': 'sum', 'columns': ['target']}], forbidden=['target'])

    def test_model_pipeline_preserves_labels_and_tunes_inside_fit(self):
        for classification in (False, True):
            state = self.state(classification)
            defaults = {'task_type': 'classification' if classification else 'regression'}
            state.update(run_operation(state, {'kind': 'features', 'parameters': {'recipes': [
                {'name': 'ratio', 'operation': 'ratio', 'columns': ['a', 'b']},
                {'name': 'frequency', 'operation': 'frequency', 'columns': ['group']},
            ]}}, defaults))
            state.update(run_operation(state, {'kind': 'preprocess', 'inputs': {
                'fit': 'fit_table', 'validation': 'validation_table', 'test': 'test_table'}}, defaults))
            self.assertIn('ratio', state['transformer'].diagnostics_['selected_columns'])
            self.assertNotIn('ratio', state['fit'])
            params = {'estimator': 'sklearn.ensemble.ExtraTreesClassifier' if classification else 'sklearn.ensemble.ExtraTreesRegressor',
                      'parameters': {'n_estimators': 8}}
            if not classification:
                params.update(search_space={'max_depth': {'kind': 'int', 'low': 2, 'high': 4}}, num_trials=2)
            state.update(run_operation(state, {'kind': 'fit', 'parameters': params}, defaults))
            result = run_operation(state, {'kind': 'predict'}, defaults)
            self.assertEqual(list(result['test_predictions']), ['target'])
            self.assertEqual(len(result['test_predictions']), 12)
            if classification:
                self.assertLessEqual(set(result['test_predictions'].target), {'yes', 'no'})
        self.assertEqual(AutoMLPlan(engine='autogluon', validation='cv', stack_levels=1).folds, 3)

    def test_model_ids_and_compiled_interfaces(self):
        for engine, expected in [('flaml', 'lgbm'), ('autogluon', 'GBM')]:
            call = ToolCall.model_validate({'kind': 'automl', 'parameters': {
                'engine': engine, 'models': ['LightGBM'],
                'model_parameters': {'LightGBM': {'learning_rate': 0.05}}}})
            self.assertEqual(call.parameters['models'], [expected])
            self.assertEqual(list(call.parameters['model_parameters']), [expected])
        plan = NodePlan.model_validate({'pipeline': {'preprocessing': 'None',
            'model': {'name': 'Constant', 'configuration': 'Training mean'}, 'training': 'No fit'},
            'order': ['constant'], 'write': [{'name': 'constant', 'description': 'Reference predictor',
                'operation': {'kind': 'constant'}, 'requires': ['invented'], 'provides': ['wrong']}]})
        self.assertIn('fit', plan.write[0].requires)
        self.assertEqual(plan.write[0].provides, ['validation_predictions', 'test_predictions'])

    def test_search_parameter_errors_are_detected_in_smoke(self):
        state = self.state(True)
        defaults = {'task_type': 'classification'}
        state['preparation']['runtime']['is_smoke_check'] = True
        state.update(run_operation(state, {'kind': 'preprocess'}, defaults))
        with self.assertRaisesRegex(ValueError, 'n_estimators'):
            run_operation(state, {'kind': 'fit', 'parameters': {
                'estimator': 'sklearn.ensemble.HistGradientBoostingClassifier',
                'search_space': {'n_estimators': {'kind': 'int', 'low': 5, 'high': 10}}}}, defaults)
        selected = run_operation(state, {'kind': 'preprocess', 'parameters': {'feature_columns': ['a']}}, defaults)
        self.assertEqual(selected['transformer'].diagnostics_['selected_columns'], ['a'])

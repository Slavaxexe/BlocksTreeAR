"""Feature measurements and trusted experiments, without external API calls."""
import time
import unittest
import json
from unittest.mock import patch

import numpy as np
import pandas as pd

from agent.eda import CATALOG, analyze_features, analysis_key
from agent.features import create_features
from agent.operations import run_operation
from agent.plans import AutoMLPlan, ToolCall
from agent.contracts import NodePlan


class OperationTests(unittest.TestCase):
    def test_automl_multiple_targets_freeze_and_refit_without_target_inputs(self):
        import tempfile
        from pathlib import Path
        import cloudpickle
        from agent.finalize import FinalRefit
        state = self.state()
        state['fit']['second'] = state['fit'].target * 2
        state['preparation']['data']['target_columns'] = ['target', 'second']
        state['preparation']['output']['columns'] = ['id', 'target', 'second']
        defaults = {'task_type': 'multioutput_regression'}
        plan = AutoMLPlan(engine='flaml', models=['lgbm'], num_trials=1)
        with tempfile.TemporaryDirectory() as temporary:
            state['preparation']['runtime'].update(work_directory=temporary,
                fit_row_positions=np.arange(len(state['fit'])))
            result = run_operation(state, {'kind': 'automl', 'parameters': plan.model_dump()}, defaults)
            for target in result['automl_diagnostics']['targets']:
                snapshot = cloudpickle.loads(Path(target['diagnostics']['model_file']).read_bytes())
                self.assertFalse({'target', 'second'} & set(snapshot['columns']))
            refitter = FinalRefit.__new__(FinalRefit)
            refitter.defaults = defaults
            refitted = refitter.multi_target(plan, result['automl_diagnostics'], state)
            self.assertEqual(list(refitted['test_predictions']), ['target', 'second'])

    def test_native_automl_forecast_keeps_snapshot_and_query_order(self):
        import tempfile
        from agent.selected_model import refit_saved
        state = self.state()
        state['fit'] = pd.DataFrame({'period': range(60), 'target': np.sin(np.arange(60) / 4)})
        state['validation'] = pd.DataFrame({'period': [60, 61, 62]})
        state['test'] = pd.DataFrame({'period': [65, 63, 64]})
        state['preparation']['data'].update(time_column='period', id_columns=[])
        state['preparation']['output']['columns'] = ['target']
        plan = AutoMLPlan(engine='flaml', models=['lgbm'], num_trials=1,
                          model_parameters={'lgbm': {'n_estimators': 8, 'lags': 3}})
        defaults = {'task_type': 'forecasting'}
        with tempfile.TemporaryDirectory() as temporary:
            state['preparation']['runtime']['work_directory'] = temporary
            result = run_operation(state, {'kind': 'automl', 'parameters': plan.model_dump()}, defaults)
            frozen = refit_saved(state, plan, result['automl_diagnostics']['selected_model_file'], defaults)
            np.testing.assert_allclose(frozen['test_predictions'], result['test_predictions'])
            self.assertEqual(len(frozen['test_predictions']), 3)

    def test_multiple_targets_and_label_spaces(self):
        from agent.task_types import task_score
        for task in ('multioutput_regression', 'multioutput_classification', 'multilabel_classification'):
            state = self.state(task != 'multioutput_regression')
            if task == 'multioutput_regression':
                state['fit']['second'] = state['fit'].a - state['fit'].b * 2
                estimator, metric = 'sklearn.linear_model.Ridge', 'RMSE'
            else:
                state['fit']['second'] = np.where(state['fit'].b > 0, 10, 20)
                if task.startswith('multilabel'):
                    state['fit']['target'] = (state['fit'].a > 0).astype(int)
                    state['fit']['second'] = (state['fit'].b > 0).astype(int)
                estimator, metric = 'sklearn.linear_model.LogisticRegression', 'Macro F1'
            state['preparation']['data']['target_columns'] = ['target', 'second']
            state['preparation']['output']['columns'] = ['id', 'target', 'second']
            state['preparation']['metric']['name'] = metric
            defaults = {'task_type': task}
            state.update(run_operation(state, {'kind': 'preprocess'}, defaults))
            state.update(run_operation(state, {'kind': 'fit', 'parameters': {'estimator': estimator}}, defaults))
            import cloudpickle
            state['model'] = cloudpickle.loads(cloudpickle.dumps(state['model']))
            result = run_operation(state, {'kind': 'predict'}, defaults)['test_predictions']
            self.assertEqual(list(result), ['target', 'second'])
            score = task_score(state['fit'].iloc[:12], result, state['preparation'], task)
            self.assertTrue(np.isfinite(score))
            if task.startswith('multilabel'):
                probability = run_operation(state, {'kind': 'predict', 'parameters': {'output': 'probabilities'}}, defaults)['test_predictions']
                self.assertTrue(((probability >= 0) & (probability <= 1)).all().all())
            proxy = analyze_features(state, {'name': 'multi', 'kind': 'task_predictive', 'columns': ['a', 'b']}, task_type=task)
            self.assertEqual(proxy['status'], 'completed')

    def test_rankers_keep_complete_query_groups_and_prediction_order(self):
        from agent.task_types import task_score
        state = self.state()
        state['fit']['query'] = np.arange(len(state['fit'])) % 15
        state['fit']['target'] = (state['fit'].a > 0).astype(int)
        for scope in ('validation', 'test'):
            state[scope]['query'] = np.arange(12) % 3
        state['preparation']['data']['query_columns'] = ['query']
        state['preparation']['metric'] = {'name': 'NDCG@5', 'direction': 'maximize'}
        defaults = {'task_type': 'ranking'}
        state.update(run_operation(state, {'kind': 'preprocess', 'parameters': {'feature_columns': ['a', 'b']}}, defaults))
        for estimator, parameters in [('lightgbm.LGBMRanker', {'n_estimators': 5, 'verbosity': -1}),
                                       ('xgboost.XGBRanker', {'n_estimators': 5}),
                                       ('catboost.CatBoostRanker', {'iterations': 5, 'verbose': False, 'allow_writing_files': False})]:
            state.update(run_operation(state, {'kind': 'fit', 'parameters': {'estimator': estimator, 'parameters': parameters}}, defaults))
            result = run_operation(state, {'kind': 'predict'}, defaults)['test_predictions']
            reference = state['fit'].iloc[:12].copy()
            reference['query'] = state['test']['query'].to_numpy()
            self.assertTrue(0 <= task_score(reference, result, state['preparation'], 'ranking') <= 1)
        measured = analyze_features(state, {'name': 'rank', 'kind': 'task_predictive', 'columns': ['a', 'b']}, task_type='ranking')
        self.assertEqual(measured['task_family'], 'ranking')

    def test_target_free_clustering_and_recursive_forecast(self):
        from agent.task_types import task_score
        state = self.state()
        state['fit'] = state['fit'].drop(columns='target')
        state['preparation']['data']['target_columns'] = []
        state['preparation']['output']['columns'] = ['id', 'cluster']
        state['preparation']['metric']['name'] = 'Silhouette'
        defaults = {'task_type': 'clustering'}
        state.update(run_operation(state, {'kind': 'preprocess', 'parameters': {'feature_columns': ['a', 'b']}}, defaults))
        state.update(run_operation(state, {'kind': 'fit', 'parameters': {'estimator': 'sklearn.cluster.KMeans', 'parameters': {'n_clusters': 2, 'n_init': 1}}}, defaults))
        result = run_operation(state, {'kind': 'predict'}, defaults)['test_predictions']
        self.assertTrue(np.isfinite(task_score(state['test'], result, state['preparation'], 'clustering')))
        fit = pd.DataFrame({'period': range(40), 'target': np.arange(40) % 2})
        query = pd.DataFrame({'period': [43, 40, 42, 41]})
        metadata = {'data': {'target_columns': ['target'], 'time_column': 'period'},
                    'output': {'columns': ['target']}, 'metric': {'name': 'RMSE'},
                    'validation': {'strategy': 'time', 'time_column': 'period'}}
        state = {'fit': fit, 'validation': query, 'test': query, 'preparation': metadata}
        call = {'kind': 'forecast', 'parameters': {'estimator': 'sklearn.tree.DecisionTreeRegressor',
                'lags': [1], 'feature_columns': ['period']}}
        result = run_operation(state, call, {'task_type': 'forecasting'})
        np.testing.assert_array_equal(result['test_predictions'].target, [1, 0, 0, 1])
        np.testing.assert_array_equal(result['validation_predictions'], result['test_predictions'])
        measured = analyze_features(state, {'name': 'time', 'kind': 'task_predictive'}, task_type='forecasting')
        self.assertEqual(measured['task_family'], 'forecasting')

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

    def test_research_recipes_are_applied_to_model_checks_and_kept_in_comparisons(self):
        state = self.state()
        recipe = {'name': 'sum_ab', 'operation': 'sum', 'columns': ['a', 'b']}
        for kind in ('predictive', 'permutation', 'interactions'):
            measured = analyze_features(state, {'name': 'check', 'kind': kind, 'columns': ['a', 'b'],
                'features': [recipe], 'repeats': 1}, task_type='regression')
            self.assertEqual(measured['baseline']['feature_columns'], ['a', 'b', 'sum_ab'])
        comparison = analyze_features(state, {'name': 'addition', 'kind': 'feature', 'columns': ['a', 'b'],
            'baseline_features': [recipe], 'features': [
                {'name': 'delta', 'operation': 'difference', 'columns': ['a', 'b']}]}, task_type='regression')
        self.assertEqual(comparison['baseline']['recipe_names'], ['sum_ab'])
        self.assertEqual(comparison['candidate']['recipe_names'], ['sum_ab', 'delta'])
        self.assertEqual(comparison['baseline']['inner_split_sha256'], comparison['candidate']['inner_split_sha256'])
        removed = analyze_features(state, {'name': 'remove_b', 'kind': 'ablation', 'columns': ['b'],
            'baseline_columns': ['a', 'b'], 'features': [recipe]}, task_type='regression')
        self.assertEqual(removed['candidate']['feature_columns'], ['a', 'sum_ab'])
        with patch('agent.eda._model', side_effect=AssertionError('Should not train')):
            with self.assertRaisesRegex(ValueError, 'every model input'):
                analyze_features(state, {'name': 'empty', 'kind': 'ablation',
                    'columns': ['a', 'b'], 'baseline_columns': ['a', 'b']})

    def test_empty_populations_and_target_summaries_do_not_produce_nan_or_leak_labels(self):
        state = self.state()
        for field in ('filters', 'comparison_filters'):
            measured = analyze_features(state, {'name': 'empty', 'kind': 'shift', 'columns': ['a'],
                field: [{'column': 'group', 'operation': 'eq', 'value': 'absent'}]})
            self.assertEqual(measured['status'], 'unavailable')
            json.dumps(measured, allow_nan=False)
        summary = analyze_features(state, {'name': 'summary', 'kind': 'groups', 'group_by': 'group',
            'columns': ['a', 'target']}, task_type='regression')
        self.assertIn('std', summary['groups'][0]['target_distribution'])
        stability = analyze_features(state, {'name': 'stable', 'kind': 'stability',
            'columns': ['a', 'target'], 'target': 'target'}, task_type='regression')
        self.assertNotIn('target', stability['features'])
        hidden = analyze_features(state, {'name': 'hidden', 'kind': 'shift', 'columns': ['target']})
        self.assertEqual(hidden['status'], 'unavailable')

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

    def test_catboost_seed_alias_is_not_added_twice(self):
        from agent.operations import _estimator
        model = _estimator('catboost.CatBoostClassifier', {'random_seed': 7}, {'num_cpus': 1})
        self.assertEqual(model.get_params()['random_seed'], 7)
        self.assertNotIn('random_state', model.get_params())
        state = self.state()
        state.update(run_operation(state, {'kind': 'preprocess'}))
        state.update(run_operation(state, {'kind': 'fit', 'parameters': {'estimator': 'catboost.CatBoost',
            'parameters': {'iterations': 2, 'verbose': False, 'allow_writing_files': False}}}))
        self.assertEqual(type(state['model']['estimator']).__name__, 'CatBoostRegressor')

    def test_flaml_fixed_replay_keeps_selected_parameters(self):
        from agent.auto_settings import model_settings
        from flaml.automl.state import SearchState
        from flaml.automl.model import LGBMEstimator
        from flaml.automl.task.factory import task_factory
        fixed = {'n_estimators': 433, 'num_leaves': 4, 'learning_rate': 0.066}
        plan = AutoMLPlan(engine='flaml', models=['lgbm'], model_parameters={'lgbm': fixed}, num_trials=1)
        _, settings = model_settings(plan)
        state = SearchState(LGBMEstimator, np.zeros((100, 3)), task_factory('classification'),
                            custom_hp=settings['lgbm'], max_iter=1, budget=10)
        for key, value in fixed.items():
            self.assertEqual(state.init_config[0][key], value)

    def test_repair_cannot_replace_stacking_with_a_blend(self):
        operation = ToolCall(kind='saved_ensemble', parameters={
            'source_nodes': ['node_001', 'node_002'], 'method': 'stacking'})
        required, provided = operation.interface()
        block = {'requires': required, 'provides': provided, 'operation': operation.model_dump()}
        changed = operation.model_dump()
        changed['parameters']['method'] = 'probabilities'
        with self.assertRaisesRegex(ValueError, 'preserve method'):
            ToolCall.model_validate(changed, context={'block': block})
        changed['parameters']['method'] = 'stacking'
        changed['parameters']['folds'] = 2
        ToolCall.model_validate(changed, context={'block': block})

    def test_identical_fit_loads_the_saved_model(self):
        from pathlib import Path
        import tempfile
        from unittest.mock import patch
        state = self.state(True)
        with tempfile.TemporaryDirectory() as temporary:
            state['preparation']['runtime'].update(run_directory=temporary, work_directory=temporary,
                                                   preparation_sha256='a' * 64)
            state.update(run_operation(state, {'kind': 'preprocess'}, {'task_type': 'classification'}))
            call = {'kind': 'fit', 'parameters': {'estimator': 'sklearn.tree.DecisionTreeClassifier'}}
            first = run_operation(state, call, {'task_type': 'classification'})
            with patch('agent.operations._fit', side_effect=AssertionError('Model was retrained')):
                second = run_operation(state, call, {'task_type': 'classification'})
            np.testing.assert_array_equal(first['model']['estimator'].predict(state['test_features']),
                                          second['model']['estimator'].predict(state['test_features']))
            self.assertEqual(len(list(Path(temporary).glob('training_cache/*/result.pkl'))), 1)

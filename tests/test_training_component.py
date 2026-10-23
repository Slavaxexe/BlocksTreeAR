"""Offline integration tests: real workers/models, deterministic upstream-Qwen responses."""
import asyncio
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import runpy
import time
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

import numpy as np
import pandas as pd

from agent.training_component import main, run_training
from agent.training_contracts import (ValidationDecision, TrainingPlan, TrainingCandidate,
                                      TrainingAnalysis, TrainingNextPlan)
from agent.training_coordinator import TrainingCoordinator
from agent.training_roles import compact_role_context
from agent.training_templates import template_context
from agent.training_backend import _model


class Decisions:
    def __init__(self, scenario, fail_analysis=False):
        self.root = SimpleNamespace(log_directory=None)
        self.scenario, self.fail_analysis = scenario, fail_analysis
    async def ask(self, schema, prompt, role):
        if schema is ValidationDecision:
            return ValidationDecision(validation_fraction=.2, inner_validation='cv', folds=3, rationale='check actual three-fold evaluation')
        if schema is TrainingPlan:
            return TrainingPlan(candidates=[
                TrainingCandidate(name='invalid_parameter', model_index=0, parameters={'alpha': -1.}, rationale='intentional validator failure'),
                TrainingCandidate(name='ridge', model_index=0, parameters={'alpha': 1.}, rationale='regularized linear baseline'),
                TrainingCandidate(name='trees', model_index=1, parameters={'n_estimators': 30, 'min_samples_leaf': 2}, rationale='provided tree family')],
                rationale='only upstream families, including a controlled parameter failure')
        if self.fail_analysis:
            raise RuntimeError('simulated Qwen outage after successful training')
        if schema is TrainingAnalysis:
            return TrainingAnalysis(observations=['invalid alpha failed; other recorded models have measured scores'],
                                    recommendation='repair' if role == 'training_analyst_error' else 'stop',
                                    rationale='measured comparison complete')
        if role == 'training_planner_repair':
            return TrainingNextPlan(next_candidates=[TrainingCandidate(name='ridge', model_index=0,
                parameters={'alpha': 1.}, rationale='use nonnegative alpha')], rationale='repair invalid alpha')
        return TrainingNextPlan(stop=True, rationale='finish with winner')


class TrainingComponentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        repo = Path(__file__).resolve().parents[1]
        generator = runpy.run_path(str(repo/'tests/fixtures/training_component/generate.py'))
        with redirect_stdout(io.StringIO()):
            generator['generate'](repo/'artifacts/training_component_inputs', 20261023)

    def fixture(self, scenario):
        return Path(__file__).resolve().parents[1]/'artifacts/training_component_inputs'/scenario/'input.json'

    def run_case(self, scenario, fail_analysis=False):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)/'run'
            result = asyncio.run(run_training(self.fixture(scenario), output, time_budget=90,
                trial_timeout=15, max_trials=4, agents=Decisions(scenario, fail_analysis)))
            self.assertEqual(result['status'], 'completed', result)
            self.assertTrue(any(trial['status'] == 'failed' for trial in result['trials']))
            completed = [trial for trial in result['trials'] if trial['status'] == 'completed']
            self.assertTrue(all(len(trial['inner_fold_scores']) == 3 for trial in completed))
            choose = max if result['direction'] == 'maximize' else min
            self.assertEqual(result['winner']['score'], choose(trial['score'] for trial in completed))
            final = result['final_refit']
            self.assertIsNone(final['score'])
            self.assertEqual(final['full_training_rows'], 600 if scenario == 'group' else 500)
            self.assertEqual(final['candidate_hash'], next(trial['candidate_hash'] for trial in completed
                             if trial['trial_id'] == result['winner']['trial_id']))
            submission = pd.read_csv(output/'submission.csv', dtype={'id': str})
            source = json.loads(self.fixture(scenario).read_text(encoding='utf-8'))
            test = pd.read_csv(self.fixture(scenario).parent/source['files']['test'], dtype={'id': str})
            self.assertListEqual(submission.id.tolist(), test.id.tolist())
            self.assertTrue(np.isfinite(submission.prediction).all())
            with np.load(output/'split.npz') as split:
                train = pd.read_csv(self.fixture(scenario).parent/source['files']['train'])
                if scenario == 'group':
                    column = source['validation_constraints']['group_columns'][0]
                    self.assertFalse(set(train.iloc[split['fit']][column]) & set(train.iloc[split['valid']][column]))
                else:
                    column = source['validation_constraints']['time_column']
                    self.assertLess(train.iloc[split['fit']][column].max(), train.iloc[split['valid']][column].min())
            if fail_analysis:
                self.assertIn('analysis_warning', result)

    def test_group_cv_error_isolation_and_full_refit(self):
        self.run_case('group', fail_analysis=True)

    def test_failed_trial_is_reviewed_before_next_family(self):
        class RepairDecisions(Decisions):
            async def ask(self, schema, prompt, role):
                if role == 'training_analyst_error':
                    assert 'InvalidParameterError' in prompt
                    assert 'traceback' in prompt
                    payload = json.loads(prompt[prompt.index('\n')+1:])
                    assert payload['input_facts']['feature_columns'] == ['x1', 'x2']
                    assert payload['input_facts']['outer_fit_rows'] == 480
                    templates = payload['model_templates']['models']
                    assert [item['family'] for item in templates] == ['ridge', 'extra_trees']
                    assert 'from sklearn.linear_model import Ridge' in templates[0]['python_source']
                    return TrainingAnalysis(observations=['Ridge alpha=-1 violates its nonnegative constraint'],
                                            recommendation='repair', rationale='confirmed invalid parameter')
                if role == 'training_planner_repair':
                    payload = json.loads(prompt[prompt.index('\n')+1:])
                    assert payload['analyst_report']['recommendation'] == 'repair'
                    return TrainingNextPlan(next_candidates=[TrainingCandidate(
                        name='fixed_ridge', model_index=0, parameters={'alpha': .1},
                        rationale='Ridge alpha must be nonnegative')], rationale='repair actual error')
                return await super().ask(schema, prompt, role)
        with tempfile.TemporaryDirectory() as temporary:
            result = asyncio.run(run_training(self.fixture('group'), Path(temporary)/'run',
                time_budget=90, trial_timeout=15, max_trials=3, agents=RepairDecisions('group')))
            self.assertEqual(result['status'], 'completed', result)
            self.assertEqual(result['repairs'][0]['status'], 'retry_queued')
            self.assertEqual([trial['candidate']['name'] for trial in result['trials']],
                             ['invalid_parameter', 'fixed_ridge', 'trees'])
            self.assertEqual(result['final_refit']['full_training_rows'], 600)
            calls = [entry['stage'] for entry in result['role_history']]
            self.assertLess(calls.index('repair_analysis'), calls.index('repair_planning'))

    def test_coordinator_stops_stale_rounds_and_reserves_final_time(self):
        policy = TrainingCoordinator(started=time.monotonic(), time_budget=100,
                                     trial_timeout=20, max_trials=10, patience_rounds=2)
        policy.direction = 'minimize'
        candidate = TrainingCandidate(name='baseline', model_index=0, parameters={}, rationale='baseline')
        self.assertTrue(policy.admit(candidate, 'ridge'))
        self.assertFalse(policy.admit(candidate, 'ridge'))
        policy.observe({'status': 'completed', 'score': 10.}, candidate.model_dump(), 'ridge', 2.)
        policy.finish_round()
        policy.finish_round()
        self.assertIsNone(policy.stopping_reason())
        policy.finish_round()
        self.assertEqual(policy.stopping_reason(), 'no_improvement')
        self.assertEqual(policy.reserve, 25.)
        policy.search_deadline = time.monotonic()+1.
        expensive = TrainingCandidate(name='expensive', model_index=0, parameters={'alpha': 2.},
                                       estimated_seconds=15., rationale='too late to start')
        self.assertFalse(policy.admit(expensive, 'ridge'))
        self.assertEqual(policy.skipped[-1]['reason'], 'insufficient_search_time')

    def test_completed_cli_report_survives_windows_legacy_encoding(self):
        buffer = io.BytesIO()
        stream = io.TextIOWrapper(buffer, encoding='cp1251')
        report = {'status': 'completed', 'explanation': 'RMSE ≈ 1.24 → полный refit'}
        with patch('sys.stdout', stream), patch('agent.training_component.run_training', AsyncMock(return_value=report)):
            code = main(['input.json', '--output-dir', 'not-created'])
        stream.flush()
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(buffer.getvalue().decode('utf-8')), report)

    def test_role_budget_adapts_to_large_upstream_eda(self):
        policy = TrainingCoordinator(started=time.monotonic(), time_budget=600, trial_timeout=120, max_trials=10)
        policy.role_durations = [17., 54., 41.]
        self.assertGreater(policy.planning_budget(), 100.)
        self.assertLessEqual(policy.planning_budget(), 180.)
        upstream = {'eda': {'upstream_data_profile': {'test_columns': [{'name': 'duplicate'}],
                    'train_columns': [{'name': 'x', 'numeric_stats': {'mean': 5., 'n_missing': 3, 'n_zeros': 1}}]},
                    'upstream_preparation_report': {'high_correlations': list(range(100))}}}
        compact = compact_role_context({'upstream_report': upstream})['upstream_report']['eda']
        self.assertNotIn('test_columns', compact['upstream_data_profile'])
        self.assertEqual(compact['upstream_data_profile']['train_columns'][0]['numeric_stats']['n_missing'], 3)
        self.assertEqual(len(compact['upstream_preparation_report']['high_correlations']), 20)
        self.assertEqual(len(upstream['eda']['upstream_preparation_report']['high_correlations']), 100)

    def test_templates_require_both_requirements_and_installed_library(self):
        models = [{'family': 'ridge'}, {'family': 'lightgbm'}, {'family': 'catboost'}]
        catalog = template_context(models, 'scikit-learn==1.5.2\nlightgbm==4.6.0\n',
                                   {'scikit-learn': '1.5.2', 'catboost': '1.2.8'})
        self.assertEqual([item['available'] for item in catalog['models']], [True, False, False])
        self.assertIn('python_source', catalog['models'][0])
        self.assertNotIn('python_source', catalog['models'][1])
        self.assertNotIn('python_source', catalog['models'][2])

    def test_qwen_can_copy_executable_lightgbm_defaults(self):
        model, parameters = _model('lightgbm', {'verbosity': -1, 'n_estimators': 3}, 42)
        x = pd.DataFrame({'x': np.arange(50, dtype=float)})
        model.fit(x, x.x * 2)
        self.assertTrue(np.isfinite(model.predict(x)).all())
        self.assertEqual(parameters['verbosity'], -1)
        with self.assertRaisesRegex(ValueError, 'Unsupported parameters'):
            _model('lightgbm', {'invented_parameter': 1}, 42)

    def test_time_cv_and_full_refit(self):
        # Same selected families are needed for this fault-injection plan.
        original = self.fixture('time')
        data = json.loads(original.read_text(encoding='utf-8'))
        data['selected_models'][1]['family'] = 'extra_trees'
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'input.json'
            for key, value in data['files'].items():
                if value:
                    data['files'][key] = str((original.parent/value).resolve())
            path.write_text(json.dumps(data), encoding='utf-8')
            previous = self.fixture
            self.fixture = lambda scenario: path
            try:
                self.run_case('time')
            finally:
                self.fixture = previous

    def test_hard_worker_timeout_is_bounded(self):
        with tempfile.TemporaryDirectory() as temporary:
            started = time.monotonic()
            result = asyncio.run(run_training(self.fixture('group'), Path(temporary)/'run', time_budget=2,
                trial_timeout=.01, max_trials=3, agents=Decisions('group')))
            self.assertEqual(result['status'], 'failed')
            self.assertLess(time.monotonic()-started, 5)

    def test_r2_winner_uses_maximum_score(self):
        original = self.fixture('group')
        data = json.loads(original.read_text(encoding='utf-8'))
        data['primary_metric'] = 'r2'
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'input.json'
            for key, value in data['files'].items():
                if value:
                    data['files'][key] = str((original.parent/value).resolve())
            path.write_text(json.dumps(data), encoding='utf-8')
            previous = self.fixture
            self.fixture = lambda scenario: path
            try:
                self.run_case('group')
            finally:
                self.fixture = previous


if __name__ == '__main__':
    unittest.main()

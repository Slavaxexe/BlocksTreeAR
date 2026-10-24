"""Feature policies must be fold-local, and a real timeout must be recoverable."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ac_eda import FeaturePolicy, FittedFeaturePolicy, extended_profile, confusion_evidence
from ac_experiment import ExperimentSpec, run_in_subprocess
from ac_run import run


class EdaRecoveryTests(unittest.TestCase):
    def test_default_policy_preserves_inputs(self):
        frame = pd.DataFrame({'x': [1., 2., 3.], 'cat': pd.Categorical(['b', 'a', 'b'])})
        pd.testing.assert_frame_equal(FittedFeaturePolicy(FeaturePolicy()).fit(frame).transform(frame), frame)

    def test_bounds_frequencies_and_categories_use_only_fit(self):
        train = pd.DataFrame({'x': np.arange(10, dtype=float), 'cat': pd.Categorical(['a']*8+['b', 'c'])})
        valid = pd.DataFrame({'x': [1000., -1000.], 'cat': pd.Categorical(['a', 'future'])})
        policy = FeaturePolicy(clip_quantile=.1/2, outlier_flags=True,
            rare_category_min_fraction=.05, frequency_columns=['cat'])
        fitted = FittedFeaturePolicy(policy).fit(train)
        before = dict(fitted.bounds)
        result = fitted.transform(valid)
        self.assertLess(result.x.max(), 10)
        self.assertGreater(result.x.min(), 0)
        self.assertEqual(result['eda_outlier__x'].tolist(), [1., 1.])
        self.assertEqual(result['eda_frequency__cat'].tolist(), [.8, 0.])
        self.assertEqual(fitted.bounds, before)
        self.assertNotIn('future', result.cat.cat.categories)

    def test_removal_interactions_and_raw_missingness(self):
        frame = pd.DataFrame({'x': [1., 2., 3., 4.], 'copy': [1., 2., 3., 4.], 'constant': 1.,
            'denominator': [0., 1., 1., 2.]})
        raw = frame.copy();raw.loc[0, 'x'] = np.nan
        policy = FeaturePolicy(drop_constants=True, drop_duplicates=True, missing_flags=True,
            interactions=[{'columns': ['x', 'denominator'], 'operation': 'ratio'}])
        fitted = FittedFeaturePolicy(policy).fit(frame, raw)
        result = fitted.transform(frame, raw)
        self.assertNotIn('constant', result)
        self.assertNotIn('copy', result)
        self.assertEqual(result['eda_interaction__0'].iloc[0], 0.)
        self.assertEqual(result['eda_missing__x'].tolist(), [1., 0., 0., 0.])
        self.assertEqual(len(result), len(frame))
        with self.assertRaises(ValueError):
            FittedFeaturePolicy(FeaturePolicy(drop_columns=['target'])).fit(frame)

    def test_extended_eda_handles_boolean_nan_inf_and_arbitrary_labels(self):
        frame = pd.DataFrame({'x': [0., 1., np.nan, np.inf, 4., 5.], 'copy': [0., 1., np.nan, np.inf, 4., 5.],
            'flag': [True, False]*3, 'cat': ['a', 'a', '-', '-', 'b', 'b'], 'target': ['yes', 'no']*3})
        profile = extended_profile(frame, frame.drop(columns='target'), ['x', 'copy', 'flag', 'cat'], 'target')
        json.dumps(profile, allow_nan=False)
        self.assertTrue(profile['duplicate_columns'])
        self.assertEqual(next(row for row in profile['numeric'] if row['feature']=='x')['nonfinite'], 1)
        evidence = confusion_evidence(frame.drop(columns='target'), np.array([0, 1]*3), ['yes', 'no'],
            [{'true': 'yes', 'predicted': 'no'}])
        self.assertEqual(evidence['pairs'][0]['fit_support'], [3, 3])

    def test_real_baseline_timeout_then_recovery_and_full_submission(self):
        from test_vitaly_classification import fixture
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory);train, test = fixture(root)
            calls = []
            def injected_timeout(spec, context, output, timeout):
                calls.append(spec)
                if len(calls) == 1:
                    return run_in_subprocess(spec, {**context, 'timeout_scope': 'experiment'}, output, .01)
                return run_in_subprocess(spec, context, output, timeout)
            with patch('ac_research.run_in_subprocess', side_effect=injected_timeout):
                report = run(root/'task.md', root/'result', no_llm=True,
                    total_budget=120, max_experiments=2, n_jobs=2)
            self.assertEqual(report['experiments'][0]['status'], 'timeout')
            recovered = report['experiments'][1]
            self.assertEqual(recovered['source'], 'timeout_recovery')
            self.assertEqual(recovered['status'], 'ok')
            self.assertEqual(recovered['spec']['training']['max_boost_rounds'], 128)
            self.assertEqual(report['final_status'], 'ok')
            self.assertEqual(report['final_refit']['full_training_rows'], len(train))
            self.assertEqual(report['submission_checks']['rows'], len(test))
            self.assertTrue((root/'submission.csv').is_file())

    def test_eda_proposal_protocol_worker_and_full_refit(self):
        from test_vitaly_classification import fixture
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory);train, test = fixture(root)
            for frame in (train, test):
                frame['constant'] = 1.;frame['copy_x'] = frame.x
            train.to_csv(root/'train.csv', index=False);test.to_csv(root/'test.csv', index=False)
            observed = []
            def qwen_stub(messages):
                payload = json.loads(messages[-1]['content'].split('MEASURED MEMORY:\n', 1)[1])
                observed.append(payload)
                reply = {'analysis': {'experiment_id': 'exp_001', 'lesson': 'EDA found constants and duplicate features'},
                    'decision': 'experiment', 'proposal': {'hypothesis': 'Remove redundant inputs and add a row interaction',
                    'why': 'Same validation, fit-local frequency and duplicate detection', 'parent': 'exp_001',
                    'eda_evidence': ['constant has one value', 'copy_x duplicates x in sampled training EDA'],
                    'spec': {'preprocessing': {'drop_constants': True, 'drop_duplicates': True,
                        'frequency_columns': ['category'], 'interactions': [{'columns': ['x', 'z'], 'operation': 'product'}]}}}}
                return json.dumps(reply), {}
            report = run(root/'task.md', root/'result', llm_fn=qwen_stub,
                total_budget=120, max_experiments=2, n_jobs=2)
            self.assertIn('feature_profile', observed[0]['EDA'])
            trial = report['experiments'][1]
            self.assertEqual(trial['source'], 'qwen')
            self.assertEqual(trial['status'], 'ok')
            removed = trial['preprocessing_fits'][0]['removed_columns']
            self.assertIn('constant', removed);self.assertIn('copy_x', removed)
            context = json.loads((root/'result/context.json').read_text(encoding='utf-8'))
            spec = ExperimentSpec.model_validate(trial['spec'])
            final = run_in_subprocess(spec, {**context, 'mode': 'final', 'n_rounds': 30}, root/'eda_final', 30)
            self.assertEqual(final['status'], 'ok')
            self.assertEqual(final['full_training_rows'], len(train))
            self.assertEqual(final['prediction_rows'], len(test))
            self.assertIn('constant', final['preprocessing_fits'][0]['removed_columns'])


if __name__ == '__main__':
    unittest.main()

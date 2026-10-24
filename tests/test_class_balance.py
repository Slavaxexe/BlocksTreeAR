"""Balance-driven actions must be measured, metric-aware and fit-local."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ac_diagnostics import class_profile, dataset_profile, fit_class_weights, prediction_monitor
from ac_experiment import ExperimentSpec
from ac_run import run


class ClassBalanceTests(unittest.TestCase):
    def test_weights_clipped_and_normalized_on_supplied_labels_only(self):
        y = np.r_[np.zeros(90, int), np.ones(9, int), np.full(1, 2)]
        weights = fit_class_weights(y, .5, 3.)
        self.assertAlmostEqual(weights.mean(), 1.)
        self.assertLessEqual(weights.max()/weights.min(), 3.)
        self.assertGreater(weights[-1], weights[0])
        self.assertIsNone(fit_class_weights(y, 0, 3.))
        self.assertAlmostEqual(fit_class_weights(np.zeros(90, int), .5, 20).mean(), 1.)

    def test_monitor_actions_depend_on_metric_and_underprediction(self):
        y = np.r_[np.zeros(90, int), np.ones(10, int)]
        p = np.tile([.99, .01], (100, 1))
        profile = class_profile(y, ['common', 'rare'])
        result = prediction_monitor(y, p, ['common', 'rare'], 'f1_macro', profile)
        self.assertTrue(result['soft_weights_worth_testing'])
        self.assertEqual(result['per_class'][1]['predicted_count'], 0)
        self.assertTrue(result['per_class'][1]['zero_recall'])
        self.assertEqual(result['top_confusions'][0]['true'], 'rare')
        self.assertFalse(prediction_monitor(y, p, ['common', 'rare'], 'log_loss', profile)['soft_weights_worth_testing'])
        self.assertFalse(prediction_monitor(y, np.eye(2)[y], ['common', 'rare'], 'f1_macro', profile)['soft_weights_worth_testing'])

    def test_no_double_class_weighting(self):
        with self.assertRaises(ValueError):
            ExperimentSpec.model_validate({'training': {'class_weight_power': .5},
                'model': {'model_id': 'lightgbm', 'params': {'class_weight': 'balanced'}}})
        with self.assertRaises(ValueError):
            ExperimentSpec.model_validate({'training': {'class_weight_power': 1.1}})

    def test_eda_detects_ambiguity_without_dropping_rows(self):
        train = pd.DataFrame({'id': range(4), 'x': [1, 1, 2, 3],
            'service': ['-', '-', 'a', 'b'], 'target': ['first', 'second', 'first', 'first']})
        test = train.drop(columns='target').iloc[:2]
        profile = dataset_profile(train, test, 'target', ['id'])
        self.assertEqual(profile['duplicates']['rows_in_conflicting_groups'], 2)
        self.assertEqual(profile['placeholder_candidates'][0]['rows'], 2)
        self.assertEqual(len(train), 4)

    def test_native_monitor_schedules_one_soft_trial_and_saves_final_balance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            n, m = 320, 30
            train = pd.DataFrame({'id': np.arange(n), 'x': np.zeros(n),
                'target': ['common']*300+['rare']*20})
            test = pd.DataFrame({'id': np.arange(n, n+m), 'x': np.zeros(m)})
            train.to_csv(root/'train.csv', index=False);test.to_csv(root/'test.csv', index=False)
            pd.DataFrame({'id': test.id, 'target': 'common'}).to_csv(root/'sample_submission.csv', index=False)
            (root/'task.md').write_text('Classification. Primary metric: Macro F1. Output class labels in submission.csv.', encoding='utf-8')
            report = run(root/'task.md', root/'run', no_llm=True, total_budget=120, max_experiments=2, n_jobs=2)
            self.assertEqual(report['final_status'], 'ok')
            self.assertEqual(report['experiments'][1]['source'], 'balance_monitor')
            self.assertEqual(report['experiments'][1]['spec']['training']['class_weight_power'], .5)
            self.assertTrue((root/'run/experiments/exp_002/balance.json').is_file())
            self.assertTrue((root/'run/balance_final.json').is_file())
            self.assertEqual(sum(report['submission_checks']['predicted_test_class_counts'].values()), m)
            self.assertEqual(report['final_refit']['full_training_rows'], n)
            from ar_memory import Memory
            from ar_config import Settings
            from ac_research import Lab
            from ac_run import Budget
            from ac_data import Features
            memory = Memory.__new__(Memory)
            memory.state = json.loads((root/'run/state.json').read_text(encoding='utf-8'))
            context = json.loads((root/'run/context.json').read_text(encoding='utf-8'))
            lab = Lab(memory, Settings(), Budget(120), None, context, root/'run')
            payload = json.loads(lab.research_memory())
            self.assertTrue(payload['BUDGET']['final_reserve_already_excluded'])
            self.assertIn('top_confusions', payload['CURRENT_DIAGNOSTICS'])
            self.assertIn('per_class', payload['BEST_DIAGNOSTICS'])
            self.assertIsNone(lab.balance_proposal())


if __name__ == '__main__':
    unittest.main()

"""Required EDA coverage and data-matched history cannot be replaced by prose."""
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import MagicMock
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ac_strategy import context_signature, previous_winner, required_trials, protocol_status
from ac_custom import apply_code
from ac_run import run


class ClassificationStrategyTests(unittest.TestCase):
    def test_nonbinding_iteration_ceiling_is_rejected(self):
        from ac_experiment import ExperimentSpec
        from ac_research import Lab
        from ac_run import Budget
        from ar_memory import Memory
        from ar_config import Settings
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory);memory = Memory(root)
            original = ExperimentSpec()
            memory.add_experiment({'id': 'exp_001', 'status': 'ok', 'cv': .6, 'seconds': 2.,
                'spec': original.model_dump(), 'spec_hash': original.key(), 'best_iters': [30, 40]})
            features = MagicMock();features.available.return_value = {'base': ''}
            lab = Lab(memory, Settings(), Budget(300), features, {'classes': ['a', 'b'], 'time_column': None}, root)
            self.assertTrue(any('non-binding' in error for error in lab.validate(
                ExperimentSpec.model_validate({'training': {'max_boost_rounds': 1000}}))))
            self.assertFalse(lab.validate(ExperimentSpec.model_validate({'training': {'max_boost_rounds': 20}})))

    def test_required_actions_derived_from_eda(self):
        profile = {'categorical': [{'feature': 'category_a', 'unique': 4}],
            'constants': ['constant_a'], 'numeric': [{'feature': 'amount', 'skew': 8.}]}
        trials = required_trials(profile, {}, {'base': '', 'oof_target_encoding': ''})
        self.assertEqual([row['id'] for row in trials], ['categorical_encoding', 'feature_quality', 'tail_or_missingness', 'model_family'])
        self.assertTrue(all(row['status'] == 'pending' for row in protocol_status([], trials)))
        record = {'id': 'exp_001', 'status': 'ok', 'cv': .7, 'spec': {'model': {'model_id': 'lightgbm'},
            'training': {}, 'blocks': ['base', 'oof_target_encoding'], 'preprocessing': {}}}
        self.assertEqual(protocol_status([record], trials)[0]['status'], 'completed')

    def test_history_requires_identical_data_labels_and_folds(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory);past = root/'artifacts/classification_20260101';past.mkdir(parents=True)
            current = root/'artifacts/classification_20260102';current.mkdir()
            pd.DataFrame({'x': [0., 1., 2.]}).to_parquet(past/'features.parquet')
            np.save(past/'labels.npy', [0, 1, 0]);np.savez(past/'folds.npz', train_0=[0, 1], valid_0=[2])
            context = {'features_path': str(past/'features.parquet'), 'labels_path': str(past/'labels.npy'),
                'folds_path': str(past/'folds.npz'), 'target': 'target', 'classes': ['a', 'b'], 'metric': 'f1_macro', 'positive_index': 1}
            (past/'context.json').write_text(json.dumps(context))
            report = {'best_single': {'id': 'exp_001'}, 'experiments': [{'id': 'exp_001', 'status': 'ok', 'cv': .8, 'spec': {}}]}
            (past/'report.json').write_text(json.dumps(report))
            other = dict(context)
            for key, name in [('features_path', 'features.parquet'), ('labels_path', 'labels.npy'), ('folds_path', 'folds.npz')]:
                shutil.copyfile(past/name, current/name);other[key] = str(current/name)
            self.assertEqual(context_signature(context), context_signature(other))
            self.assertEqual(previous_winner(root, current, other)[1]['reference_cv'], .8)
            np.save(current/'labels.npy', [1, 1, 0])
            self.assertIsNone(previous_winner(root, current, other)[1])

    def test_generated_astype_float_executes_in_isolated_process(self):
        frame = pd.DataFrame({'x': [0., 1., 2.]})
        code = "def add_features(df):\n    out = df.copy()\n    out['positive'] = (out['x'] > 0).astype(float)\n    return out\n"
        self.assertEqual(apply_code(code, frame).positive.tolist(), [0., 1., 1.])

    def test_native_required_coverage_precedes_free_search(self):
        from test_vitaly_classification import fixture
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory);train, test = fixture(root)
            rng = np.random.default_rng(55)
            for frame in (train, test):
                frame['constant'] = 1.;frame['copy_x'] = frame.x
                frame['long_tail'] = rng.lognormal(0, 4, len(frame))
            train.to_csv(root/'train.csv', index=False);test.to_csv(root/'test.csv', index=False)
            report = run(root/'task.md', root/'result', no_llm=True,
                total_budget=300, max_experiments=5, n_jobs=2)
            self.assertEqual(report['final_status'], 'ok')
            protocol = report['facts']['search.protocol']
            self.assertTrue(protocol['enabled'])
            self.assertTrue(all(row['status'] == 'completed' for row in protocol['trials']), protocol)
            self.assertTrue(any(row['spec']['model']['model_id'] == 'catboost' for row in report['experiments']))
            self.assertTrue(any('oof_target_encoding' in row['spec']['blocks'] for row in report['experiments']))
            self.assertTrue(all(row['status'] == 'ok' for row in report['experiments']))
            self.assertEqual(report['submission_checks']['rows'], len(test))


if __name__ == '__main__':
    unittest.main()

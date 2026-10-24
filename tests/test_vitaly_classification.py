"""Native classifier branch checks, chronological leakage guards, unchanged regression."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

VITALY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(VITALY))
from ac_task import detect_task_type, primary_metric, read_task
from ac_data import Features, detect_time, design_folds, discover_entity, period_values
from ac_models import score, fit_predict
from ac_run import run
from ac_custom import check_code


def fixture(root, temporal=False, probability=False, multiclass_probability=False):
    rng = np.random.default_rng(417)
    n, m = 360, 60
    if temporal:
        n, m = 300, 60
        train = pd.DataFrame({'id': [f'{i:05d}' for i in range(n)], 'asset': [f'a{i%60}' for i in range(n)],
            'period': np.repeat(np.arange(1, 6), 60), 'x': rng.normal(size=n), 'category': [f'g{i%4}' for i in range(n)]})
        labels = np.tile(np.arange(60)%3, 5)
        test = pd.DataFrame({'id': [f't{i:04d}' for i in range(m)], 'asset': [f'a{i}' for i in range(m)],
            'period': np.repeat(6, m), 'x': rng.normal(size=m), 'category': [f'g{i%4}' for i in range(m)]})
    else:
        train = pd.DataFrame({'id': [f'{i:05d}' for i in range(n)], 'x': rng.normal(size=n),
                              'z': rng.normal(size=n), 'category': [f'g{i%4}' for i in range(n)]})
        test = pd.DataFrame({'id': [f't{i:04d}' for i in range(m)], 'x': rng.normal(size=m),
                             'z': rng.normal(size=m), 'category': [f'g{i%4}' for i in range(m)]})
        labels = np.digitize(train.x, [-.4, .4])
    train['target'] = np.asarray(['low', 'mid', 'high'])[labels]
    metric = 'Macro F1'
    if probability:
        train['target'] = np.where(labels == 0, 2, 9)
        sample = pd.DataFrame({'id': test.id, 'risk': .5})
        metric = 'ROC-AUC'
    elif multiclass_probability:
        sample = pd.DataFrame({'id': test.id, 'low': 0., 'mid': 0., 'high': 0.})
        metric = 'Log Loss'
    else:
        sample = pd.DataFrame({'id': test.id, 'target': 'low'})
    train.to_csv(root/'train.csv', index=False);test.to_csv(root/'test.csv', index=False)
    sample.to_csv(root/'sample_submission.csv', index=False)
    text = ('# Classification task\nPredict categorical target `target`.\n'
            'Files: train.csv, test.csv, sample_submission.csv.\n'
            f'## Primary metric\n{metric}\n'
            + ('Output probabilities in submission.csv. Positive class: 2\n' if probability else
               'Output probabilities in submission.csv.\n' if multiclass_probability else
               'Output class labels in submission.csv.\n'))
    (root/'task.md').write_text(text, encoding='utf-8')
    return train, test


class VitalyClassificationTests(unittest.TestCase):
    def test_routing_metric_and_numeric_regression(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory);fixture(root, probability=True)
            self.assertEqual(detect_task_type(root/'task.md'), 'classification')
            task = read_task(root/'task.md')
            self.assertEqual(task.primary_metric if hasattr(task, 'primary_metric') else task.metric, 'roc_auc')
            self.assertEqual(task.positive_label, '2')
            self.assertEqual(task.prediction_columns, ['risk'])
            (root/'task.md').write_text('Regression: predict numeric target. Primary metric RMSE.', encoding='utf-8')
            self.assertEqual(detect_task_type(root/'task.md'), 'regression')
        self.assertEqual(primary_metric('Primary metric: Accuracy. Also report Macro F1.')[0], 'accuracy')

    def test_temporal_features_never_use_future_or_validation_labels(self):
        frame = pd.DataFrame({'asset': ['a', 'b']*4, 'period': np.repeat([1, 2, 3, 4], 2),
                              'category': ['g']*8, 'x': np.arange(8)})
        source = np.arange(6)
        generator = Features(frame, 'target', [], 2, 'period', ['asset'], ['category']).fit(
            source, np.array([0, 1, 0, 1, 1, 0]), ['base', 'entity_history', 'group_history'])
        before = generator.transform([2, 3])
        changed = Features(frame, 'target', [], 2, 'period', ['asset'], ['category']).fit(
            source, np.array([0, 1, 1, 0, 0, 1]), ['base', 'entity_history', 'group_history']).transform([2, 3])
        # Changing all labels at/after period2 must not change period2 histories.
        columns = [name for name in before if name.startswith(('entity_', 'group'))]
        pd.testing.assert_frame_equal(before[columns], changed[columns])
        self.assertEqual(generator.transform([6, 7]).entity_count.tolist(), [3., 3.])
        self.assertTrue(np.isfinite(generator.transform([6, 7]).entity_p0).all())

    def test_no_time_encoding_is_cross_fitted_inside_source_only(self):
        frame = pd.DataFrame({'category': ['a']*60, 'x': np.arange(60)})
        source = np.arange(40)
        y = np.tile([0, 1], 20)
        generator = Features(frame, 'target', [], 2, group_columns=['category']).fit(source, y, ['oof_target_encoding'])
        actual = generator.transform([40, 41])
        self.assertTrue(np.allclose(actual.group0_p0, .5))
        for row, subset in generator.crossfit.items():self.assertNotIn(row, source[subset])

    def test_datetime_and_group_validation_boundaries(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory);train, test = fixture(root, temporal=True)
            for frame in [train, test]:frame['date'] = pd.to_datetime('2025-01-01')+pd.to_timedelta(frame.pop('period')*30, unit='D')
            info = detect_time(train, test, 'target', ['id'])
            self.assertEqual(info['col'], 'date')
            y = np.tile(np.arange(60)%3, 5)
            entity = discover_entity(train, test, ['asset', 'x', 'category'], 'date')
            self.assertIsNotNone(entity)
            folds = design_folds(train, y, 'date', entity['key'])
            periods = period_values(train.date).to_numpy()
            for fold in folds['folds']:
                self.assertLess(periods[fold['train']].max(), periods[fold['valid']].min())
                self.assertLess(periods[fold['fit']].max(), periods[fold['early_stop']].min())
            folds = design_folds(train, y, entity_key=['asset'])
            for fold in folds['folds']:
                self.assertFalse(set(train.asset.iloc[fold['train']]) & set(train.asset.iloc[fold['valid']]))

    def test_class_probability_metrics_and_ast(self):
        y = np.array([0, 1, 2, 0, 1, 2])
        p = np.eye(3)[y]*.9+.1/3
        self.assertEqual(score('f1_macro', y, p), 1.)
        self.assertLess(score('log_loss', y, p), .1)
        self.assertEqual(score('roc_auc', y, p), 1.)
        self.assertTrue(check_code('def add_features(df):\n    return df.mean()'))
        self.assertTrue(check_code('def add_features(df):\n    import os\n    return df'))

    def test_native_families_align_classes_missing_from_a_fold(self):
        x = pd.DataFrame({'x': np.linspace(-2, 2, 80), 'cat': pd.Categorical(['a', 'b']*40)})
        y = np.where(x.x > 0, 2, 0)
        for family in ['lightgbm', 'catboost', 'xgboost', 'extra_trees', 'random_forest', 'logistic_regression', 'sgd']:
            with self.subTest(family=family):
                parameters = {'n_estimators': 40} if family in {'extra_trees', 'random_forest'} else {}
                probabilities, iteration, _ = fit_predict(family, 'cross_entropy', parameters, x.iloc[:60], y[:60],
                    x.iloc[60:], y[60:], [x], 3, n_jobs=2, n_rounds=30)
                self.assertEqual(probabilities[0].shape, (80, 3))
                self.assertTrue(np.allclose(probabilities[0].sum(axis=1), 1))
                self.assertTrue((probabilities[0][:, 1] < 1e-8).all())
                self.assertTrue(np.isfinite(probabilities[0]).all())

    def run_native(self, temporal=False, probability=False, multiclass_probability=False):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory);train, test = fixture(root, temporal, probability, multiclass_probability)
            report = run(root/'task.md', root/'result', no_llm=True, total_budget=120,
                         max_experiments=3, n_jobs=2)
            self.assertEqual(report['final_status'], 'ok', report)
            self.assertTrue(report['submission_checks']['ok'])
            self.assertTrue(all(value['status'] == 'ok' for value in report['experiments']))
            self.assertEqual(report['final_refit']['full_training_rows'], len(train))
            result = pd.read_csv(root/'submission.csv', dtype={'id': str})
            self.assertEqual(result.id.tolist(), test.id.tolist())
            self.assertEqual(len(result), len(test))
            if probability:
                self.assertTrue(result.risk.between(0, 1).all())
            elif multiclass_probability:
                self.assertTrue(np.allclose(result[['low', 'mid', 'high']].sum(axis=1), 1))
            else:
                self.assertTrue(set(result.target).issubset(set(train.target)))
            if temporal:
                self.assertEqual(report['validation']['kind'], 'time_rolling_origin')
                self.assertTrue(any('entity_history' in value['spec']['blocks'] for value in report['experiments']))

    def test_native_temporal_classification_research_full_refit(self):self.run_native(temporal=True)
    def test_native_binary_probability_full_refit(self):self.run_native(probability=True)
    def test_native_multiclass_probability_full_refit(self):self.run_native(multiclass_probability=True)

    def test_regression_entry_still_calls_original_runner(self):
        spec = importlib.util.spec_from_file_location('vitaly_entry', VITALY/'solution.py')
        module = importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'task.md'
            path.write_text('Regression target price. Primary metric RMSE.', encoding='utf-8')
            with patch('ar_run.run') as original:
                self.assertEqual(module.main([str(path), '--budget', '120', '--no-llm']), 0)
                original.assert_called_once_with(str(path), no_llm=True, total_budget=120., llm_calls=None, write_next_to_task=True)


if __name__ == '__main__':unittest.main()

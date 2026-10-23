"""Real external recipe + our workers; no API, tiny datasets and bounded models."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd

from agent.margo_recipe import MargoRecipeTransformer, register_margo
from agent.training_backend import run_job


SOURCE = Path(__file__).resolve().parents[2]/'AlfaHakaton-margo'


@unittest.skipUnless((SOURCE/'data_preparation/transformer.py').is_file(), 'Requires the external margo checkout')
class MargoRecipeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        x = np.arange(50, dtype=float)
        x[::7] = np.nan
        self.train = pd.DataFrame({'id': [f'{i:05d}' for i in range(50)], 'value': x,
                                  'kind': ['a', 'b']*25, 'y': np.arange(50)*2.+1})
        self.train.to_csv(self.root/'train.csv', index=False)
        pd.DataFrame({'id': [f'{i:05d}' for i in range(50, 60)], 'value': np.arange(10),
                      'kind': ['a', 'new']*5}).to_csv(self.root/'test.csv', index=False)
        recipe = {'columns_to_drop': ['id'], 'column_transforms': [
            {'column': 'value', 'operations': ['impute_median', 'passthrough']},
            {'column': 'kind', 'operations': ['one_hot_encode']}], 'new_features': []}
        self._write('recipe.json', recipe)
        self.preparation = {'provider': 'margo', 'source_root': str(SOURCE),
            'recipe': str(self.root/'recipe.json'), 'split': str(self.root/'split.json'), 'feature_names': [],
            'source_hashes': {str(p.relative_to(SOURCE)).replace('\\', '/'): hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in (SOURCE/'data_preparation').glob('*.py')}}
        self._write('split.json', {'fit': list(range(40)), 'valid': list(range(40, 50))})
        np.savez(self.root/'split.npz', fit=np.arange(40), valid=np.arange(40, 50))

    def _write(self, name, value):
        (self.root/name).write_text(json.dumps(value), encoding='utf-8')

    def execute(self, family, parameters, inner='holdout'):
        self._write('input.json', {'task_type': 'regression', 'files': {'train': str(self.root/'train.csv'),
            'test': str(self.root/'test.csv')}, 'target': 'y', 'id_columns': ['id'], 'primary_metric': 'rmse',
            'selected_models': [{'family': family, 'parameters': {}}], 'validation_constraints': {'strategy': 'random'},
            'submission': {'filename': 'submission.csv', 'prediction_column': 'prediction'}, 'preparation': self.preparation})
        candidate = {'name': family, 'model_index': 0, 'parameters': parameters, 'rationale': 'bounded integration'}
        job = {'input_path': str(self.root/'input.json'), 'split_path': str(self.root/'split.npz'),
               'candidate': candidate, 'validation': {'inner_validation': inner, 'folds': 3, 'seed': 42},
               'phase': 'evaluate', 'output_dir': str(self.root/f'{family}-evaluate')}
        fit_rows = []
        original_fit = MargoRecipeTransformer.fit
        def recorded_fit(instance, x, y=None):
            fit_rows.append(len(x))
            return original_fit(instance, x, y)
        with patch.object(MargoRecipeTransformer, 'fit', recorded_fit):
            evaluated = run_job(job, 45)
            final = run_job({**job, 'phase': 'final', 'output_dir': str(self.root/f'{family}-final'),
                             'source_result_dir': job['output_dir']}, 45)
        self.assertEqual(evaluated['preparation_fit_rows'], 40)
        self.assertEqual(final['preparation_fit_rows'], 50)
        self.assertEqual(final['full_training_rows'], 50)
        self.assertIsNone(final['score'])
        submission = pd.read_csv(Path(final['submission_path']), dtype={'id': str})
        self.assertEqual(submission.id.tolist(), [f'{i:05d}' for i in range(50, 60)])
        self.assertTrue(np.isfinite(submission.prediction).all())
        return evaluated, final, fit_rows

    def test_fold_local_recipe_and_all_row_final_refit(self):
        evaluated, final, fit_rows = self.execute('ridge', {'alpha': 1.}, 'cv')
        self.assertEqual(len(evaluated['inner_fold_scores']), 3)
        self.assertTrue(all(size < 40 for size in fit_rows[:3]))
        self.assertEqual(fit_rows[-2:], [40, 50])
        model = joblib.load(final['model_path'])
        self.assertAlmostEqual(model.named_steps['upstream'].transformer_._params['value__impute_median'], self.train.value.median())

    def test_automl_search_recipe_uses_inner_fit_only(self):
        for family, parameters in [('flaml', {'estimator_list': ['rf'], 'max_iter': 2}),
                                    ('autogluon', {'hyperparameters': {'RF': {'n_estimators': 3}}})]:
            with self.subTest(family=family):
                _, _, fit_rows = self.execute(family, parameters)
                self.assertEqual(fit_rows, [32, 40, 50])


if __name__ == '__main__':
    unittest.main()

"""Regression coverage for ID preservation and prediction semantics in both blend paths."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
from alfa_agent.worker import combine, blend_prediction_type

class BlendContractTests(unittest.TestCase):
    def test_composite_ids_preserved_and_predictions_averaged(self):
        a = pd.DataFrame({'uuid': ['001', '002'], 'number': [90, 91], 'y': [1., 5.], 'z': [4., 6.]})
        b = a.assign(y=[3., 9.], z=[8., 10.])
        before = a.copy(deep=True)
        out = combine([a, b], [1, 3], False, id_columns=['uuid', 'number'], reference=a,
                      prediction_type='numeric')
        pd.testing.assert_frame_equal(out[['uuid','number']], a[['uuid','number']])
        self.assertEqual(out.y.tolist(), [2.5, 8.])
        self.assertEqual(out.z.tolist(), [7., 9.])
        pd.testing.assert_frame_equal(a, before)

    def test_probability_endpoints_are_averaged_not_voted(self):
        a = pd.DataFrame({'no': [1,0], 'yes': [0,1]})
        b = pd.DataFrame({'no': [0,1], 'yes': [1,0]})
        out = combine([a,b], [1,3], True, prediction_type='numeric')
        np.testing.assert_allclose(out, [[.25,.75],[.75,.25]])
        np.testing.assert_allclose(out.sum(axis=1), 1.)

    def test_float_string_boolean_and_nullable_labels_vote_per_row(self):
        for values in ([.5, 1.5], ['cat','dog'], [True,False], pd.array([2,3], dtype='Int64')):
            a = pd.DataFrame({'label': values})
            b = a.iloc[::-1].reset_index(drop=True)
            out = combine([a,b,b], [2,2,1], True, prediction_type='labels')
            self.assertEqual(out.label.tolist(), b.label.tolist())
            tie = combine([a,b], [1,1], True, prediction_type='labels')
            self.assertEqual(tie.label.tolist(), a.label.tolist())

    def test_ids_can_be_absent_from_predictions_and_index_can_be_nondefault(self):
        a = pd.DataFrame({'y': pd.array([1,2], dtype='Int64')}, index=[8,9])
        out = combine([a,a], [1,1], False, id_columns=['id'],
                      reference=pd.DataFrame({'id': ['x','y']}), prediction_type='numeric')
        self.assertEqual(out.index.tolist(), [8,9])
        self.assertEqual(out.y.tolist(), [1.,2.])

    def test_rejects_mismatched_ids_even_if_all_models_have_same_wrong_order(self):
        ref = pd.DataFrame({'id': ['x','y'], 'n': [1,2], 'y': [1.,2.]})
        wrong = ref.iloc[::-1].reset_index(drop=True)
        for frames in ([ref,wrong], [wrong,wrong]):
            with self.assertRaisesRegex(ValueError, 'identifier/order'):
                combine(frames, [1,1], False, id_columns=['id','n'], reference=ref)

    def test_invalid_members_fail_closed(self):
        a = pd.DataFrame({'y': [1.,2.]})
        cases = [([a,a.iloc[:1]], [1,1]), ([a,a.rename(columns={'y':'z'})], [1,1]),
                 ([a,a.set_axis([1,0])], [1,1]), ([a,a], [1]), ([a,a], [1,0]),
                 ([a,a], [1,-1]), ([a,a], [1,float('inf')]),
                 ([a,a.assign(y=[np.nan,2])], [1,1]), ([a,a.assign(y=[np.inf,2])], [1,1]),
                 ([pd.concat([a,a],axis=1)], [1]), ([], [])]
        for frames, weights in cases:
            with self.subTest(weights=weights), self.assertRaises(ValueError):
                combine(frames, weights, False, prediction_type='numeric')

    def test_high_cardinality_labels_and_ids_do_not_need_class_by_row_matrix(self):
        n = 20000
        a = pd.DataFrame({'id': [f'id-{i}' for i in range(n)], 'y': [f'class-{i}' for i in range(n)]})
        b = a.assign(y=a.y.iloc[::-1].to_numpy())
        out = combine([a,b], [1,2], True, id_columns=['id'], reference=a, prediction_type='labels')
        self.assertEqual(out.y.tolist(), b.y.tolist())
        pd.testing.assert_series_equal(out.id, a.id)

    def test_numeric_mode_preserves_only_identical_text_metadata(self):
        a = pd.DataFrame({'y':[1.], 'comment':['constant']})
        pd.testing.assert_frame_equal(combine([a,a],[1,1],False,prediction_type='numeric'), a)
        with self.assertRaises(ValueError):
            combine([a,a.assign(comment='other')],[1,1],False,prediction_type='numeric')

    def test_validation_selection_and_final_writer_share_id_safe_blend(self):
        import pickle
        from tests.test_pipeline import fixture
        from alfa_agent.contracts import Preparation
        from alfa_agent.data import file_hash
        from alfa_agent.storage import save
        from alfa_agent.worker import prepare, blend, blend_final
        with tempfile.TemporaryDirectory() as temporary:
            task = Path(temporary).resolve()
            _, answers = fixture(task)
            root = task/'run'; root.mkdir()
            draft = dict(answers['PreparationDraft'])
            (root/'prepare.py').write_text(draft.pop('code'))
            (root/'extraction.json').write_text(json.dumps({'output':answers['OutputFacts']}))
            save(root/'preparation.json', Preparation(**draft, task_directory=str(task), task_type='regression',
                                                      code_sha256=file_hash(root/'prepare.py')))
            prepare(root)
            train = pd.read_csv(task/'samples.psv', sep='|', dtype={'record':str})
            reference = train.iloc[-8:].reset_index(drop=True)
            test = pd.read_json(task/'query.jsonl', lines=True, dtype={'record':str})
            nodes = []
            for i, offset in enumerate([-2.,2.]):
                node_id = f'node_{i:03d}'
                node = root/'nodes'/node_id; node.mkdir(parents=True)
                valid = reference[['record','yield_value']].copy()
                valid['yield_value'] += offset
                predicted = test[['record']].assign(yield_value=10.+offset)
                (node/'predictions.pkl').write_bytes(pickle.dumps({'validation':valid,'test':predicted}))
                nodes.append({'node_id':node_id, 'result':{'status':'completed','value':9.}})
            (root/'delta.json').write_text(json.dumps({'nodes':nodes}))
            blend(root, 2)
            plan = json.loads((root/'blend.json').read_text())
            self.assertTrue(plan['selected'])
            self.assertEqual(plan['value'],7.)
            blend_final(root)
            out = pd.read_csv(root/'final/result.tsv', sep='\t', dtype={'record':str})
            self.assertEqual(out.record.tolist(),test.record.tolist())
            self.assertEqual(out.estimate.tolist(),[10.]*len(test))
            # Refitted member predictions go through the same combination and writer.
            for i in range(2):
                node = root/'nodes'/f'node_{i:03d}'
                predicted = test[['record']].assign(yield_value=20.+[-2.,2.][i])
                (node/'refit_predictions.pkl').write_bytes(pickle.dumps(predicted))
            blend_final(root)
            out = pd.read_csv(root/'final/result.tsv', sep='\t', dtype={'record':str})
            self.assertEqual(out.estimate.tolist(),[20.]*len(test))

    def test_semantics_from_contract_not_values(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prep = SimpleNamespace(prediction_scope='row', extraction='extraction.json',
                                   task_type='classification', metric=SimpleNamespace(name='custom'))
            for text, metric, expected in [('class probabilities','accuracy','numeric'), ('class labels','ROC AUC','labels'),
                                            ('','ROC AUC','numeric'), ('','macro F1','labels')]:
                (root/'extraction.json').write_text(json.dumps({'output': {'prediction_type':text}}))
                prep.metric.name = metric
                self.assertEqual(blend_prediction_type(root, prep), expected)
            prep.metric.name = 'unknown custom classification metric'
            with self.assertRaisesRegex(ValueError, 'ambiguous'):
                blend_prediction_type(root, prep)
            prep.task_type = 'regression'
            self.assertEqual(blend_prediction_type(root,prep), 'numeric')
            prep.prediction_scope = 'batch'
            with self.assertRaisesRegex(ValueError, 'row-wise'):
                blend_prediction_type(root,prep)

if __name__ == '__main__':
    unittest.main()

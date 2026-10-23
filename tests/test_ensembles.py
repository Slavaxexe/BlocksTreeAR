"""Saved blends and OOF combiners: real tiny models, no model API calls."""
import asyncio
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from agent.assembly import assemble, execute, initialize
from agent.contracts import Assembly, Block, Hypothesis, Idea, Pipeline, Start
from agent.data import file_hash
from agent.operations import run_operation
from agent.predictions import load_pair, save_pair
from agent.storage import atomic_bytes, save
from agent.ensemble_data import subset, concatenate


class EnsembleTests(unittest.TestCase):
    def setup_run(self, root, classification):
        rng = np.random.default_rng(42)
        fit = pd.DataFrame({'x': rng.normal(size=100), 'z': rng.normal(size=100)})
        fit['y'] = np.where(fit.x > 0, 'yes', 'no') if classification else 3 * fit.x + fit.z
        meta = fit.iloc[80:].copy().reset_index(drop=True)
        fit = fit.iloc[:80].copy()
        query = fit.drop(columns='y').iloc[:12].copy()
        combined = pd.concat([query, meta.drop(columns='y')], ignore_index=True)
        preparation = {'data': {'target_columns': ['y'], 'id_columns': []},
            'metric': {'name': 'Macro F1' if classification else 'RMSE', 'direction': 'maximize' if classification else 'minimize'},
            'output': {'columns': ['y']}, 'validation': {'strategy': 'stratified_random' if classification else 'random'}}
        save(root / 'preparation.json', preparation)
        sha = file_hash(root / 'preparation.json')
        defaults = {'task_type': 'classification' if classification else 'regression'}
        for number in (0, 1):
            node = root / 'nodes' / f'node_{number:03d}'
            node.mkdir(parents=True)
            operations = [('preprocess', {'kind': 'preprocess'}), ('fit', {'kind': 'fit', 'parameters': {
                'estimator': 'sklearn.ensemble.ExtraTreesClassifier' if classification else 'sklearn.ensemble.ExtraTreesRegressor',
                'parameters': {'n_estimators': 3 + number, 'max_depth': 2 + number}}}), ('predict', {'kind': 'predict'})]
            from agent.plans import ToolCall
            blocks = []
            for name, value in operations:
                call = ToolCall.model_validate(value)
                required, provided = call.interface()
                blocks.append(Block(name=name, description=name, requires=required, provides=provided, operation=call))
            start = Start(node_id=node.name, purpose='baseline', preparation_sha256=sha,
                hypothesis=Hypothesis(rationale='Tiny model', expected_effect='Measure', pipeline=Pipeline(data={}, preprocessing={}, model={}, training={})),
                assembly=Assembly(blocks=blocks, write_blocks=[b.name for b in blocks]), outputs={'validation': 'validation_predictions.csv', 'submission': 'submission.csv'}, timeout_seconds=60.)
            save(node / 'start.json', start)
            initialize(node, start)
            for block in blocks:
                code = f'from agent.operations import run_operation\ndef run(state):\n    return run_operation(state, {block.operation.model_dump()!r}, {defaults!r})\n'
                atomic_bytes(node / 'modules' / (block.name + '.py'), code.encode())
            code_sha = assemble(node, start)
            metadata = {**preparation, 'runtime': {'deadline': time.monotonic() + 60, 'num_cpus': 1,
                'run_directory': str(root), 'work_directory': str(node), 'preparation_sha256': sha}}
            valid, test = execute(node, fit, combined, query, metadata)
            save_pair(node, {'validation': subset(valid, np.arange(12)), 'meta': subset(valid, np.arange(12, 32)), 'test': test}, sha, code_sha, 'validation_predictions.csv')
            save(node / 'finish.json', {'status': 'completed', 'preparation_sha256': sha, 'code_sha256': code_sha})
        work = root / 'nodes' / 'node_002'
        work.mkdir()
        metadata['runtime'].update(work_directory=str(work), deadline=time.monotonic() + 60)
        metadata['runtime'].update(meta_rows=20, outer_validation_rows=12)
        return {'fit': fit, 'validation': combined, 'test': query, 'preparation': metadata,
                'auxiliary': {'ensemble_meta': meta}}, defaults, sha

    def test_cached_and_conditional_blends_never_retrain(self):
        with tempfile.TemporaryDirectory() as temporary:
            state, defaults, sha = self.setup_run(Path(temporary), False)
            base = {'source_nodes': ['node_000', 'node_001'], 'method': 'mean'}
            pairs = [load_pair(Path(temporary), node, sha) for node in base['source_nodes']]
            with patch('agent.operations._fit', side_effect=AssertionError('Cached blending retrained')):
                result = run_operation(state, {'kind': 'saved_ensemble', 'parameters': base}, defaults)
                np.testing.assert_allclose(result['test_predictions'].y, (pairs[0]['test'].y + pairs[1]['test'].y) / 2)
                result = run_operation(state, {'kind': 'saved_ensemble', 'parameters': {**base, 'method': 'conditional',
                    'weights': [0., 1.], 'rules': [{'conditions': [{'column': 'x', 'operation': 'lt', 'value': 0}], 'weights': [1., 0.]}]}}, defaults)
                expected = np.where(state['test'].x < 0, pairs[0]['test'].y, pairs[1]['test'].y)
                np.testing.assert_allclose(result['test_predictions'].y, expected)
            with self.assertRaises(ValueError):
                load_pair(Path(temporary), 'node_000', '0' * 64)
            from agent.search import Search
            request = Idea(purpose='improve', parent_id='node_000', change='Combine saved sources',
                expected_effect='Measure complementarity', ensemble=base)
            proposal = asyncio.run(Search.ensemble_plan(None, request))
            self.assertEqual(proposal.assembly.blocks[0].operation.kind, 'saved_ensemble')
            self.assertEqual(proposal.hypothesis.pipeline.model['sources'], base['source_nodes'])

    def test_learned_combiners_never_retrain_sources(self):
        for classification in (False, True):
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                state, defaults, _ = self.setup_run(root, classification)
                original_hash = file_hash(root / 'nodes/node_000/assembly.json')
                base = {'source_nodes': ['node_000', 'node_001'], 'method': 'stacking', 'passthrough': True}
                with patch('agent.assembly.execute', side_effect=AssertionError('Ensemble retrained a source')):
                    for method in ('stacking', 'gating'):
                        result = run_operation(state, {'kind': 'saved_ensemble', 'parameters': {**base, 'method': method}}, defaults)
                        self.assertEqual(len(result['test_predictions']), 12)
                        self.assertFalse(result['validation_predictions'].attrs['meta_held_out'])
                self.assertEqual(file_hash(root / 'nodes/node_000/assembly.json'), original_hash)
                facts = json.loads((root / 'nodes/node_002/ensemble_full_diagnostics.json').read_text())
                self.assertEqual(facts['oof_rows'], 20)
                self.assertFalse(facts['source_retraining'])
                # A learned source must not reuse its in-sample meta predictions for another stack.
                state['auxiliary']['ensemble_meta'] = state['auxiliary']['ensemble_meta'].iloc[:10]
                with self.assertRaisesRegex(ValueError, 'same ensemble holdout'):
                    run_operation(state, {'kind': 'saved_ensemble', 'parameters': base}, defaults)

    def test_conditional_votes_align_classes_when_subsets_omit_labels(self):
        from agent.ensembles import conditional
        from agent.plans import EnsemblePlan
        state = {'validation': pd.DataFrame({'x': [-1, 1]}), 'test': pd.DataFrame({'x': [-1, 1]}),
                 'preparation': {'data': {'target_columns': ['y'], 'id_columns': []},
                    'output': {'columns': ['y']}, 'metric': {'name': 'Macro F1'}}}
        pairs = [{'validation': pd.DataFrame({'y': labels}), 'test': pd.DataFrame({'y': labels})}
                 for labels in (['no', 'yes'], ['no', 'no'])]
        plan = EnsemblePlan(source_nodes=['node_000', 'node_001'], method='conditional',
            rules=[{'conditions': [{'column': 'x', 'operation': 'lt', 'value': 0}], 'weights': [0., 1.]}])
        result = conditional(pairs, state, {'task_type': 'classification'}, plan, 'vote')
        self.assertEqual(result['test_predictions'].y.tolist(), ['no', 'yes'])
        self.assertEqual(result['test_predictions'].attrs['probabilities']['values'].shape, (2, 2))
        for pair in pairs:
            for scope in pair:
                pair[scope]['y'] = pair[scope].y.map({'no': 0, 'yes': 1})
        result = conditional(pairs, state, {'task_type': 'classification'}, plan, 'vote')
        self.assertTrue(pd.api.types.is_integer_dtype(result['test_predictions'].y))

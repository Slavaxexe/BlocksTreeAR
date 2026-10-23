"""Real research execution, evidence memory and mocked-API integration; no external API."""

import asyncio
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import httpx
import numpy as np
from openai import AsyncOpenAI
import pandas as pd
from pydantic import ValidationError

from agent.agents import JSONAgents
from agent.config import Settings
from agent.contracts import Preparation, ResearchPlan, ResearchReview
from agent.data import file_hash
from agent.research import Research
from agent.research_worker import execute, inspect
from agent.runner import Options, run
from agent.storage import atomic_bytes, fingerprint, save
from agent.worker import prepare
from viewer import collect


LOADER = '''import numpy as np
import pandas as pd

def load_data(task_directory, config):
    return {name: pd.read_csv(task_directory / path) for name, path in config['data']['files'].items()}

def make_split(train, config):
    return np.arange(6), np.arange(6, 8)

def score(reference, predictions, config):
    return float(np.mean(np.abs(reference['target'] - predictions['target'])))

def write_submission(test, predictions, tables, config, path):
    pd.DataFrame({'id': test['id'], 'target': predictions['target']}).to_csv(path, index=False)
'''

CHECKS = [{'name': 'rows', 'kind': 'quality'}]


MODEL = '''import pandas as pd

def run(state):
    mean = float(state['fit']['target'].mean())
    return {name + '_predictions': pd.DataFrame({'target': [mean] * len(state[name])})
            for name in ('validation', 'test')}
'''


def fixture(directory):
    inputs, root = directory / 'inputs', directory / 'run'
    inputs.mkdir()
    root.mkdir()
    pd.DataFrame({'id': range(8), 'feature': range(8), 'flag': [True, False] * 4,
                  'target': [0, 0, 1, 1, 2, 2, 900001, 900002]}).to_csv(inputs / 'train.csv', index=False)
    pd.DataFrame({'id': [8, 9], 'feature': [8, 9]}).to_csv(inputs / 'test.csv', index=False)
    text = ('Regression predicts target from train.csv and test.csv. The identifier is id. '
            'Metric MAE, lower is better. Write submission.csv with columns id,target.')
    task = inputs / 'task.md'
    task.write_text(text, encoding='utf-8')
    draft = {'rationale': 'Use the supplied files and hold out the last two rows.',
             'data': {'files': {'train': 'train.csv', 'test': 'test.csv'}, 'target_columns': ['target'], 'id_columns': ['id']},
             'validation': {'strategy': 'ordered', 'group_columns': []},
             'metric': {'name': 'MAE', 'direction': 'minimize', 'definition': 'mean(abs(error))', 'status': 'ready'},
             'output': {'filename': 'submission.csv', 'format': 'CSV', 'columns': ['id', 'target']}, 'code': LOADER}
    atomic_bytes(root / 'prepare.py', LOADER.encode())
    save(root / 'preparation.json', Preparation(task_directory=str(inputs), code_sha256=file_hash(root / 'prepare.py'),
         **{k: v for k, v in draft.items() if k != 'code'}))
    return task, root, draft


def review(statement='Measured row count', **updates):
    return {'outcome': 'supported', 'summary': 'The recorded measurement supports the statement.',
            'insights': [{'statement': statement, 'kind': 'observation', 'evidence_keys': ['rows'],
                          'implication': 'Budget the following experiments for this data size.'}],
            **updates}


class WorkerTests(unittest.TestCase):
    def test_structure_then_fit_hide_holdout_targets_and_record_actual_measurements(self):
        with tempfile.TemporaryDirectory() as temporary:
            _, root, _ = fixture(Path(temporary))
            inspect(root)
            self.assertFalse((root / 'split.npz').exists())
            profile = json.loads((root / 'research/profile.json').read_text())
            self.assertEqual(profile['target_counts']['target']['0'], 2)
            self.assertEqual(profile['tables']['train']['columns']['flag']['frequent_values'], {'True': 4, 'False': 4})
            for phase in ('structure', 'fit'):
                if phase == 'fit':
                    prepare(root)
                node = root / 'research/nodes/research_000'
                save(node / 'start.json', {'phase': phase, 'analyses': CHECKS, 'analysis_sha256': fingerprint(CHECKS),
                     'preparation_sha256': file_hash(root / 'preparation.json')})
                execute(root, node)
                values = json.loads((node / 'measurements.json').read_text())['values']
                self.assertEqual(values['rows']['rows_used'], 8 if phase == 'structure' else json.loads((root / 'preparation.json').read_text())['validation']['model_fit_rows'])
                self.assertEqual(values['rows']['target_available'], phase == 'fit')
                self.assertFalse(json.loads((node / 'measurements.json').read_text())['target_access']['validation'])
                self.assertFalse(json.loads((node / 'measurements.json').read_text())['target_access']['test'])

    def test_review_rejects_invented_evidence_and_unknown_superseded_insights(self):
        context = {'measurements': {'actual_rows': 8}, 'insight_ids': []}
        with self.assertRaises(ValidationError):
            ResearchReview.model_validate(review(), context=context)
        with self.assertRaises(ValidationError):
            ResearchReview.model_validate(review(supersedes=['invented']),
                context={'measurements': {'rows': 8}, 'insight_ids': []})


class ResearchTests(unittest.IsolatedAsyncioTestCase):
    async def test_two_slots_measurements_memory_supersession_and_cancellation(self):
        with tempfile.TemporaryDirectory() as temporary:
            _, root, _ = fixture(Path(temporary))
            inspect(root)
            agents = JSONAgents(Settings('fake', 'https://qwen.example/v1', 'model'), time.monotonic() + 20, 100)
            shared, entered, peak, calls = {}, 0, 0, {}
            both = asyncio.Event()

            async def ask(client, schema, prompt, role, **kwargs):
                client.requests += 1
                client.budget['requests'] += 1
                context = json.loads(prompt.rsplit('\n', 1)[1])
                if schema is ResearchPlan:
                    return ResearchPlan(question=client.node_id, hypothesis='Rows exist', method='Count rows',
                                        decision='Set training budget', analyses=CHECKS, parent_id='research_999')
                return ResearchReview.model_validate(review(open_questions=['Explain duplicates']), context=kwargs['validation'])

            async def process(args, cwd, log, deadline, timeout):
                nonlocal entered, peak
                key = cwd.name
                calls[key] = calls.get(key, 0) + 1
                entered += 1
                peak = max(peak, entered)
                if entered == 2:
                    both.set()
                await asyncio.wait_for(both.wait(), 2)
                entered -= 1
                start = json.loads((cwd / 'start.json').read_text())
                save(cwd / 'measurements.json', {'values': {'rows': 8}, 'analysis_sha256': start['analysis_sha256'],
                     'preparation_sha256': start['preparation_sha256']})

            research = Research(agents, root, Options(max_llm_calls=100, research_parallel=2), shared, process, lambda message: None, lambda: None)
            with patch.object(JSONAgents, 'ask', new=ask):
                await research.bootstrap()
            self.assertEqual(peak, 2)
            errors = [p.read_text() for p in (root / 'research/nodes').glob('*/error.json')]
            self.assertEqual(research.revision, 2, errors)
            self.assertEqual(len(shared['data_research']['insights']), 1)  # Same observation is stored once.
            self.assertEqual(research.questions, ['Explain duplicates'])
            self.assertTrue((root / 'research/nodes/research_000/start.json').exists())

            first = research.insights[0]['id']
            result = review('Refined observation', supersedes=[first], resolves=['Explain duplicates'])
            completed = asyncio.create_task(asyncio.sleep(0, result={'status': 'completed', 'review':
                ResearchReview(**result).model_dump(), 'duration_seconds': .1}))
            row = {'node_id': 'research_002', 'phase': 'fit', 'stage': 'review', 'parent_id': 'research_000'}
            research.nodes.append(row)
            research.active[completed] = row
            await completed
            research.collect({completed})
            self.assertFalse(research.insights[0]['active'])
            self.assertEqual(len(research.view()['insights']), 1)
            self.assertEqual(research.questions, [])

            pending = asyncio.create_task(asyncio.Event().wait())
            research.active[pending] = {'node_id': 'research_003', 'phase': 'fit', 'stage': 'planning'}
            await research.close('time_budget')
            self.assertEqual(research.report()['researching_nodes'], [])
            self.assertEqual(json.loads((root / 'research/nodes/research_003/finish.json').read_text())['status'], 'cancelled')

    async def test_exhausted_research_budget_preserves_model_calls(self):
        with tempfile.TemporaryDirectory() as temporary:
            _, root, _ = fixture(Path(temporary))
            agents = JSONAgents(Settings('fake', 'https://qwen.example/v1', 'model'), time.monotonic()+5, 8)
            research = Research(agents, root, Options(max_llm_calls=8, research_parallel=2), {}, None, print, lambda: None)
            research.fill()
            self.assertEqual(research.tasks, [])
            self.assertEqual(research.stop_reason, 'llm_limit')
            self.assertEqual(agents.remaining_calls, 8)

    async def test_pipeline_bootstraps_then_updates_baker_and_monitoring(self):
        with tempfile.TemporaryDirectory() as temporary:
            task, fixture_root, draft = fixture(Path(temporary))
            output = fixture_root.parent / 'pipeline'
            text, contexts, running_counts = task.read_text(), [], []
            evidence = lambda fields: {field: [text] for field in fields}
            replies = {
                'ProblemFacts': {'task_type': 'regression', 'target_column': 'target', 'evidence': evidence(['task_type', 'target_column'])},
                'DataFacts': {'train_file': 'train.csv', 'test_file': 'test.csv', 'id_column': 'id', 'evidence': evidence(['train_file', 'test_file', 'id_column'])},
                'MetricFacts': {'name': 'MAE', 'direction': 'minimize', 'evidence': evidence(['name', 'direction'])},
                'OutputFacts': {'submission_file': 'submission.csv', 'columns': ['id', 'target'], 'evidence': evidence(['submission_file', 'columns'])},
                'PreparationDraft': draft,
                'Program': {'code': MODEL},
                'Analysis': {'summary': 'Measured constant baseline', 'findings': [], 'next_ideas': []},
            }
            spec = {'operation': {'kind': 'constant'}, 'name': 'predict', 'source_node': None, 'description': 'Predict using the mean fit target',
                    'requires': ['fit', 'validation', 'test', 'preparation'], 'provides': ['validation_predictions', 'test_predictions']}

            def handler(request):
                body = json.loads(request.content)
                title = json.loads(body['messages'][0]['content'].split('Schema: ', 1)[1])['title']
                context = json.loads(body['messages'][1]['content'].rsplit('\n', 1)[1])
                contexts.append((title, context))
                if title == 'Program':
                    running_counts.append(len(json.loads((output / 'report.json').read_text())['running_nodes']))
                if title == 'ResearchPlan':
                    self.assertTrue((task.parent / 'submission.csv').is_file(), 'Cheap baseline must be published before EDA')
                    reply = {'question': context['node_id'], 'hypothesis': 'The input has rows', 'method': 'Count rows',
                             'decision': 'Budget fitting', 'analyses': CHECKS}
                elif title == 'ResearchReview':
                    reply = review('Rows measured by ' + context['plan']['node_id'])
                elif title == 'Ideas':
                    if 'baseline' in context['purposes']:
                        purposes = ['baseline']
                    else:
                        purposes = ['improve'] * context['max_new_ideas']
                        explores = context['occupied_slots'].get('explore', 0) + sum(
                            item['purpose'] == 'explore' for item in context['queued_ideas'])
                        if not explores:
                            purposes[-1] = 'explore'
                    reply = {'ideas': [{'purpose': purpose, 'parent_id': context['leader_id'],
                        'change': f"Mean model {len(contexts)}-{i}", 'change_blocks': ['predict'], 'expected_effect': 'Measure baseline'}
                        for i, purpose in enumerate(purposes)]}
                elif title == 'NodePlan':
                    blocks = [spec]
                    if context['idea']['purpose'] != 'baseline':
                        blocks = [
                            {'name': 'preprocess', 'description': 'Encode fit features',
                             'operation': {'kind': 'preprocess'},
                             'requires': ['fit', 'validation', 'test', 'preparation'],
                             'provides': ['fit_features', 'fit_targets', 'validation_features', 'test_features', 'transformer']},
                            {'name': 'fit', 'description': 'Fit a configured small forest',
                             'operation': {'kind': 'fit', 'parameters': {'estimator': 'sklearn.ensemble.ExtraTreesRegressor',
                                           'parameters': {'n_estimators': len(contexts)}}},
                             'requires': ['fit_features', 'fit_targets', 'transformer', 'fit', 'preparation'], 'provides': ['model']},
                            {'name': 'predict', 'description': 'Predict using fitted forest', 'operation': {'kind': 'predict'},
                             'requires': ['model', 'validation_features', 'test_features', 'preparation'],
                             'provides': ['validation_predictions', 'test_predictions']},
                        ]
                    reply = {'pipeline': {'preprocessing': 'Use the supplied fit rows',
                             'model': {'name': 'Mean predictor', 'configuration': f'Constant target mean, version {len(contexts)}'},
                             'training': 'Calculate the mean fit target'},
                             'order': [block['name'] for block in blocks], 'reuse': [], 'write': blocks}
                else:
                    reply = replies[title]
                return httpx.Response(200, json={'id': 'mock', 'object': 'chat.completion', 'created': 0, 'model': 'model',
                    'choices': [{'index': 0, 'finish_reason': 'stop', 'message': {'role': 'assistant', 'content': json.dumps(reply)}}]})

            def client(**kwargs):
                return AsyncOpenAI(api_key='fake', base_url='https://qwen.example/v1', max_retries=0,
                                   http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))

            with patch('agent.agents.AsyncOpenAI', side_effect=client):
                root, report = await run(task, Options(time_budget=70, experiment_timeout=15, output_dir=output,
                    max_experiments=7, max_research_nodes=4, max_llm_calls=120, research_parallel=2),
                    Settings('fake', 'https://qwen.example/v1', 'model', timeout_seconds=5))
            self.assertEqual(report.status, 'ready', report.errors + report.warnings)
            self.assertEqual(report.max_parallel_nodes, 6)
            self.assertEqual(report.max_parallel_research, 2)
            self.assertEqual(report.researching_nodes, [])
            self.assertEqual(report.research_revision, 4, report.warnings)
            preparations = [context for title, context in contexts if title == 'PreparationDraft']
            self.assertEqual(len(preparations), 2)
            self.assertEqual(preparations[1]['data_research']['revision'], 2)
            baker = [context for title, context in contexts if title == 'Ideas']
            self.assertIn('published_cheap_baseline', report.checks)
            first_baker = next(index for index, (title, _) in enumerate(contexts) if title == 'Ideas')
            self.assertTrue(any(title == 'ResearchPlan' and context['phase'] == 'fit' for title, context in contexts[:first_baker]))
            self.assertGreaterEqual(baker[0]['data_research']['revision'], 2)
            self.assertEqual(baker[-1]['data_research']['revision'], 4)
            self.assertNotIn('Program', [title for title, _ in contexts])
            self.assertTrue((task.parent / 'submission.csv').exists())
            snapshot = collect(root)['runs'][0]
            self.assertEqual(len(snapshot['research_nodes']), 4)
            self.assertEqual(snapshot['research_context']['revision'], 4)
            split_hash = file_hash(root / 'split.npz')
            for node in snapshot['research_nodes']:
                if node['phase'] == 'fit':
                    self.assertFalse(node['measurements']['target_access']['validation'])
                    self.assertEqual(node['measurements']['values']['rows']['rows_used'],
                                     json.loads((root / 'preparation.json').read_text())['validation']['model_fit_rows'])
            self.assertEqual(split_hash, json.loads((root / 'preparation.json').read_text())['validation']['split_sha256'])


if __name__ == '__main__':
    unittest.main()

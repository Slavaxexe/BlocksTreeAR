"""Mocked Qwen HTTP, real block assemblies and child processes. No external API."""

import asyncio
import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import httpx
import numpy as np
from openai import AsyncOpenAI
import pandas as pd

from agent.assembly import BlockError, execute, implementation_hash
from agent.config import Settings
from agent.context import preparation_view
from agent.contracts import Assembly, Finish, Preparation, Start
from agent.data import check_code, file_hash
from agent.runner import Options, process, run
from agent.storage import load

PREPARE = '''import numpy as np
import pandas as pd

def load_data(task_directory, config):
    files = config['data']['files']
    return {'train': pd.read_csv(task_directory / files['train'], sep='|', dtype={'record': str}),
            'test': pd.read_json(task_directory / files['test'], lines=True, dtype={'record': str})}

def score(reference, predictions, config):
    column = config['data']['target_columns'][0]
    return float(np.mean(np.abs(reference[column].to_numpy() - predictions[column].to_numpy())) + 7)

def write_submission(test, predictions, tables, config, path):
    column = config['data']['target_columns'][0]
    pd.DataFrame({'record': test['record'].to_numpy(), 'estimate': predictions[column].to_numpy(),
                  'comment': 'baseline'}).to_csv(path, sep='\\t', index=False)
'''

PROGRAM = '''import pandas as pd

def run(state):
    fit, validation, test, preparation = (state[k] for k in ('fit', 'validation', 'test', 'preparation'))
    column = preparation['data']['target_columns'][0]
    value = float(fit[column].mean())
    return {'validation_predictions': pd.DataFrame({column: [value] * len(validation)}),
            'test_predictions': pd.DataFrame({column: [value] * len(test)})}
'''


def block(name='predict', source=None):
    return {'name': name, 'source_node': source, 'description': 'Predict the target in supplied row order',
            'requires': ['fit', 'validation', 'test', 'preparation'],
            'provides': ['validation_predictions', 'test_predictions']}


def proposal(index=0, purpose='baseline', parent=None):
    return {'purpose': purpose, 'parent_id': parent,
            'hypothesis': {'rationale': f'Experiment {index}', 'expected_effect': 'Measure the proposed approach',
                'pipeline': {'data': {}, 'preprocessing': {}, 'model': {'family': 'constant', 'version': index}, 'training': {}}},
            'assembly': {'blocks': [block(source=parent)], 'edit_block': 'predict'}}


def fixture(directory):
    train = pd.DataFrame({'record': [f'{i:04d}' for i in range(40)], 'signal': np.arange(40),
                          'yield_value': np.arange(40) * 2.})
    test = pd.DataFrame({'record': [f'{i:04d}' for i in range(100, 105)], 'signal': np.arange(5)})
    train.to_csv(directory / 'samples.psv', sep='|', index=False)
    test.to_json(directory / 'query.jsonl', orient='records', lines=True)
    text = ('A regression task predicts yield_value. Training file samples.psv uses |; query.jsonl uses JSON lines. '
            'The identifier is record. Keep the last 8 ordered training rows for validation. '
            'The custom score is mean(abs(y_true-y_pred)) + 7, lower is better. '
            'Write result.tsv with columns record,estimate,comment; comment must be baseline; preserve query order.')
    path = directory / 'condition.md'
    path.write_text(text, encoding='utf-8')
    evidence = lambda fields: {field: [text] for field in fields}
    return path, {
        'ProblemFacts': {'task_type': 'regression', 'target_column': 'yield_value',
                         'evidence': evidence(['task_type', 'target_column'])},
        'DataFacts': {'train_file': 'samples.psv', 'test_file': 'query.jsonl', 'id_column': 'record',
                      'evidence': evidence(['train_file', 'test_file', 'id_column'])},
        'MetricFacts': {'name': 'custom absolute error + seven', 'direction': 'minimize',
                        'definition': 'mean(abs(y_true-y_pred)) + 7', 'evidence': evidence(['name', 'direction', 'definition'])},
        'OutputFacts': {'submission_file': 'result.tsv', 'columns': ['record', 'estimate', 'comment'],
                        'format': 'TSV', 'evidence': evidence(['submission_file', 'columns', 'format'])},
        'PreparationDraft': {'rationale': 'Read the actual formats; ordered validation.',
            'data': {'files': {'train': 'samples.psv', 'test': 'query.jsonl'}, 'target_columns': ['yield_value'], 'id_columns': ['record']},
            'validation': {'strategy': 'ordered_tail', 'rows': 8},
            'metric': {'name': 'custom absolute error + seven', 'direction': 'minimize',
                       'definition': 'mean(abs(y_true-y_pred)) + 7', 'status': 'ready'},
            'output': {'filename': 'result.tsv', 'format': 'TSV', 'columns': ['record', 'estimate', 'comment']}, 'code': PREPARE},
    }


class PipelineTests(unittest.IsolatedAsyncioTestCase):
    async def exercise(self, directory, *, repairs=0, full_repair=False, network=False, blocked=False,
                       prepare_repairs=0, missing_metric=False, search=False, repair_reused=False, expire=False,
                       planning_duplicates=False):
        path, answers = fixture(directory)
        if missing_metric:
            answers['PreparationDraft']['metric']['status'] = 'needs_implementation'
        root = directory / 'artifacts' / 'run'
        requests, contexts, saved_starts = [], [], {}
        four_ready, release_research = asyncio.Event(), asyncio.Event()
        simultaneous = peak = 0
        program_counts, prepare_count, batch = {}, 0, 0

        async def handler(request):
            nonlocal simultaneous, peak, prepare_count, batch
            body = json.loads(request.content)
            title = json.loads(body['messages'][0]['content'].split('Schema: ', 1)[1])['title']
            context = json.loads(body['messages'][1]['content'].rsplit('\n', 1)[1])
            requests.append(title)
            contexts.append((title, context))
            if title == 'PreparationDraft':
                answer = answers[title]
                if prepare_count < prepare_repairs:
                    answer = {**answer, 'code': PREPARE.replace("files = config['data']['files']", 'raise ValueError("repair preparation")')}
                prepare_count += 1
            elif title == 'Proposals':
                if not batch:
                    answer = {'proposals': [proposal()]}
                else:
                    self.assertIn('summary', context['experiments'][0]['analysis'])
                    self.assertNotIn('preparation_code', context)
                    self.assertNotIn('used', context)
                    self.assertTrue(all('sha256' not in str(b) for b in context['module_catalog'].values()))
                    if planning_duplicates:
                        if planning_duplicates == 'always' or batch < 3:
                            repeated = proposal(0, 'improve', 'node_000')
                            repeated['assembly']['blocks'][0]['source_node'] = None
                            answer = {'proposals': [repeated]}
                        else:
                            answer = {'proposals': [proposal(1, 'improve', context['leader_id'])]}
                    elif repair_reused:
                        plan = proposal(1, 'improve', 'node_000')
                        plan['assembly'] = {'blocks': [
                            {'name': 'shift', 'source_node': None, 'description': 'Supply a marker',
                             'requires': [], 'provides': ['marker']}, block(source='node_000')], 'edit_block': 'shift'}
                        answer = {'proposals': [plan]}
                    else:
                        index = len(list((root / 'nodes').glob('node_*')))
                        answer = {'proposals': [proposal(index + i, purpose, context['leader_id'])
                                                for i, purpose in enumerate(context['purposes'])]}
                        if batch > 1:
                            self.assertIn('node_004', [n['node_id'] for n in context['pending']])
                batch += 1
            elif title == 'Program':
                node_id = context['node_id']
                node = root / 'nodes' / node_id
                saved = (node / 'start.json').read_bytes()
                self.assertEqual(saved_starts.setdefault(node_id, saved), saved)
                self.assertNotIn('preparation_code', context)
                self.assertNotIn('evidence', json.dumps(context))
                self.assertNotIn('source_hashes', json.dumps(context))
                state = json.loads((root / 'report.json').read_text())
                purposes = [json.loads((root / 'nodes' / n / 'start.json').read_text())['purpose'] for n in state['running_nodes']]
                self.assertLessEqual(purposes.count('improve'), 3)
                self.assertLessEqual(purposes.count('explore'), 1)
                count = program_counts.get(node_id, 0)
                program_counts[node_id] = count + 1
                if network and count == 0:
                    raise httpx.ConnectError('temporary connection error', request=request)
                if search and node_id in {'node_001', 'node_002', 'node_003', 'node_004'}:
                    simultaneous += 1
                    peak = max(peak, simultaneous)
                    if simultaneous == 4:
                        four_ready.set()
                    await asyncio.wait_for(four_ready.wait(), 15)
                    if node_id == 'node_004':
                        await asyncio.wait_for(release_research.wait(), 60)
                    simultaneous -= 1
                code = PROGRAM
                if repair_reused:
                    if node_id == 'node_000':
                        code = PROGRAM.replace("value = float(fit[column].mean())",
                            'if state.get("marker") == "bad":\n        raise ValueError("downstream failure")\n    value = float(fit[column].mean())')
                    elif context['target_block']['name'] == 'shift':
                        code = 'def run(state):\n    return {"marker": "bad"}\n'
                    else:
                        self.assertEqual(context['target_block']['name'], 'predict')
                        self.assertEqual(context['failed_stage'], 'check')
                        self.assertIn('downstream failure', context['execution_error'])
                if search:
                    value = {'node_001': 71., 'node_002': 69., 'node_003': 0., 'node_004': 10., 'node_005': 50.}.get(node_id)
                    if value is not None:
                        code = code.replace('float(fit[column].mean())', str(value))
                if count < repairs:
                    code = code.replace('value = float(fit[column].mean())', 'raise ValueError("repair me")')
                if full_repair and count:
                    self.assertEqual(context['failed_stage'], 'execution')
                    self.assertIn('full-only failure', context['execution_error'])
                answer = {'blocked_reason': 'Cannot implement the saved plan'} if blocked else {'code': code}
            elif title == 'Analysis':
                if context['experiment']['result']['stage'] != 'implementation':
                    self.assertTrue(context['modules'])
                self.assertNotIn('start_sha256', json.dumps(context))
                node_id = context['experiment']['node_id']
                if node_id == 'node_005':
                    release_research.set()
                answer = {'summary': 'Observed result', 'findings': ['predict produced measured predictions'],
                          'next_ideas': ['Improve the predict block']}
            else:
                answer = answers[title]
            return httpx.Response(200, json={'id': 'test', 'object': 'chat.completion', 'created': 0, 'model': 'model',
                'choices': [{'index': 0, 'finish_reason': 'stop', 'message': {'role': 'assistant', 'content': json.dumps(answer)}}]})

        def client(**kwargs):
            return AsyncOpenAI(api_key='fake-secret', base_url='https://qwen.example/v1', max_retries=0,
                               http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
        full_calls = 0
        async def subprocess(*args):
            nonlocal full_calls
            if full_repair and '--start' in args[0]:
                full_calls += 1
                if full_calls == 1:
                    raise ValueError('full-only failure')
            return await process(*args)
        from agent.runner import summarize
        offset = [0.]
        def merge(*args):
            result = summarize(*args)
            if expire and args[1].node_id == 'node_001':
                offset[0] = 200.
            return result
        with patch('agent.agents.AsyncOpenAI', side_effect=client), patch('agent.runner.process', side_effect=subprocess), \
                patch('agent.runner.summarize', side_effect=merge), \
                patch('agent.runner.time', SimpleNamespace(monotonic=lambda: time.monotonic() + offset[0])):
            result = await run(path, Options(time_budget=100, experiment_timeout=20, output_dir=root,
                max_experiments=6 if search else 2 if repair_reused or planning_duplicates else 1),
                Settings('fake-secret', 'https://qwen.example/v1', 'model'))
        for node_id, saved in saved_starts.items():
            self.assertEqual((root / 'nodes' / node_id / 'start.json').read_bytes(), saved)
        return result, requests, contexts, peak

    async def test_unseen_formats_custom_metric_ids_and_real_assembly(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (root, report), _, _, _ = await self.exercise(directory)
            self.assertEqual(report.status, 'ready', report.errors)
            node = root / 'nodes/node_000'
            finish = load(node / 'finish.json', Finish)
            self.assertAlmostEqual(finish.value, 47.)
            self.assertEqual(finish.code_sha256, implementation_hash(node))
            self.assertEqual(finish.start_sha256, file_hash(node / 'start.json'))
            output = pd.read_csv(directory / 'result.tsv', sep='\t', dtype={'record': str})
            self.assertEqual(output.record.tolist(), ['0100', '0101', '0102', '0103', '0104'])
            self.assertEqual(list(output), ['record', 'estimate', 'comment'])
            self.assertTrue((node / 'analysis.json').exists())
            self.assertEqual(report.analyzing_nodes, [])
            self.assertNotIn('fake-secret', (root / 'report.json').read_text())

    async def test_three_code_attempts_and_preserved_error_history(self):
        for repairs in (2, 3):
            with self.subTest(repairs=repairs), tempfile.TemporaryDirectory() as temporary:
                (root, report), requests, _, _ = await self.exercise(Path(temporary), repairs=repairs)
                self.assertEqual(requests.count('Program'), 3)
                self.assertEqual(report.status, 'ready' if repairs == 2 else 'failed', report.errors)
                node = root / 'nodes/node_000'
                self.assertEqual(len(list((node / 'attempts').glob('*'))), 3)
                self.assertIn('repair me', (node / 'attempts/01/error.json').read_text())
                self.assertTrue((node / 'start.json').exists())

    async def test_full_execution_error_is_repaired_in_same_node(self):
        with tempfile.TemporaryDirectory() as temporary:
            (root, report), requests, _, _ = await self.exercise(Path(temporary), full_repair=True)
            self.assertEqual(report.status, 'ready', report.errors)
            self.assertEqual(requests.count('Program'), 2)
            self.assertEqual(load(root / 'nodes/node_000/finish.json', Finish).stage, 'execution')
            self.assertEqual(json.loads((root / 'nodes/node_000/attempts/01/error.json').read_text())['stage'], 'execution')

    async def test_reuse_and_repair_only_the_actual_failing_block(self):
        with tempfile.TemporaryDirectory() as temporary:
            (root, report), _, contexts, _ = await self.exercise(Path(temporary), repair_reused=True)
            self.assertEqual(report.status, 'ready', report.errors)
            child = root / 'nodes/node_001'
            self.assertEqual((child / 'modules/shift.py').read_bytes(), (child / 'attempts/01/shift.py').read_bytes())
            self.assertEqual([c['target_block']['name'] for title, c in contexts if title == 'Program' and c['node_id'] == 'node_001'],
                             ['shift', 'predict'])
            self.assertIn('downstream failure', (root / 'nodes/node_000/modules/predict.py').read_text())
            self.assertNotIn('downstream failure', (child / 'modules/predict.py').read_text())
            self.assertEqual(load(child / 'start.json', Start).assembly.edit_block, 'shift')
            self.assertEqual(load(child / 'finish.json', Finish).status, 'completed')

    async def test_four_workers_three_improve_one_explore_and_rolling_feedback(self):
        with tempfile.TemporaryDirectory() as temporary:
            (root, report), _, _, peak = await self.exercise(Path(temporary), search=True)
            self.assertEqual(report.status, 'ready', report.errors)
            self.assertEqual(peak, 4)
            starts = [load(p, Start) for p in sorted((root / 'nodes').glob('*/start.json'))]
            self.assertEqual([s.purpose for s in starts[:5]], ['baseline', 'improve', 'improve', 'improve', 'explore'])
            self.assertEqual(starts[5].purpose, 'improve')
            self.assertEqual(report.best_node, 'node_001')
            self.assertEqual((Path(temporary) / 'result.tsv').read_bytes(), (root / 'nodes/node_001/result.tsv').read_bytes())
            self.assertEqual(report.running_nodes, [])

    async def test_duplicate_batch_retries_with_fresh_context_and_continues_search(self):
        with tempfile.TemporaryDirectory() as temporary:
            (root, report), requests, _, _ = await self.exercise(Path(temporary), planning_duplicates=True)
            self.assertEqual(report.stop_reason, 'max_experiments', report.errors)
            self.assertEqual(report.errors, [])
            self.assertEqual(requests.count('Proposals'), 4)
            self.assertEqual(load(root / 'nodes/node_001/finish.json', Finish).status, 'completed')
            self.assertTrue((root / 'planning/round_001/response_02.json').exists())
            self.assertTrue((root / 'planning/round_002/response_01.json').exists())

    async def test_duplicate_batches_have_bounded_retries_and_preserve_best(self):
        with tempfile.TemporaryDirectory() as temporary:
            (root, report), requests, _, _ = await self.exercise(Path(temporary), planning_duplicates='always')
            self.assertEqual(report.stop_reason, 'planning_stalled', report.errors)
            self.assertEqual(requests.count('Proposals'), 7)
            self.assertEqual(report.best_node, 'node_000')
            self.assertEqual(report.errors, [])
            self.assertEqual((Path(temporary) / 'result.tsv').read_bytes(), (root / 'nodes/node_000/result.tsv').read_bytes())

    async def test_network_retry_and_blocked_plan_keep_trace(self):
        for mode in ('network', 'blocked'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temporary:
                (root, report), requests, _, _ = await self.exercise(Path(temporary), **{mode: True})
                finish = load(root / 'nodes/node_000/finish.json', Finish)
                if mode == 'network':
                    self.assertEqual(report.status, 'ready', report.errors)
                    self.assertEqual(requests.count('Program'), 2)
                    self.assertTrue(any(c.status == 'api_error' and c.node_id == 'node_000' for c in report.llm_calls))
                else:
                    self.assertEqual(finish.stage, 'implementation')
                    self.assertIsNone(finish.code_sha256)
                    self.assertTrue((root / 'nodes/node_000/start.json').exists())

    async def test_deadline_cancels_workers_records_results_and_preserves_submission(self):
        with tempfile.TemporaryDirectory() as temporary:
            (root, report), _, _, _ = await self.exercise(Path(temporary), search=True, expire=True)
            self.assertEqual(report.stop_reason, 'time_budget', report.errors)
            self.assertEqual(report.running_nodes, [])
            self.assertEqual(report.analyzing_nodes, [])
            self.assertFalse(report.planning)
            for path in (root / 'nodes').glob('*/start.json'):
                self.assertTrue((path.parent / 'finish.json').exists(), path.parent.name)
            self.assertEqual((Path(temporary) / 'result.tsv').read_bytes(),
                             (root / 'nodes' / report.best_node / 'result.tsv').read_bytes())

    async def test_preparation_three_attempts_and_unknown_metric(self):
        with tempfile.TemporaryDirectory() as temporary:
            (root, report), requests, _, _ = await self.exercise(Path(temporary), prepare_repairs=2, missing_metric=True)
            self.assertEqual(requests.count('PreparationDraft'), 3)
            self.assertEqual(report.status, 'partial', report.errors)
            self.assertIsNone(load(root / 'nodes/node_000/finish.json', Finish).value)

    async def test_changed_inputs_and_changed_modules_are_rejected(self):
        from agent.worker import smoke
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (root, report), _, _, _ = await self.exercise(directory)
            self.assertEqual(report.status, 'ready', report.errors)
            node = root / 'nodes/node_000'
            original = (node / 'finish.json').read_bytes()
            with (directory / 'samples.psv').open('a') as stream:
                stream.write('9999|10|20\n')
            with self.assertRaisesRegex(ValueError, 'Input files changed'):
                smoke(node, root / 'preparation.json')
            self.assertEqual((node / 'finish.json').read_bytes(), original)
            (node / 'modules/constant.py').write_text('def run(state): return {}', encoding='utf-8')
            with self.assertRaises(BlockError):
                implementation_hash(node)

    async def test_compact_profile_keeps_all_columns_types_and_missing_counts(self):
        with tempfile.TemporaryDirectory() as temporary:
            (root, _), _, _, _ = await self.exercise(Path(temporary))
            prep = load(root / 'preparation.json', Preparation)
            view = preparation_view(prep)
            for role, table in prep.profile.items():
                columns = {name: dtype for dtype, names in view['profile'][role]['columns_by_dtype'].items() for name in names}
                self.assertEqual(columns, {name: c['dtype'] for name, c in table['columns'].items()})
                self.assertEqual(view['profile'][role]['missing'], {name:c['missing'] for name,c in table['columns'].items() if c['missing']})
            self.assertNotIn('source_hashes', view)

    @unittest.skipUnless(os.name == 'nt', 'Windows system libraries require SystemRoot')
    async def test_windows_systemroot_allows_sklearn_import(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            await process(['-c', 'import os, asyncio, _overlapped; from sklearn.linear_model import Ridge; print(bool(os.environ.get("SystemRoot"))); print(Ridge.__name__)'],
                          root, root / 'import.log', time.monotonic()+30, 20)
            self.assertEqual((root / 'import.log').read_text().splitlines(), ['True', 'Ridge'])

    async def test_process_timeout_and_clean_environment(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch.dict(os.environ, {'OPENAI_API_KEY': 'must-not-reach-worker', 'OTHER_SECRET': 'hidden'}):
                await process(['-c', 'import os; print(dict(os.environ))'], root, root / 'env.log', time.monotonic()+5, 2)
            self.assertNotIn('must-not-reach-worker', (root / 'env.log').read_text())
            self.assertNotIn('OTHER_SECRET', (root / 'env.log').read_text())
            with self.assertRaisesRegex(TimeoutError, 'Process exceeded'):
                await process(['-c', 'import time; time.sleep(5)'], root, root / 'timeout.log', time.monotonic()+5, .1)

    def test_assembly_rejects_disconnected_order_and_missing_reuse_source(self):
        for blocks, edit in (([block(), block('predict')], 'predict'),
                             ([{**block(), 'requires': ['absent']}], 'predict'),
                             ([block(), {**block('features'), 'provides': ['x']}], 'features')):
            with self.subTest(blocks=blocks), self.assertRaises(ValueError):
                Assembly(blocks=blocks, edit_block=edit)
        check_code('class CoupledModel:\n    def fit(self, state): return state\n\ndef run(state): return CoupledModel().fit(state)', ('run',))
        for code in ('import subprocess\n' + PROGRAM, 'print(1)\n' + PROGRAM, 'x = list()\n' + PROGRAM):
            with self.assertRaises(ValueError):
                check_code(code, ('run',))


if __name__ == '__main__':
    unittest.main()


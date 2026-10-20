"""Mocked Qwen HTTP, real generated programs and real subprocesses on an unseen task."""

import asyncio
import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import httpx
import numpy as np
from openai import AsyncOpenAI
import pandas as pd

from alfa_agent.config import Settings
from alfa_agent.contracts import Delta, Finish, Hypothesis, Metric, Preparation, Start
from alfa_agent.data import check_code, file_hash
from alfa_agent.runner import Options, process, run, summarize
from alfa_agent.storage import load

PREPARE = '''import numpy as np
import pandas as pd

def load_data(task_directory, config):
    files = config['data']['files']
    return {'train': pd.read_csv(task_directory / files['train'], sep='|', dtype={'record': str}),
            'test': pd.read_json(task_directory / files['test'], lines=True, dtype={'record': str})}

def make_split(train, config):
    return np.arange(len(train) - 8), np.arange(len(train) - 8, len(train))

def score(reference, predictions, config):
    column = config['data']['target_columns'][0]
    return float(np.mean(np.abs(reference[column].to_numpy() - predictions[column].to_numpy())) + 7)

def write_submission(test, predictions, tables, config, path):
    column = config['data']['target_columns'][0]
    pd.DataFrame({'record': test['record'].to_numpy(), 'estimate': predictions[column].to_numpy(),
                  'comment': 'baseline'}).to_csv(path, sep='\\t', index=False)
'''
PROGRAM = '''import pandas as pd

def fit_predict(fit, validation, test, preparation):
    column = preparation['data']['target_columns'][0]
    value = float(fit[column].mean())
    return (pd.DataFrame({column: [value] * len(validation)}),
            pd.DataFrame({column: [value] * len(test)}))
'''


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
    path.write_text(text)
    evidence = lambda fields: {field: [text] for field in fields}
    answers = {
        'ProblemFacts': {'task_type': 'regression', 'target_column': 'yield_value',
                         'evidence': evidence(['task_type', 'target_column'])},
        'DataFacts': {'train_file': 'samples.psv', 'test_file': 'query.jsonl', 'id_column': 'record',
                      'evidence': evidence(['train_file', 'test_file', 'id_column'])},
        'MetricFacts': {'name': 'custom absolute error + seven', 'direction': 'minimize',
                        'definition': 'mean(abs(y_true-y_pred)) + 7', 'evidence': evidence(['name', 'direction', 'definition'])},
        'OutputFacts': {'submission_file': 'result.tsv', 'columns': ['record', 'estimate', 'comment'],
                        'format': 'TSV', 'evidence': evidence(['submission_file', 'columns', 'format'])},
        'PreparationDraft': {'rationale': 'Read the described formats and use the required ordered split.',
            'data': {'files': {'train': 'samples.psv', 'test': 'query.jsonl'},
                     'target_columns': ['yield_value'], 'id_columns': ['record']},
            'validation': {'strategy': 'ordered_tail', 'rows': 8},
            'metric': {'name': 'custom absolute error + seven', 'direction': 'minimize',
                       'definition': 'mean(abs(y_true-y_pred)) + 7', 'status': 'ready'},
            'output': {'filename': 'result.tsv', 'format': 'TSV', 'columns': ['record', 'estimate', 'comment']},
            'code': PREPARE},
        'Hypothesis': {'rationale': 'Constant baseline', 'expected_effect': 'Exercise the workflow',
            'pipeline': {'data': {'source': 'preparation'}, 'preprocessing': {},
                         'model': {'family': 'constant', 'strategy': 'mean'}, 'training': {}}},
        'Program': {'code': PROGRAM, 'notes': ['The target name is taken from preparation.']},
    }
    return path, answers


class PipelineTests(unittest.IsolatedAsyncioTestCase):
    async def exercise(self, directory, *, repair=False, missing_metric=False, blocked=False, broken_metric=False,
                       invalid_program=False, broken_code=False, execution_timeout=False):
        path, answers = fixture(directory)
        if missing_metric:
            answers['PreparationDraft']['metric']['status'] = 'needs_implementation'
            answers['PreparationDraft']['code'] = PREPARE.replace(
                'return float(np.mean(np.abs(reference[column].to_numpy() - predictions[column].to_numpy())) + 7)',
                'raise NotImplementedError("Metric definition is incomplete")')
        if broken_metric:
            answers['PreparationDraft']['code'] = PREPARE.replace(
                'return float(np.mean(np.abs(reference[column].to_numpy() - predictions[column].to_numpy())) + 7)',
                'raise ValueError("invalid metric definition")')
        if blocked:
            answers['Hypothesis'] = {**answers['Hypothesis'], 'pipeline': {**answers['Hypothesis']['pipeline'], 'model': {'family': 'unavailable_model'}}}
            answers['Program'] = {'blocked_reason': 'The planned model is unavailable.', 'notes': ['No model substitution was made.']}
        if invalid_program:
            answers['Program'] = {'code': ''}
        if broken_code:
            answers['Program']['code'] = PROGRAM.replace('value = float(fit[column].mean())', 'raise ValueError("cannot run")')
        seen, saved_starts = [], []
        run_directory = directory / 'artifacts' / 'run'
        def handler(request):
            body = json.loads(request.content)
            title = json.loads(body['messages'][0]['content'].split('Schema: ', 1)[1])['title']
            answer = answers[title]
            if title == 'Program':
                saved = (run_directory / 'nodes/node_000/start.json').read_bytes()
                saved_starts.append(saved)
                self.assertTrue(all(value == saved for value in saved_starts))
                context = json.loads(body['messages'][1]['content'].rsplit('\n', 1)[1])
                self.assertEqual(context['start'], json.loads(saved))
                self.assertIn('task_text', context)
                self.assertIn('preparation_code', context)
                self.assertEqual(context['start']['context']['installed_libraries'], context['installed_libraries'])
            if title == 'Program' and repair and title not in seen:
                answer = {**answers['Program'], 'code': PROGRAM.replace('value = float(fit[column].mean())', 'raise ValueError("repair me")')}
            elif title == 'Program' and repair:
                self.assertIn('repair me', body['messages'][-1]['content'])
            seen.append(title)
            return httpx.Response(200, json={'id': 'test', 'object': 'chat.completion', 'created': 0,
                'model': 'model', 'choices': [{'index': 0, 'finish_reason': 'stop',
                'message': {'role': 'assistant', 'content': json.dumps(answer)}}]})
        def client(**kwargs):
            return AsyncOpenAI(api_key='fake-secret', base_url='https://qwen.example/v1', max_retries=0,
                               http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
        async def launch(arguments, *args):
            if execution_timeout and '--start' in arguments:
                raise TimeoutError('Experiment time limit')
            return await process(arguments, *args)
        with patch('alfa_agent.agents.AsyncOpenAI', side_effect=client), patch('alfa_agent.runner.process', side_effect=launch):
            result = await run(path, Options(time_budget=40, experiment_timeout=12, output_dir=run_directory, max_nodes=1),
                               Settings('fake-secret', 'https://qwen.example/v1', 'model'))
        return result, seen

    async def test_new_names_formats_custom_metric_and_three_column_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (root, report), seen = await self.exercise(directory)
            self.assertEqual(report.status, 'ready', report.errors)
            self.assertEqual(len(seen), 7)
            prep = load(root / 'preparation.json', Preparation)
            self.assertEqual(prep.profile['train']['rows'], 40)
            self.assertEqual(prep.profile['test']['rows'], 5)
            self.assertEqual(prep.profile['train']['columns']['record']['dtype'], 'object')
            self.assertEqual(prep.validation['validation_rows'], 8)
            self.assertEqual(prep.validation['fit_rows'], 32)
            node = root / 'nodes/node_000'
            start, finish = load(node / 'start.json', Start), load(node / 'finish.json', Finish)
            self.assertEqual(start.preparation_sha256, file_hash(root / 'preparation.json'))
            self.assertEqual(start.schema_version, 3)
            self.assertEqual(start.implementation, 'experiment.py')
            self.assertEqual(finish.start_sha256, file_hash(node / 'start.json'))
            self.assertEqual(finish.code_sha256, file_hash(node / 'experiment.py'))
            self.assertEqual(finish.notes, ['The target name is taken from preparation.'])
            self.assertAlmostEqual(finish.value, 47.)  # Fit mean=31; validation target mean=71; +7.
            self.assertEqual(finish.evaluation_status, 'completed')
            submission = pd.read_csv(node / 'result.tsv', sep='\t', dtype={'record': str})
            self.assertEqual(list(submission), ['record', 'estimate', 'comment'])
            self.assertEqual(list(submission.record), ['0100', '0101', '0102', '0103', '0104'])
            self.assertEqual((directory / 'result.tsv').read_bytes(), (node / 'result.tsv').read_bytes())
            self.assertFalse((root / 'task.json').exists())
            self.assertFalse((root / 'nodes/node_001').exists())
            self.assertNotIn('fake-secret', (root / 'report.json').read_text())

    async def test_code_repair_preserves_the_already_saved_start(self):
        with tempfile.TemporaryDirectory() as temporary:
            (root, report), seen = await self.exercise(Path(temporary), repair=True)
            self.assertEqual(report.status, 'ready', report.errors)
            self.assertEqual(seen.count('Program'), 2)
            self.assertTrue(any('code repair' in warning for warning in report.warnings))
            self.assertNotIn('repair me', (root / 'nodes/node_000/experiment.py').read_text())

    async def test_unimplemented_metric_keeps_predictions_and_never_invents_score(self):
        with tempfile.TemporaryDirectory() as temporary:
            (root, report), _ = await self.exercise(Path(temporary), missing_metric=True)
            self.assertEqual(report.status, 'partial', report.errors)
            self.assertEqual(load(root / 'preparation.json', Preparation).status, 'partial')
            finish = load(root / 'nodes/node_000/finish.json', Finish)
            self.assertEqual(finish.status, 'completed')
            self.assertIsNone(finish.value)
            self.assertEqual(finish.evaluation_status, 'unavailable')
            self.assertTrue((root / 'nodes/node_000/result.tsv').is_file())
            self.assertFalse((root / 'best.json').exists())

    async def test_blocked_implementation_preserves_baker_plan_and_records_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            (root, report), seen = await self.exercise(Path(temporary), blocked=True)
            self.assertEqual(report.status, 'failed')
            node = root / 'nodes/node_000'
            start, finish = load(node / 'start.json', Start), load(node / 'finish.json', Finish)
            self.assertEqual(start.hypothesis.pipeline.model['family'], 'unavailable_model')
            self.assertEqual((finish.status, finish.stage), ('failed', 'implementation'))
            self.assertIn('unavailable', finish.error)
            self.assertIsNone(finish.code_sha256)
            self.assertEqual(finish.notes, ['No model substitution was made.'])
            self.assertEqual(seen.count('Program'), 1)
            self.assertFalse((node / 'experiment.py').exists())

    async def test_invalid_implementation_response_still_has_finish(self):
        with tempfile.TemporaryDirectory() as temporary:
            (root, report), seen = await self.exercise(Path(temporary), invalid_program=True)
            self.assertEqual(report.status, 'failed')
            finish = load(root / 'nodes/node_000/finish.json', Finish)
            self.assertEqual(finish.stage, 'implementation')
            self.assertEqual(seen.count('Program'), 2)
            self.assertIsNone(finish.code_sha256)

    async def test_failed_check_records_attempted_code_and_stops_after_one_repair(self):
        with tempfile.TemporaryDirectory() as temporary:
            (root, report), seen = await self.exercise(Path(temporary), broken_code=True)
            self.assertEqual(report.status, 'failed')
            node = root / 'nodes/node_000'
            finish = load(node / 'finish.json', Finish)
            self.assertEqual((finish.status, finish.stage), ('failed', 'check'))
            self.assertEqual(finish.code_sha256, file_hash(node / 'experiment.py'))
            self.assertEqual(seen.count('Program'), 2)
            self.assertFalse((root / 'best.json').exists())

    async def test_execution_timeout_has_phase_plan_hash_and_code_hash(self):
        with tempfile.TemporaryDirectory() as temporary:
            (root, report), _ = await self.exercise(Path(temporary), execution_timeout=True)
            self.assertEqual(report.status, 'failed')
            node = root / 'nodes/node_000'
            finish = load(node / 'finish.json', Finish)
            self.assertEqual((finish.status, finish.stage), ('timeout', 'execution'))
            self.assertEqual(finish.start_sha256, file_hash(node / 'start.json'))
            self.assertEqual(finish.code_sha256, file_hash(node / 'experiment.py'))

    async def test_changed_inputs_are_rejected_without_touching_completed_result(self):
        from alfa_agent.worker import smoke
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (root, report), _ = await self.exercise(directory)
            self.assertEqual(report.status, 'ready', report.errors)
            finish_path = root / 'nodes/node_000/finish.json'
            original = finish_path.read_bytes()
            with (directory / 'samples.psv').open('a') as stream:
                stream.write('9999|10|20\n')
            with self.assertRaisesRegex(ValueError, 'Input files changed'):
                smoke(root / 'nodes/node_000', root / 'preparation.json')
            self.assertEqual(finish_path.read_bytes(), original)

    async def test_execution_rejects_plan_or_code_changed_after_the_check(self):
        from alfa_agent.worker import evaluate
        with tempfile.TemporaryDirectory() as temporary:
            (root, report), _ = await self.exercise(Path(temporary))
            self.assertEqual(report.status, 'ready', report.errors)
            node = root / 'nodes/node_000'
            start_path, code_path = node / 'start.json', node / 'experiment.py'
            start_bytes, code_bytes = start_path.read_bytes(), code_path.read_bytes()
            hashes = {'start_sha256': file_hash(start_path), 'code_sha256': file_hash(code_path)}
            for path, content, message in ((start_path, start_bytes + b'\n', 'Saved plan changed'),
                                           (code_path, code_bytes + b'\n# changed\n', 'Experiment source changed')):
                with self.subTest(path=path.name):
                    path.write_bytes(content)
                    result = evaluate(start_path, lambda *args: self.fail('Changed artifacts must not execute'), **hashes)
                    self.assertEqual(result.status, 'failed')
                    self.assertIn(message, result.error)
                    start_path.write_bytes(start_bytes)
                    code_path.write_bytes(code_bytes)

    async def test_metric_exception_keeps_real_predictions_without_fake_score(self):
        with tempfile.TemporaryDirectory() as temporary:
            (root, report), _ = await self.exercise(Path(temporary), broken_metric=True)
            self.assertEqual(report.status, 'partial', report.errors)
            finish = load(root / 'nodes/node_000/finish.json', Finish)
            self.assertEqual(finish.status, 'completed')
            self.assertEqual(finish.evaluation_status, 'failed')
            self.assertIsNone(finish.value)
            self.assertTrue((root / 'nodes/node_000/result.tsv').is_file())
            self.assertTrue(any('invalid metric definition' in w for w in report.warnings))

    async def test_process_timeout_and_clean_environment(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch.dict(os.environ, {'OPENAI_API_KEY': 'secret-must-not-reach-worker', 'OTHER_SECRET': 'hidden'}):
                await process(['-c', 'import os; print(dict(os.environ))'], root, root / 'env.log', time.monotonic()+5, 2)
            output = (root / 'env.log').read_text()
            self.assertNotIn('secret-must-not-reach-worker', output)
            self.assertNotIn('OTHER_SECRET', output)
            with self.assertRaises(TimeoutError):
                await process(['-c', 'import time; time.sleep(5)'], root, root / 'timeout.log', time.monotonic()+5, .1)

    def test_code_interface_and_import_guard(self):
        for code in ('def fit_predict(:\n    pass', 'import subprocess\n' + PROGRAM, 'print("top-level execution")\n' + PROGRAM,
                     PROGRAM.replace('fit_predict', 'wrong_function')):
            with self.subTest(code=code[:30]), self.assertRaises(ValueError):
                check_code(code, ('fit_predict',))


class TreeTests(unittest.IsolatedAsyncioTestCase):
    async def exercise(self, directory, *, blocked=False, stop_api=False, stop_time=False):
        path, answers = fixture(directory)
        root = directory / 'artifacts' / 'tree'
        plans, code = [], []
        for index, (family, value) in enumerate((('constant', 'float(fit[column].mean())'),
                                                ('linear', None), ('constant', '0.'), ('constant', '50.'))):
            plans.append({**answers['Hypothesis'], 'pipeline': {**answers['Hypothesis']['pipeline'],
                          'model': {'family': family, 'value': value}}})
            if index == 1:
                code.append('from sklearn.linear_model import LinearRegression\nimport pandas as pd\n'
                    'def fit_predict(fit, validation, test, preparation):\n'
                    '    column = preparation["data"]["target_columns"][0]\n'
                    '    model = LinearRegression().fit(fit[["signal"]], fit[column])\n'
                    '    return pd.DataFrame({column:model.predict(validation[["signal"]])}), pd.DataFrame({column:model.predict(test[["signal"]])})\n')
            else:
                code.append(PROGRAM.replace('float(fit[column].mean())', value))
        baker_calls, starts = [], {}
        def handler(request):
            body = json.loads(request.content)
            title = json.loads(body['messages'][0]['content'].split('Schema: ', 1)[1])['title']
            context = json.loads(body['messages'][1]['content'].rsplit('\n', 1)[1])
            if title == 'Hypothesis':
                index = len(baker_calls)
                if index:
                    self.assertEqual(context['delta'], json.loads((root / 'delta.json').read_text()))
                    self.assertEqual(context['parent']['start']['node_id'], 'node_000' if index == 1 else 'node_001')
                    self.assertIn('code', context['parent'])
                baker_calls.append(index)
                if stop_api and index == 2:
                    return httpx.Response(503, json={'error': {'message': 'Unavailable'}})
                answer = plans[index]
            elif title == 'Program':
                node_id = context['start']['node_id']
                index = int(node_id.split('_')[1])
                saved = (root / 'nodes' / node_id / 'start.json').read_bytes()
                starts[node_id] = saved
                self.assertEqual(context['start'], json.loads(saved))
                answer = {'blocked_reason': 'Candidate failed', 'notes': ['Keep searching']} if blocked and index == 2 else {'code': code[index], 'notes': [f'Candidate {index}']}
            else:
                answer = answers[title]
            return httpx.Response(200, json={'id': 'test', 'object': 'chat.completion', 'created': 0,
                'model': 'model', 'choices': [{'index': 0, 'finish_reason': 'stop',
                'message': {'role': 'assistant', 'content': json.dumps(answer)}}]})
        def client(**kwargs):
            return AsyncOpenAI(api_key='fake-secret', base_url='https://qwen.example/v1', max_retries=0,
                               http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
        # Expire only the controller's clock after a measured winner, without sleeping or changing asyncio's clock.
        from types import SimpleNamespace
        offset = [0.]
        def record(*args):
            result = summarize(*args)
            if stop_time and result[0].node_id == 'node_001':
                offset[0] = 100.
            return result
        with patch('alfa_agent.agents.AsyncOpenAI', side_effect=client), patch('alfa_agent.runner.summarize', side_effect=record), \
             patch('alfa_agent.runner.time', SimpleNamespace(monotonic=lambda: time.monotonic() + offset[0])):
            directory, report = await run(path, Options(time_budget=60, experiment_timeout=12, output_dir=root, max_nodes=4),
                                          Settings('fake-secret', 'https://qwen.example/v1', 'model'))
        for node_id, saved in starts.items():
            self.assertEqual((root / 'nodes' / node_id / 'start.json').read_bytes(), saved)
        return directory, report

    async def test_feedback_creates_branches_and_publishes_best_instead_of_latest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, report = await self.exercise(Path(temporary))
            self.assertEqual((report.status, report.stop_reason, report.completed_nodes), ('ready', 'node_limit', 4), report.errors)
            delta = load(root / 'delta.json', Delta)
            self.assertEqual([n.parent_id for n in delta.nodes], [None, 'node_000', 'node_001', 'node_001'])
            self.assertEqual(delta.best_node_id, 'node_001')
            self.assertEqual([round(n.result.value) for n in delta.nodes], [47, 7, 78, 28])
            self.assertAlmostEqual(delta.nodes[1].comparisons[0].improvement, 40.)
            self.assertEqual(len(delta.nodes[3].comparisons), 3)
            self.assertTrue(any(c.path == 'pipeline.model.family' for c in delta.nodes[1].comparisons[0].changes))
            for node in delta.nodes:
                self.assertEqual(node.result.node_id, node.node_id)
                self.assertEqual(json.loads((root / 'nodes' / node.node_id / 'delta.json').read_text()), node.model_dump(mode='json'))
            best = json.loads((root / 'best.json').read_text())
            self.assertEqual(best['node_id'], 'node_001')
            self.assertEqual((Path(temporary) / 'result.tsv').read_bytes(), (root / best['submission_file']).read_bytes())

    async def test_failed_candidate_has_delta_and_does_not_stop_following_nodes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, report = await self.exercise(Path(temporary), blocked=True)
            self.assertEqual((report.status, report.completed_nodes, report.failed_nodes), ('ready', 3, 1), report.errors)
            delta = load(root / 'delta.json', Delta)
            self.assertEqual(delta.nodes[2].result.stage, 'implementation')
            self.assertTrue(all(not c.comparable and c.improvement is None for c in delta.nodes[2].comparisons))
            self.assertEqual(delta.nodes[3].parent_id, 'node_001')
            self.assertEqual(delta.best_node_id, 'node_001')

    async def test_api_or_budget_stop_preserves_measured_winner(self):
        for mode in ('stop_api', 'stop_time'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temporary:
                root, report = await self.exercise(Path(temporary), **{mode: True})
                self.assertEqual(report.status, 'partial', report.errors)
                self.assertEqual(report.stop_reason, 'llm_error' if mode == 'stop_api' else 'time_budget')
                self.assertEqual(report.completed_nodes, 2)
                best = json.loads((root / 'best.json').read_text())
                self.assertEqual(best['node_id'], 'node_001')
                self.assertEqual((Path(temporary) / 'result.tsv').read_bytes(), (root / best['submission_file']).read_bytes())

    def test_comparisons_handle_maximization_missing_direction_and_different_preparation(self):
        hypothesis = Hypothesis(rationale='test', expected_effect='test', pipeline={'data': {}, 'preprocessing': {}, 'model': {}, 'training': {}})
        metric = Metric(name='custom', direction='maximize', status='ready')
        first = Start(hypothesis=hypothesis, preparation_sha256='a'*64, outputs={}, timeout_seconds=12.)
        result = Finish(status='completed', metric=metric, value=-5., evaluation_status='completed', duration_seconds=0.,
                        preparation_sha256='a'*64, start_sha256='b'*64, code_sha256='c'*64, validation_file='v', submission_file='s')
        _, history = summarize(Delta(preparation_sha256='a'*64, metric=metric), first, result)
        second = first.model_copy(update={'node_id': 'node_001', 'parent_id': 'node_000'})
        row, history = summarize(history, second, result.model_copy(update={'node_id': 'node_001', 'value': -2.}))
        self.assertEqual(row.comparisons[0].improvement, 3.)
        self.assertEqual(history.best_node_id, 'node_001')
        row, _ = summarize(history, second, result.model_copy(update={'preparation_sha256': 'd'*64}))
        self.assertTrue(all(not c.comparable for c in row.comparisons))
        unknown = metric.model_copy(update={'direction': None})
        no_direction = result.model_copy(update={'metric': unknown})
        _, history = summarize(Delta(preparation_sha256='a'*64, metric=unknown), first, no_direction)
        row, history = summarize(history, second, no_direction.model_copy(update={'node_id': 'node_001'}))
        self.assertIsNone(history.best_node_id)
        self.assertEqual(row.comparisons[0].reason, 'Optimization direction unknown')


if __name__ == '__main__':
    unittest.main()

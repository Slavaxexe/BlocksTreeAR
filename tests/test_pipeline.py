"""Mocked Qwen HTTP, real generated programs and real subprocesses on an unseen task."""

import asyncio
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
from zipfile import ZipFile

import httpx
import numpy as np
from openai import AsyncOpenAI
import pandas as pd

from alfa_agent.config import Settings
from alfa_agent.agents import baker_context, implementation_context
from alfa_agent.contracts import Delta, Finish, Hypotheses, Hypothesis, Metric, Preparation, RunReport, Start
from alfa_agent.data import check_code, file_hash, profile
from alfa_agent.runner import Options, process, run, summarize
from alfa_agent.storage import load
from package_solution import build

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
                       invalid_program=False, broken_code=False, execution_timeout=False, full_repair=False,
                       json_failure=False, network_failure=False, ast_failure=False, refused_once=False,
                       refused_after_error=False, max_nodes=1):
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
        if ast_failure:
            answers['Program']['code'] = 'COLUMNS = list()\n' + PROGRAM
        seen, saved_starts = [], []
        run_directory = directory / 'artifacts' / 'run'
        def handler(request):
            body = json.loads(request.content)
            title = json.loads(body['messages'][0]['content'].split('Schema: ', 1)[1])['title']
            if title == 'Program' and seen.count(title) < int(network_failure):
                seen.append(title)
                raise httpx.ConnectError('temporary disconnection', request=request)
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
                self.assertNotIn('extraction', context)
                self.assertEqual(context['runtime_contract']['rows'], {'fit': 32, 'validation': 8, 'test': 5})
            if title == 'Program' and json_failure and title not in seen:
                answer = {'code': ''}
            elif title == 'Program' and refused_once:
                if title not in seen:
                    answer = {'blocked_reason': 'Fit has no features or targets.'}
                else:
                    self.assertEqual(context['last_refusal']['source'], 'model_claim')
                    self.assertFalse(context['last_refusal']['verified'])
                    self.assertNotIn('execution_error', context)
                    self.assertIn('features', context['runtime_contract']['fit'])
            elif title == 'Program' and refused_after_error:
                count = seen.count(title)
                if count == 0:
                    answer = {'code': PROGRAM.replace('value = float(fit[column].mean())', 'raise ValueError("repair me")')}
                elif count == 1:
                    answer = {'blocked_reason': 'Fit has no features or targets.'}
                else:
                    self.assertIn('repair me', context['execution_error'])
                    self.assertIn('repair me', context['previous_code'])
                    self.assertEqual(context['failed_stage'], 'check')
                    self.assertFalse(context['last_refusal']['verified'])
            elif title == 'Program' and repair and seen.count(title) < int(repair):
                answer = {**answers['Program'], 'code': PROGRAM.replace('value = float(fit[column].mean())', 'raise ValueError("repair me")')}
            elif title == 'Program' and repair:
                self.assertIn('repair me', body['messages'][-1]['content'])
            if title == 'Program' and full_repair and seen.count(title):
                self.assertEqual(context['failed_stage'], 'execution')
                self.assertIn('full run failed', context['execution_error'])
            seen.append(title)
            return httpx.Response(200, json={'id': 'test', 'object': 'chat.completion', 'created': 0,
                'model': 'model', 'choices': [{'index': 0, 'finish_reason': 'stop',
                'message': {'role': 'assistant', 'content': json.dumps(answer)}}]})
        def client(**kwargs):
            return AsyncOpenAI(api_key='fake-secret', base_url='https://qwen.example/v1', max_retries=0,
                               http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
        full_attempts = 0
        async def launch(arguments, *args):
            nonlocal full_attempts
            if '--check' in arguments and (broken_code or execution_timeout):
                if broken_code:
                    raise ValueError('cannot run')
                return
            if full_repair and '--start' in arguments:
                full_attempts += 1
                if full_attempts == 1:
                    raise ValueError('full run failed')
            if execution_timeout and '--start' in arguments:
                raise TimeoutError('Experiment time limit')
            return await process(arguments, *args)
        with patch('alfa_agent.agents.AsyncOpenAI', side_effect=client), patch('alfa_agent.runner.process', side_effect=launch):
            result = await run(path, Options(time_budget=40, experiment_timeout=12, output_dir=run_directory, max_nodes=max_nodes),
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
            (root, report), seen = await self.exercise(Path(temporary), repair=3)
            self.assertEqual(report.status, 'ready', report.errors)
            self.assertEqual(seen.count('Program'), 4)
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
            self.assertEqual(seen.count('Program'), 20)
            self.assertFalse((node / 'experiment.py').exists())

    async def test_unverified_refusal_is_reconsidered_without_changing_start(self):
        with tempfile.TemporaryDirectory() as temporary:
            (root, report), seen = await self.exercise(Path(temporary), refused_once=True)
            self.assertEqual(report.status, 'ready', report.errors)
            self.assertEqual(seen.count('Program'), 2)
            finish = load(root / 'nodes/node_000/finish.json', Finish)
            self.assertEqual((finish.status, finish.value), ('completed', 47.))

    async def test_failed_baseline_stops_before_any_search_hypothesis(self):
        with tempfile.TemporaryDirectory() as temporary:
            (root, report), seen = await self.exercise(Path(temporary), blocked=True, max_nodes=4)
            self.assertEqual((report.status, report.stop_reason), ('failed', 'baseline_failed'))
            self.assertEqual(seen.count('Program'), 20)
            self.assertNotIn('Hypotheses', seen)
            self.assertEqual([p.name for p in (root / 'nodes').iterdir()], ['node_000'])
            self.assertEqual(report.failed_nodes, 1)
            self.assertTrue(report.errors)

    async def test_refusal_does_not_overwrite_the_last_observed_error_or_code(self):
        with tempfile.TemporaryDirectory() as temporary:
            (root, report), seen = await self.exercise(Path(temporary), refused_after_error=True)
            self.assertEqual(report.status, 'ready', report.errors)
            self.assertEqual(seen.count('Program'), 3)
            self.assertEqual(load(root / 'nodes/node_000/finish.json', Finish).value, 47.)

    async def test_broken_baseline_scorer_stops_search_but_keeps_predictions(self):
        with tempfile.TemporaryDirectory() as temporary:
            (root, report), seen = await self.exercise(Path(temporary), broken_metric=True, max_nodes=4)
            self.assertEqual((report.status, report.stop_reason), ('partial', 'baseline_failed'))
            self.assertNotIn('Hypotheses', seen)
            self.assertTrue((root / 'nodes/node_000/result.tsv').is_file())

    async def test_invalid_implementation_response_still_has_finish(self):
        with tempfile.TemporaryDirectory() as temporary:
            (root, report), seen = await self.exercise(Path(temporary), invalid_program=True)
            self.assertEqual(report.status, 'failed')
            finish = load(root / 'nodes/node_000/finish.json', Finish)
            self.assertEqual(finish.stage, 'implementation')
            self.assertEqual(seen.count('Program'), 2)
            self.assertIsNone(finish.code_sha256)

    async def test_request_cap_includes_json_repairs_and_records_last_attempted_code(self):
        with tempfile.TemporaryDirectory() as temporary:
            (root, report), seen = await self.exercise(Path(temporary), broken_code=True, json_failure=True)
            self.assertEqual(report.status, 'failed')
            node = root / 'nodes/node_000'
            finish = load(node / 'finish.json', Finish)
            self.assertEqual((finish.status, finish.stage), ('failed', 'check'))
            self.assertEqual(finish.code_sha256, file_hash(node / 'experiment.py'))
            self.assertEqual(seen.count('Program'), 20)
            self.assertEqual(sum(call.role == 'implementer' for call in report.llm_calls), 20)
            self.assertEqual(sum(call.status == 'invalid_json' for call in report.llm_calls), 1)
            self.assertFalse((root / 'best.json').exists())

    async def test_full_execution_error_can_be_repaired_in_the_same_node(self):
        with tempfile.TemporaryDirectory() as temporary:
            (root, report), seen = await self.exercise(Path(temporary), full_repair=True)
            self.assertEqual(report.status, 'ready', report.errors)
            self.assertEqual(seen.count('Program'), 2)
            self.assertEqual(load(root / 'nodes/node_000/finish.json', Finish).stage, 'execution')

    async def test_connection_errors_retry_within_three_request_cap(self):
        for failures in (1, 2, 3):
            with self.subTest(failures=failures), tempfile.TemporaryDirectory() as temporary:
                (_, report), seen = await self.exercise(Path(temporary), network_failure=failures)
                self.assertEqual(report.status, 'failed' if failures == 3 else 'ready', report.errors)
                self.assertEqual(seen.count('Program'), min(failures + 1, 3))
                calls = [call for call in report.llm_calls if call.role == 'implementer']
                self.assertEqual([call.status for call in calls], ['api_error'] * failures + (['completed'] if failures < 3 else []))

    async def test_rejected_source_and_actionable_check_error_are_preserved(self):
        with tempfile.TemporaryDirectory() as temporary:
            (root, report), seen = await self.exercise(Path(temporary), ast_failure=True)
            self.assertEqual(report.status, 'failed')
            node = root / 'nodes/node_000'
            finish = load(node / 'finish.json', Finish)
            self.assertEqual((finish.stage, seen.count('Program')), ('check', 20))
            self.assertEqual(finish.code_sha256, file_hash(node / 'experiment.py'))
            self.assertIn('COLUMNS = list()', (node / 'experiment.py').read_text())
            self.assertIn('Move computed values inside a function', (node / 'execution.log').read_text())

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
            with self.assertRaisesRegex(TimeoutError, 'time limit'):
                await process(['-c', 'import time; time.sleep(5)'], root, root / 'timeout.log', time.monotonic()+5, .1)

    def test_code_interface_and_import_guard(self):
        check_code(PROGRAM.replace('    column = ', '    columns = [name for name in fit.columns]\n    column = '), ('fit_predict',))
        for code in ('def fit_predict(:\n    pass', 'import subprocess\n' + PROGRAM, 'print("top-level execution")\n' + PROGRAM,
                     PROGRAM.replace('fit_predict', 'wrong_function')):
            with self.subTest(code=code[:30]), self.assertRaises(ValueError):
                check_code(code, ('fit_predict',))


class TreeTests(unittest.IsolatedAsyncioTestCase):
    async def exercise(self, directory, *, blocked=False, stop_api=False, stop_time=False, max_nodes=4,
                       wrong_batch=False, duplicate_batch=False, repair_both=False, planning_race=False, retry_race=False,
                       stuck_explore=False):
        path, answers = fixture(directory)
        root = directory / 'artifacts' / 'tree'
        plans, code = [], []
        candidates = [('constant', 'float(fit[column].mean())'), ('linear', None), ('constant', '0.'),
                      ('constant', '50.')] + [('constant', f'{50 + i}.') for i in range(4, max_nodes)]
        for index, (family, value) in enumerate(candidates):
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
        starts, requests, wrong_sent, rejected = {}, {}, False, 0
        gate, second_recorded, pair_ready = asyncio.Event(), asyncio.Event(), asyncio.Event()
        first_started, first_recorded = asyncio.Event(), asyncio.Event()
        simultaneous, peak = 0, 0
        if planning_race or retry_race:
            plans[2], code[2] = plans[1], code[1]
            plans[1], code[1] = plans[3], code[3]
            plans[3] = {**plans[3], 'pipeline': {**plans[3]['pipeline'], 'model': {'family': 'constant', 'value': '49.'}}}
            code[3] = PROGRAM.replace('float(fit[column].mean())', '49.')
        async def handler(request):
            nonlocal wrong_sent, rejected, simultaneous, peak
            body = json.loads(request.content)
            title = json.loads(body['messages'][0]['content'].split('Schema: ', 1)[1])['title']
            context = json.loads(body['messages'][1]['content'].rsplit('\n', 1)[1])
            if title in {'Hypothesis', 'Hypotheses'}:
                self.assertEqual(context['runtime_contract']['rows'], {'fit': 32, 'validation': 8, 'test': 5})
                index = len(list((root / 'nodes').glob('*/start.json')))
                if index:
                    measured = json.loads((root / 'delta.json').read_text())['nodes']
                    self.assertNotIn('delta', context)
                    self.assertNotIn('extraction', context)
                    for row, original in zip(context['history'], measured[:len(context['history'])]):
                        self.assertEqual(row['node_id'], original['node_id'])
                        self.assertEqual({**context['parent']['pipeline'], **row['pipeline_changes']}, original['hypothesis']['pipeline'])
                        self.assertEqual(row['score'], original['result']['value'])
                        self.assertNotIn('rationale', row)
                    self.assertIn('code', context['parent'])
                    if stuck_explore and context['requested_purposes'] == ['explore']:
                        await asyncio.wait_for(first_started.wait(), 8)
                        await asyncio.wait_for(first_recorded.wait(), 8)
                        self.assertTrue((root / 'nodes/node_001/start.json').exists())
                        if context['correction']['explore']['repeats'] >= 2:
                            self.assertEqual(context['correction']['explore']['action'], 'new_direction')
                            self.assertEqual(len(body['messages']), 2)
                    if index == 3 and not repair_both and not stuck_explore:
                        if retry_race and wrong_sent:
                            self.assertEqual(context['parent']['node_id'], 'node_002')
                            self.assertEqual(len(context['history']), 3)
                            self.assertEqual(context['pending'], [])
                            self.assertTrue(body['messages'][1]['content'].startswith('PREVIOUS RESPONSE REJECTED:'))
                            self.assertIn('improve=node_002', body['messages'][-1]['content'])
                        else:
                            self.assertEqual(context['parent']['node_id'], 'node_001')
                            if not stop_api:
                                self.assertIn('node_002', [p['node_id'] for p in context['pending']])
                            if planning_race or retry_race:
                                gate.set()
                                await asyncio.wait_for(second_recorded.wait(), 8)
                if stop_api and index == 3:
                    gate.set()
                    return httpx.Response(503, json={'error': {'message': 'Unavailable'}})
                purposes = context['requested_purposes'] if index else []
                proposals = [{'purpose': purpose,
                    'parent_id': context['parent']['node_id'] if purpose == 'improve' else 'node_000',
                    'hypothesis': plans[index + offset]} for offset, purpose in enumerate(purposes)]
                if stuck_explore and index:
                    answer = {'hypotheses': [{**p, 'hypothesis': plans[0]} if p['purpose'] == 'explore' else p for p in proposals]}
                elif retry_race and index == 3 and not wrong_sent:
                    wrong_sent = True
                    answer = {'hypotheses': [{**p, 'hypothesis': plans[2]} for p in proposals]}
                elif title == 'Hypotheses' and rejected < int(duplicate_batch):
                    if rejected:
                        self.assertTrue(body['messages'][1]['content'].startswith('PREVIOUS RESPONSE REJECTED:'))
                        for message in (body['messages'][1], body['messages'][-1]):
                            self.assertIn('improve=node_000', message['content'])
                            self.assertIn('explore=node_000', message['content'])
                        self.assertEqual(len(body['messages']), 2 if rejected >= 2 else 4)
                        if rejected >= 2:
                            self.assertTrue(all(c['action'] == 'new_direction' for c in context['correction'].values()))
                    rejected += 1
                    answer = {'hypotheses': [{**p, 'hypothesis': plans[0]} for p in proposals]}
                elif title == 'Hypotheses' and wrong_batch and not wrong_sent:
                    wrong_sent = True
                    answer = {'hypotheses': proposals[:-1]}
                else:
                    answer = {'hypotheses': proposals} if index else plans[index]
            elif title == 'Program':
                node_id = context['start']['node_id']
                index = int(node_id.split('_')[1])
                saved = (root / 'nodes' / node_id / 'start.json').read_bytes()
                starts[node_id] = saved
                self.assertEqual(context['start'], json.loads(saved))
                self.assertNotIn('delta', context)
                self.assertNotIn('parent', context)
                self.assertNotIn('extraction', context)
                if index:
                    self.assertEqual(context['parent_reference']['node_id'], context['start']['parent_id'])
                    self.assertEqual(set(context['parent_reference']), {'node_id', 'pipeline', 'code'})
                    self.assertNotIn("if __name__", context['parent_reference']['code'])
                if index in (1, 2) and not wrong_batch and not stuck_explore:
                    self.assertTrue((root / 'nodes/node_001/start.json').exists())
                    self.assertTrue((root / 'nodes/node_002/start.json').exists())
                state = json.loads((root / 'report.json').read_text())
                self.assertLessEqual(len(state['running_nodes']), 2)
                requests[node_id] = requests.get(node_id, 0) + 1
                simultaneous += 1
                peak = max(peak, simultaneous)
                try:
                    if index == 1:
                        first_started.set()
                    if index in (1, 2) and requests[node_id] == 1 and not wrong_batch and not stuck_explore:
                        if simultaneous == 2:
                            pair_ready.set()
                        await asyncio.wait_for(pair_ready.wait(), 8)
                    if index == 2 and not repair_both and not stuck_explore:
                        await gate.wait()  # This older sibling waits until node_003 has actually finished.
                finally:
                    simultaneous -= 1
                answer = {'blocked_reason': 'Candidate failed', 'notes': ['Keep searching']} if blocked and index == 2 else {'code': code[index], 'notes': [f'Candidate {index}']}
                if repair_both and index in (1, 2):
                    if requests[node_id] == 1 and index == 1:
                        answer = {'code': ''}
                    elif requests[node_id] == 1:
                        raise httpx.ConnectError('temporary disconnection', request=request)
                    else:
                        answer = {'code': 'COLUMNS = list()\n' + code[index]}
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
            if result[0].node_id == 'node_001':
                first_recorded.set()
            if result[0].node_id == 'node_003':
                gate.set()
            if result[0].node_id == 'node_002':
                second_recorded.set()
            if stop_time and result[0].node_id == 'node_001':
                offset[0] = 100.
            return result
        real_sleep = asyncio.sleep
        async def backoff(seconds):
            await real_sleep(0)  # Keep scheduling races, without waiting for mock HTTP retries.
        with patch('alfa_agent.agents.AsyncOpenAI', side_effect=client), patch('alfa_agent.runner.summarize', side_effect=record), \
             patch('alfa_agent.runner.time', SimpleNamespace(monotonic=lambda: time.monotonic() + offset[0])), \
             patch('alfa_agent.agents.asyncio.sleep', side_effect=backoff):
            directory, report = await run(path, Options(time_budget=90, experiment_timeout=12, output_dir=root, max_nodes=max_nodes),
                                          Settings('fake-secret', 'https://qwen.example/v1', 'model'))
        for node_id, saved in starts.items():
            self.assertEqual((root / 'nodes' / node_id / 'start.json').read_bytes(), saved)
        self.assertLessEqual(peak, 2)
        if len(starts) > 1 and not stuck_explore:
            self.assertGreaterEqual(peak, 2)
        if repair_both:
            self.assertEqual([requests['node_001'], requests['node_002']], [20, 20])
        return directory, report

    async def test_feedback_creates_branches_and_publishes_best_instead_of_latest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, report = await self.exercise(Path(temporary))
            self.assertEqual((report.status, report.stop_reason, report.completed_nodes), ('ready', 'node_limit', 4), report.errors)
            delta = load(root / 'delta.json', Delta)
            by_id = {n.node_id:n for n in delta.nodes}
            self.assertEqual([n.parent_id for n in sorted(delta.nodes, key=lambda n:n.node_id)], [None, 'node_000', 'node_000', 'node_001'])
            self.assertLess([n.node_id for n in delta.nodes].index('node_003'), [n.node_id for n in delta.nodes].index('node_002'))
            self.assertEqual(delta.best_node_id, 'node_001')
            self.assertEqual([round(n.result.value) for n in sorted(delta.nodes, key=lambda n:n.node_id)], [47, 7, 78, 28])
            self.assertAlmostEqual(delta.nodes[1].comparisons[0].improvement, 40.)
            self.assertEqual(len(by_id['node_002'].comparisons), 3)
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
            failed = next(n for n in delta.nodes if n.node_id == 'node_002')
            self.assertEqual(failed.result.stage, 'implementation')
            self.assertTrue(all(not c.comparable and c.improvement is None for c in failed.comparisons))
            self.assertEqual(next(n for n in delta.nodes if n.node_id == 'node_003').parent_id, 'node_001')
            self.assertEqual(delta.best_node_id, 'node_001')

    async def test_api_or_budget_stop_preserves_measured_winner(self):
        for mode in ('stop_api', 'stop_time'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temporary:
                root, report = await self.exercise(Path(temporary), **{mode: True})
                self.assertEqual(report.status, 'partial', report.errors)
                self.assertEqual(report.stop_reason, 'llm_error' if mode == 'stop_api' else 'time_budget')
                self.assertEqual(report.completed_nodes, 3 if mode == 'stop_api' else 2)
                if mode == 'stop_api':
                    self.assertTrue(all(c.error == 'HTTP 503' for c in report.llm_calls if c.status == 'api_error'))
                self.assertEqual(len(load(root / 'delta.json', Delta).nodes), 3)
                best = json.loads((root / 'best.json').read_text())
                self.assertEqual(best['node_id'], 'node_001')
                self.assertEqual((Path(temporary) / 'result.tsv').read_bytes(), (root / best['submission_file']).read_bytes())

    async def test_ten_node_cap_batches_and_final_single_hypothesis(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, report = await self.exercise(Path(temporary), max_nodes=10, wrong_batch=True)
            self.assertEqual((report.status, report.completed_nodes, report.failed_nodes), ('ready', 10, 0), report.errors)
            delta = load(root / 'delta.json', Delta)
            self.assertEqual(len(delta.nodes), 10)
            self.assertEqual(delta.best_node_id, 'node_001')
            self.assertFalse((root / 'nodes/node_010').exists())
            starts = [load(root / 'nodes' / f'node_{index:03d}' / 'start.json', Start) for index in range(10)]
            self.assertEqual([s.context['purpose'] for s in starts], ['baseline'] + ['improve', 'explore']*4 + ['improve'])
            for index in (1, 3, 5, 7):
                if index != 1:
                    self.assertEqual(starts[index].context['delta_nodes'], starts[index+1].context['delta_nodes'])
            self.assertEqual([starts[i].parent_id for i in (2, 4, 6, 8)], ['node_000']*4)
            calls = [call for call in report.llm_calls if call.role == 'baker']
            self.assertEqual(sum(call.status == 'completed' for call in calls), 6)
            self.assertEqual(sum(call.status == 'invalid_json' for call in calls), 1)

    async def test_parallel_nodes_have_independent_twenty_request_budgets(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, report = await self.exercise(Path(temporary), repair_both=True)
            self.assertEqual((report.status, report.completed_nodes, report.failed_nodes), ('ready', 2, 2), report.errors)
            for node_id in ('node_001', 'node_002'):
                calls = [c for c in report.llm_calls if c.node_id == node_id]
                self.assertEqual(len(calls), 20)
            self.assertEqual(report.running_nodes, [])
            self.assertEqual(report.queued_nodes, [])
            self.assertFalse(report.planning)

    async def test_baker_repairs_duplicate_proposals_with_specific_feedback(self):
        with tempfile.TemporaryDirectory() as temporary:
            _, report = await self.exercise(Path(temporary), duplicate_batch=True)
            self.assertEqual((report.status, report.completed_nodes), ('ready', 4), report.errors)
            rejected = [c for c in report.llm_calls if c.role == 'baker' and c.status == 'invalid_json']
            self.assertEqual(len(rejected), 1)
            self.assertIn('improve=node_000', rejected[0].error)
            self.assertIn('explore=node_000', rejected[0].error)

    async def test_baker_can_recover_on_the_tenth_request_without_growing_history(self):
        with tempfile.TemporaryDirectory() as temporary:
            _, report = await self.exercise(Path(temporary), duplicate_batch=9)
            self.assertEqual((report.status, report.completed_nodes), ('ready', 4), report.errors)
            rejected = [c for c in report.llm_calls if c.role == 'baker' and c.status == 'invalid_json']
            self.assertEqual([c.attempt for c in rejected], list(range(1, 10)))
            self.assertTrue(any(c.role == 'baker' and c.status == 'completed' and c.attempt == 10 for c in report.llm_calls))

    async def test_baker_stops_after_ten_rejections_and_preserves_baseline(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, report = await self.exercise(Path(temporary), duplicate_batch=10)
            self.assertEqual((report.status, report.stop_reason, report.completed_nodes), ('partial', 'llm_error', 1), report.errors)
            self.assertEqual(len([c for c in report.llm_calls if c.status == 'invalid_json']), 10)
            self.assertEqual(len(load(root / 'delta.json', Delta).nodes), 1)
            self.assertEqual(json.loads((root / 'best.json').read_text())['node_id'], 'node_000')
            self.assertFalse((root / 'nodes/node_001').exists())
            self.assertFalse(report.planning)

    async def test_stuck_explore_does_not_block_improve_or_stop_the_tree(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, report = await self.exercise(Path(temporary), stuck_explore=True)
            self.assertEqual((report.status, report.stop_reason, report.completed_nodes), ('ready', 'node_limit', 4), report.errors)
            starts = [load(p, Start) for p in sorted((root / 'nodes').glob('*/start.json'))]
            self.assertEqual([s.context['purpose'] for s in starts], ['baseline', 'improve', 'improve', 'improve'])
            self.assertEqual(len([c for c in report.llm_calls if c.role == 'baker' and c.status == 'invalid_json']), 20)
            self.assertTrue(any('explore' in warning for warning in report.warnings))
            self.assertFalse(report.errors)

    async def test_baker_retry_refreshes_leader_results_pending_and_validation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, report = await self.exercise(Path(temporary), retry_race=True)
            self.assertEqual((report.status, report.completed_nodes), ('ready', 4), report.errors)
            start = load(root / 'nodes/node_003/start.json', Start)
            self.assertEqual((start.parent_id, start.context['delta_nodes']), ('node_002', 3))
            rejected = [c for c in report.llm_calls if c.role == 'baker' and c.status == 'invalid_json']
            self.assertEqual(len(rejected), 1)
            self.assertIn('improve=node_002', rejected[0].error)

    async def test_results_are_merged_while_baker_uses_an_older_snapshot(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, report = await self.exercise(Path(temporary), planning_race=True)
            self.assertEqual((report.status, report.completed_nodes), ('ready', 4), report.errors)
            delta = load(root / 'delta.json', Delta)
            self.assertEqual(delta.best_node_id, 'node_002')
            start = load(root / 'nodes/node_003/start.json', Start)
            self.assertEqual((start.parent_id, start.context['delta_nodes']), ('node_001', 2))
            self.assertEqual([n.node_id for n in delta.nodes], ['node_000','node_001','node_002','node_003'])

    def test_proposal_roles_and_parents_are_validated_without_model_registry(self):
        hypothesis = {'rationale': 'test', 'expected_effect': 'test',
            'pipeline': {'data': {}, 'preprocessing': {}, 'model': {'custom': 'anything'}, 'training': {}}}
        proposal = {'purpose': 'improve', 'parent_id': 'node_001', 'hypothesis': hypothesis}
        context = {'hypothesis_count': 1, 'parent_ids': ['node_000','node_001'], 'leader_id': 'node_001'}
        self.assertEqual(Hypotheses.model_validate_json(json.dumps({'hypotheses':[proposal]}), context=context).hypotheses[0].parent_id, 'node_001')
        with self.assertRaisesRegex(ValueError, 'improve=node_000'):
            Hypotheses.model_validate_json(json.dumps({'hypotheses':[proposal]}),
                context={**context, 'used_pipelines': {'node_000': hypothesis['pipeline']}})
        for changes in ({'parent_id':'node_999'}, {'parent_id':'node_000'}, {'purpose':'explore'}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                Hypotheses.model_validate_json(json.dumps({'hypotheses':[{**proposal, **changes}]}), context=context)

    def test_valid_proposal_survives_duplicate_or_malformed_sibling(self):
        hypothesis = {'rationale': 'test', 'expected_effect': 'test',
            'pipeline': {'data': {}, 'preprocessing': {}, 'model': {'custom': 'new'}, 'training': {}}}
        improve = {'purpose': 'improve', 'parent_id': 'node_000', 'hypothesis': hypothesis}
        duplicate = {**hypothesis, 'pipeline': {**hypothesis['pipeline'], 'model': {'custom': 'old'}}}
        for sibling in ({'purpose': 'explore', 'parent_id': 'node_000', 'hypothesis': duplicate},
                        {'purpose': 'explore', 'parent_id': 'node_000', 'hypothesis': hypothesis},
                        {'purpose': 'explore', 'hypothesis': {}}):
            rejected = {}
            context = {'accept_partial': True, 'purposes': ['improve', 'explore'], 'rejected': rejected,
                'parent_ids': ['node_000'], 'leader_id': 'node_000',
                'used_pipelines': {'node_000': duplicate['pipeline']}}
            with self.subTest(sibling=sibling):
                result = Hypotheses.model_validate_json(json.dumps({'hypotheses': [improve, sibling]}), context=context)
                self.assertEqual([p.purpose for p in result.hypotheses], ['improve'])
                self.assertIn('explore', rejected)
        reversed_pair = [{**improve, 'hypothesis': duplicate}, {**improve, 'purpose': 'explore'}]
        result = Hypotheses.model_validate_json(json.dumps({'hypotheses': reversed_pair}), context={**context, 'rejected': {}})
        self.assertEqual([p.purpose for p in result.hypotheses], ['explore'])

    def test_compact_baker_context_preserves_arbitrary_pipeline_changes_and_requirements(self):
        pipeline = {'data': {'tables': {'events': 'custom.arrow'}}, 'preprocessing': {'custom': [1, None]},
                    'model': {'arbitrary_model': {'x': 7}}, 'training': {'arbitrary_parameter': {'y': False}}}
        hypothesis = {'pipeline': pipeline, 'rationale': 'past narrative ' * 1000, 'expected_effect': 'past expectation'}
        changed = {**pipeline, 'data': {'tables': {'events': 'other.parquet'}},
                   'training': {'arbitrary_parameter': {'y': 0}}}
        row = {'node_id': 'node_000', 'parent_id': None, 'hypothesis': {**hypothesis, 'pipeline': changed},
               'result': {'status': 'failed', 'stage': 'check', 'value': None, 'duration_seconds': 12.5,
                          'error': 'Traceback detail\nValueError: custom failure'}}
        refused = {**row, 'node_id': 'node_003', 'result': {'status': 'failed', 'stage': 'implementation',
                   'error': 'Implementation blocked: alleged missing data'}}
        context = {'task_text': 'Preserve this arbitrary task constraint.',
            'preparation': {'metric': {'definition': 'a bespoke scoring formula'}},
            'parent': {'start': {'node_id': 'node_001', 'hypothesis': hypothesis}, 'finish': {'value': .5}, 'code': PROGRAM},
            'delta': {'nodes': [row, refused]}, 'pending': [{'node_id': 'node_002', 'parent_id': 'node_001', 'hypothesis': hypothesis}]}
        compact = baker_context(context)
        restored = {**compact['parent']['pipeline'], **compact['history'][0]['pipeline_changes']}
        self.assertEqual(json.dumps(restored, sort_keys=True), json.dumps(changed, sort_keys=True))
        self.assertEqual(compact['pending'][0]['pipeline_changes'], {})
        self.assertEqual(compact['task_text'], context['task_text'])
        self.assertEqual(compact['preparation'], context['preparation'])
        self.assertEqual(compact['history'][0]['error'], 'ValueError: custom failure')
        self.assertEqual(compact['history'][0]['duration_seconds'], 12.5)
        self.assertEqual(compact['history'][0]['error_source'], 'runtime')
        self.assertEqual(compact['history'][1]['error_source'], 'model_claim')
        self.assertEqual(compact['history'][1]['source'], 'nodes/node_003/finish.json')
        self.assertEqual(compact['pending'][0]['source'], 'nodes/node_002/start.json')
        self.assertLess(len(json.dumps(compact)), len(json.dumps(context)) / 2)

    def test_implementation_brief_excludes_misleading_ancestor_failures(self):
        current = {'node_id': 'node_001', 'hypothesis': {'pipeline': {'model': {'family': 'arbitrary'}}}}
        parent = {'start': {'node_id': 'node_000', 'hypothesis': {'pipeline': {'model': {'family': 'constant'}},
                            'rationale': 'misleading ancestor narrative'}},
                  'finish': {'error': 'alleged missing data', 'notes': ['irrelevant refusal']}, 'code': PROGRAM}
        context = implementation_context({'start': current, 'parent': parent, 'preparation': {},
                                          'extraction': {'unused': 'repeated source quotes'}})
        self.assertEqual(context['start'], current)
        self.assertEqual(context['parent_reference']['pipeline']['model']['family'], 'constant')
        self.assertNotIn('misleading', json.dumps(context))
        self.assertNotIn('alleged', json.dumps(context))
        self.assertNotIn('extraction', context)

    def test_profile_measures_cardinality_without_rejecting_structured_cells(self):
        result = profile({'train': pd.DataFrame({'category': ['a', 'a', None], 'structured': [[1], [2], None]})})
        self.assertEqual(result['train']['columns']['category'], {'dtype': 'object', 'missing': 1, 'distinct': 1})
        self.assertIsNone(result['train']['columns']['structured']['distinct'])

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


class SubmissionTests(unittest.TestCase):
    def test_platform_budgets_are_large_and_explicit_overrides_win(self):
        from solution import main
        with patch('solution.cli', return_value=0) as cli:
            main(['task.md'])
            arguments = cli.call_args.args[0]
            for option, value in {'--time-budget': '10500', '--experiment-timeout': '1200', '--max-nodes': '100'}.items():
                self.assertEqual(arguments[arguments.index(option) + 1], value)
            main(['task.md', '--time-budget=300', '--experiment-timeout', '60', '--max-nodes', '1'])
            self.assertEqual(cli.call_args.args[0], ['baseline', '--task', 'task.md', '--time-budget=300',
                '--experiment-timeout', '60', '--max-nodes', '1'])

    def test_zip_contains_only_root_modules_and_original_requirements(self):
        with tempfile.TemporaryDirectory() as temporary:
            archive = build(Path(temporary) / 'submission.zip')
            with ZipFile(archive) as bundle:
                self.assertEqual(set(bundle.namelist()), {'solution.py', 'requirements.txt', 'alfa_cli.py',
                    'alfa_agents.py', 'alfa_config.py', 'alfa_contracts.py', 'alfa_data.py', 'alfa_qwen.py',
                    'alfa_runner.py', 'alfa_storage.py', 'alfa_worker.py'})
                self.assertEqual(bundle.read('requirements.txt'), Path('vendor/requirements_data_folder.txt').read_bytes())
                for name in bundle.namelist():
                    if name.endswith('.py'):
                        self.assertNotIn('alfa_agent.', bundle.read(name).decode())
            self.assertLess(archive.stat().st_size, 100 * 1024**2)

    def test_export_runs_from_task_directory_without_the_package_or_repository(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with ZipFile(build(root / 'submission.zip')) as bundle:
                bundle.extractall(root)
            task, answers = fixture(root)
            task.rename(root / 'task.md')
            (root / 'mock_answers.json').write_text(json.dumps(answers))
            driver = '''import importlib.util,json,runpy,sys
from pathlib import Path
from unittest.mock import patch
import httpx
from openai import AsyncOpenAI
assert importlib.util.find_spec('alfa_agent') is None
answers=json.loads(Path('mock_answers.json').read_text())
def handler(request):
    body=json.loads(request.content)
    assert body['model']=='qwen/qwen3.8-27b'
    title=json.loads(body['messages'][0]['content'].split('Schema: ',1)[1])['title']
    context=json.loads(body['messages'][1]['content'].rsplit('\\n',1)[1])
    if title=='Hypotheses':
        count=sum(1 for _ in Path('artifacts').glob('*/nodes/*/start.json'))
        proposals=[]
        for offset,purpose in enumerate(context['requested_purposes']):
            hypothesis=json.loads(json.dumps(answers['Hypothesis']))
            hypothesis['pipeline']['model']={'family':'constant','value':count+offset}
            proposals.append({'purpose':purpose,'parent_id':context['parent']['node_id'],'hypothesis':hypothesis})
        answer={'hypotheses':proposals}
    elif title=='Program':
        index=int(context['start']['node_id'].split('_')[1])
        answer={**answers[title]}
        if index: answer['code']=answer['code'].replace('float(fit[column].mean())',str(index)+'.')
    else: answer=answers[title]
    return httpx.Response(200,json={'id':'mock','object':'chat.completion','created':0,'model':'model',
        'choices':[{'index':0,'finish_reason':'stop','message':{'role':'assistant','content':json.dumps(answer)}}]})
def client(**kwargs):
    return AsyncOpenAI(**kwargs,http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
sys.argv=['solution.py','task.md','--max-nodes','3']
with patch('alfa_agents.AsyncOpenAI',side_effect=client):
    runpy.run_path('solution.py',run_name='__main__')
'''
            env = {'PATH': os.defpath, 'PYTHONPATH': '', 'PYTHONDONTWRITEBYTECODE': '1',
                   'OPENAI_API_KEY': 'fake-secret', 'OPENAI_BASE_URL': 'https://qwen.example/v1',
                   'OMP_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2'}
            result = subprocess.run([sys.executable, '-c', driver], cwd=root, env=env, capture_output=True, text=True, timeout=90)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            output = next((root / 'artifacts').iterdir())
            report = load(output / 'report.json', RunReport)
            self.assertEqual((report.status, report.completed_nodes), ('ready', 3))
            self.assertIn('working_baseline', report.checks)
            self.assertEqual(load(output / 'delta.json', Delta).best_node_id, 'node_000')
            submission = pd.read_csv(root / 'result.tsv', sep='\t', dtype={'record': str})
            self.assertEqual(list(submission.record), ['0100', '0101', '0102', '0103', '0104'])


if __name__ == '__main__':
    unittest.main()

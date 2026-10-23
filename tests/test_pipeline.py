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
from unittest.mock import AsyncMock, patch
from zipfile import ZipFile

import httpx
import numpy as np
from openai import AsyncOpenAI
import pandas as pd

from alfa_agent.config import Settings
from alfa_agent.agents import JSONAgents, baker_context, implementation_context
from alfa_agent.contracts import CodeReview, Delta, Finish, Hypotheses, Hypothesis, Metric, OutputBinding, Preparation, Program, ReviewRecord, RunReport, Start
from alfa_agent.data import LIBRARIES, check_code, file_hash, library_context, profile
from alfa_agent.runner import Options, RepairProgress, experiment, process, review_source, run, summarize
from alfa_agent.storage import load
from alfa_agent.worker import check_predictor, diagnostics, prediction_frame, validate_submission
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

def fit_model(fit, preparation):
    column = preparation['data']['target_columns'][0]
    value = float(fit[column].mean())
    def predict(features):
        return pd.DataFrame({column: [value] * len(features)})
    return predict
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
        'CodeReview': {'status': 'pass', 'findings': []},
    }
    return path, answers


class PipelineTests(unittest.IsolatedAsyncioTestCase):
    async def test_empty_and_truncated_llm_answers_are_repaired(self):
        for first in ([], [{'index': 0, 'finish_reason': 'length',
                            'message': {'role': 'assistant', 'content': json.dumps({'code': PROGRAM})}}]):
            seen = []
            def handler(request):
                seen.append(request)
                choices = first if len(seen) == 1 else [{'index': 0, 'finish_reason': 'stop',
                    'message': {'role': 'assistant', 'content': json.dumps({'code': PROGRAM})}}]
                return httpx.Response(200, json={'id': 'test', 'object': 'chat.completion', 'created': 0,
                                               'model': 'model', 'choices': choices})
            def client(**kwargs):
                return AsyncOpenAI(**kwargs, http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
            agents = JSONAgents(Settings('fake', 'http://proxy/v1', 'model'), time.monotonic() + 10)
            with patch('alfa_agent.agents.AsyncOpenAI', side_effect=client):
                result = await agents.ask(Program, 'Write code', 'implementer', code=True)
            self.assertEqual(result.code, PROGRAM)
            self.assertEqual(len(seen), 2)
            self.assertEqual(agents.calls[0].status, 'invalid_json')

    async def test_json_object_is_recovered_from_surrounding_prose(self):
        reply = 'Here is the fix:\n```json\n' + json.dumps({'code': PROGRAM}) + '\n```\nIt converts categories.'
        def handler(request):
            return httpx.Response(200, json={'id': 'test', 'object': 'chat.completion', 'created': 0, 'model': 'model',
                'choices': [{'index': 0, 'finish_reason': 'stop', 'message': {'role': 'assistant', 'content': reply}}]})
        def client(**kwargs):
            return AsyncOpenAI(**kwargs, http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
        agents = JSONAgents(Settings('fake', 'http://proxy/v1', 'model'), time.monotonic() + 10)
        with patch('alfa_agent.agents.AsyncOpenAI', side_effect=client):
            self.assertEqual((await agents.ask(Program, 'Write code', 'implementer', code=True)).code, PROGRAM)

    async def test_llm_concurrency_gate_serializes_requests_of_one_run(self):
        active, peak = 0, 0
        async def handler(request):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(.05)
            active -= 1
            return httpx.Response(200, json={'id': 'test', 'object': 'chat.completion', 'created': 0, 'model': 'model',
                'choices': [{'index': 0, 'finish_reason': 'stop', 'message': {'role': 'assistant', 'content': json.dumps({'code': PROGRAM})}}]})
        def client(**kwargs):
            return AsyncOpenAI(**kwargs, http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
        agents = JSONAgents(Settings('fake', 'http://proxy/v1', 'model'), time.monotonic() + 10, concurrency=1)
        nodes = [JSONAgents(agents.settings, agents.deadline, shared=agents, node_id=f'node_00{i}') for i in range(3)]
        with patch('alfa_agent.agents.AsyncOpenAI', side_effect=client):
            await asyncio.gather(*(node.ask(Program, 'Write code', 'implementer', code=True) for node in nodes))
        self.assertEqual(peak, 1)

    async def test_reply_without_message_is_logged_and_retried_and_content_parts_are_joined(self):
        cases = (([{'index': 0, 'finish_reason': 'stop'}], 2),  # HTTP 200 without message
                 ([{'index': 0, 'finish_reason': 'stop', 'message': {'role': 'assistant', 'content': None}}], 2),
                 ([{'index': 0, 'finish_reason': 'stop', 'message': {'role': 'assistant', 'content': [
                     {'type': 'text', 'text': json.dumps({'code': PROGRAM})[:20]},
                     {'type': 'text', 'text': json.dumps({'code': PROGRAM})[20:]}]}}], 1))
        for first, expected in cases:
            seen = []
            def handler(request):
                seen.append(request)
                choices = first if len(seen) == 1 else [{'index': 0, 'finish_reason': 'stop',
                    'message': {'role': 'assistant', 'content': json.dumps({'code': PROGRAM})}}]
                return httpx.Response(200, json={'id': 'test', 'object': 'chat.completion', 'created': 0,
                    'model': 'model', 'choices': choices, 'usage': {'prompt_tokens': None, 'completion_tokens': None,
                    'total_tokens': None}})
            def client(**kwargs):
                return AsyncOpenAI(**kwargs, http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
            agents = JSONAgents(Settings('fake', 'http://proxy/v1', 'model'), time.monotonic() + 10)
            with patch('alfa_agent.agents.AsyncOpenAI', side_effect=client):
                result = await agents.ask(Program, 'Write code', 'implementer', code=True)
            self.assertEqual((result.code, len(seen)), (PROGRAM, expected))
            self.assertEqual([c.status for c in agents.calls], ['invalid_json', 'completed'][2 - expected:])

    async def test_json_mode_rejected_with_http_400_is_retried_without_it(self):
        bodies = []
        def handler(request):
            body = json.loads(request.content)
            bodies.append(body)
            if 'response_format' in body:
                return httpx.Response(400, json={'error': {'message': 'response_format unsupported'}})
            return httpx.Response(200, json={'id': 'test', 'object': 'chat.completion', 'created': 0, 'model': 'model',
                'choices': [{'index': 0, 'finish_reason': 'stop', 'message': {'role': 'assistant',
                'content': json.dumps({'status': 'pass', 'findings': []})}}]})
        def client(**kwargs):
            return AsyncOpenAI(**kwargs, http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
        agents = JSONAgents(Settings('fake', 'http://proxy/v1', 'model'), time.monotonic() + 10)
        with patch('alfa_agent.agents.AsyncOpenAI', side_effect=client):
            result = await agents.ask(CodeReview, 'Review', 'reviewer', validation={'review_code': PROGRAM})
        self.assertEqual((result.status, ['response_format' in b for b in bodies]), ('pass', [True, False]))

    async def exercise(self, directory, *, repair=False, missing_metric=False, blocked=False, broken_metric=False,
                       invalid_program=False, broken_code=False, execution_timeout=False, full_repair=False,
                       json_failure=False, network_failure=False, ast_failure=False, refused_once=False,
                       refused_after_error=False, wrong_columns=False, semantic_error=False, preparation_error=False,
                       review_warning=False, review_unavailable=False, never_fix=False, max_nodes=1,
                       final_refit=False, refit_failure=False, autopilot=False):
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
            self.assertEqual(body['reasoning_effort'], 'none')
            self.assertEqual(body['chat_template_kwargs'], {'enable_thinking': False})
            self.assertEqual('response_format' in body, title not in {'PreparationDraft', 'Program'})  # JSON-mode except code roles
            if title == 'Program' and seen.count(title) < int(network_failure):
                seen.append(title)
                raise httpx.ConnectError('temporary disconnection', request=request)
            answer = answers[title]
            if title == 'PreparationDraft' and wrong_columns and title not in seen:
                answer = {**answer, 'output': {**answer['output'], 'columns': ['wrong']}}
            if title == 'PreparationDraft' and preparation_error and (title not in seen or preparation_error == 'always'):
                answer = {**answer, 'code': PREPARE.replace(' + 7)', ' + 8)')}
            if title == 'CodeReview':
                review_context = json.loads(body['messages'][1]['content'].rsplit('\n', 1)[1])
                kind = review_context['kind']
                self.assertEqual(set(review_context), {'kind', 'code', 'contract', 'previous_findings'} |
                                 ({'plan', 'smoke'} if kind == 'experiment' else {'requirements', 'execution'}))
                self.assertEqual(body['max_tokens'], 8192)
                if kind == 'experiment' and review_unavailable:
                    return httpx.Response(401, json={'error': {'message': 'fake-secret'}})
                quote = 'float(fit[column].median())' if kind == 'experiment' else ' + 8)'
                if (semantic_error if kind == 'experiment' else preparation_error) and quote in review_context['code']:
                    answer = {'status': 'revise', 'findings': [{'severity': 'error', 'code_quote': quote,
                        'requirement': 'Use the specified mean / metric formula', 'explanation': 'Implemented a different statistic',
                        'check': 'Compare against hand-calculated values on a small table'}]}
                elif kind == 'experiment' and review_warning:
                    answer = {'status': 'pass', 'findings': [{'severity': 'warning', 'code_quote': 'mean()',
                        'requirement': 'Finite predictions', 'explanation': 'All-missing targets could require a fallback',
                        'check': 'Check whether targets can be missing'}]}
            if title in {'PreparationDraft', 'Hypothesis', 'Program'}:
                payload = json.loads(body['messages'][1]['content'].rsplit('\n', 1)[1])
                self.assertEqual(payload['allowed_libraries'], LIBRARIES)
                self.assertIn('scikit-learn', payload['installed_libraries'])
            if title == 'Program':
                saved = (run_directory / 'nodes/node_000/start.json').read_bytes()
                saved_starts.append(saved)
                self.assertTrue(all(value == saved for value in saved_starts))
                context = json.loads(body['messages'][1]['content'].rsplit('\n', 1)[1])
                self.assertEqual(context['start'], json.loads(saved))
                self.assertIn('task_text', context)
                self.assertIn('toolkit', context)
                self.assertNotIn('preparation_code', context)
                self.assertEqual(context['start']['context']['installed_libraries'], context['installed_libraries'])
                self.assertNotIn('extraction', context)
                self.assertEqual(context['runtime_contract']['rows'], {'fit': 32, 'validation': 8, 'test': 5})
                if semantic_error and (title not in seen or never_fix):
                    answer = {**answer, 'code': PROGRAM.replace('mean()', 'median()')}
                elif semantic_error:
                    self.assertEqual(context['failed_stage'], 'review')
                    self.assertEqual(context['review_findings'][0]['code_quote'], 'float(fit[column].median())')
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
                answer = {**answers['Program'], 'code': PROGRAM.replace('value = float(fit[column].mean())',
                          f'raise ValueError("repair me attempt {seen.count(title)}")')}
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
            if refit_failure and '--refit' in arguments:
                raise TimeoutError('Final refit exceeded remaining budget')
            if semantic_error and '--start' in arguments:
                self.assertNotIn('.median()', Path(arguments[0]).read_text())
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
            result = await run(path, Options(time_budget=40, experiment_timeout=12, output_dir=run_directory, max_nodes=max_nodes, final_refit=final_refit,
                                                     autopilot=autopilot),
                               Settings('fake-secret', 'https://qwen.example/v1', 'model'))
        return result, seen

    async def test_semantic_review_repairs_runnable_code_before_full_execution(self):
        with tempfile.TemporaryDirectory() as temporary:
            (root, report), seen = await self.exercise(Path(temporary), semantic_error=True)
            self.assertEqual(report.status, 'ready', report.errors)
            self.assertEqual(seen.count('Program'), 2)
            node = root / 'nodes/node_000'
            review = load(node / 'review.json', ReviewRecord)
            self.assertEqual(review.result.status, 'pass')
            self.assertEqual(review.code_sha256, file_hash(node / 'experiment.py'))
            self.assertEqual(load(node / 'finish.json', Finish).value, 47.)

    async def test_autopilot_controller_baseline_runs_without_generated_code(self):
        with tempfile.TemporaryDirectory() as temporary:
            (root, report), seen = await self.exercise(Path(temporary), autopilot=True, final_refit=True)
            self.assertEqual(report.status, 'ready', report.errors)
            self.assertNotIn('Program', seen)
            self.assertNotIn('Hypothesis', seen)
            self.assertEqual(seen.count('CodeReview'), 1)  # preparation review only
            start, finish = load(root / 'nodes/node_000/start.json', Start), load(root / 'nodes/node_000/finish.json', Finish)
            self.assertEqual((start.source, finish.status), ('controller', 'completed'))
            self.assertTrue(np.isfinite(finish.value))
            preparation = load(root / 'preparation.json', Preparation)
            self.assertEqual(preparation.profile['target']['yield_value']['distinct'], 40)
            self.assertEqual(json.loads((root / 'finalization.json').read_text())['status'], 'completed')
            self.assertEqual(len(pd.read_csv(Path(temporary) / 'result.tsv', sep='\t')), 5)

    async def test_autopilot_failed_controller_source_is_repaired_by_the_implementer(self):
        from alfa_agent import autopilot
        broken = lambda preparation, threads: ('def fit_model(fit, preparation):\n    raise ValueError("template broke")\n',
                                               autopilot.template(preparation, threads)[1])
        with tempfile.TemporaryDirectory() as temporary, patch('alfa_agent.runner.template', side_effect=broken):
            (root, report), seen = await self.exercise(Path(temporary), autopilot=True)
            self.assertEqual(report.status, 'ready', report.errors)
            self.assertEqual(seen.count('Program'), 1)
            self.assertEqual(load(root / 'nodes/node_000/finish.json', Finish).value, 47.)

    def test_shift_split_detects_later_test_period_and_strong_covariate_shift(self):
        from alfa_agent.contracts import DataBinding
        from alfa_agent.worker import shift_split
        data = DataBinding(files={}, target_columns=['y'], id_columns=['id'])
        rng = np.random.RandomState(0)
        train = pd.DataFrame({'id': range(400), 'period': np.repeat([1, 2, 3, 4], 100), 'x': rng.rand(400), 'y': rng.rand(400)})
        test = pd.DataFrame({'id': range(400, 500), 'period': 5, 'x': rng.rand(100)})
        fit, valid, report = shift_split(train, test, data, {'strategy': 'random'}, np.arange(300), np.arange(300, 400))
        self.assertEqual((report['temporal_column'], report.get('override')), ('period', None))  # already temporal
        mixed = rng.permutation(400)
        fit, valid, report = shift_split(train, test, data, {'strategy': 'random'}, mixed[:320], mixed[320:])
        self.assertEqual(report['override'], 'temporal_holdout')
        self.assertTrue((train.period.iloc[valid] == 4).all() and (train.period.iloc[fit] < 4).all())
        dated = train.assign(period=pd.date_range('2020-01-01', periods=400).astype(str))
        later = test.assign(period=pd.date_range('2021-03-01', periods=100).astype(str))
        self.assertEqual(shift_split(dated, later, data, {}, mixed[:320], mixed[320:])[2]['temporal_column'], 'period')
        shifted = test.assign(period=rng.randint(1, 5, 100), x=rng.rand(100) + 3)
        fit, valid, report = shift_split(train, shifted, data, {'strategy': 'random'}, mixed[:320], mixed[320:])
        self.assertGreater(report['adversarial_auc'], .8)
        self.assertEqual((report['override'], report['shifted_features'][0], len(valid)), ('adversarial_holdout', 'x', 80))
        self.assertNotIn('override', shift_split(train, shifted, data, {'strategy': 'group'}, mixed[:320], mixed[320:])[2])

    def test_line_edits_apply_bottom_up_and_reject_misnumbered_or_overlapping_ranges(self):
        from alfa_agent.contracts import LineEdit
        from alfa_agent.runner import apply_edits
        code = 'import pandas as pd\n\ndef fit_model(fit, preparation):\n    value = 1\n    return value'
        edits = [LineEdit(from_line=4, to_line=4, first_line='value = 1', new='    value = 2\n    value += 1'),
                 LineEdit(from_line=1, to_line=1, first_line='import  pandas as pd ', new='import numpy as np\nimport pandas as pd')]
        self.assertEqual(apply_edits(code, edits).split('\n')[4:6], ['    value = 2', '    value += 1'])
        with self.assertRaisesRegex(ValueError, "actually: '    return value'"):
            apply_edits(code, [LineEdit(from_line=5, to_line=5, first_line='value = 1', new='')])
        with self.assertRaisesRegex(ValueError, 'overlap'):
            apply_edits(code, [LineEdit(from_line=2, to_line=4, first_line='', new=''), LineEdit(from_line=4, to_line=5, first_line='', new='')])
        with self.assertRaisesRegex(ValueError, 'outside'):
            apply_edits(code, [LineEdit(from_line=6, to_line=6, first_line='', new='')])

    def test_api_reference_reads_installed_signatures_error_names_first(self):
        from alfa_agent.library_docs import api_reference
        code = ('import numpy as np\nfrom sklearn.ensemble import RandomForestRegressor\nimport sklearn.linear_model as lm\n'
                'def fit_model(fit, preparation):\n    a = lm.Ridge(alpha=1.)\n    b = RandomForestRegressor(n_estimators=10)\n'
                '    return np.mean')
        reference = api_reference(code, "TypeError: RandomForestRegressor.__init__() got an unexpected keyword argument 'x'")
        self.assertEqual(list(reference)[:2], ['sklearn.ensemble.RandomForestRegressor', 'sklearn.linear_model.Ridge'])
        self.assertIn('n_estimators', reference['sklearn.ensemble.RandomForestRegressor'])
        self.assertIn('.fit(', reference['sklearn.ensemble.RandomForestRegressor'])
        toolkit = api_reference('from alfa_agent.toolkit import automl\ndef fit_model(fit, preparation):\n'
                                '    return automl(fit, preparation, "flaml", 10, features=1)', "unexpected keyword argument 'features'")
        self.assertIn('time_limit', toolkit['alfa_agent.toolkit.automl'])

    def test_toolkit_utilities_encode_per_library_split_by_shift_and_expose_probabilities(self):
        from alfa_agent.toolkit import Features, class_scale, gbdt, inner_score, labels, output, split
        rng = np.random.RandomState(0)
        fit = pd.DataFrame({'id': range(300), 'city': rng.choice(['a', 'b', None], 300), 'x': rng.randn(300),
                            'period': np.repeat([1, 2, 3], 100)})
        fit['y'] = np.where(fit.x > 0, 'hi', 'lo')
        prep = {'data': {'target_columns': ['y'], 'id_columns': ['id']}, 'metric': {'name': 'accuracy', 'direction': 'maximize'},
                'task_type': 'classification', 'profile': {'target': {'y': {'distinct': 2, 'counts': {'hi': 1, 'lo': 1}}}},
                'validation': {'shift': {'temporal_column': 'period'}},
                'scorer': lambda ref, pred, cfg: float((ref['y'].to_numpy() == pred['y'].to_numpy()).mean())}
        batch = pd.DataFrame({'id': [7, 8], 'city': ['zzz', None], 'x': [1., -1.], 'period': [4, 4]})
        strings, codes = Features(fit, prep, kind='string'), Features(fit, prep, kind='codes')
        self.assertEqual((strings.categorical, list(strings(batch).city)), (['city'], ['nan', 'nan']))
        self.assertEqual(list(codes(batch).city), [-1, -1])
        self.assertNotIn('y', strings.train)
        target, y, classes = labels(fit, prep)
        train, valid = split(fit, prep)
        self.assertTrue((fit.period.iloc[valid] == 3).all() and (fit.period.iloc[train] < 3).all())
        self.assertEqual(inner_score(classes[y[valid]], fit, valid, prep), 1.)
        self.assertEqual(len(class_scale(np.eye(2)[y[valid]], classes, fit, valid, prep)), 2)
        self.assertEqual(list(output(batch, prep, ['hi', 'lo']).columns), ['id', 'y'])
        predictor = gbdt(fit, prep)
        self.assertEqual((predictor.proba(batch).shape, list(predictor.classes)), ((2, 2), ['hi', 'lo']))
        from alfa_agent.toolkit import target_stats
        with self.assertRaisesRegex(ValueError, 'Join new features'):
            Features(fit, prep, transform=lambda frame, ref, training: target_stats(ref, frame, ['city'], 'y', training=training))

    def test_target_stats_never_use_a_training_row_own_or_later_targets(self):
        from alfa_agent.toolkit import target_stats
        fit = pd.DataFrame({'key': ['a', 'a', 'a', 'b', 'b'], 'year': [1, 2, 3, 1, 2], 'y': [10., 20., 30., 5., 7.]})
        fit = pd.concat([fit] * 5, ignore_index=True).assign(y=lambda d: d.y + np.arange(len(d)) * 1e-3)
        stats = target_stats(fit, fit, ['key'], 'y', training=True, time='year')
        first = fit.year == 1
        self.assertTrue(stats.loc[first, 'ts_key_mean'].isna().all())  # nothing earlier than the first period
        row = fit.index[(fit.key == 'a') & (fit.year == 3)][0]
        earlier = fit[(fit.key == 'a') & (fit.year < 3)].y
        self.assertAlmostEqual(stats.loc[row, 'ts_key_mean'], earlier.mean())
        later = target_stats(fit, pd.DataFrame({'key': ['a', 'c'], 'year': [4, 4]}), ['key'], 'y', time='year')
        self.assertEqual((later['ts_key_count'].tolist(), np.isnan(later['ts_key_mean'].iloc[1])), ([15., 0.], True))
        folds = target_stats(fit, fit, ['key'], 'y', training=True)
        self.assertTrue(all(folds.loc[i, 'ts_key_mean'] != fit.loc[fit.key == fit.key[i], 'y'].mean() for i in range(5)))

    def test_blend_combines_numbers_by_weighted_mean_and_labels_by_weighted_vote(self):
        from alfa_agent.worker import combine
        numbers = [pd.DataFrame({'y': [1., 3.]}), pd.DataFrame({'y': [3., 7.]})]
        self.assertEqual(combine(numbers, [3, 1], False)['y'].tolist(), [1.5, 4.])
        labels = [pd.DataFrame({'y': ['a', 'b']}), pd.DataFrame({'y': ['b', 'b']}), pd.DataFrame({'y': ['b', 'a']})]
        self.assertEqual(combine(labels, [3, 1, 1], True)['y'].tolist(), ['a', 'b'])
        codes = [pd.DataFrame({'y': [0, 2]}), pd.DataFrame({'y': [1, 2]})]
        self.assertEqual(combine(codes, [1, 2], True)['y'].tolist(), [1, 2])  # class codes are voted, not averaged

    async def test_code_identical_to_a_measured_node_is_not_executed_again(self):
        from alfa_agent.runner import code_key
        with tempfile.TemporaryDirectory() as temporary:
            (root, report), seen = await self.exercise(Path(temporary), max_nodes=1)
            measured = {code_key(PROGRAM): 'node_000'}
            node = root / 'nodes/node_009'
            node.mkdir()
            start = load(root / 'nodes/node_000/start.json', Start).model_copy(update={'node_id': 'node_009'})
            from alfa_agent.storage import save
            save(node / 'start.json', start)
            agent = JSONAgents(Settings('fake', 'http://proxy/v1', 'model'), time.monotonic() + 20)
            agent.implement = AsyncMock(return_value=Program(code=PROGRAM + '\n# same code, new comment'))
            context = {'preparation': json.loads((root / 'preparation.json').read_text()), 'task_text': ''}
            with patch('alfa_agent.runner.process', new=AsyncMock()) as child:
                finish = await experiment(node, start, context, agent, time.monotonic() + 20, lambda *a: None, measured=measured)
            self.assertEqual(finish.status, 'failed')
            self.assertIn('already measured as node_000', finish.error)
            child.assert_not_called()

    async def test_full_refit_uses_all_train_without_overwriting_validation_score(self):
        with tempfile.TemporaryDirectory() as temporary:
            (root, report), _ = await self.exercise(Path(temporary), final_refit=True)
            self.assertEqual(report.status, 'ready', report.errors)
            final, best = (json.loads((root / name).read_text()) for name in ('finalization.json', 'best.json'))
            self.assertEqual((final['status'], final['fit_input_rows']), ('completed', 40))
            self.assertEqual(best['value'], 47.)
            self.assertFalse(best['refit_scored'])
            self.assertEqual(load(root / 'nodes/node_000/finish.json', Finish).value, 47.)
            self.assertEqual(pd.read_csv(Path(temporary) / 'result.tsv', sep='\t').estimate.tolist(), [39.] * 5)
            self.assertEqual(pd.read_csv(root / best['validation_submission_file'], sep='\t').estimate.tolist(), [31.] * 5)

    async def test_failed_refit_keeps_validated_submission(self):
        with tempfile.TemporaryDirectory() as temporary:
            (root, report), _ = await self.exercise(Path(temporary), final_refit=True, refit_failure=True)
            self.assertEqual(report.status, 'ready', report.errors)
            self.assertEqual(json.loads((root / 'finalization.json').read_text())['status'], 'timeout')
            self.assertEqual((Path(temporary) / 'result.tsv').read_bytes(), (root / 'nodes/node_000/result.tsv').read_bytes())

    async def test_plan_conflict_routes_to_baker_without_mutating_start(self):
        from types import SimpleNamespace
        from alfa_agent.contracts import PlanConflict
        from alfa_agent.storage import save
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            node = root / 'nodes/node_001'
            node.mkdir(parents=True)
            (_, answers) = fixture(root)
            hypothesis = Hypothesis.model_validate(answers['Hypothesis'])
            hypothesis.pipeline.training['thresholds'] = 'Tune on outer validation labels'
            start = Start(node_id='node_001', parent_id='node_000', preparation_sha256='a' * 64,
                          hypothesis=hypothesis, outputs={}, timeout_seconds=10.)
            save(node / 'start.json', start)
            before = (node / 'start.json').read_bytes()
            (root / 'prepare.py').write_text(PREPARE)
            agent = SimpleNamespace(max_calls=20, requests=0, settings=SimpleNamespace(api_key='fake'),
                implement=AsyncMock(return_value=Program(plan_conflict=PlanConflict(
                    missing_input='validation_targets', plan_quote='Tune on outer validation labels'))))
            with patch('alfa_agent.runner.process', new=AsyncMock()) as child:
                finish = await experiment(node, start, {'preparation': answers['PreparationDraft']}, agent,
                                          time.monotonic() + 20, lambda *args: None)
            self.assertEqual(finish.status, 'needs_replan')
            self.assertFalse(finish.replan['verified'])
            self.assertEqual((node / 'start.json').read_bytes(), before)
            child.assert_not_called()
            context = {'parent': {'start': start.model_dump(), 'finish': {'value': 0}, 'code': ''},
                       'preparation': {}, 'delta': {'nodes': [{'node_id': start.node_id, 'parent_id': start.parent_id,
                       'hypothesis': hypothesis.model_dump(), 'result': finish.model_dump()}]}, 'pending': []}
            compact = baker_context(context)
            self.assertEqual(compact['replan_requests'][0]['node_id'], 'node_001')
            self.assertEqual(compact['history'][0]['error_source'], 'model_request')

    def test_no_progress_ignores_comments_but_distinct_repairs_keep_budget(self):
        progress = RepairProgress()
        self.assertEqual(progress.observe('check', PROGRAM, ValueError('same'))['unchanged_failures'], 1)
        progress.observe('check', PROGRAM + '\n# cosmetic', ValueError('same'))
        with self.assertRaisesRegex(ValueError, 'Repair stalled'):
            progress.observe('check', PROGRAM + '\n\n# another comment', ValueError('same'))
        for index in range(20):
            progress.observe('check', PROGRAM + f'\nVERSION = {index}', ValueError('same'))

    def test_frozen_predictor_catches_batch_categories_and_preserves_fitted_stochastic_model(self):
        from sklearn.neural_network import MLPRegressor
        features = pd.DataFrame({'category': ['a', 'b', 'c'] * 12})
        broken = lambda frame: pd.DataFrame({'y': frame['category'].astype('category').cat.codes.to_numpy()})
        with self.assertRaisesRegex(ValueError, 'Frozen predictor changed'):
            check_predictor(broken, features, prediction_frame(broken, features.copy()))
        numeric = pd.DataFrame({'x': np.arange(36, dtype=float)})
        model = MLPRegressor(hidden_layer_sizes=(4,), max_iter=5, random_state=1).fit(numeric, numeric.x)
        predict = lambda frame: pd.DataFrame({'y': model.predict(frame)})
        check_predictor(predict, numeric, prediction_frame(predict, numeric.copy()))

    def test_auxiliary_class_diagnostics_do_not_replace_custom_metric(self):
        from types import SimpleNamespace
        preparation = SimpleNamespace(data=SimpleNamespace(target_columns=['y']), task_type='classification')
        measured = diagnostics(pd.DataFrame({'y': ['a', 'a', 'a', 'b']}), pd.DataFrame({'y': ['a', 'b', 'b', 'b']}), preparation)
        self.assertAlmostEqual(measured['classification']['b']['precision'], 1/3)
        self.assertEqual(measured['classification']['b']['recall'], 1.)
        self.assertEqual(measured['classification']['b']['support'], 1)
        integer = diagnostics(pd.DataFrame({'y': [0, 1]}), pd.DataFrame({'y': [0., 1.]}), preparation)
        self.assertEqual(integer['classification']['1.0']['f1'], 1.)
        self.assertEqual(diagnostics(pd.DataFrame({'y': [0, 1]}), pd.DataFrame({'y': [.2, .7]}), preparation), {})

    def test_predictor_rejects_input_mutation_and_nonfinite_values(self):
        features = pd.DataFrame({'x': [1., 2.]})
        def mutate(frame):
            frame['x'] = 0
            return pd.DataFrame({'y': [0., 0.]})
        with self.assertRaisesRegex(ValueError, 'mutated'):
            prediction_frame(mutate, features.copy())
        with self.assertRaisesRegex(ValueError, 'non-finite'):
            prediction_frame(lambda frame: pd.DataFrame({'y': [float('inf'), 1.]}), features.copy())

    def test_baker_diagnostics_stay_bounded_and_include_leader_parent_deltas(self):
        pipeline = {'data': {}, 'preprocessing': {}, 'model': {}, 'training': {}}
        hypothesis = {'pipeline': pipeline}
        classes = {str(i): {'f1': i/100, 'precision': .5, 'recall': .4, 'support': i+1} for i in range(100)}
        rows = [{'node_id': f'node_{i:03}', 'parent_id': None, 'hypothesis': hypothesis, 'result': {'value': i,
                'diagnostics': {'classification': classes, 'parent_delta': {k: {'f1': .1} for k in classes}}}} for i in range(10)]
        context = {'preparation': {}, 'parent': {'start': {'node_id': 'node_000', 'hypothesis': hypothesis},
                   'finish': {'value': 0}, 'code': ''}, 'delta': {'nodes': rows}, 'pending': []}
        brief = baker_context(context)['diagnostics']
        self.assertEqual([r['node_id'] for r in brief], ['node_000', 'node_008', 'node_009'])
        self.assertTrue(all(len(r['weak_classes']) == len(r['parent_delta']) == 3 for r in brief))

    async def test_preparation_review_repairs_wrong_metric_before_any_node(self):
        with tempfile.TemporaryDirectory() as temporary:
            (root, report), seen = await self.exercise(Path(temporary), preparation_error=True)
            self.assertEqual(report.status, 'ready', report.errors)
            self.assertEqual(seen.count('PreparationDraft'), 2)
            self.assertEqual(load(root / 'preparation_review.json', ReviewRecord).result.status, 'pass')
            self.assertEqual(load(root / 'nodes/node_000/finish.json', Finish).value, 47.)

    async def test_review_warnings_pass_and_unavailable_review_does_not_publish(self):
        for unavailable in (False, True):
            with self.subTest(unavailable=unavailable), tempfile.TemporaryDirectory() as temporary:
                (root, report), _ = await self.exercise(Path(temporary), review_warning=True, review_unavailable=unavailable)
                finish = load(root / 'nodes/node_000/finish.json', Finish)
                review = load(root / 'nodes/node_000/review.json', ReviewRecord)
                if unavailable:
                    self.assertEqual((report.status, finish.stage), ('failed', 'review'))
                    self.assertIsNone(review.result)
                    self.assertNotIn('fake-secret', review.error)
                    self.assertFalse((Path(temporary) / 'result.tsv').exists())
                else:
                    self.assertEqual(report.status, 'ready')
                    self.assertTrue(any('Reviewer warning:' in note for note in finish.notes))

    async def test_repeated_rejected_code_reuses_review_and_respects_total_node_budget(self):
        with tempfile.TemporaryDirectory() as temporary:
            (root, report), seen = await self.exercise(Path(temporary), semantic_error=True, never_fix=True)
            calls = [c for c in report.llm_calls if c.node_id == 'node_000']
            self.assertEqual(len(calls), 4)
            self.assertEqual(sum(c.role == 'reviewer' for c in calls), 1)
            self.assertEqual(seen.count('Program'), 3)
            self.assertEqual(load(root / 'nodes/node_000/finish.json', Finish).stage, 'review')
            self.assertFalse((root / 'best.json').exists())

    async def test_review_quotes_must_exist_and_uncertain_findings_cannot_block(self):
        finding = {'severity': 'warning', 'code_quote': 'mean()', 'requirement': 'Finite outputs',
                   'explanation': 'Possible missing targets', 'check': 'Inspect missing targets'}
        for status, quote in [('revise', 'mean()'), ('pass', 'invented()')]:
            with self.subTest(status=status, quote=quote), self.assertRaises(ValueError):
                CodeReview.model_validate({'status': status, 'findings': [{**finding, 'code_quote': quote}]},
                                          context={'review_code': PROGRAM})

    async def test_failed_preparation_review_never_starts_baseline(self):
        with tempfile.TemporaryDirectory() as temporary:
            (root, report), seen = await self.exercise(Path(temporary), preparation_error='always')
            self.assertEqual((report.status, report.stop_reason), ('failed', 'preparation_failed'))
            self.assertNotIn('Hypothesis', seen)
            self.assertEqual(load(root / 'preparation.json', Preparation).status, 'blocked')

    async def test_review_cache_invalidates_changed_code_or_contract(self):
        agent = JSONAgents(Settings('fake', 'http://proxy/v1', 'model'), time.monotonic() + 10)
        agent.review = AsyncMock(return_value=CodeReview(status='pass'))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, record = root / 'experiment.py', root / 'review.json'
            source.write_text(PROGRAM)
            context = {'kind': 'experiment', 'code': PROGRAM, 'plan': {'model': 'mean'}}
            await review_source(agent, record, source, context)
            await review_source(agent, record, source, context)
            self.assertEqual(agent.review.await_count, 1)
            await review_source(agent, record, source, {**context, 'plan': {'model': 'median'}})
            source.write_text(PROGRAM + '\n')
            await review_source(agent, record, source, context)
            self.assertEqual(agent.review.await_count, 3)

    async def test_preparation_repairs_columns_that_disagree_with_task(self):
        with tempfile.TemporaryDirectory() as temporary:
            (_, report), seen = await self.exercise(Path(temporary), wrong_columns=True)
            self.assertEqual(report.status, 'ready')
            self.assertEqual(seen.count('PreparationDraft'), 2)

    async def test_new_names_formats_custom_metric_and_three_column_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (root, report), seen = await self.exercise(directory)
            self.assertEqual(report.status, 'ready', report.errors)
            self.assertEqual(len(seen), 9)  # Four extractors, preparation/review, Baker, implementation/review.
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
            self.assertEqual(seen.count('Program'), 2)  # a repeated refusal returns the plan to Baker
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
            self.assertEqual(seen.count('Program'), 2)  # a repeated refusal returns the plan to Baker
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
            self.assertEqual(seen.count('Program'), 8)  # every unusable answer is retried within one request's attempts
            self.assertIsNone(finish.code_sha256)

    async def test_request_cap_includes_json_repairs_and_records_last_attempted_code(self):
        with tempfile.TemporaryDirectory() as temporary:
            (root, report), seen = await self.exercise(Path(temporary), broken_code=True, json_failure=True)
            self.assertEqual(report.status, 'failed')
            node = root / 'nodes/node_000'
            finish = load(node / 'finish.json', Finish)
            self.assertEqual((finish.status, finish.stage), ('failed', 'check'))
            self.assertEqual(finish.code_sha256, file_hash(node / 'experiment.py'))
            self.assertEqual(seen.count('Program'), 4)
            self.assertEqual(sum(call.role == 'implementer' for call in report.llm_calls), 4)
            self.assertEqual(sum(call.status == 'invalid_json' for call in report.llm_calls), 1)
            self.assertFalse((root / 'best.json').exists())

    async def test_full_execution_error_can_be_repaired_in_the_same_node(self):
        with tempfile.TemporaryDirectory() as temporary:
            (root, report), seen = await self.exercise(Path(temporary), full_repair=True)
            self.assertEqual(report.status, 'ready', report.errors)
            self.assertEqual(seen.count('Program'), 2)
            self.assertEqual(load(root / 'nodes/node_000/finish.json', Finish).stage, 'execution')

    async def test_connection_errors_retry_within_eight_request_cap(self):
        for failures in (1, 7, 8):
            with self.subTest(failures=failures), tempfile.TemporaryDirectory() as temporary, \
                    patch('alfa_agent.agents.asyncio.sleep', new=AsyncMock()):
                (_, report), seen = await self.exercise(Path(temporary), network_failure=failures)
                self.assertEqual(report.status, 'failed' if failures == 8 else 'ready', report.errors)
                self.assertEqual(seen.count('Program'), min(failures + 1, 8))
                calls = [call for call in report.llm_calls if call.role == 'implementer']
                self.assertEqual([call.status for call in calls], ['api_error'] * failures + (['completed'] if failures < 8 else []))

    async def test_rejected_source_and_actionable_check_error_are_preserved(self):
        with tempfile.TemporaryDirectory() as temporary:
            (root, report), seen = await self.exercise(Path(temporary), ast_failure=True)
            self.assertEqual(report.status, 'failed')
            node = root / 'nodes/node_000'
            finish = load(node / 'finish.json', Finish)
            self.assertEqual((finish.stage, seen.count('Program')), ('check', 3))
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
        check_code(PROGRAM.replace('    column = ', '    columns = [name for name in fit.columns]\n    column = '), ('fit_model',))
        for code in ('def fit_model(:\n    pass', 'import subprocess\n' + PROGRAM, 'print("top-level execution")\n' + PROGRAM,
                     PROGRAM.replace('fit_model', 'wrong_function')):
            with self.subTest(code=code[:30]), self.assertRaises(ValueError):
                check_code(code, ('fit_model',))

    async def test_timeout_stops_library_child_processes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            child_code = "import time; from pathlib import Path; time.sleep(1); Path('orphan').touch()"
            parent_code = (f'import subprocess, sys, time; subprocess.Popen([sys.executable, "-c", {child_code!r}]); '
                           'print("spawned", flush=True); time.sleep(10)')
            with self.assertRaises(TimeoutError):
                await process(['-c', parent_code], root, root / 'timeout.log', time.monotonic() + 5, .5)
            self.assertIn('spawned', (root / 'timeout.log').read_text())
            await asyncio.sleep(1)
            self.assertFalse((root / 'orphan').exists())

    def test_organizer_libraries_and_custom_models_are_allowed(self):
        requirements = Path('vendor/requirements_data_folder.txt').read_text().splitlines()
        distributions = {line.split('==')[0].split('[')[0] for line in requirements if '==' in line}
        self.assertEqual(set(LIBRARIES), distributions)
        for root in LIBRARIES.values():
            check_code(f'import {root}\n' + PROGRAM, ('fit_model',))
        check_code('from torch import nn\nclass Network(nn.Module):\n    def forward(self, x):\n        return x\n' +
                   PROGRAM, ('fit_model',))
        with self.assertRaisesRegex(ValueError, 'Unsupported generated import'):
            check_code('import unapproved_package\n' + PROGRAM, ('fit_model',))

    def test_missing_local_library_is_still_permitted(self):
        from importlib.metadata import PackageNotFoundError
        def installed(name):
            if name == 'catboost':
                raise PackageNotFoundError(name)
            return '1.0'
        with patch('alfa_agent.data.version', side_effect=installed):
            context = library_context()
        self.assertEqual(context['allowed_libraries']['catboost'], 'catboost')
        self.assertNotIn('catboost', context['installed_libraries'])
        self.assertIn('lightgbm', context['installed_libraries'])


class TreeTests(unittest.IsolatedAsyncioTestCase):
    async def test_json_mode_rejected_with_http_400_is_retried_without_it(self):
        bodies = []
        def handler(request):
            body = json.loads(request.content)
            bodies.append(body)
            if 'response_format' in body:
                return httpx.Response(400, json={'error': {'message': 'response_format unsupported'}})
            return httpx.Response(200, json={'id': 'test', 'object': 'chat.completion', 'created': 0, 'model': 'model',
                'choices': [{'index': 0, 'finish_reason': 'stop', 'message': {'role': 'assistant',
                'content': json.dumps({'status': 'pass', 'findings': []})}}]})
        def client(**kwargs):
            return AsyncOpenAI(**kwargs, http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
        agents = JSONAgents(Settings('fake', 'http://proxy/v1', 'model'), time.monotonic() + 10)
        with patch('alfa_agent.agents.AsyncOpenAI', side_effect=client):
            result = await agents.ask(CodeReview, 'Review', 'reviewer', validation={'review_code': PROGRAM})
        self.assertEqual((result.status, ['response_format' in b for b in bodies]), ('pass', [True, False]))

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
                    'def fit_model(fit, preparation):\n'
                    '    column = preparation["data"]["target_columns"][0]\n'
                    '    model = LinearRegression().fit(fit[["signal"]], fit[column])\n'
                    '    def predict(features):\n'
                    '        return pd.DataFrame({column:model.predict(features[["signal"]])})\n'
                    '    return predict\n')
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
            self.assertEqual(body['reasoning_effort'], 'none')
            self.assertEqual(body['chat_template_kwargs'], {'enable_thinking': False})
            context = json.loads(body['messages'][1]['content'].rsplit('\n', 1)[1])
            if title in {'Hypothesis', 'Hypotheses'}:
                self.assertEqual(context['allowed_libraries'], LIBRARIES)
                self.assertIn('scikit-learn', context['installed_libraries'])
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
                        answer = {'code': f'ATTEMPT = {requests[node_id]}\nCOLUMNS = list()\n' + code[index]}
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
            directory, report = await run(path, Options(time_budget=90, experiment_timeout=12, output_dir=root, max_nodes=max_nodes, final_refit=False),
                                          Settings('fake-secret', 'https://qwen.example/v1', 'model'))
        for node_id, saved in starts.items():
            self.assertEqual((root / 'nodes' / node_id / 'start.json').read_bytes(), saved)
        self.assertLessEqual(peak, 2)
        if len(starts) > 1 and not stuck_explore:
            self.assertGreaterEqual(peak, 2)
        if repair_both:
            self.assertEqual([requests['node_001'], requests['node_002']], [12, 12])
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
                self.assertEqual(len(calls), 12)
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
    def test_submission_checks_serialized_columns_and_row_count(self):
        output = OutputBinding(filename='submission.csv', format='CSV', columns=['id', 'prediction'])
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / output.filename
            for content in ['', 'id,prediction\n', 'prediction,id\n1,001\n',
                            'id,prediction\n001,1,extra\n', 'id,prediction\n001,1\n002,2\n']:
                path.write_text(content)
                with self.subTest(content=content), self.assertRaises(ValueError):
                    validate_submission(path, output, 1)
            path.write_text('id,prediction\n001,1\n')
            validate_submission(path, output, 1)

    def test_platform_budgets_are_large_and_explicit_overrides_win(self):
        from solution import main
        with patch('solution.cli', return_value=0) as cli:
            main(['task.md'])
            arguments = cli.call_args.args[0]
            for option, value in {'--time-budget': '7200', '--experiment-timeout': '1200', '--max-nodes': '1000',
                                  '--executors': '6', '--threads': '2'}.items():
                self.assertEqual(arguments[arguments.index(option) + 1], value)
            main(['task.md', '--time-budget=300', '--experiment-timeout', '60', '--max-nodes', '1'])
            self.assertEqual(cli.call_args.args[0], ['baseline', '--task', 'task.md', '--platform-output', '--time-budget=300',
                '--experiment-timeout', '60', '--max-nodes', '1', '--executors', '6', '--threads', '2'])

    def test_default_entry_never_uses_fallback_or_masks_agent_exit(self):
        import solution
        self.assertIs(solution.ENABLE_FALLBACK, False)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            task = root / 'task.md'
            task.write_text('Unseen task')
            for existing in (False, True):
                if existing:
                    (root / 'submission.csv').write_text('id,target\n1,0\n')
                with self.subTest(existing=existing), patch('solution.cli', return_value=2), \
                        patch('alfa_agent.fallback.emergency') as emergency:
                    self.assertEqual(solution.main([str(task)]), 2)
                    emergency.assert_not_called()
                    self.assertEqual((root / 'submission.csv').exists(), existing)

    def test_default_entry_propagates_failure_without_creating_submission(self):
        from solution import main
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch('solution.cli', side_effect=RuntimeError('agent failed')), \
                    patch('alfa_agent.fallback.emergency') as emergency:
                with self.assertRaisesRegex(RuntimeError, 'agent failed'):
                    main([str(root / 'task.md')])
                emergency.assert_not_called()
            self.assertFalse((root / 'submission.csv').exists())

    def test_default_entry_preserves_only_agent_output(self):
        from solution import main
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            content = 'id,target\n001,42\n'
            def agent(arguments):
                (root / 'submission.csv').write_text(content)
                return 0
            with patch('solution.cli', side_effect=agent), patch('alfa_agent.fallback.emergency') as emergency:
                self.assertEqual(main([str(root / 'task.md')]), 0)
                emergency.assert_not_called()
            self.assertEqual((root / 'submission.csv').read_text(), content)

    def test_solution_writes_a_fallback_submission_only_when_explicitly_enabled(self):
        from solution import main
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / 'nested' / 'participants'
            data.mkdir(parents=True)
            rng = np.random.RandomState(0)
            train = pd.DataFrame({'id': range(300), 'city': rng.choice(['a', 'b'], 300), 'x': rng.randn(300)})
            train['price'] = train.x * 10 + (train.city == 'a') * 3 + rng.randn(300)
            test = pd.DataFrame({'id': range(300, 340), 'city': rng.choice(['a', 'b'], 40), 'x': rng.randn(40)})
            train.to_csv(data / 'train.csv', index=False)
            test.to_csv(data / 'test.csv', index=False)
            pd.DataFrame({'id': test.id, 'price': 0.}).to_csv(data / 'sample_submission.csv', index=False)
            (root / 'task.md').write_text('Predict price; RMSE. Files in nested/participants.')
            with patch('solution.ENABLE_FALLBACK', True), \
                    patch('solution.cli', side_effect=RuntimeError('LLM contract failed')):
                with self.assertRaises(RuntimeError):
                    main([str(root / 'task.md')])
            result = pd.read_csv(root / 'submission.csv')
            self.assertEqual((list(result.columns), len(result)), (['id', 'price'], 40))
            self.assertEqual(result.id.tolist(), test.id.tolist())
            self.assertLess(np.sqrt(np.mean((result.price - (test.x * 10 + (test.city == 'a') * 3)) ** 2)), 5)

    def test_zip_contains_only_root_modules_and_original_requirements(self):
        with tempfile.TemporaryDirectory() as temporary:
            archive = build(Path(temporary) / 'submission.zip')
            with ZipFile(archive) as bundle:
                self.assertEqual(set(bundle.namelist()), {'solution.py', 'requirements.txt', 'alfa_cli.py',
                    'alfa_agents.py', 'alfa_autopilot.py', 'alfa_fallback.py', 'alfa_library_docs.py', 'alfa_toolkit.py', 'alfa_config.py', 'alfa_contracts.py', 'alfa_data.py',
                    'alfa_qwen.py', 'alfa_runner.py', 'alfa_storage.py', 'alfa_worker.py'})
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
            # Exercise the platform contract with nested inputs and a nonstandard target/ID.
            replacements = {'samples.psv': 'data/train/samples.psv', 'query.jsonl': 'data/test/query.jsonl',
                            'result.tsv': 'submission.csv', 'TSV': 'CSV'}
            for source, target in list(replacements.items())[:2]:
                (root / target).parent.mkdir(parents=True, exist_ok=True)
                (root / source).rename(root / target)
            text, serialized = task.read_text(), json.dumps(answers)
            for source, target in replacements.items():
                text, serialized = text.replace(source, target), serialized.replace(source, target)
            task.write_text(text)
            answers = json.loads(serialized)
            answers['PreparationDraft']['code'] = answers['PreparationDraft']['code'].replace("sep='\\t'", "sep=','")
            task.rename(root / 'task.md')
            (root / 'mock_answers.json').write_text(json.dumps(answers))
            driver = '''import importlib.util,json,runpy,sys
from pathlib import Path
from unittest.mock import patch
import httpx
from openai import AsyncOpenAI
assert importlib.util.find_spec('alfa_agent') is None
answers=json.loads(Path('mock_answers.json').read_text())
preparations=0
def handler(request):
    global preparations
    assert str(request.url)=='http://llm-proxy:8080/v1/chat/completions'
    body=json.loads(request.content)
    assert body['model']=='qwen/qwen3.8-27b'
    title=json.loads(body['messages'][0]['content'].split('Schema: ',1)[1])['title']
    assert body['reasoning_effort']=='none'
    assert body['chat_template_kwargs']=={'enable_thinking':False}
    assert body['temperature']==0 and body['max_tokens']==(8192 if title=='CodeReview' else 32768) and ('response_format' in body)==(title not in ('PreparationDraft','Program'))
    context=json.loads(body['messages'][1]['content'].rsplit('\\n',1)[1])
    if title=='PreparationDraft':
        preparations+=1
        assert context['platform_output']['filename']=='submission.csv'
        assert {'data/train/samples.psv','data/test/query.jsonl'} <= {f['path'] for f in context['inventory']}
        answer=json.loads(json.dumps(answers[title]))
        if preparations==1: answer['output']['filename']='wrong.csv'
        else: assert 'submission.csv' in context['execution_error']
    elif title=='Hypotheses':
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
                   'OPENAI_API_KEY': 'fake-secret', 'OPENAI_BASE_URL': 'http://llm-proxy:8080/v1',
                   'OMP_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2'}
            result = subprocess.run([sys.executable, '-c', driver], cwd=root, env=env, capture_output=True, text=True, timeout=90)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            output = next((root / 'artifacts').iterdir())
            report = load(output / 'report.json', RunReport)
            self.assertEqual((report.status, report.completed_nodes), ('ready', 3))
            self.assertIn('working_baseline', report.checks)
            self.assertEqual(load(output / 'delta.json', Delta).best_node_id, 'node_000')
            submission = pd.read_csv(root / 'submission.csv', dtype={'record': str})
            self.assertEqual(list(submission), ['record', 'estimate', 'comment'])
            self.assertEqual(list(submission.record), ['0100', '0101', '0102', '0103', '0104'])


if __name__ == '__main__':
    unittest.main()

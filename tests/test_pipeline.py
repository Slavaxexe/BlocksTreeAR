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
from alfa_agent.contracts import Finish, Preparation, Start
from alfa_agent.data import check_code, file_hash
from alfa_agent.runner import Options, process, run
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
        'Program': {'code': PROGRAM},
    }
    answers['Program']['hypothesis'] = answers['Hypothesis']
    return path, answers


class PipelineTests(unittest.IsolatedAsyncioTestCase):
    async def exercise(self, directory, *, repair=False, missing_metric=False, revise_plan=False, broken_metric=False):
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
        if revise_plan:
            answers['Hypothesis'] = {**answers['Hypothesis'], 'pipeline': {**answers['Hypothesis']['pipeline'], 'model': {'family': 'unavailable_model'}}}
        seen = []
        run_directory = directory / 'artifacts' / 'run'
        def handler(request):
            body = json.loads(request.content)
            title = json.loads(body['messages'][0]['content'].split('Schema: ', 1)[1])['title']
            answer = answers[title]
            if title == 'Program' and repair and title not in seen:
                answer = {**answers['Program'], 'code': PROGRAM.replace('value = float(fit[column].mean())', 'raise ValueError("repair me")')}
            elif title == 'Program' and repair:
                self.assertFalse((run_directory / 'nodes/node_000/start.json').exists())
                self.assertIn('repair me', body['messages'][-1]['content'])
            seen.append(title)
            return httpx.Response(200, json={'id': 'test', 'object': 'chat.completion', 'created': 0,
                'model': 'model', 'choices': [{'index': 0, 'finish_reason': 'stop',
                'message': {'role': 'assistant', 'content': json.dumps(answer)}}]})
        def client(**kwargs):
            return AsyncOpenAI(api_key='fake-secret', base_url='https://qwen.example/v1', max_retries=0,
                               http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
        with patch('alfa_agent.agents.AsyncOpenAI', side_effect=client):
            result = await run(path, Options(time_budget=40, experiment_timeout=12, output_dir=run_directory),
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
            self.assertEqual(start.implementation.sha256, file_hash(node / 'experiment.py'))
            self.assertAlmostEqual(finish.value, 47.)  # Fit mean=31; validation target mean=71; +7.
            self.assertEqual(finish.evaluation_status, 'completed')
            submission = pd.read_csv(node / 'result.tsv', sep='\t', dtype={'record': str})
            self.assertEqual(list(submission), ['record', 'estimate', 'comment'])
            self.assertEqual(list(submission.record), ['0100', '0101', '0102', '0103', '0104'])
            self.assertEqual((directory / 'result.tsv').read_bytes(), (node / 'result.tsv').read_bytes())
            self.assertFalse((root / 'task.json').exists())
            self.assertFalse((root / 'nodes/node_001').exists())
            self.assertNotIn('fake-secret', (root / 'report.json').read_text())

    async def test_code_repair_happens_before_final_start_and_is_bounded(self):
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

    async def test_start_records_realizer_plan_instead_of_unimplemented_baker_proposal(self):
        with tempfile.TemporaryDirectory() as temporary:
            (root, report), _ = await self.exercise(Path(temporary), revise_plan=True)
            self.assertEqual(report.status, 'ready', report.errors)
            start = load(root / 'nodes/node_000/start.json', Start)
            self.assertEqual(start.hypothesis.pipeline.model['family'], 'constant')
            self.assertNotIn('unavailable_model', start.model_dump_json())

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


if __name__ == '__main__':
    unittest.main()

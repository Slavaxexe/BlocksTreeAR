"""Stage-one proof: real task text → mocked HTTP → sourced contracts → real CSV checks."""

import asyncio
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import httpx
from openai import AsyncOpenAI
from pydantic import ValidationError

from alfa_agent.agents import JSONAgents, extract_task
from alfa_agent.config import Settings
from alfa_agent.contracts import Extraction, ProblemFacts
from alfa_agent.storage import load

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ['Normal', 'Generic', 'Exploits', 'Fuzzers', 'DoS', 'Reconnaissance', 'Analysis', 'Backdoor', 'Shellcode', 'Worms']


def expected(name):
    """Manually specified facts for the two public tasks; never supplied to real Qwen."""
    path = ROOT / 'data/raw' / name / 'task.local.md'
    text = path.read_text(encoding='utf-8')
    line = lambda needle: next(row for row in text.splitlines() if needle in row)
    classification = name == 'classification'
    target = 'attack_cat' if classification else 'site_eui'
    problem = dict(task_type='classification' if classification else 'regression', target_column=target,
                   target_unit=None if classification else 'kBtu/ft² за год', class_labels=CLASSES if classification else [], warnings=[])
    problem['evidence'] = {'task_type': [line('многоклассовой классификации' if classification else 'регрессионную модель')],
                           'target_column': [line('Целевая переменная' if classification else 'годовое удельное энергопотребление')]}
    if classification:
        start = text.index('Целевая переменная')
        problem['evidence']['class_labels'] = [text[start:text.index('\n\n', start)]]
    else:
        problem['evidence']['target_unit'] = [line('**kBtu/ft² за год**')]
    data = dict(train_file='train.csv', test_file='test.csv', sample_submission_file='sample_submission.csv', id_column='id',
                dictionary_file='data_dictionary.csv', encoding='UTF-8', delimiter=',', decimal='.')
    data['evidence'] = {key: [line('`' + value + '`')] for key, value in data.items() if key.endswith('_file')}
    data['evidence']['id_column'] = [line('идентификатор `id`' if classification else '| `id` |')]
    start = text.index('CSV используют')
    csv_quote = text[start:text.index('\n\n', start)]
    data['evidence'].update({key: [csv_quote] for key in ('encoding', 'delimiter', 'decimal')})
    metric = dict(name='macro_f1' if classification else 'rmse', direction='maximize' if classification else 'minimize',
                  secondary_metrics=['accuracy', 'weighted_f1'] if classification else [])
    metric['evidence'] = {'name': [line('Основная метрика —' if classification else 'Метрика оценки —')],
                          'direction': [line('большее значение лучше' if classification else 'Меньшее значение лучше')]}
    if classification:
        metric['evidence']['secondary_metrics'] = [line('Дополнительно рассчитываются')]
    start = text.index('Для каждого `id`')
    constraints = text[start:]
    output = dict(submission_file='submission.csv', columns=['id', target], prediction_type='label' if classification else 'number',
                  preserve_test_order=True, unique_ids=True)
    output['evidence'] = {'submission_file': [line('Файл `submission.csv`')], 'columns': ['id,' + target],
                          **{key: [constraints] for key in ('prediction_type', 'preserve_test_order', 'unique_ids')}}
    return path, text, dict(problem=problem, data=data, metric=metric, output=output)


class ExtractionTests(unittest.IsolatedAsyncioTestCase):
    async def test_two_public_tasks_through_http_schema_source_and_csv_checks_without_training(self):
        for name in ('regression', 'classification'):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                path, text, facts = expected(name)
                settings = Settings('fake-secret', 'https://qwen.example/v1', 'model')
                seen = []
                def handler(request):
                    body = json.loads(request.content)
                    title = json.loads(body['messages'][0]['content'].split('Schema: ', 1)[1])['title']
                    role = {'ProblemFacts': 'problem', 'DataFacts': 'data', 'MetricFacts': 'metric', 'OutputFacts': 'output'}[title]
                    seen.append(role)
                    self.assertEqual(body['chat_template_kwargs'], {'enable_thinking': False})
                    self.assertEqual(json.loads(body['messages'][-1]['content'].rsplit('\n', 1)[1]), {'task_text': text})
                    return httpx.Response(200, json={'id': 'test', 'object': 'chat.completion', 'created': 0, 'model': 'model',
                        'choices': [{'index': 0, 'finish_reason': 'stop', 'message': {'role': 'assistant', 'content': json.dumps(facts[role])}}]})
                def client(**kwargs):
                    return AsyncOpenAI(api_key=settings.api_key, base_url=settings.base_url, max_retries=0,
                                       http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
                output = Path(temporary) / name
                with patch('alfa_agent.agents.AsyncOpenAI', side_effect=client), \
                        patch.object(JSONAgents, 'bake', side_effect=AssertionError('No baker in stage one')), \
                        patch('asyncio.create_subprocess_exec', side_effect=AssertionError('No workers in stage one')):
                    report = await extract_task(path, output, settings)
                self.assertEqual(report.status, 'ready', report.errors)
                self.assertCountEqual(seen, ['problem', 'data', 'metric', 'output'])
                extracted = load(output / 'extraction.json', Extraction)
                self.assertEqual(extracted.core_task().problem.target_column, facts['problem']['target_column'])
                self.assertEqual(extracted.metric.secondary_metrics, facts['metric']['secondary_metrics'])
                self.assertEqual(extracted.output.columns, facts['output']['columns'])
                self.assertFalse((output / 'nodes').exists())
                self.assertNotIn('fake-secret', (output / 'report.json').read_text())

    async def test_four_extractors_are_concurrent_and_receive_no_bootstrap_answers(self):
        _, text, facts = expected('regression')
        agents = JSONAgents(Settings('fake', 'https://qwen.example/v1', 'model'), time.monotonic() + 5)
        barrier, entered = asyncio.Event(), []
        async def ask(schema, prompt, role, *, source):
            entered.append(role)
            if len(entered) == 4:
                barrier.set()
            await asyncio.wait_for(barrier.wait(), .5)
            self.assertNotIn('observed_files', prompt)
            return schema.model_validate(facts[role], context={'source': source})
        with tempfile.TemporaryDirectory() as temporary, patch.object(agents, 'ask', side_effect=ask):
            extraction = await agents.extract(text, Path(temporary))
        self.assertEqual(extraction.problem.target_column, 'site_eui')

    def test_fabricated_or_unrelated_evidence_and_unknown_fields_are_rejected(self):
        _, text, facts = expected('regression')
        for change in ('fabricated', 'wrong_target', 'missing_target', 'missing_evidence', 'extra_field'):
            value = json.loads(json.dumps(facts['problem']))
            if change == 'fabricated':
                value['evidence']['target_column'] = ['a quote not present in the task']
            elif change == 'wrong_target':
                value['target_column'] = 'invented_target'
            elif change == 'missing_target':
                value.pop('target_column')
            elif change == 'missing_evidence':
                value['evidence'].pop('task_type')
            else:
                value['model_family'] = 'extra_trees'
            with self.subTest(change=change), self.assertRaises(ValidationError):
                ProblemFacts.model_validate_json(json.dumps(value), context={'source': text})

    def test_primary_metric_and_output_columns_cannot_be_silently_changed(self):
        _, text, facts = expected('classification')
        for change in ('secondary_as_primary', 'wrong_direction', 'reversed_columns'):
            value = json.loads(json.dumps(facts))
            if change == 'secondary_as_primary':
                value['metric']['name'] = 'accuracy'
            elif change == 'wrong_direction':
                value['metric']['direction'] = 'minimize'
            else:
                value['output']['columns'].reverse()
            with self.subTest(change=change), self.assertRaises(ValidationError):
                Extraction.model_validate(value, context={'source': text})

    async def test_csv_conflict_leaves_a_failed_report_instead_of_an_accepted_task(self):
        path, text, facts = expected('regression')
        agents_result = Extraction.model_validate(facts, context={'source': text})
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / 'extract'
            with patch.object(JSONAgents, 'extract', return_value=agents_result), \
                    patch('alfa_agent.data.read_tables', side_effect=ValueError('CSV contract conflict')):
                report = await extract_task(path, output, Settings('fake', 'https://qwen.example/v1', 'model'))
            self.assertEqual(report.status, 'failed')
            self.assertEqual(report.errors, ['CSV contract conflict'])
            self.assertFalse((output / 'extraction.json').exists())
            self.assertTrue((output / 'report.json').is_file())


if __name__ == '__main__':
    unittest.main()

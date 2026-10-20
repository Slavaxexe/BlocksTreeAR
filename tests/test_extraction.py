"""Stage one extracts source-backed facts without opening datasets."""

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
from alfa_agent.contracts import DataFacts, Extraction, ProblemFacts
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
    def test_bad_quote_diagnostic_identifies_the_case_mismatch(self):
        with self.assertRaises(ValidationError) as caught:
            ProblemFacts.model_validate({'warnings': ['missing values'], 'evidence': {'warnings': ['missing values']}},
                context={'source': 'Missing values'})
        self.assertIn("['missing values']", str(caught.exception))
        self.assertIn('preserving case', str(caught.exception))

    async def test_proxy_markdown_wrapper_is_removed_but_unknown_fields_are_rejected(self):
        _, text, facts = expected('regression')
        for extra in (False, True):
            value = {**facts['problem'], **({'unexpected': 'keep me invalid'} if extra else {})}
            def handler(request):
                body = json.loads(request.content)
                self.assertNotIn('response_format', body)
                self.assertEqual(body['reasoning_effort'], 'none')
                return httpx.Response(200, json={'id': 'proxy', 'object': 'chat.completion', 'created': 0,
                    'model': 'model', 'choices': [{'index': 0, 'finish_reason': 'stop', 'message': {
                        'role': 'assistant', 'reasoning_content': 'This is not the JSON to parse.',
                        'content': '```json\n' + json.dumps(value) + '\n```'}}]})
            def client(**kwargs):
                return AsyncOpenAI(**kwargs, http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
            agents = JSONAgents(Settings('fake', 'https://qwen.example/v1', 'model'), time.monotonic() + 5)
            with patch('alfa_agent.agents.AsyncOpenAI', side_effect=client):
                if extra:
                    from alfa_agent.qwen import QwenError
                    with self.assertRaises(QwenError):
                        await agents.ask(ProblemFacts, text, 'problem', source=text, attempts=1)
                else:
                    result = await agents.ask(ProblemFacts, text, 'problem', source=text, attempts=1)
                    self.assertEqual(result.target_column, 'site_eui')

    async def test_two_public_tasks_through_http_and_source_checks_without_reading_data(self):
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
                    self.assertEqual(body['reasoning_effort'], 'none')
                    self.assertNotIn('response_format', body)
                    self.assertEqual(json.loads(body['messages'][-1]['content'].rsplit('\n', 1)[1]), {'task_text': text})
                    return httpx.Response(200, json={'id': 'test', 'object': 'chat.completion', 'created': 0, 'model': 'model',
                        'choices': [{'index': 0, 'finish_reason': 'stop', 'message': {'role': 'assistant', 'content': json.dumps(facts[role])}}]})
                def client(**kwargs):
                    return AsyncOpenAI(api_key=settings.api_key, base_url=settings.base_url, max_retries=0,
                                       http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
                output = Path(temporary) / name
                with patch('alfa_agent.agents.AsyncOpenAI', side_effect=client), \
                        patch.object(JSONAgents, 'plan_baseline', side_effect=AssertionError('No baker in stage one')), \
                        patch('asyncio.create_subprocess_exec', side_effect=AssertionError('No workers in stage one')):
                    report = await extract_task(path, output, settings)
                self.assertEqual(report.status, 'ready', report.errors)
                self.assertCountEqual(seen, ['problem', 'data', 'metric', 'output'])
                extracted = load(output / 'extraction.json', Extraction)
                self.assertEqual(extracted.problem.target_column, facts['problem']['target_column'])
                self.assertEqual(extracted.metric.secondary_metrics, facts['metric']['secondary_metrics'])
                self.assertEqual(extracted.output.columns, facts['output']['columns'])
                self.assertFalse((output / 'nodes').exists())
                self.assertFalse((output / 'task.json').exists())
                self.assertTrue((output / 'extraction' / 'problem.json').is_file())
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
        for change in ('fabricated', 'wrong_target', 'missing_evidence', 'extra_field'):
            value = json.loads(json.dumps(facts['problem']))
            if change == 'fabricated':
                value['evidence']['target_column'] = ['a quote not present in the task']
            elif change == 'wrong_target':
                value['target_column'] = 'invented_target'
            elif change == 'missing_evidence':
                value['evidence'].pop('task_type')
            else:
                value['model_family'] = 'extra_trees'
            with self.subTest(change=change), self.assertRaises(ValidationError):
                ProblemFacts.model_validate_json(json.dumps(value), context={'source': text})

    def test_arbitrary_metrics_and_output_shapes_are_not_execution_constraints(self):
        _, text, facts = expected('classification')
        facts['metric']['name'] = 'Accuracy'
        facts['metric']['direction'] = 'minimize'
        facts['output']['columns'].reverse()
        # Source presence is checked here; interpreting requirements belongs to preparation.
        result = Extraction.model_validate(facts, context={'source': text})
        self.assertEqual(result.metric.name, 'Accuracy')
        self.assertEqual(result.output.columns, ['attack_cat', 'id'])

    async def test_unavailable_inputs_and_custom_metric_do_not_block_extraction(self):
        text = 'Train: mystery.parquet. Test: holdout.jsonl. Score = median(abs(error)) + 3; direction unstated.'
        facts = dict(problem={'evidence': {}},
            data={'train_file': 'mystery.parquet', 'test_file': 'holdout.jsonl',
                  'evidence': {'train_file': [text], 'test_file': [text]}},
            metric={'name': 'custom', 'definition': 'median(abs(error)) + 3',
                    'evidence': {'name': [text], 'definition': [text]}},
            output={'evidence': {}})
        extracted = Extraction.model_validate(facts, context={'source': text})
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'task.md'
            path.write_text(text)
            output = Path(temporary) / 'run'
            with patch.object(JSONAgents, 'extract', return_value=extracted), \
                    patch('alfa_agent.data.inventory', side_effect=AssertionError('No data inspection during extraction')):
                report = await extract_task(path, output, Settings('fake', 'https://qwen.example/v1', 'model'))
            self.assertEqual(report.status, 'ready', report.errors)
            self.assertTrue((output / 'extraction.json').is_file())
            self.assertFalse((output / 'preparation.json').exists())
            self.assertIsNone(extracted.metric.direction)

    def test_nested_evidence_is_normalized_without_weakening_source_validation(self):
        text = 'Input a.tsv is TSV; b.jsonl is JSON Lines.'
        facts = {'formats': {'a.tsv': 'TSV', 'b.jsonl': 'JSON Lines'},
                 'evidence': {'formats': {'a.tsv': [text], 'b.jsonl': text}}}
        result = DataFacts.model_validate(facts, context={'source': text})
        self.assertEqual(result.evidence['formats'], [text])
        facts['evidence']['formats']['b.jsonl'] = 'fabricated source quote'
        with self.assertRaises(ValidationError):
            DataFacts.model_validate(facts, context={'source': text})

    def test_long_evidence_and_empty_field_annotations_do_not_reject_real_quotes(self):
        text = '\n'.join(f'Input part {i}: supplied data.' for i in range(12))
        value = {'warnings': text.splitlines(), 'class_labels': [],
                 'evidence': {'warnings': text.splitlines(), 'class_labels': [text]}}
        result = ProblemFacts.model_validate(value, context={'source': text})
        self.assertEqual(len(result.evidence['warnings']), 12)
        self.assertNotIn('class_labels', result.evidence)


if __name__ == '__main__':
    unittest.main()

"""Read-only viewer checks; no browser, training or external API requests."""

import json
from pathlib import Path
import tempfile
import unittest

from viewer import collect, page


class ViewerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.run = self.root / 'regression' / 'run-1'
        self.node = self.run / 'nodes' / 'node_000'
        self.node.mkdir(parents=True)
        self.write(self.run / 'task.json', {'problem': {'task_type': 'regression'},
                                          'metric': {'name': 'rmse', 'direction': 'minimize'}})
        self.write(self.node / 'start.json', {'node_id': 'node_000', 'parent_id': None, 'source': 'baseline'})

    @staticmethod
    def write(path, data):
        path.write_text(json.dumps(data), encoding='utf-8')

    def test_new_finish_appears_on_next_read_without_losing_start_or_parent(self):
        initial = collect(self.root)['runs'][0]
        self.assertIsNone(initial['nodes'][0]['finish'])
        self.write(self.node / 'finish.json', {'status': 'completed', 'value': 4.2})
        updated = collect(self.root)['runs'][0]
        self.assertEqual(updated['nodes'][0]['finish']['value'], 4.2)
        self.assertIsNone(updated['nodes'][0]['start']['parent_id'])
        self.assertEqual(updated['id'], 'regression/run-1')

    def test_partial_json_does_not_hide_other_nodes_or_read_credentials(self):
        (self.run / '.env').write_text('OPENAI_API_KEY=viewer-must-not-read-this', encoding='utf-8')
        (self.node / 'finish.json').write_text('{unfinished', encoding='utf-8')
        snapshot = collect(self.root)
        self.assertEqual(len(snapshot['runs'][0]['nodes']), 1)
        self.assertTrue(snapshot['runs'][0]['warnings'])
        self.assertNotIn('viewer-must-not-read-this', json.dumps(snapshot))

    def test_export_cannot_execute_html_from_model_text(self):
        payload = '</script><img src=x onerror="alert(1)">'
        self.write(self.node / 'start.json', {'node_id': 'node_000', 'hypothesis': {'rationale': payload}})
        document = page(collect(self.root)).decode()
        embedded = document.split('<script id="initial-data" type="application/json">', 1)[1].split('</script>', 1)[0]
        self.assertNotIn('<', embedded)
        self.assertEqual(json.loads(embedded)['runs'][0]['nodes'][0]['start']['hypothesis']['rationale'], payload)

    def test_selected_run_directory_and_empty_root_are_supported(self):
        self.assertEqual(len(collect(self.run)['runs']), 1)
        self.assertEqual(collect(self.root / 'not-created-yet')['runs'], [])

    def test_symlink_cannot_read_json_outside_selected_root(self):
        outside = self.root / 'outside.json'
        self.write(outside, {'secret': 'outside-file'})
        (self.node / 'finish.json').symlink_to(outside)
        snapshot = collect(self.run)
        self.assertIsNone(snapshot['runs'][0]['nodes'][0]['finish'])
        self.assertNotIn('outside-file', json.dumps(snapshot))

    def test_partial_extraction_is_visible_before_task_or_report_exists(self):
        extraction = self.root / 'extraction' / 'run-1'
        extraction.mkdir(parents=True)
        facts = {'train_file': 'train.csv', 'evidence': {'train_file': ['Файл train.csv']}}
        self.write(extraction / 'data.json', facts)
        run = collect(extraction)['runs'][0]
        self.assertEqual(run['kind'], 'extraction')
        self.assertEqual(run['blocks']['data'], facts)
        self.assertIsNone(run['task'])
        self.assertIsNone(run['extraction'])
        self.assertIsNone(run['report'])
        self.write(extraction / 'metric.json', {'name': 'rmse', 'evidence': {'name': ['RMSE']}})
        updated = collect(self.root)['runs']
        self.assertEqual(len(updated), 2)
        self.assertEqual(next(r for r in updated if r['kind'] == 'extraction')['blocks']['metric']['name'], 'rmse')

    def test_failed_extraction_report_is_visible_without_any_accepted_block(self):
        extraction = self.root / 'failed'
        extraction.mkdir()
        report = {'status': 'failed', 'task_sha256': 'hash', 'checks': [],
                  'errors': ['timeout'], 'llm_calls': [{'role': 'problem', 'status': 'api_error'}]}
        self.write(extraction / 'report.json', report)
        run = collect(extraction)['runs'][0]
        self.assertEqual(run['kind'], 'extraction')
        self.assertEqual(run['report'], report)
        self.assertTrue(all(value is None for value in run['blocks'].values()))

    def test_multiple_roots_keep_extraction_and_experiments_distinct(self):
        extraction = self.root / 'extraction'
        extraction.mkdir()
        facts = {'target_column': 'value', 'evidence': {'target_column': ['value']}}
        self.write(extraction / 'problem.json', facts)
        self.write(extraction / 'extraction.json', {'schema_version': 1, 'problem': facts})
        self.write(extraction / 'task.json', {'problem': {'target_column': 'value'}})
        runs = collect(self.root / 'regression', extraction)['runs']
        self.assertEqual({r['id'] for r in runs}, {'regression/run-1', 'extraction/extraction'})
        extracted = next(r for r in runs if r['kind'] == 'extraction')
        self.assertEqual(extracted['extraction']['problem'], facts)
        self.assertEqual(extracted['task']['problem']['target_column'], 'value')

    def test_new_layout_is_one_run_with_preparation_node_and_source(self):
        root = self.root / 'artifacts'
        run = root / 'run-new'
        blocks = run / 'extraction'
        node = run / 'nodes' / 'node_000'
        blocks.mkdir(parents=True)
        node.mkdir(parents=True)
        facts = {'target_column': 'price', 'evidence': {'target_column': ['predict price']}}
        self.write(blocks / 'problem.json', facts)
        self.write(run / 'extraction.json', {'problem': facts, 'metric': {'name': 'custom'}})
        self.write(run / 'report.json', {'status': 'running', 'stage': 'baseline', 'task_sha256': 'hash'})
        self.write(run / 'preparation.json', {'status': 'ready', 'profile': {'train': {'rows': 123}}})
        self.write(node / 'start.json', {'node_id': 'node_000', 'parent_id': None, 'source': 'qwen'})
        (run / 'prepare.py').write_text('def load_data(): pass')
        (node / 'experiment.py').write_text('def fit_predict(): pass')
        snapshot = collect(root)
        self.assertEqual(len(snapshot['runs']), 1)
        collected = snapshot['runs'][0]
        self.assertEqual(collected['blocks']['problem'], facts)
        self.assertEqual(collected['preparation']['profile']['train']['rows'], 123)
        self.assertEqual(collected['preparation_code'], 'def load_data(): pass')
        self.assertEqual(collected['nodes'][0]['code'], 'def fit_predict(): pass')
        self.assertEqual(collected['task']['metric']['name'], 'custom')

    def test_partially_extracted_new_run_is_visible_without_assembled_json(self):
        root = self.root / 'artifacts'
        blocks = root / 'run-new' / 'extraction'
        blocks.mkdir(parents=True)
        self.write(blocks / 'metric.json', {'name': 'MAE', 'evidence': {'name': ['MAE']}})
        snapshot = collect(root)
        self.assertEqual(len(snapshot['runs']), 1)
        self.assertEqual(snapshot['runs'][0]['id'], 'run-new')
        self.assertIsNone(snapshot['runs'][0]['extraction'])


if __name__ == '__main__':
    unittest.main()

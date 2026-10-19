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


if __name__ == '__main__':
    unittest.main()

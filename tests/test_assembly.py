"""Recombination, optional blocks and immutable versions across multiple sources."""

from pathlib import Path
import tempfile
import unittest

import pandas as pd

from agent.assembly import BlockError, assemble, execute, initialize
from agent.contracts import Assembly, Start
from agent.data import file_hash
from agent.storage import save


FEATURES = '''def run(state):
    return {'x_valid': state['validation']['signal'].to_numpy(),
            'x_test': state['test']['signal'].to_numpy()}
'''
MODEL = '''import pandas as pd

class CoupledModel:
    def predict(self, values):
        return values * 2

def run(state):
    model = CoupledModel()
    return {'validation_predictions': pd.DataFrame({'target': model.predict(state['x_valid'])}),
            'test_predictions': pd.DataFrame({'target': model.predict(state['x_test'])})}
'''
CALIBRATION = '''def run(state):
    return {'validation_predictions': state['validation_predictions'],
            'test_predictions': state['test_predictions']}
'''


class AssemblyTests(unittest.TestCase):
    def test_mix_two_sources_omit_optional_block_and_preserve_source_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            features = {'name': 'features', 'source_node': 'node_001', 'description': 'Select signal arrays',
                        'requires': ['validation', 'test'], 'provides': ['x_valid', 'x_test']}
            model = {'name': 'model', 'source_node': 'node_002', 'description': 'Coupled model',
                     'requires': ['x_valid', 'x_test'], 'provides': ['validation_predictions', 'test_predictions']}
            calibrate = {'name': 'calibrate', 'source_node': None, 'description': 'Optional identity calibration',
                         'requires': ['validation_predictions', 'test_predictions'],
                         'provides': ['validation_predictions', 'test_predictions']}
            for node_id, name, code in [('node_001', 'features', FEATURES), ('node_002', 'model', MODEL)]:
                path = root / 'nodes' / node_id / 'modules' / f'{name}.py'
                path.parent.mkdir(parents=True)
                path.write_text(code, encoding='utf-8')
            hashes = {b['name']: file_hash(root / 'nodes' / b['source_node'] / 'modules' / (b['name'] + '.py'))
                      for b in (features, model)}
            for number, blocks, edit in [('003', [features, model, calibrate], 'calibrate'),
                                         ('004', [features, model], 'model')]:
                node = root / 'nodes' / ('node_' + number)
                start = Start(node_id=node.name, parent_id='node_002', purpose='improve', preparation_sha256='a'*64,
                    context={'source_hashes': hashes}, assembly=Assembly(blocks=blocks, edit_block=edit),
                    hypothesis={'rationale': 'Recombine known modules', 'expected_effect': 'Keep predictions',
                                'pipeline': {'data': {}, 'preprocessing': {}, 'model': {}, 'training': {}}},
                    outputs={}, timeout_seconds=10.)
                save(node / 'start.json', start)
                initialize(node, start)
                if edit == 'calibrate':
                    (node / 'modules/calibrate.py').write_text(CALIBRATION, encoding='utf-8')
                assemble(node, start)
                validation, test = execute(node, pd.DataFrame(), pd.DataFrame({'signal': [3, 7]}),
                                           pd.DataFrame({'signal': [5]}), {})
                self.assertEqual(validation.target.tolist(), [6, 14])
                self.assertEqual(test.target.tolist(), [10])
                for b in (features, model):
                    self.assertEqual(file_hash(node / 'modules' / (b['name'] + '.py')), hashes[b['name']])
            source = root / 'nodes/node_001/modules/features.py'
            source.write_text(FEATURES + '\n# changed source\n', encoding='utf-8')
            with self.assertRaisesRegex(BlockError, 'Source block changed'):
                initialize(root / 'nodes/node_005', start)

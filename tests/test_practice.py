"""The held-out protocol and scorer: no external API, no model fitting."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from evaluation import practice
from evaluation.catalog import CATALOG, FILES


class PracticeTests(unittest.TestCase):
    def test_all_five_built_datasets_have_valid_splits_and_checksums(self):
        with redirect_stdout(io.StringIO()):
            practice.verify(list(CATALOG))
        self.assertEqual(len(CATALOG), 5)

    def test_perfect_predictions_score_perfectly(self):
        for slug, spec in CATALOG.items():
            with self.subTest(dataset=slug):
                result = practice.score(slug, practice.PRIVATE / slug / 'test_labels.csv')
                self.assertAlmostEqual(result['value'], 0 if spec['direction'] == 'minimize' else 1)

    def test_metric_definitions(self):
        self.assertAlmostEqual(practice.metric_value([0., 4.], [0., 0.], CATALOG['concrete_strength']), np.sqrt(8))
        self.assertAlmostEqual(practice.metric_value([0., 3.], [0., 0.], CATALOG['bike_demand']), np.log(4) / np.sqrt(2))
        self.assertAlmostEqual(practice.metric_value([0, 1, 0, 1], [.1, .9, .8, .2], CATALOG['bank_subscription']), .75)
        self.assertEqual(practice.metric_value(['ham', 'spam'], ['ham', 'ham'], CATALOG['sms_spam']), 0)
        # All seven classes must count, including a class absent from predictions.
        beans = CATALOG['dry_bean']
        self.assertAlmostEqual(practice.metric_value(beans['labels'], ['BOMBAY'] * 7, beans), (2 / 8) / 7)

    def test_scorer_rejects_invalid_submissions(self):
        spec = CATALOG['bank_subscription']
        truth = pd.DataFrame({'id': ['first', 'second'], 'subscribed': [0, 1]})
        valid = truth.assign(subscribed=[.2, .8])
        mutations = [
            valid.iloc[::-1], valid.iloc[:1], valid.assign(extra=0),
            valid[['subscribed', 'id']], valid.assign(id=['first', 'first']),
            valid.assign(id=['first', '']), valid.assign(id=['first', 'foreign']),
            valid.assign(subscribed=[float('nan'), .8]), valid.assign(subscribed=[float('inf'), .8]),
            valid.assign(subscribed=[-.1, .8]), valid.assign(subscribed=[.2, 1.1]),
            valid.assign(subscribed=['yes', 'no']), valid.assign(subscribed=['', .8]),
        ]
        for index, table in enumerate(mutations):
            with self.subTest(case=index), self.assertRaises(ValueError):
                practice.validate_prediction(table, truth, spec)
        with self.assertRaisesRegex(ValueError, 'nonnegative'):
            practice.validate_prediction(pd.DataFrame({'id': ['a'], 'rental_count': [-1.]}),
                                         pd.DataFrame({'id': ['a'], 'rental_count': [1]}), CATALOG['bike_demand'])
        with self.assertRaisesRegex(ValueError, 'Unknown'):
            practice.validate_prediction(pd.DataFrame({'id': ['a'], 'message_class': ['unknown']}),
                                         pd.DataFrame({'id': ['a'], 'message_class': ['spam']}), CATALOG['sms_spam'])

    def test_export_contains_only_participant_allowlist_and_does_not_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / 'input'
            practice.export('sms_spam', destination)
            self.assertEqual({p.name for p in destination.iterdir()}, set(FILES))
            for name in FILES:
                self.assertEqual(practice.sha256(destination / name), practice.sha256(practice.PUBLIC / 'sms_spam' / name))
            with self.assertRaises(FileExistsError):
                practice.export('sms_spam', destination)

    def test_scorer_rejects_tampered_hidden_labels(self):
        with tempfile.TemporaryDirectory() as temporary:
            private = Path(temporary)
            destination = private / 'sms_spam'
            destination.mkdir()
            source = practice.PRIVATE / 'sms_spam'
            (destination / 'manifest.json').write_bytes((source / 'manifest.json').read_bytes())
            (destination / 'test_labels.csv').write_bytes((source / 'test_labels.csv').read_bytes() + b'\n')
            with patch.object(practice, 'PRIVATE', private), self.assertRaisesRegex(ValueError, 'checksum'):
                practice.score('sms_spam', practice.PUBLIC / 'sms_spam' / 'sample_submission.csv')

    @unittest.skipUnless(all((practice.CACHE / spec['cache']).exists() for spec in CATALOG.values()),
                         'Download source cache with evaluation.practice build before the offline rebuild check')
    def test_offline_rebuild_is_byte_reproducible(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch.object(practice, 'PUBLIC', root / 'public'), patch.object(practice, 'PRIVATE', root / 'private'):
                with redirect_stdout(io.StringIO()):
                    practice.build(list(CATALOG), offline=True)
            for slug in CATALOG:
                stored = json.loads((practice.PRIVATE / slug / 'manifest.json').read_text(encoding='utf-8'))
                rebuilt = json.loads((root / 'private' / slug / 'manifest.json').read_text(encoding='utf-8'))
                self.assertEqual(stored, rebuilt, slug)
                for name in FILES:
                    self.assertEqual(practice.sha256(practice.PUBLIC / slug / name),
                                     practice.sha256(root / 'public' / slug / name))


if __name__ == '__main__':
    unittest.main()

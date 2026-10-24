"""Transient Windows locks must not stop the launcher or freeze viewer time."""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.vitaly_state import publish_manifest, save_manifest
from tools.vitaly_viewer import snapshot
from ac_experiment import ExperimentSpec
from ac_research import Proposal, research_loop


class RunStatusTests(unittest.TestCase):
    def test_transient_windows_lock_retried_without_corrupting_json(self):
        original = os.replace
        calls = []
        def replace(source, target):
            calls.append(1)
            if len(calls) < 3:raise PermissionError('Sharing violation')
            return original(source, target)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'state.json'
            with patch('tools.vitaly_state.os.replace', side_effect=replace), patch('tools.vitaly_state.time.sleep'):
                save_manifest(path, {'status': 'completed'})
            self.assertEqual(json.loads(path.read_text()), {'status': 'completed'})
            self.assertEqual(len(calls), 3)

    def test_persistent_lock_returns_control_and_keeps_previous_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'state.json'
            path.write_text('{"status":"running"}')
            with patch('tools.vitaly_state.os.replace', side_effect=PermissionError('locked')), patch('tools.vitaly_state.time.sleep'):
                self.assertFalse(publish_manifest(path, {'status': 'completed'}))
            self.assertEqual(json.loads(path.read_text())['status'], 'running')
            self.assertEqual([p.name for p in path.parent.iterdir()], ['state.json'])

    def test_completed_report_recovers_stale_launcher_status(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory);artifact = root/'artifacts/run';artifact.mkdir(parents=True)
            (artifact/'report.json').write_text(json.dumps({'final_status': 'ok', 'elapsed_s': 120.,
                'submission_checks': {'ok': True}, 'final_cv': .6}))
            manifest = root/'artifacts/latest_launch.json'
            manifest.write_text(json.dumps({'status': 'running', 'elapsed_seconds': 3.,
                'budget_seconds': 1800., 'artifacts': str(artifact)}))
            result = snapshot(root)
            self.assertEqual(result['launch']['status'], 'completed')
            self.assertEqual(result['launch']['elapsed_seconds'], 120.)
            self.assertEqual(result['launch']['remaining_seconds'], 1680.)

    def test_new_experiment_not_started_without_measured_budget(self):
        lab = MagicMock()
        record = {'id': 'exp_001', 'cv': .6, 'status': 'ok'}
        lab.run.return_value = record
        lab.mem.state = {'experiments': [record]}
        lab.mem.ok_experiments.return_value = [record]
        lab.settings.max_experiments = 2
        lab.budget.total_s = 1800.
        lab.left.return_value = 20.
        lab.balance_proposal.return_value = None
        lab.recovery_proposal.return_value = None
        lab.history_proposal.return_value = None
        lab.protocol_proposal.return_value = None
        lab.heuristic.return_value = Proposal(hypothesis='test', why='measured', spec=ExperimentSpec())
        lab.estimate_seconds.return_value = 40.
        llm = MagicMock();llm.enabled = False
        self.assertEqual(research_loop(lab, llm, lambda message: None, lambda record: None), 'time_budget')
        self.assertEqual(lab.run.call_count, 1)  # baseline only
        self.assertEqual(lab.mem.log.call_args.kwargs['reason'], 'insufficient_measured_research_budget')


if __name__ == '__main__':
    unittest.main()

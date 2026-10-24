"""Measured recipes remain explicit choices and survive the planner unchanged."""
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from agent.contracts import Ideas
from agent.feature_trials import candidate_id, candidate_view, check_recipes


class FeatureTrialTests(unittest.TestCase):
    def test_untested_recipe_needs_an_explicit_decision(self):
        candidate = {'features':[{'name':'frequency_x','operation':'frequency','columns':['x']}],
                     'trial_status':'untested'}
        identity = candidate_id(candidate)
        context = {'purposes':['improve'],'parents':['node_000'],'leader':'node_000',
                   'feature_candidates':[candidate]}
        with self.assertRaises(ValueError):
            Ideas.model_validate({},context=context)
        response = Ideas.model_validate({'feature_decisions':[{'candidate_id':identity,'action':'defer',
            'reason':'First finish the pending model experiment'}]},context=context)
        self.assertEqual(response.feature_decisions[0].candidate_id, identity)

    def test_recipe_is_tracked_by_executed_content_and_must_survive_planning(self):
        recipe = {'name':'frequency_x','operation':'frequency','columns':['x']}
        candidate = {'features':[recipe]}
        empty = SimpleNamespace(assembly=SimpleNamespace(model_dump=lambda:{'blocks':[]}))
        with self.assertRaises(ValueError):
            check_recipes(empty,[candidate])
        assembly = {'blocks':[{'operation':{'kind':'features','parameters':{'recipes':[recipe]}}}]}
        assembly['blocks'] += [{'operation':{'kind':kind, 'inputs': {'fit':'fit_table'} if kind == 'preprocess' else {}}}
                               for kind in ('preprocess','fit','predict')]
        proposal = SimpleNamespace(assembly=SimpleNamespace(model_dump=lambda:assembly))
        check_recipes(proposal,[candidate])
        row = SimpleNamespace(node_id='node_001',assembly=proposal.assembly,
                              result=SimpleNamespace(status='completed',value=12.))
        with tempfile.TemporaryDirectory() as temporary:
            result = candidate_view([candidate],Path(temporary),[row],[],{})[0]
        self.assertEqual(result['trial_status'],'tested')
        self.assertEqual(result['trials'][0]['node_id'],'node_001')


if __name__ == '__main__':
    unittest.main()

"""Measured recipes remain explicit choices and survive the planner unchanged."""
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from agent.contracts import Ideas, validated_ideas
from agent.feature_trials import candidate_id, candidate_view, check_recipes


class FeatureTrialTests(unittest.TestCase):
    def test_one_invalid_hypothesis_does_not_discard_valid_work(self):
        idea = {'purpose': 'improve', 'parent_id': 'node_000', 'change': 'Try a different regularization',
                'expected_effect': 'Measure the change'}
        result, rejected = validated_ideas({'ideas': [idea, {**idea, 'parent_id': 'node_999'}]},
            {'purposes': ['improve'], 'parents': ['node_000'], 'leader': 'node_000', 'max_new_ideas': 2})
        self.assertEqual(len(result.ideas), 1)
        self.assertEqual(rejected[0]['index'], 1)
        with self.assertRaises(ValueError):
            validated_ideas({'ideas': [{**idea, 'parent_id': 'node_999'}]},
                {'purposes': ['improve'], 'parents': ['node_000'], 'leader': 'node_000'})

    def test_omitted_recipe_remains_deferred_without_rejecting_the_response(self):
        candidate = {'features':[{'name':'frequency_x','operation':'frequency','columns':['x']}],
                     'trial_status':'untested'}
        identity = candidate_id(candidate)
        context = {'purposes':['improve'],'parents':['node_000'],'leader':'node_000',
                   'feature_candidates':[candidate]}
        default = Ideas.model_validate({},context=context)
        self.assertEqual(default.feature_decisions[0].candidate_id, identity)
        self.assertEqual(default.feature_decisions[0].action, 'defer')
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

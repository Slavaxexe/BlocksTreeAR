"""Context selection preserves evidence, exact configuration and local-only Baker."""
import copy
import json
import time
import unittest
from unittest.mock import AsyncMock, patch

from agent.agents import JSONAgents
from agent.config import Settings
from agent.context import (analyzer_context, compact, experiment_index, prompt_schema,
                           repair_context, select_experiments, table)
from agent.contracts import Ideas


class ContextTests(unittest.TestCase):
    def test_long_history_keeps_leader_diversity_recent_failure_and_complete_index(self):
        cards = [{'node_id': f'node_{n:03d}', 'parent_id': 'node_000',
                  'hypothesis': {'pipeline': {'model': {'name': 'forest' if n != 1 else 'linear'}}},
                  'result': {'status': 'completed', 'value': n / 100, 'duration_seconds': n}}
                 for n in range(100)]
        cards[-1]['result'].update(status='failed', value=None)
        before = copy.deepcopy(cards)
        selected = select_experiments(cards, 'node_090', 'maximize')
        self.assertLessEqual(len(selected), 8)
        self.assertTrue({'node_090', 'node_099', 'node_098', 'node_001'} <= {row['node_id'] for row in selected})
        self.assertEqual(len(experiment_index(cards)['rows']), 100)
        self.assertEqual(cards, before)

    def test_compaction_preserves_table_values_property_names_and_meaningful_zeroes(self):
        rows = [{'group': name, 'score': score, 'tested': False} for name, score in [('a', 0), ('b', None), ('c', .5)]]
        packed = table(rows)
        self.assertEqual([dict(zip(packed['columns'], row)) for row in packed['rows']], rows)
        self.assertEqual(compact({'zero': 0, 'flag': False, 'empty': [], 'text': 'a' * 5000}),
                         {'zero': 0, 'flag': False, 'text': 'a' * 5000})
        schema = {'title': 'Root', 'properties': {'title': {'type': 'string', 'title': 'Title'},
                                                  'description': {'type': 'string', 'description': 'Keep this meaning'}}}
        self.assertEqual(set(prompt_schema(schema)['properties']), {'title', 'description'})
        self.assertEqual(prompt_schema(schema)['properties']['description']['description'], 'Keep this meaning')

    def test_repair_filters_research_and_analyzer_uses_actual_configuration_once(self):
        call = {'kind': 'fit', 'parameters': {'estimator': 'sklearn.ensemble.ExtraTreesRegressor', 'parameters': {'n_estimators': 20}}}
        block = {'name': 'fit', 'requires': ['fit_features', 'fit_targets'], 'provides': ['model'], 'operation': call}
        hypothesis = {'rationale': 'Fit a small forest', 'research_ids': ['r1'],
                      'pipeline': {'model': {'name': 'ExtraTrees'}, 'training': {'description': 'Fit only'}}}
        context = {'requirements': {}, 'preparation': {}, 'installed_libraries': {}, 'node_id': 'node_001',
            'purpose': 'improve', 'hypothesis': hypothesis, 'assembly': {'blocks': [block]}, 'target_block': block,
            'previous_operation': call, 'execution_error': 'Invalid parameter', 'failed_stage': 'check',
            'experiment_timeout_seconds': 60, 'operations': {'fit': {}, 'automl': {}},
            'data_research': {'insights': [{'id': 'r1', 'statement': 'Relevant'}, {'id': 'r2', 'statement': 'Other'}]}}
        selected = repair_context(context)
        self.assertEqual([item['id'] for item in selected['data_research']['insights']], ['r1'])
        self.assertNotIn('assembly', selected)
        row = {'node_id': 'node_001', 'parent_id': 'node_000', 'hypothesis': hypothesis,
               'assembly': {'blocks': [block]}, 'result': {'status': 'completed', 'value': 1.}}
        analyzed = analyzer_context({**context, 'experiment': row, 'modules': {'fit': 'do not send this wrapper'},
            'tool_configurations': {'fit': call}, 'automl_diagnostics': [{'configuration': call, 'completed_trials': 2}]})
        self.assertNotIn('modules', analyzed)
        self.assertNotIn('configuration', analyzed['automl_diagnostics'][0])
        self.assertEqual(analyzed['saved_operations'][0]['operation']['parameters']['parameters']['n_estimators'], 20)
        self.assertEqual(analyzed['automl_diagnostics'][0]['completed_trials'], 2)


class BakerTests(unittest.IsolatedAsyncioTestCase):
    async def test_baker_has_no_tools_or_archive_and_keeps_validation_sources(self):
        client = JSONAgents(Settings('fake', 'https://example.test/v1', 'model'), time.monotonic() + 5)
        context = {'purposes': ['improve'], 'available_parents': ['node_000'], 'leader_id': 'node_000',
                   'max_new_ideas': 1, 'previous_ideas': [], 'prediction_sources': [], 'experiments': []}
        with patch.object(client, 'ask', new=AsyncMock(return_value=Ideas())) as ask:
            await client.propose_ideas(context)
        self.assertFalse(ask.call_args.kwargs['reference'])
        payload = json.loads(ask.call_args.args[1].rsplit('\n', 1)[1])
        self.assertNotIn('archive_tool', payload)
        self.assertEqual(ask.call_args.kwargs['validation']['parents'], ['node_000'])

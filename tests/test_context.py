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
    def test_baker_can_select_a_ready_eda_build_without_copying_its_recipes(self):
        from agent.builds import register_eda
        from pathlib import Path
        import tempfile
        from agent.contracts import Ideas
        with tempfile.TemporaryDirectory() as temporary:
            configuration = {'features': [], 'preprocess': {'feature_columns': ['x']},
                'fit': {'estimator': 'sklearn.linear_model.Ridge', 'parameters': {'alpha': 1.}}}
            build = register_eda(Path(temporary), 'eda_' + 'b' * 16, configuration, {})
            context = {'purposes': ['improve'], 'parents': ['node_000'], 'leader': 'node_000',
                'max_new_ideas': 1, 'build_ids': [build['id']], 'builds': {build['id']: build},
                'feature_candidates': [{'id': 'feature_test', 'features': [], 'build_id': build['id'],
                                        'trial_status': 'untested'}]}
            answer = Ideas.model_validate({'ideas': [{'purpose': 'improve', 'parent_id': 'node_000',
                'change': 'Check measured recipe on outer validation', 'expected_effect': 'Measure transfer',
                'build_id': build['id']}]}, context=context)
            self.assertEqual(answer.ideas[0].feature_candidate_ids, [])
            self.assertEqual(answer.ideas[0].build_id, build['id'])

    def test_research_focus_keeps_schema_but_limits_detailed_statistics(self):
        from agent.context import research_profile_view
        profile = {'tables': {'train': {'rows': 10, 'columns': {
            name: {'dtype': 'float', 'unique': 4, 'missing': 1,
                   'quantiles': {'0.0': 0, '0.5': 2, '1.0': 4}}
            for name in ['relevant', 'other']}}}}
        view = research_profile_view(profile, detailed=False, focus_columns={'relevant'})
        facts = view['tables']['train']
        self.assertEqual(facts['columns_by_dtype']['float'], ['relevant', 'other'])
        self.assertEqual(set(facts['quantiles']), {'relevant'})

    def test_guidance_is_selected_by_actual_work_not_the_whole_catalog(self):
        from agent.guidance import augment_prompt, observed_rules
        prompt = 'Plan.\n' + json.dumps({'idea': {'change': 'Train CatBoostRegressor'},
            'operations': {'ablation': {}, 'history': {}}})
        updated, ids = augment_prompt(prompt, 'node_planner', {})
        self.assertEqual(ids, ['catboost_class'])
        self.assertIn('execution_guidance', json.loads(updated.rsplit('\n', 1)[1]))
        self.assertEqual(observed_rules("CatBoost.__init__() got an unexpected keyword argument"), ['catboost_class'])
        self.assertEqual(augment_prompt(prompt, 'metric', {}), (prompt, []))

    def test_compact_research_retains_claims_and_recipes_without_duplicate_evidence(self):
        from agent.context import research_memory_view
        text = 'A complete measured statement ' * 100
        memory = {'insights': [{'id': 'r1', 'statement': text, 'evidence': {'large': list(range(100)),
                               'score': {'kind': 'feature', 'baseline': {'value': 50}, 'candidate': {'value': 45}}},
                               'implication': 'Try this recipe'}], 'feature_candidates': [{'features': ['recipe']}],
                  'open_questions': ['Literal question']}
        view = research_memory_view(memory, review=True)
        self.assertEqual(view['insights'][0]['statement'], text)
        self.assertEqual(view['open_questions'], ['Literal question'])
        self.assertNotIn('evidence', view['insights'][0])
        self.assertNotIn('feature_candidates', view)
        self.assertEqual(view['insights'][0]['measured'][0]['candidate']['value'], 45)

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
        self.assertIn('operations', payload)
        self.assertIn('purpose', payload['operations']['fit'])
        self.assertNotIn('source_node=node_...', payload['operations']['fit'].get('purpose', ''))
        self.assertNotIn('profile', payload.get('preparation', {}))

    def test_research_memory_view_optimizations_for_baker_and_researcher(self):
        from agent.context import research_memory_view
        memory = {
            'revision': 1,
            'open_questions': ['How to cluster?'],
            'coverage': {'entities': 1},
            'insights': [{
                'id': 'res_1:0',
                'phase': 'fit',
                'statement': 'Target encoding building_class improves proxy score',
                'kind': 'feature',
                'importance': 'high',
                'confidence': 'high',
                'implication': 'Add target encoding',
                'evidence': {
                    'stat': {
                        'kind': 'feature',
                        'scope': 'fit',
                        'rows_used': 5000,
                        'improvement_percent': 12.3456,
                        'baseline': {'metric': 'rmse', 'value': 45.123456, 'inner_train_rows': 5000, 'inner_validation_rows': 1000},
                        'candidate': {'metric': 'rmse', 'value': 39.54321, 'inner_train_rows': 5000, 'inner_validation_rows': 1000}
                    }
                }
            }]
        }
        # Baker view: clean insight with improvement_percent, no open_questions/coverage, no raw measured tables
        baker_view = research_memory_view(memory, baker=True)
        self.assertNotIn('open_questions', baker_view)
        self.assertNotIn('coverage', baker_view)
        self.assertEqual(baker_view['insights'][0]['improvement_percent'], 12.35)
        self.assertNotIn('measured', baker_view['insights'][0])

        # Researcher view: keeps open_questions, coverage, and measured, but strips redundant inner_ split keys and rounds floats
        res_view = research_memory_view(memory, baker=False)
        self.assertIn('open_questions', res_view)
        self.assertIn('coverage', res_view)
        self.assertIn('measured', res_view['insights'][0])
        measured = res_view['insights'][0]['measured'][0]
        self.assertEqual(measured['improvement_percent'], 12.35)
        self.assertEqual(measured['baseline']['value'], 45.1235)
        self.assertNotIn('inner_train_rows', measured['baseline'])
        self.assertNotIn('inner_validation_rows', measured['candidate'])


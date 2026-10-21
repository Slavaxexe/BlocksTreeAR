"""Baker edits a priority queue while experiments and research keep running."""

import asyncio
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

from pydantic import ValidationError

from agent.agents import JSONAgents
from agent.config import Settings
from agent.contracts import Idea, Ideas, IdeaUpdate
from agent.runner import Options
from agent.search import Search


def idea(change='Try a different model', priority=50, purpose='improve'):
    return Idea(purpose=purpose, parent_id=None if purpose == 'baseline' else 'node_000',
                change=change, expected_effect='Test generalization', priority=priority)


def queued(index, priority=50, purpose='improve'):
    return {'id': f'idea_{index:03d}', 'status': 'queued', 'research_revision': 1,
            'idea': idea(f'Change {index}', priority, purpose).model_dump()}


def delta():
    return SimpleNamespace(nodes=[], metric=SimpleNamespace(direction='minimize'))


class QueueTests(unittest.IsolatedAsyncioTestCase):
    def make_search(self, directory, shared=None):
        agents = JSONAgents(Settings('fake', 'https://qwen.example/v1', 'model'), time.monotonic()+20, 100)
        return Search(agents, Path(directory), Options(parallel=2, max_experiments=10),
                      shared if shared is not None else {}, lambda message, **kwargs: None)

    async def collect_baker(self, search):
        task = search.baker
        await asyncio.wait_for(task, 2)
        search.collect({task})

    async def test_fresh_research_keeps_queued_ideas(self):
        with tempfile.TemporaryDirectory() as temporary:
            search = self.make_search(temporary, {'data_research': {'revision': 2}})
            record = queued(0, purpose='explore')
            search.queue.append(record)
            search.records.append(record)
            occupied = [SimpleNamespace(purpose=p) for p in ('improve', 'explore')]
            search.fill(delta(), occupied, 0)
            self.assertEqual(list(search.queue), [record])
            self.assertEqual(record['status'], 'queued')
            self.assertIsNone(search.baker)

    async def test_inflight_baker_reply_is_kept_and_next_request_gets_fresh_context(self):
        with tempfile.TemporaryDirectory() as temporary:
            shared = {'data_research': {'revision': 1, 'insights': ['first']}}
            search = self.make_search(temporary, shared)
            entered, release = asyncio.Event(), asyncio.Event()
            contexts = []

            async def propose(context):
                contexts.append(json.loads(json.dumps(context)))
                entered.set()
                await release.wait()
                return Ideas(ideas=[idea(f'Baseline {len(contexts)}', purpose='baseline')])

            with patch.object(search.agents, 'propose_ideas', side_effect=propose):
                search.request_ideas()
                await asyncio.wait_for(entered.wait(), 2)
                shared['data_research'] = {'revision': 2, 'insights': ['first', 'new evidence']}
                release.set()
                await self.collect_baker(search)
                self.assertEqual(search.records[0]['research_revision'], 1)
                self.assertEqual(search.misses, 0)
                self.assertEqual(len(search.queue), 1)
                search.queue.clear()
                search.request_ideas()
                await self.collect_baker(search)
            self.assertEqual(contexts[0]['data_research']['revision'], 1)
            self.assertEqual(contexts[1]['data_research'], shared['data_research'])
            saved = json.loads((Path(temporary) / 'planning/batch_002/request.json').read_text())
            self.assertEqual(saved['data_research']['insights'], ['first', 'new evidence'])

    async def test_retry_wait_refreshes_context_before_call(self):
        with tempfile.TemporaryDirectory() as temporary:
            shared = {'data_research': {'revision': 1}}
            search = self.make_search(temporary, shared)
            search.failures = 1
            entered, release = asyncio.Event(), asyncio.Event()

            async def wait(seconds):
                entered.set()
                await release.wait()

            with patch('agent.search.asyncio.sleep', side_effect=wait), \
                    patch.object(search.agents, 'propose_ideas', return_value=Ideas()) as propose:
                search.request_ideas()
                await asyncio.wait_for(entered.wait(), 2)
                shared['data_research'] = {'revision': 3}
                release.set()
                await self.collect_baker(search)
            self.assertEqual(propose.call_args.args[0]['data_research']['revision'], 3)

    async def test_priority_and_removal_edit_only_waiting_ideas(self):
        with tempfile.TemporaryDirectory() as temporary:
            search = self.make_search(temporary)
            first, second = queued(0), queued(1)
            search.queue.extend([first, second])
            search.records.extend([first, second])
            updates = [IdeaUpdate(idea_id='idea_000', action='remove', reason='The new analysis refuted it'),
                       IdeaUpdate(idea_id='idea_001', action='prioritize', priority=90, reason='Addresses the observed error')]
            search.baker = asyncio.create_task(asyncio.sleep(0, result=Ideas(updates=updates)))
            await self.collect_baker(search)
            self.assertEqual(first['status'], 'removed')
            self.assertEqual(first['decision']['reason'], updates[0].reason)
            self.assertEqual(search.report()['queued_ideas'][0]['priority'], 90)
            search.queue.remove(second)
            second['status'] = 'planning'
            search.apply_updates([IdeaUpdate(idea_id=second['id'], action='remove', reason='Arrived too late')])
            self.assertEqual(second['status'], 'planning')
            self.assertEqual(second['decision']['action'], 'prioritize')
            second['status'] = 'planned'
            search.apply_updates([IdeaUpdate(idea_id=second['id'], action='remove', reason='Already running')])
            self.assertEqual(second['status'], 'planned')

    async def test_dispatch_selects_highest_priority_with_a_free_purpose_slot(self):
        with tempfile.TemporaryDirectory() as temporary:
            search = self.make_search(temporary)
            low, blocked, high = queued(0, 10), queued(1, 100, 'explore'), queued(2, 90)
            search.queue.extend([low, blocked, high])
            planner = SimpleNamespace(plan_node=AsyncMock(return_value=None))
            with patch('agent.search.JSONAgents', return_value=planner):
                search.dispatch([SimpleNamespace(purpose='explore')])
                await asyncio.gather(*search.tasks)
            self.assertEqual(high['status'], 'planning')
            self.assertEqual(list(search.queue), [low, blocked])

    async def test_new_analysis_requests_baker_even_with_a_full_queue(self):
        with tempfile.TemporaryDirectory() as temporary:
            shared = {'data_research': {'revision': 1}}
            search = self.make_search(temporary, shared)
            search.next_node = 1
            records = [queued(0), queued(1)]
            search.queue.extend(records)
            search.records.extend(records)
            row = SimpleNamespace(node_id='node_000', result=SimpleNamespace(status='completed', value=1),
                                  analysis_status='completed', analysis={'summary': 'New error analysis'})
            history = SimpleNamespace(nodes=[row], metric=SimpleNamespace(direction='minimize'))
            occupied = [SimpleNamespace(purpose=p) for p in ('improve', 'explore')]
            with patch('agent.search.block_catalog', return_value={}), \
                    patch('agent.search.idea_history_view', side_effect=lambda item: {'analysis': item.analysis}), \
                    patch.object(search.agents, 'propose_ideas', return_value=Ideas()) as propose:
                search.fill(history, occupied, 1)
                await self.collect_baker(search)
                self.assertEqual(propose.call_args.args[0]['experiments'][0]['analysis'], row.analysis)
                search.fill(history, occupied, 1)
                self.assertIsNone(search.baker)
                shared['data_research'] = {'revision': 2}
                search.fill(history, occupied, 1)
                await self.collect_baker(search)
            self.assertEqual(propose.call_count, 2)
            self.assertEqual(propose.call_args.args[0]['data_research']['revision'], 2)
            self.assertEqual(list(search.queue), records)

    async def test_feedback_during_baker_is_coalesced_into_one_followup(self):
        with tempfile.TemporaryDirectory() as temporary:
            shared = {'data_research': {'revision': 1}}
            search = self.make_search(temporary, shared)
            search.next_node = 1
            records = [queued(0), queued(1)]
            search.queue.extend(records)
            search.records.extend(records)
            row = SimpleNamespace(node_id='node_000', result=SimpleNamespace(status='completed', value=1),
                                  analysis_status='completed', analysis={'summary': 'First analysis'})
            history = SimpleNamespace(nodes=[row], metric=SimpleNamespace(direction='minimize'))
            occupied = [SimpleNamespace(purpose=p) for p in ('improve', 'explore')]
            entered, release = asyncio.Event(), asyncio.Event()
            contexts = []

            async def propose(context):
                contexts.append(json.loads(json.dumps(context)))
                entered.set()
                await release.wait()
                return Ideas(updates=[IdeaUpdate(idea_id='idea_000', action='prioritize', priority=80,
                                                reason='Supported by measured evidence')])

            with patch('agent.search.block_catalog', return_value={}), \
                    patch('agent.search.idea_history_view', side_effect=lambda item: {'analysis': item.analysis}), \
                    patch.object(search.agents, 'propose_ideas', side_effect=propose) as mock:
                search.fill(history, occupied, 1)
                await asyncio.wait_for(entered.wait(), 2)
                first_call = search.baker
                row.analysis = {'summary': 'Latest analysis'}
                shared['data_research'] = {'revision': 3, 'insights': ['Latest evidence']}
                search.fill(history, occupied, 2)
                search.fill(history, occupied, 3)
                self.assertIs(search.baker, first_call)
                self.assertEqual(mock.call_count, 1)
                release.set()
                await self.collect_baker(search)
                self.assertEqual(records[0]['idea']['priority'], 80)
                search.fill(history, occupied, 3)
                await self.collect_baker(search)
                search.fill(history, occupied, 3)
                self.assertIsNone(search.baker)
            self.assertEqual(len(contexts), 2)
            self.assertEqual(contexts[1]['feedback_revision'], {'experiments': 3, 'data_research': 3})
            self.assertEqual(contexts[1]['experiments'][0]['analysis'], row.analysis)
            self.assertEqual(list(search.queue), records)


class ContractTests(unittest.TestCase):
    def test_fewer_ideas_and_updates_only_are_valid_but_unknown_ids_are_not(self):
        context = {'purposes': ['improve', 'explore'], 'max_new_ideas': 3,
                   'parents': ['node_000'], 'leader': 'node_000', 'idea_ids': ['idea_004', 'idea_005']}
        self.assertEqual(len(Ideas.model_validate({'ideas': [idea().model_dump()]}, context=context).ideas), 1)
        update = {'idea_id': 'idea_004', 'action': 'prioritize', 'priority': 80, 'reason': 'New evidence'}
        self.assertEqual(Ideas.model_validate({'updates': [update]}, context=context).ideas, [])
        started = {**update, 'idea_id': 'idea_005', 'action': 'remove'}
        self.assertEqual(Ideas.model_validate({'updates': [started]}, context=context).updates[0].idea_id, 'idea_005')
        with self.assertRaises(ValidationError):
            Ideas.model_validate({'updates': [{**update, 'idea_id': 'idea_999'}]}, context=context)


if __name__ == '__main__':
    unittest.main()

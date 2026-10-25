"""Real measurement workers, mocked HTTP, and research failure isolation through finalization."""
import asyncio
import copy
import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx
import numpy as np
import pandas as pd
from openai import AsyncOpenAI
from pydantic import ValidationError

from alfa_agent import agents, research, runner
from alfa_agent.config import Settings
from alfa_agent.contracts import Preparation
from alfa_agent.data import file_hash, input_hashes
from alfa_agent.storage import save
from alfa_agent.worker import prepare
from tests.test_pipeline import fixture, PROGRAM


def prepared(root):
    root = root.resolve()
    task, answers = fixture(root)
    directory = root / 'run'
    directory.mkdir()
    draft = dict(answers['PreparationDraft'])
    code = draft.pop('code')
    (directory / 'prepare.py').write_text(code)
    save(directory / 'preparation.json', Preparation(**draft, task_directory=str(root), task_type='regression',
                                                    code_sha256=file_hash(directory / 'prepare.py')))
    prepare(directory)
    return directory


def request(root, analyses):
    node = root / 'research' / 'research_000'
    node.mkdir(parents=True, exist_ok=True)
    plan = research.ResearchPlan(question='Measure actual data', analyses=analyses).model_dump()
    research.write_json(node / 'start.json', {'plan': plan, 'plan_sha256': research.fingerprint(plan),
                                             'preparation_sha256': file_hash(root / 'preparation.json')})
    return node


class MeasurementTests(unittest.TestCase):
    def test_worker_only_exposes_fit_labels_and_preserves_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = prepared(Path(tmp))
            protected = [directory/'preparation.json', directory/'prepare.py', directory/'split.npz',
                         Path(tmp)/'samples.psv', Path(tmp)/'query.jsonl']
            before = [file_hash(p) for p in protected]
            node = request(directory, [{'kind': 'association', 'columns': ['signal'], 'target': 'yield_value'},
                                       {'kind': 'quality', 'columns': ['signal'], 'dataset': 'validation'},
                                       {'kind': 'shift', 'columns': ['signal']}])
            original = research.measure
            def measure(tables, preparation, req):
                self.assertEqual(len(tables['fit']), 32)
                self.assertEqual(len(tables['validation']), 8)
                self.assertIn('yield_value', tables['fit'])
                for name in ('validation', 'test'):
                    self.assertNotIn('yield_value', tables[name])
                return original(tables, preparation, req)
            with patch.object(research, 'measure', side_effect=measure):
                research.execute(directory, node)
            measured = json.loads((node/'measurements.json').read_text())
            self.assertEqual(measured['values']['m0']['values']['signal']['spearman'], 1.)
            self.assertEqual(measured['values']['m0']['rows_used'], 32)
            self.assertEqual(measured['values']['m1']['rows_used'], 8)
            self.assertEqual([file_hash(p) for p in protected], before)
            self.assertFalse((Path(tmp)/'result.tsv').exists())

    def test_bad_columns_partial_measurements_and_changed_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = prepared(Path(tmp))
            node = request(directory, [{'kind': 'quality', 'columns': ['yield_value']},
                                       {'kind': 'quality', 'columns': ['signal']}])
            research.execute(directory, node)
            values = json.loads((node/'measurements.json').read_text())['values']
            self.assertEqual(values['m0']['status'], 'failed')
            self.assertEqual(values['m1']['status'], 'completed')
            (directory/'split.npz').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'split changed'):
                research.execute(directory, node)

    def test_structured_plans_and_evidence_reject_label_leakage_and_fake_citations(self):
        for item in [{'kind':'association','columns':['x'],'target':'y','dataset':'validation'},
                     {'kind':'quality','columns':['x'],'target':'y'},
                     {'kind':'quality','columns':['x'],'code':'arbitrary'},
                     {'kind':'quality','columns':['x','x']}]:
            with self.subTest(item=item), self.assertRaises(ValidationError):
                research.Analysis.model_validate(item)
        value = {'insights':[{'statement':'Observed', 'implication':'Test', 'limitations':'Sampled', 'evidence_keys':['m0']}]}
        for context in ({}, {'m0':{'status':'failed'}}, {'m0':{'status':'unavailable'}}):
            with self.assertRaises(ValidationError):
                research.ResearchReview.model_validate(value, context={'measurements':context})
        research.ResearchReview.model_validate(value, context={'measurements':{'m0':{'status':'completed'}}})

    def test_categorical_and_numeric_catalog_is_finite_and_sampled(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = prepared(Path(tmp))
            preparation = Preparation.model_validate_json((directory/'preparation.json').read_text())
            frame = pd.DataFrame({'x':np.arange(6000,dtype=float),'x2':np.arange(6000,dtype=float)*2,
                                  'group':['a','b']*3000,'yield_value':[0,1]*3000})
            frame.loc[0,'x']=np.inf
            tables={'fit':frame,'validation':frame.drop(columns='yield_value'),'test':frame.drop(columns='yield_value').iloc[:100]}
            classification=preparation.model_copy(update={'task_type':'classification'})
            for item in [{'kind':'quality','columns':['x','group']}, {'kind':'shift','columns':['x','group']},
                         {'kind':'correlation','columns':['x','x2']},
                         {'kind':'association','columns':['x','group'],'target':'yield_value'}]:
                result=research.measure(tables,classification,research.Analysis.model_validate(item))
                self.assertEqual(result['status'],'completed')
                self.assertEqual(result['rows_used'],5000)
                json.dumps(result,allow_nan=False)


class ResearchLifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def exercise(self, root, case='success'):
        task, answers=fixture(root)
        review_seen=asyncio.Event()
        contexts=[]
        async def handler(req):
            body=json.loads(req.content)
            title=json.loads(body['messages'][0]['content'].split('Schema: ',1)[1])['title']
            context=json.loads(body['messages'][1]['content'].rsplit('\n',1)[1])
            if title in ('ResearchPlan','ResearchReview') and case=='api_error':
                return httpx.Response(401,json={'error':{'message':'research-only injected error'}})
            if title=='ResearchPlan':
                direction=context['direction']
                if direction=='correlation': direction='quality'
                analysis={'kind':direction,'columns':['signal']}
                if direction=='association': analysis['target']='yield_value'
                answer={'question':'Inspect signal '+direction,'analyses':[analysis]}
            elif title=='ResearchReview':
                review_seen.set()
                key=next(k for k,v in context['measurements']['values'].items() if v['status']=='completed')
                answer={'insights':[{'statement':'Measured signal '+context['question'], 'evidence_keys':[key],
                                    'implication':'Compare a nonlinear feature in a new model.', 'limitations':'Sampled fit evidence, no model gain measured.'}],
                        'open_questions':['Does a nonlinear feature improve the fixed metric?']}
            elif title=='Hypotheses':
                contexts.append(context)
                h=copy.deepcopy(answers['Hypothesis'])
                h['pipeline']['model']['strategy']='median'
                answer={'hypotheses':[{'purpose':p,'parent_id':context['parent']['node_id'],'hypothesis':h}
                                      for p in context['requested_purposes']]}
            elif title=='Program':
                if context['start']['node_id']=='node_000' and case=='success':
                    await asyncio.wait_for(review_seen.wait(),15)
                    await asyncio.sleep(.05)
                answer=copy.deepcopy(answers[title])
                if context['start']['node_id']!='node_000':
                    answer['code']=PROGRAM.replace('fit[column].mean()','fit[column].median()')
            else:
                answer=answers[title]
            return httpx.Response(200,json={'id':'mock','object':'chat.completion','created':0,'model':'mock',
                'choices':[{'index':0,'finish_reason':'stop','message':{'role':'assistant','content':json.dumps(answer)}}]})
        def client(**kw):
            return AsyncOpenAI(**kw,http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
        original_process=runner.process
        original_write=research.write_json
        original_create=asyncio.create_task
        async def process(arguments,*args):
            if 'alfa_agent.research' in arguments and case=='timeout':
                raise TimeoutError('injected research timeout')
            return await original_process(arguments,*args)
        def write(path,value):
            if path.name=='context.json' and case=='artifact_error':
                raise OSError('injected read-only research artifact')
            return original_write(path,value)
        def create(coro,**kw):
            t=original_create(coro,**kw)
            if getattr(coro,'cr_code',None) and coro.cr_code.co_name=='investigate' and case=='cancel_before_start':
                t.cancel()
            return t
        with patch.object(agents,'AsyncOpenAI',side_effect=client), patch.object(runner,'process',side_effect=process), \
                patch.object(research,'write_json',side_effect=write), patch.object(asyncio,'create_task',side_effect=create):
            directory,report=await runner.run(task,runner.Options(time_budget=50,experiment_timeout=8,max_nodes=2,
                executors=1,threads=1,llm_concurrency=8,research_parallel=2,output_dir=root/'artifacts'),
                Settings('fake','http://mock/v1','mock'))
        return directory,report,contexts

    async def test_measured_findings_reach_planner_and_finalization(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory,report,contexts=await self.exercise(Path(tmp))
            self.assertEqual((report.status,report.stage),('ready','complete'), report.errors)
            self.assertGreaterEqual(report.research['completed'],1)
            self.assertEqual(report.research['running'],[])
            self.assertTrue(contexts[0]['data_research']['insights'])
            self.assertEqual(json.loads((directory/'finalization.json').read_text())['status'],'completed')
            self.assertTrue((Path(tmp)/'result.tsv').is_file())
            self.assertTrue(any(call.role=='data_reviewer' for call in report.llm_calls))

    async def test_research_failures_never_prevent_search_or_finalization(self):
        for case in ('api_error','timeout','artifact_error','cancel_before_start'):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as tmp:
                directory,report,_=await self.exercise(Path(tmp),case)
                self.assertEqual((report.status,report.stage),('ready','complete'), report.errors)
                self.assertTrue((Path(tmp)/'result.tsv').is_file())
                self.assertEqual(report.research['running'],[])
                self.assertEqual(json.loads((directory/'finalization.json').read_text())['status'],'completed')
                self.assertFalse(report.errors)

    async def test_close_before_entry_and_during_active_work_drains_tasks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            parent=agents.JSONAgents(Settings('fake','http://mock/v1','mock'),time.monotonic()+30,100,concurrency=8)
            for before_entry in (True,False):
                shared={}
                pool=research.Research(parent,root,shared,None,lambda message:None)
                entered=asyncio.Event()
                async def investigate(row,client):
                    entered.set()
                    await asyncio.Event().wait()
                with patch.object(pool,'investigate',side_effect=investigate):
                    pool.start()
                    if not before_entry:
                        await asyncio.wait_for(entered.wait(),1)
                    await asyncio.wait_for(pool.close(),1)
                self.assertEqual(pool.active,{})
                self.assertTrue(pool.task.done())
                self.assertEqual(pool.report()['running'],[])
                if not before_entry:
                    self.assertEqual(pool.report()['cancelled'],2)

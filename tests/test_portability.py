"""Regression cases for platform output limits, worker environment and finalization."""
import asyncio
from contextlib import redirect_stderr
import io
import json
import os
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import httpx
import numpy as np
import pandas as pd
from openai import AsyncOpenAI

from alfa_agent import agents, runner, worker
from alfa_agent.config import Settings
from alfa_agent.contracts import CodeReview
from alfa_agent.data import check_code
from alfa_agent.qwen import ask_qwen
from tests.test_pipeline import fixture


def reply(content='{"status":"pass"}', finish='stop'):
    return httpx.Response(200, json={'id':'test', 'object':'chat.completion', 'created':0, 'model':'mock',
        'choices':[{'index':0, 'finish_reason':finish, 'message':{'role':'assistant','content':content}}]})


class RequestTests(unittest.IsolatedAsyncioTestCase):
    async def test_platform_limit_is_used_for_every_role_and_after_truncation(self):
        for role in ('reviewer', 'preparation_reviewer', 'preparer', 'implementer', 'problem'):
            with self.subTest(role=role):
                bodies=[]
                def handler(request):
                    body=json.loads(request.content); bodies.append(body)
                    # Same diagnostic backend policy that rejected the previous default of 32768.
                    if (body.get('max_tokens') or 0)>16384:
                        return httpx.Response(400, json={'error':{'message':'max_tokens exceeds backend limit'}})
                    if len(bodies)==1:
                        return reply('{"status":', 'length')
                    return reply()
                def client(**kwargs):
                    return AsyncOpenAI(**kwargs, http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
                settings=Settings.from_env(Path('/nonexistent/.env'), environ={
                    'OPENAI_API_KEY':'fake', 'OPENAI_BASE_URL':'http://mock/v1', 'QWEN_MAX_TOKENS':'32768'})
                settings.validate()
                with patch.object(agents, 'AsyncOpenAI', side_effect=client):
                    result=await agents.JSONAgents(settings,time.monotonic()+10).ask(CodeReview,'Check',role,attempts=2)
                self.assertEqual(result.status,'pass')
                self.assertEqual(len(bodies),2)
                for body in bodies:
                    self.assertNotIn('max_tokens',body)
                    self.assertEqual(body['reasoning_effort'],'low')

    def test_sync_request_uses_platform_output_limit(self):
        def handler(request):
            self.assertNotIn('max_tokens',json.loads(request.content))
            return reply('ok')
        result=ask_qwen(Settings('fake','http://mock/v1','mock'),'Hello',
            http_client=httpx.Client(transport=httpx.MockTransport(handler)))
        self.assertEqual(result.text,'ok')

    async def test_worker_retains_directory_environment_without_credentials(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); output=root/'environment.json'
            names=('HOME','USERPROFILE','HOMEDRIVE','HOMEPATH','TMPDIR','TEMP','TMP','APPDATA','LOCALAPPDATA')
            supplied={name:str(root/name) for name in names}
            for value in supplied.values(): Path(value).mkdir()
            supplied.update(OPENAI_API_KEY='private-key',OPENAI_BASE_URL='private-address')
            script='import json,os,pathlib; pathlib.Path("environment.json").write_text(json.dumps(dict(os.environ)))'
            with patch.dict(os.environ,supplied), patch.dict(runner.CHILD,threads=2):
                await runner.process(['-c',script],root,root/'child.log',time.monotonic()+10,10)
            child=json.loads(output.read_text())
            for name in names: self.assertEqual(child[name],supplied[name])
            for name in ('OPENAI_API_KEY','OPENAI_BASE_URL'): self.assertNotIn(name,child)
            for name in ('HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','HF_DATASETS_OFFLINE'): self.assertEqual(child[name],'1')
            for name in ('ALFA_THREADS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): self.assertEqual(child[name],'2')

    async def test_report_write_failure_does_not_interrupt_finalization(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve(); task,answers=fixture(root)
            def handler(request):
                body=json.loads(request.content)
                title=json.loads(body['messages'][0]['content'].split('Schema: ',1)[1])['title']
                return reply(json.dumps(answers[title]))
            def client(**kwargs):
                return AsyncOpenAI(**kwargs,http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
            original_save=runner.save
            failures=[]
            def save(path,value):
                if path.name=='report.json' and getattr(value,'completed_nodes',0)>0:
                    failures.append(path)
                    raise PermissionError('injected write failure with fake-key')
                return original_save(path,value)
            stderr=io.StringIO()
            with patch.object(agents,'AsyncOpenAI',side_effect=client), patch.object(runner,'save',side_effect=save), redirect_stderr(stderr):
                directory,report=await runner.run(task,runner.Options(time_budget=40,max_nodes=1,output_dir=root/'run'),
                    Settings('fake-key','http://mock/v1','mock'))
            self.assertGreater(len(failures),1)
            self.assertEqual(report.stage,'complete')
            self.assertTrue((root/'result.tsv').exists())
            self.assertEqual(json.loads((directory/'finalization.json').read_text())['status'],'completed')
            self.assertEqual(sum('report.json could not' in warning for warning in report.warnings),1)
            self.assertEqual(stderr.getvalue().count('report.json could not'),1)
            self.assertNotIn('fake-key',stderr.getvalue())


class SplitAndImportTests(unittest.TestCase):
    def test_standard_library_imports_accepted_like_blocks(self):
        for name in ('abc','types','numbers','contextlib','operator'):
            with self.subTest(name=name):
                check_code(f'import {name}\ndef fit_model(fit, preparation):\n return None',('fit_model',))

    def test_temporal_holdout_is_nonempty_with_ties_missing_values_and_two_periods(self):
        data=SimpleNamespace(target_columns=['target'], id_columns=[])
        for periods,fit,valid in (
            ([1]*9+[2],[0,1,2,3,4,5,6,9],[7,8]),
            ([1]*9+[2,None],[0,1,2,3,4,5,6,9,10],[7,8]),
            ([1,2],[1],[0]),
            (list(range(10)),[0,1,2,3,4,5,6,9],[7,8]),
        ):
            with self.subTest(periods=periods):
                train=pd.DataFrame({'period':periods,'target':range(len(periods))})
                test=pd.DataFrame({'period':[20,20]})
                left,right,report=worker.shift_split(train,test,data,{},np.array(fit),np.array(valid))
                worker.validate_split(left,right,len(train))
                self.assertEqual(report['override'],'temporal_holdout')
                self.assertLess(train.iloc[left].period.max(),train.iloc[right].period.min())
                self.assertEqual(len(left)+len(right),train.period.notna().sum())
                if periods==list(range(10)):
                    np.testing.assert_array_equal(left,np.arange(8))
                    np.testing.assert_array_equal(right,np.arange(8,10))

    def test_preparation_revalidates_automatic_split(self):
        # The automatic stage must not bypass the same contract applied to generated indices.
        from alfa_agent.contracts import Preparation
        from alfa_agent.data import file_hash
        from alfa_agent.storage import save
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve(); task,answers=fixture(root)
            draft=answers['PreparationDraft']
            code=root/'prepare.py';code.write_text(draft['code'])
            prep=Preparation.model_validate({**{k:v for k,v in draft.items() if k!='code'},'task_directory':str(root),'code_sha256':file_hash(code)})
            save(root/'preparation.json',prep)
            with patch.object(worker,'SHIFT_CHECK',True), patch.object(worker,'shift_split',return_value=(np.array([],dtype=int),np.array([0]),{})):
                with self.assertRaisesRegex(ValueError,'nonempty'):
                    worker.prepare(root)

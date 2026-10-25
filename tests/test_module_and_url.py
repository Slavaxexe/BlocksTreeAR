"""Real child-process serialization and actual OpenAI request URL regressions; no network."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import typing
import unittest
from unittest.mock import patch

import cloudpickle
import httpx
from joblib.externals import cloudpickle as joblib_cloudpickle
from openai import AsyncOpenAI

from alfa_agent import agents, qwen
from alfa_agent.config import Settings
from alfa_agent.contracts import CodeReview
from alfa_agent.data import file_hash, module
from alfa_agent.qwen import ask_qwen

CODE='''from __future__ import annotations
from dataclasses import dataclass
FACTOR = 2

def square(value):
    return value * value * FACTOR

@dataclass
class Offset:
    amount: int
    def __call__(self, value):
        return square(value) + self.amount

def fit_model(fit, preparation):
    return Offset(3)
'''


def response():
    return httpx.Response(200,json={'id':'test','object':'chat.completion','created':0,'model':'mock',
        'choices':[{'index':0,'finish_reason':'stop','message':{'role':'assistant','content':'{"status":"pass"}'}}]})


class URLTests(unittest.IsolatedAsyncioTestCase):
    async def test_sync_and_async_preserve_path_and_query_values(self):
        cases=(
            ('http://proxy:8080/v1','/v1/chat/completions',{}),
            ('http://proxy:8080/team%20space/v1/','/team space/v1/chat/completions',{}),
            ('http://proxy/v1?team=example','/v1/chat/completions',{'team':['example']}),
            ('http://proxy/v1/?team=a%2Fb%2Bc&tag=one&tag=two&empty=','/v1/chat/completions',
                {'team':['a/b+c'],'tag':['one','two'],'empty':['']}),
            ('http://name:pass@proxy/v1?team=example#local','/v1/chat/completions',{'team':['example']}),
        )
        for url,path,query in cases:
            with self.subTest(url=url):
                settings=Settings.from_env(Path('/nonexistent/.env'),environ={'OPENAI_API_KEY':'fake','OPENAI_BASE_URL':url})
                settings.validate();seen=[]
                def handler(request):
                    seen.append(request)
                    self.assertEqual(request.url.path,path)
                    self.assertEqual({k:request.url.params.get_list(k) for k in request.url.params},query)
                    self.assertEqual(request.headers['authorization'],
                        'Basic bmFtZTpwYXNz' if '@' in url else 'Bearer fake')
                    return response()
                def client(**kwargs):
                    if 'http_client' not in kwargs:
                        kwargs['http_client']=httpx.AsyncClient(transport=httpx.MockTransport(handler))
                    return AsyncOpenAI(**kwargs)
                ask_qwen(settings,'hello',http_client=httpx.Client(transport=httpx.MockTransport(handler)))
                with patch.object(agents,'AsyncOpenAI',side_effect=client), patch.object(qwen,'DefaultAsyncHttpxClient',
                        side_effect=lambda **kw: httpx.AsyncClient(transport=httpx.MockTransport(handler),**kw)):
                    answer=await agents.JSONAgents(settings,time.monotonic()+5).ask(CodeReview,'Review','reviewer')
                self.assertEqual(answer.status,'pass');self.assertEqual(len(seen),2)
                self.assertEqual(settings.base_url,url)


class ModuleTests(unittest.TestCase):
    def test_functions_and_dataclasses_unpickle_without_importable_generated_module(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); source=root/'experiment.py';source.write_text(CODE)
            generated=module(source,file_hash(source))
            self.assertIs(sys.modules[generated.__name__],generated)
            self.assertEqual(typing.get_type_hints(generated.Offset),{'amount':int})
            for serializer in (cloudpickle,joblib_cloudpickle):
                with self.subTest(serializer=serializer.__name__):
                    payload=root/'payload.pkl';payload.write_bytes(serializer.dumps((generated.square,generated.Offset(3))))
                    driver='import pickle,sys; f,p=pickle.load(open(sys.argv[1],"rb")); assert f(4)==32; assert p(4)==35'
                    child=subprocess.run([sys.executable,'-c',driver,str(payload)],cwd=root,
                        env={'PATH':os.defpath,'PYTHONDONTWRITEBYTECODE':'1'},capture_output=True,text=True,timeout=20)
                    self.assertEqual(child.returncode,0,child.stderr)

    def test_real_joblib_processes_execute_generated_functions_and_callable_classes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); source=root/'experiment.py';source.write_text(CODE)
            generated=module(source,file_hash(source))
            from joblib import Parallel, delayed
            with Parallel(n_jobs=2,backend='loky',timeout=20) as pool:
                self.assertEqual(pool(delayed(generated.square)(x) for x in range(4)),[0,2,8,18])
                self.assertEqual(pool(delayed(generated.Offset(3))(x) for x in range(4)),[3,5,11,21])

    def test_loading_error_restores_previous_module_and_hash_check_is_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            source=Path(tmp)/'experiment.py';source.write_text(CODE)
            good=module(source,file_hash(source))
            with self.assertRaisesRegex(ValueError,'changed'):
                module(source,'0'*64)
            source.write_text('raise RuntimeError("injected")')
            name='generated_experiment_'+file_hash(source)[:12]
            marker=object();sys.modules[name]=marker
            try:
                with self.assertRaisesRegex(RuntimeError,'injected'):module(source,file_hash(source))
                self.assertIs(sys.modules[name],marker)
                self.assertIs(sys.modules[good.__name__],good)
            finally:sys.modules.pop(name,None)

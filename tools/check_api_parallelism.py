"""Small staged concurrency probe; no retries and stops at a failed batch."""
import asyncio
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import sys
import time

from openai import AsyncOpenAI, APIStatusError

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agent.config import Settings

async def main():
    settings = Settings.from_env()
    settings.validate()
    results = []
    # At most 63 short requests, never retry 429 or increase load after a failure.
    async with AsyncOpenAI(api_key=settings.api_key, base_url=settings.base_url,
                           max_retries=0, timeout=45) as client:
        for parallel in (1, 2, 4, 8, 16, 32):
            ready = asyncio.Event()
            async def one(index):
                await ready.wait()
                started = time.monotonic()
                try:
                    reply = await client.chat.completions.create(
                        model=settings.model,
                        messages=[{'role': 'user', 'content': 'Reply with exactly OK.'}],
                        max_tokens=128, temperature=0,
                        extra_body={'enable_thinking': False, 'reasoning_effort': 'low',
                            'chat_template_kwargs': {'enable_thinking': False, 'reasoning_effort': 'low'}})
                    usage = reply.usage.model_dump() if reply.usage else {}
                    choice = reply.choices[0] if reply.choices else None
                    return {'status': 'ok' if choice and (choice.message.content or '').strip() else 'invalid',
                            'seconds': round(time.monotonic() - started, 3),
                            'finish_reason': choice.finish_reason if choice else None, 'usage': usage}
                except Exception as error:
                    return {'status': f'HTTP {error.status_code}' if isinstance(error, APIStatusError) else type(error).__name__,
                            'seconds': round(time.monotonic() - started, 3)}
            jobs = [asyncio.create_task(one(i)) for i in range(parallel)]
            started = time.monotonic()
            ready.set()
            replies = await asyncio.gather(*jobs)
            duration = time.monotonic() - started
            latencies = sorted(r['seconds'] for r in replies)
            row = {'parallel': parallel, 'statuses': dict(Counter(r['status'] for r in replies)),
                   'batch_seconds': round(duration, 3), 'median_seconds': statistics.median(latencies),
                   'max_seconds': max(latencies), 'requests_per_second': round(parallel / duration, 3),
                   'reported_cost': sum(r.get('usage', {}).get('cost', 0) or 0 for r in replies),
                   'replies': replies}
            results.append(row)
            print(json.dumps({k: v for k, v in row.items() if k != 'replies'}, ensure_ascii=False), flush=True)
            if any(r['status'] != 'ok' for r in replies):
                break
            await asyncio.sleep(1)
    target = ROOT / 'artifacts' / ('api-parallelism-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '.json')
    target.parent.mkdir(exist_ok=True)
    target.write_text(json.dumps({'model': settings.model, 'results': results,
        'notes': 'Short requests only; success is a tested lower bound, not a documented provider quota.'},
        ensure_ascii=False, indent=2), encoding='utf-8')
    print('Saved: ' + str(target), flush=True)

asyncio.run(main())

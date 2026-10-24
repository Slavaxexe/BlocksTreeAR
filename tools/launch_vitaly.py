"""Run this solution with isolated task files, a hard timeout, and a live manifest."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def main():
    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo))
    from ar_config import Settings, load_env
    load_env(repo)
    def save(path, value):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        pending = path.with_suffix('.json.tmp')
        pending.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
        pending.replace(path)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('solution', type=Path)
    parser.add_argument('task', type=Path)
    parser.add_argument('--budget', type=float, default=10800)
    parser.add_argument('--cpu', type=int, default=8)
    parser.add_argument('--max-experiments', type=int, default=60)
    parser.add_argument('--manifest', type=Path, default=None)
    args = parser.parse_args()
    solution, task = args.solution.resolve(), args.task.resolve()
    if not (solution/'solution.py').is_file() or not task.is_file():
        parser.error('Need solution.py and an existing task Markdown')
    settings = Settings()
    if not settings.api_key or not settings.base_url or not settings.model:
        parser.error('Configure OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL in environment or root .env')
    launch = solution/'artifacts'/('local_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ'))
    inputs = launch/'task'
    inputs.mkdir(parents=True)
    for name in ('train.csv', 'test.csv', 'sample_submission.csv', 'data_dictionary.csv'):
        source = task.parent/name
        if source.is_file():
            try:
                os.link(source, inputs/name)
            except OSError:
                shutil.copyfile(source, inputs/name)
    shutil.copyfile(task, inputs/'task.md')
    env = dict(os.environ)
    env.pop('PYTHONPATH', None)
    env.update(OPENAI_API_KEY=settings.api_key, OPENAI_BASE_URL=settings.base_url, OPENAI_MODEL=settings.model,
        PYTHONIOENCODING='utf-8', PYTHONUNBUFFERED='1', OMP_NUM_THREADS=str(args.cpu),
        OPENBLAS_NUM_THREADS=str(args.cpu), MKL_NUM_THREADS=str(args.cpu))
    # Resource selection only; the external Python sources remain unchanged.
    bootstrap = ('import os,sys,runpy; os.cpu_count=lambda:'+str(args.cpu)+'; '
                 'sys.path.insert(0,'+repr(str(solution))+'); sys.argv='+repr([
                    str(solution/'solution.py'), str(inputs/'task.md'), '--budget', str(args.budget),
                    '--time-budget', str(args.budget), '--experiment-timeout', '900', '--parallel', '1',
                    '--parallel-agents', '4', '--max-experiments', str(args.max_experiments)])+'; '
                 'runpy.run_path('+repr(str(solution/'solution.py'))+',run_name="__main__")')
    sys.path.insert(0, str(solution))
    from ac_task import detect_task_type
    task_type = detect_task_type(inputs/'task.md')
    def research_directories():
        return list((solution/'artifacts').glob('classification_*' if task_type == 'classification' else 'autoresearch_*'))
    before = {str(path) for path in research_directories()}
    manifest_path = args.manifest.resolve() if args.manifest else solution/'artifacts/latest_launch.json'
    started = time.monotonic()
    manifest = {'status': 'starting', 'started_at': datetime.now(timezone.utc).isoformat(),
        'launch_directory': str(launch), 'solution_directory': str(solution), 'task': str(inputs/'task.md'),
        'console_log': str(launch/'console.log'), 'budget_seconds': args.budget, 'cpu': args.cpu,
        'real_qwen_configured': True, 'credentials_written': False, 'artifacts': None,
        'task_type': task_type}
    save(manifest_path, manifest)
    with (launch/'console.log').open('wb') as log:
        process = subprocess.Popen([sys.executable, '-u', '-c', bootstrap], cwd=solution, env=env,
            stdout=log, stderr=subprocess.STDOUT, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        manifest.update(status='running', pid=process.pid)
        print('Started PID', process.pid, 'log:', launch/'console.log', flush=True)
        while process.poll() is None:
            manifest['elapsed_seconds'] = time.monotonic()-started
            new = [path for path in research_directories() if str(path) not in before]
            if new:
                manifest['artifacts'] = str(max(new, key=lambda path: path.stat().st_mtime))
            save(manifest_path, manifest)
            if manifest['elapsed_seconds'] >= args.budget:
                import psutil
                try:
                    for child in reversed(psutil.Process(process.pid).children(recursive=True)):
                        try:
                            child.kill()
                        except psutil.Error:
                            pass
                except psutil.Error:
                    pass
                process.kill()
                process.wait()
                manifest['status'] = 'time_limit'
                break
            time.sleep(1)
        manifest.update(exit_code=process.wait(), elapsed_seconds=time.monotonic()-started)
        if manifest['status'] != 'time_limit':
            manifest['status'] = 'completed' if manifest['exit_code'] == 0 else 'failed'
        save(manifest_path, manifest)
    print(json.dumps(manifest, ensure_ascii=False), flush=True)
    return 0 if manifest['status'] == 'completed' else 2


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    raise SystemExit(main())

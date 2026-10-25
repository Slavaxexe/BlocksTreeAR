"""Export a flat platform ZIP without credentials, datasets or run artifacts."""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import zipfile


ENTRY = '''"""Platform entry: python solution.py /path/to/task.md."""
import argparse
import asyncio
from pathlib import Path
import sys

from config import Settings
from runner import Options, run


def submission_options(time_budget=10500):
    return Options(time_budget=time_budget, experiment_timeout=600,
        final_refit_seconds=900, eda_seconds=900,
        max_experiments=None, max_research_nodes=None, max_llm_calls=None,
        parallel=6, parallel_agents=8, research_parallel=2, research_timeout=120)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('task', nargs='?', type=Path, default=Path('task.md'))
    parser.add_argument('--time-budget', type=float, default=10500)
    args = parser.parse_args(argv)
    settings = None
    try:
        settings = Settings.from_env()
        settings.validate()
        directory, report = asyncio.run(run(args.task, submission_options(args.time_budget), settings))
        print(report.model_dump_json(indent=2))
        print(f'Artifacts: {directory}')
        output = args.task.resolve().parent / 'submission.csv'
        return 0 if output.is_file() and output.stat().st_size else 2
    except Exception as error:
        message = str(error)
        if settings and settings.api_key:
            message = message.replace(settings.api_key, '[REDACTED]')
        print(message, file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
'''


def flatten(text):
    text = re.sub(r'from \.(\w+) import', r'from \1 import', text)
    text = text.replace('agent.', '')
    return text.replace('Path(__file__).resolve().parents[1]', 'Path(__file__).resolve().parent')


def build(archive):
    source = Path(__file__).resolve().parent
    archive = archive.resolve()
    if archive.exists():
        raise FileExistsError(f'Archive already exists: {archive}')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    stage = archive.parent / 'tmp' / (archive.stem + '_' + stamp)
    stage.mkdir(parents=True)
    for path in sorted((source / 'agent').glob('*.py')):
        if path.stem in {'__init__', '__main__'}:
            continue
        text = path.read_text(encoding='utf-8')
        if path.stem == 'config':
            first = text.index('        path = env_file')
            last = text.index('\n    def validate', first)
            text = text[:first] + '''        # Connection settings come only from the platform environment.
        values = os.environ if environ is None else environ
        return cls(api_key=(values.get('OPENAI_API_KEY') or '').strip(),
                   base_url=(values.get('OPENAI_BASE_URL') or '').strip(),
                   model=(values.get('OPENAI_MODEL') or 'qwen/qwen3.8-27b').strip(),
                   timeout_seconds=400., max_tokens=None)
''' + text[last:]
            text = text.replace('from dotenv import dotenv_values\n', '')
        if path.stem == 'runner':
            original = '                    check_code(draft.code,'
            assert text.count(original) == 1
            text = text.replace(original,
                "                    draft = draft.model_copy(update={'output': draft.output.model_copy(update={'filename': 'submission.csv'})})\n" + original)
        if path.stem == 'search':
            original = '        explore = int(options.parallel > 1)'
            assert text.count(original) == 1
            text = text.replace(original, '        explore = min(2, max(0, options.parallel - 1))')
        text = flatten(text)
        tree = ast.parse(text, filename=path.name)
        assert not any(isinstance(node, ast.ImportFrom) and node.level for node in ast.walk(tree)), path
        (stage / path.name).write_text(text, encoding='utf-8')
    for name in ('ml_reference.py', 'ml_reference_data.py', 'ml_reference_notes.py', 'ml_reference_tools.py'):
        (stage / name).write_bytes((source / name).read_bytes())
    (stage / 'requirements.txt').write_bytes((source / 'baseline' / 'requirements_data.txt').read_bytes())
    (stage / 'solution.py').write_text(ENTRY, encoding='utf-8')
    for path in stage.glob('*.py'):
        compile(path.read_text(encoding='utf-8'), path.name, 'exec')
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as zipped:
        for path in sorted(stage.iterdir()):
            zipped.write(path, path.name)
    with zipfile.ZipFile(archive) as zipped:
        assert zipped.testzip() is None
        assert all('/' not in name and '\\' not in name for name in zipped.namelist())
    assert archive.stat().st_size < 100_000_000
    manifest = {'archive': str(archive), 'stage': str(stage), 'bytes': archive.stat().st_size,
                'sha256': hashlib.sha256(archive.read_bytes()).hexdigest()}
    (stage.parent / (archive.stem + '_build.json')).write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path(__file__).with_name('submission_6.zip'))
    print(json.dumps(build(parser.parse_args().output)))

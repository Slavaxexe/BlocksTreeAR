"""Create a root-only submission folder. Never create an archive or bundle dependencies."""
import argparse
import ast
import hashlib
import json
from pathlib import Path


BOOTSTRAP = '''"""Restore our Python packages in a temporary runtime directory, without installs."""
import hashlib
from pathlib import Path
import tempfile

from bundle_manifest import FILES, REQUIREMENTS_SHA256
from bundle_resources import RESOURCES


def materialize():
    source = Path(__file__).resolve().parent
    runtime = Path(tempfile.mkdtemp(prefix='alfa_regression_'))
    for filename, relative, expected in FILES:
        data = (source/filename).read_bytes()
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError('Submission source hash mismatch: '+filename)
        target = runtime/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    requirements = (source/'requirements.txt').read_bytes()
    if hashlib.sha256(requirements).hexdigest() != REQUIREMENTS_SHA256:
        raise ValueError('The organizer requirements.txt was modified')
    (runtime/'requirements.txt').write_bytes(requirements)
    for relative, text in RESOURCES.items():
        target = runtime/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding='utf-8')
    return runtime
'''

ENTRY = '''"""Platform entry point: python solution.py task.md"""
import sys
sys.dont_write_bytecode = True

from bundle_bootstrap import materialize


if __name__ == '__main__':
    runtime = materialize()
    sys.path.insert(0, str(runtime))
    from agent.platform_pipeline import main
    raise SystemExit(main(runtime_root=runtime))
'''


def agent_sources(root):
    directory = root/'agent'
    pending = ['platform_pipeline', 'training_worker', 'margo_preparation_worker', 'model_selection_worker', '__init__']
    found = set()
    while pending:
        name = pending.pop()
        if name in found:
            continue
        path = directory/(name+'.py')
        if not path.is_file():
            raise ValueError('Missing own module: '+str(path))
        found.add(name)
        for node in ast.walk(ast.parse(path.read_text(encoding='utf-8-sig'))):
            dependency = None
            if isinstance(node, ast.ImportFrom):
                if node.level == 1 and not node.module:
                    pending.extend(alias.name for alias in node.names if (directory/(alias.name+'.py')).is_file())
                if node.level == 1 and node.module:
                    dependency = node.module.split('.')[0]
                elif node.module and node.module.startswith('agent.'):
                    dependency = node.module.split('.')[1]
            if dependency and (directory/(dependency+'.py')).is_file():
                pending.append(dependency)
    return [directory/(name+'.py') for name in sorted(found)]


def build(output, margo_root, selection_root, *, replace=False):
    root = Path(__file__).resolve().parents[1]
    requirements = (root/'requirements.txt').read_bytes()
    organizer_sha256 = 'ac27e467b9c5c6b8b6ca2a042a67e2beeb7d08e4d12d3852408909d2d5a07b0e'
    if hashlib.sha256(requirements).hexdigest() != organizer_sha256:
        raise ValueError('requirements.txt must equal the organizer original byte for byte')
    output = Path(output).resolve()
    if output.exists() and not replace:
        raise ValueError('Destination must be a new directory: '+str(output))
    if output.exists():
        for path in output.iterdir():
            if not path.is_file() or not (path.name in {'solution.py', 'requirements.txt'} or
                                         (path.name.startswith('bundle_') and path.suffix == '.py')):
                raise ValueError('Refusing to replace a directory containing unrelated files: '+str(path))
    payload, entries = {}, []
    for prefix, package, paths in (
        ('max', 'agent', agent_sources(root)),
        ('margo', 'data_preparation', sorted((margo_root/'data_preparation').glob('*.py'))),
        ('selection', 'model_selection', sorted((selection_root/'model_selection').glob('*.py'))),
    ):
        if not paths:
            raise ValueError('Missing package '+package)
        for path in paths:
            data = path.read_bytes()
            ast.parse(data, filename=str(path))
            filename = f'bundle_{prefix}_{path.stem}.py'
            payload[filename] = data
            entries.append((filename, f'{package}/{path.name}', hashlib.sha256(data).hexdigest()))
    skill = (selection_root/'model_selection/SKILL.md').read_text(encoding='utf-8')
    payload['bundle_resources.py'] = ('"""Own model-selection methodology, preserved as text."""\nRESOURCES = '+
        repr({'model_selection/SKILL.md': skill})+'\n').encode('utf-8')
    digest = hashlib.sha256(requirements).hexdigest()
    payload['bundle_manifest.py'] = ('"""Exact source mapping for the temporary runtime."""\nFILES = '+repr(entries)+
        '\nREQUIREMENTS_SHA256 = '+repr(digest)+'\n').encode('utf-8')
    payload['bundle_bootstrap.py'] = BOOTSTRAP.encode('utf-8')
    payload['solution.py'] = ENTRY.encode('utf-8')
    payload['requirements.txt'] = requirements
    output.mkdir(parents=True, exist_ok=replace)
    for path in output.iterdir():
        if path.name not in payload:
            path.unlink()
    for filename, data in payload.items():
        (output/filename).write_bytes(data)
    summary = {'directory': str(output), 'files': len(payload), 'source_modules': len(entries),
               'bytes': sum(map(len, payload.values())), 'requirements_sha256': digest,
               'layout': 'root .py files and organizer requirements.txt only', 'archive_created': False}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    vendor = Path(__file__).resolve().parents[1]/'vendor'
    parser.add_argument('--margo-root', type=Path, default=vendor)
    parser.add_argument('--model-selection-root', type=Path, default=vendor)
    parser.add_argument('--replace', action='store_true', help='Update an existing folder containing only this builder payload')
    args = parser.parse_args()
    build(args.output, args.margo_root.resolve(), args.model_selection_root.resolve(), replace=args.replace)

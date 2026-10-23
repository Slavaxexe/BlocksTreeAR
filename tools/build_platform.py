"""Build the current agent as root-only Python modules for the platform."""
import argparse
import ast
import hashlib
from pathlib import Path
import re
import zipfile


def build(destination):
    root = Path(__file__).resolve().parents[1]
    modules = {file.stem: ('max_cli' if file.stem == '__main__' else f'max_{file.stem}')
               for file in (root / 'agent').glob('*.py') if file.stem != '__init__'}
    payload = {}
    for name, flat in modules.items():
        code = (root / 'agent' / f'{name}.py').read_text(encoding='utf-8')
        for original, replacement in modules.items():
            code = re.sub(r'\bfrom \.' + re.escape(original) + r' import\b', f'from {replacement} import', code)
            code = code.replace(f'agent.{original}', replacement)
        code = code.replace('Path(__file__).resolve().parents[1]', 'Path(__file__).resolve().parent')
        if name == '__main__':
            code = code.replace('"--time-budget", type=float, default=0', '"--time-budget", type=float, default=4800')
            code = code.replace('"--experiment-timeout", type=float, default=0', '"--experiment-timeout", type=float, default=1200')
            code = code.replace('"--max-code-attempts", type=int, default=0', '"--max-code-attempts", type=int, default=3')
            code = code.replace('args = parser.parse_args(argv)',
                'args = parser.parse_args(argv)\n'
                '    if not __import__("math").isfinite(args.time_budget) or args.time_budget <= 0:\n'
                '        args.time_budget = 4800.\n'
                '    args.time_budget = min(args.time_budget, 4800.)')
            code = code.replace('Секунды на запуск; 0 — без ограничения.', 'Секунды на запуск; максимум 4800 (80 минут).')
        if name == 'config':
            code = code.replace('values.get("QWEN_TIMEOUT_SECONDS") or "0"', 'values.get("QWEN_TIMEOUT_SECONDS") or "180"')
            code = code.replace('values.get("QWEN_MAX_TOKENS") or "0"', 'values.get("QWEN_MAX_TOKENS") or "12288"')
            code = code.replace('values.get("OPENAI_MODEL") or ""', 'values.get("OPENAI_MODEL") or "qwen/qwen3.8-27b"')
        ast.parse(code, filename=flat + '.py')
        payload[flat + '.py'] = code.encode('utf-8')
    entry = (root / 'solution.py').read_text(encoding='utf-8').replace('agent.__main__', 'max_cli')
    payload['solution.py'] = entry.encode('utf-8')
    payload['requirements.txt'] = (root / 'requirements.txt').read_bytes()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, data in sorted(payload.items()):
            archive.writestr(name, data)
    digest = hashlib.sha256(destination.read_bytes()).hexdigest()
    print(f'{destination}\nFiles: {len(payload)}\nBytes: {destination.stat().st_size}\nSHA256: {digest}')
    return digest


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('output', type=Path)
    build(parser.parse_args().output.resolve())

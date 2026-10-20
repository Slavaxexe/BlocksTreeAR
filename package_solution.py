"""Build the flat ZIP copied by the platform: python package_solution.py [output.zip]."""

import argparse
from pathlib import Path
import re
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parent


def build(destination: Path, requirements: Path | None = None) -> Path:
    modules = {p.stem: 'alfa_cli' if p.stem == '__main__' else 'alfa_' + p.stem
               for p in sorted((ROOT / 'alfa_agent').glob('*.py')) if p.stem != '__init__'}
    sources = {'solution.py': (ROOT / 'solution.py').read_text()}
    sources.update({export + '.py': (ROOT / 'alfa_agent' / (name + '.py')).read_text()
                    for name, export in modules.items()})
    # Only the export changes imports and package-root paths; development sources stay intact.
    for name, code in sources.items():
        code = re.sub(r'(?m)^(\s*)from \.(\w+) import ',
                      lambda match: match[1] + 'from ' + modules[match[2]] + ' import ', code)
        for original, export in modules.items():
            code = code.replace('alfa_agent.' + original, export)
        code = code.replace('Path(__file__).resolve().parents[1]', 'Path(__file__).resolve().parent')
        compile(code, name, 'exec')
        sources[name] = code
    sources['requirements.txt'] = (requirements or ROOT / 'vendor/requirements_data_folder.txt').read_bytes()
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(destination, 'w', compression=ZIP_DEFLATED) as archive:
        for name, content in sources.items():
            archive.writestr(name, content)
    return destination


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', nargs='?', type=Path, default=Path('artifacts/submission.zip'))
    parser.add_argument('--requirements', type=Path, help='Original requirements.txt supplied by the organizers')
    args = parser.parse_args()
    result = build(args.output, args.requirements)
    print(f'{result.resolve()} ({result.stat().st_size:,} bytes)')

"""Keep the existing root-level compatibility modules in sync with agent sources."""
import re
from pathlib import Path


def sync():
    root = Path(__file__).resolve().parents[1]
    modules = {path.stem: ('agent_main' if path.stem == '__main__' else path.stem)
               for path in (root/'agent').glob('*.py') if path.stem != '__init__'}
    for name, flat in modules.items():
        code = (root/'agent'/(name+'.py')).read_text(encoding='utf-8')
        for original, replacement in modules.items():
            code = re.sub(r'\bfrom \.'+re.escape(original)+r' import\b', 'from '+replacement+' import', code)
            code = re.sub(r'\bfrom \. import '+re.escape(original)+r'\b', 'import '+replacement, code)
            code = code.replace('agent.'+original, replacement)
        code = code.replace('Path(__file__).resolve().parents[1]', 'Path(__file__).resolve().parent')
        destination = root/(flat+'.py')
        if destination.exists() and destination.read_text(encoding='utf-8') == code:
            continue
        destination.write_text(code, encoding='utf-8')
    print('Root compatibility modules synchronized:', len(modules))


if __name__ == '__main__':
    sync()

"""Assemble immutable block snapshots and execute an ordered state pipeline."""

import hashlib
import json
from pathlib import Path
import traceback
import time
from types import MappingProxyType

import pandas as pd

from .contracts import INITIAL_STATE, Start
from .data import check_code, file_hash, module
from .storage import atomic_bytes, load


WRAPPER = '''from pathlib import Path
from agent.assembly import execute


def fit_predict(fit, validation, test, preparation, auxiliary=None):
    return execute(Path(__file__).resolve().parent, fit, validation, test, preparation, auxiliary)


if __name__ == '__main__':
    from agent.worker import main
    main(fit_predict)
'''


class BlockError(ValueError):
    def __init__(self, block, message):
        self.block = block
        super().__init__(f'Block {block}: {message}')


def block_catalog(directory: Path, nodes) -> dict:
    catalog = {}
    for row in nodes:
        manifest = directory / 'nodes' / row.node_id / 'assembly.json'
        if manifest.exists():
            saved = json.loads(manifest.read_text(encoding='utf-8'))
            catalog[row.node_id] = saved['blocks']
    return catalog


def block_path(node: Path, name: str) -> Path:
    local = node / 'modules' / (name + '.py')
    if local.exists():
        return local
    start = load(node / 'start.json', Start)
    block = next(block for block in start.assembly.blocks if block.name == name)
    source = node.parent / block.source_node
    return block_path(source, name)


def initialize(node: Path, start: Start):
    """Reused blocks stay in their owner node; repairs write a local replacement."""
    (node / 'modules').mkdir(parents=True)
    for block in start.assembly.blocks:
        if block.source_node is None:
            continue
        source = block_path(node.parent / block.source_node, block.name)
        if file_hash(source) != start.context['source_hashes'][block.name]:
            raise BlockError(block.name, 'Source block changed after planning')
    atomic_bytes(node / start.implementation, WRAPPER.encode('utf-8'))


def assemble(node: Path, start: Start) -> str:
    blocks = []
    for block in start.assembly.blocks:
        path = block_path(node, block.name)
        try:
            check_code(path.read_text(encoding='utf-8'), ('run',))
        except (OSError, ValueError) as error:
            raise BlockError(block.name, str(error)) from error
        actual = block.model_dump()
        if path.with_suffix('.json').exists():
            actual['operation'] = json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
        blocks.append({**actual, 'sha256': file_hash(path)})
    manifest = {'schema_version': 2, 'write_blocks': start.assembly.write_blocks, 'blocks': blocks}
    atomic_bytes(node / 'assembly.json', json.dumps(manifest, ensure_ascii=False, indent=2).encode('utf-8'))
    return implementation_hash(node)


def implementation_hash(node: Path) -> str:
    """Bind the wrapper, ordered manifest and every actual block to the executed result."""
    manifest = json.loads((node / 'assembly.json').read_text(encoding='utf-8'))
    digest = hashlib.sha256((node / 'experiment.py').read_bytes())
    digest.update((node / 'assembly.json').read_bytes())
    for block in manifest['blocks']:
        path = block_path(node, block['name'])
        if file_hash(path) != block['sha256']:
            raise BlockError(block['name'], 'Assembled block changed after its check')
        digest.update(path.read_bytes())
    return digest.hexdigest()


def sources(node: Path) -> dict[str, str]:
    start = load(node / 'start.json', Start)
    return {block.name: block_path(node, block.name).read_text(encoding='utf-8') for block in start.assembly.blocks}


def execute(node: Path, fit, validation, test, preparation, auxiliary=None):
    start = load(node / 'start.json', Start)
    manifest = json.loads((node / 'assembly.json').read_text(encoding='utf-8'))
    if [b['name'] for b in manifest['blocks']] != [b.name for b in start.assembly.blocks]:
        raise ValueError('Assembly order differs from the saved plan')
    state = dict(fit=fit, validation=validation, test=test, preparation=preparation, auxiliary=auxiliary or {})
    for position, block in enumerate(manifest['blocks']):
        name = block['name']
        state['preparation'] = preparation
        runtime = preparation.get('runtime', {})
        if (block.get('operation') or {}).get('kind') in {'fit', 'automl', 'forecast'} and runtime.get('deadline'):
            stages = sum((item.get('operation') or {}).get('kind') in {'fit', 'automl', 'forecast'}
                         for item in manifest['blocks'][position:])
            # Each model gets a share of the remaining training time, not the full node budget.
            remaining = max(0., runtime['deadline'] - time.monotonic()) / stages
            state['preparation'] = {**preparation, 'runtime': {**runtime,
                'deadline': time.monotonic() + remaining, 'time_budget_seconds': remaining,
                'remaining_training_stages': stages}}
        atomic_bytes(node / 'progress.json', json.dumps({'block': name}).encode('utf-8'))
        try:
            missing = set(block['requires']) - state.keys()
            if missing:
                raise ValueError(f'Missing state inputs: {sorted(missing)}')
            program = module(block_path(node, name), block['sha256'])
            updates = program.run(MappingProxyType(state))
            if not isinstance(updates, dict):
                raise ValueError('run(state) must return a dict of state updates')
            if updates.get('blocked_reason'):
                raise ValueError(f"Block could not produce its outputs: {updates['blocked_reason']}")
            if not set(block['provides']) <= updates.keys():
                raise ValueError('run(state) must return a dict containing every declared provides key')
            empty = [key for key in block['provides'] if updates[key] is None]
            if empty:
                raise ValueError(f'Required outputs are None: {empty}. Fix this producer before running later blocks')
            if updates.keys() & INITIAL_STATE:
                raise ValueError('Blocks cannot replace shared fit/validation/test/preparation/auxiliary inputs')
            state.update(updates)
        except Exception as error:
            detail = {'block': name, 'error': traceback.format_exc()[-5000:]}
            atomic_bytes(node / 'block_error.json', json.dumps(detail, ensure_ascii=False).encode('utf-8'))
            raise BlockError(name, str(error)) from error
    for key, rows in (('validation_predictions', len(validation)), ('test_predictions', len(test))):
        if not isinstance(state[key], pd.DataFrame) or len(state[key]) != rows:
            name = next(b['name'] for b in reversed(manifest['blocks']) if key in b['provides'])
            message = f'{key} must be a DataFrame with {rows} rows in the original order'
            atomic_bytes(node / 'block_error.json', json.dumps({'block': name, 'error': message}).encode('utf-8'))
            raise BlockError(name, message)
    return state['validation_predictions'], state['test_predictions']

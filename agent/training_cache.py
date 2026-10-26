"""Run-local trained models, shared by identical operations across worker processes."""
from contextlib import contextmanager
import hashlib
import copy
import os
from pathlib import Path
import time

import cloudpickle

from .storage import atomic_bytes, save


@contextmanager
def locked(path, deadline):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+b') as stream:
        if path.stat().st_size == 0:
            stream.write(b'0')
            stream.flush()
        while True:
            try:
                stream.seek(0)
                if os.name == 'nt':
                    import msvcrt
                    msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError as error:
                if error.errno not in (11, 13):
                    raise
                if time.monotonic() >= deadline:
                    raise TimeoutError('Waiting for an identical training operation exceeded the node budget')
                time.sleep(.1)
        try:
            yield
        finally:
            if os.name == 'nt':
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def cached(state, call, defaults, train):
    runtime = state['preparation'].get('runtime', {})
    if runtime.get('is_smoke_check') or not runtime.get('run_directory'):
        return train(state)
    keys = ('fit_features', 'fit_targets', 'transformer') if call.kind == 'fit' else ('fit', 'validation', 'test')
    key = hashlib.sha256(cloudpickle.dumps((1, runtime['preparation_sha256'], call.kind,
        call.parameters, defaults, [state[key] for key in keys]))).hexdigest()
    directory = Path(runtime['run_directory']) / 'training_cache' / key
    path = directory / 'result.pkl'
    with locked(directory / 'training.lock', runtime['deadline']):
        reused = path.exists()
        if reused:
            result = cloudpickle.loads(path.read_bytes())['result']
        else:
            preparation = {**state['preparation'], 'runtime': {**runtime, 'work_directory': str(directory)}}
            result = train({**state, 'preparation': preparation})
            artifact = {'result': result}
            if call.kind == 'fit':
                transformer = copy.copy(state['transformer'])
                # Keep the fitted preprocessing, not another copy of the training table.
                transformer.__dict__.pop('fit_source_', None)
                artifact['transformer'] = transformer
                artifact['feature_recipes'] = state['fit'].attrs.get('feature_recipes', [])
            atomic_bytes(path, cloudpickle.dumps(artifact))
        save(Path(runtime['work_directory']) / ('training_' + key[:12] + '.json'),
             {'kind': call.kind, 'cache': str(directory.relative_to(runtime['run_directory'])), 'reused': reused})
        return result

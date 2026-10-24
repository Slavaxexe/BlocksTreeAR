"""Local viewer state must never interrupt the training watchdog."""
import json
import os
from pathlib import Path
import tempfile
import time


def save_manifest(path, value, attempts=8):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                prefix='.'+path.name+'.', suffix='.tmp', delete=False) as stream:
            pending = Path(stream.name)
            json.dump(value, stream, ensure_ascii=False, indent=2)
        for attempt in range(attempts):
            try:
                os.replace(pending, path)
                return
            except PermissionError:
                if attempt+1 == attempts:raise
                time.sleep(min(.02*2**attempt, .25))
    finally:
        if pending is not None:
            try:pending.unlink(missing_ok=True)
            except OSError:pass


def publish_manifest(path, value):
    """Keep the last complete snapshot; retry on the next watchdog tick."""
    try:
        save_manifest(path, value)
        return True
    except OSError:
        return False

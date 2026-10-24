"""Validated, atomic artifacts; interrupted writes never become valid nodes."""

import hashlib
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import tempfile
import time

from max_contracts import Contract


def canonical(value) -> str:
    if isinstance(value, Contract):
        value = value.model_dump(mode="json")
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)


def fingerprint(value) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def atomic_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        # A viewer or antivirus can briefly keep a Windows file open without delete sharing.
        for attempt in range(6):
            try:
                os.replace(temporary, path)
                break
            except PermissionError as error:
                if getattr(error, 'winerror', None) not in {5, 32, 33} or attempt == 5:
                    raise
                time.sleep(min(0.02 * 2 ** attempt, 0.2))
    finally:
        Path(temporary).unlink(missing_ok=True)


def save(path: Path, value: Contract | dict) -> None:
    # Re-validate even locally constructed artifacts at the persistence boundary.
    canonical(value)  # Also rejects non-finite numbers nested in open-ended configuration blocks.
    if isinstance(value, Contract):
        type(value).model_validate_json(value.model_dump_json())
        value = value.model_dump(mode='json')
    atomic_bytes(path, json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2).encode('utf-8'))


def load(path: Path, schema):
    return schema.model_validate_json(path.read_text(encoding="utf-8"))


def append_event(path: Path, event: str, **details):
    record = {'time_utc': datetime.now(timezone.utc).isoformat(), 'event': event, **details}
    try:
        with path.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
    except OSError as error:
        print(f'Runtime log unavailable ({event}): {error}', file=sys.stderr)

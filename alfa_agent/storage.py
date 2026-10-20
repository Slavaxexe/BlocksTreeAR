"""Validated, atomic artifacts; interrupted writes never become valid nodes."""

import json
import os
from pathlib import Path
import tempfile

from .contracts import Contract


def canonical(value) -> str:
    if isinstance(value, Contract):
        value = value.model_dump(mode="json")
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)


def atomic_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def save(path: Path, value: Contract) -> None:
    # Re-validate even locally constructed artifacts at the persistence boundary.
    canonical(value)  # Also rejects non-finite numbers nested in open-ended configuration blocks.
    type(value).model_validate_json(value.model_dump_json())
    atomic_bytes(path, value.model_dump_json(indent=2).encode())


def load(path: Path, schema):
    return schema.model_validate_json(path.read_text(encoding="utf-8"))

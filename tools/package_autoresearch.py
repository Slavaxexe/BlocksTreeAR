"""Собрать ZIP для платформы: python tools/package_autoresearch.py dist/solution.zip

В корне архива solution.py (точка входа), рядом пакеты autoresearch/ и model_selection/ (основной пайплайн),
agent/ (прежний агент — запасной путь для неподдержанных задач) и requirements.txt.
Не попадают: данные, artifacts/, тесты, .env (ключ платформа передаёт переменными окружения).
"""
import hashlib
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INCLUDE_DIRS = ["autoresearch", "model_selection", "agent"]
INCLUDE_FILES = ["solution.py", "requirements.txt"]
SKIP_PARTS = {"__pycache__", "tests", "contracts", "artifacts", "catboost_info"}


def files():
    for f in INCLUDE_FILES:
        yield ROOT / f
    for d in INCLUDE_DIRS:
        for p in sorted((ROOT / d).rglob("*")):
            if p.is_file() and not (set(p.relative_to(ROOT).parts) & SKIP_PARTS) and p.suffix in (".py", ".md", ".html"):
                yield p


def build(dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    names = []
    with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for p in files():
            rel = p.relative_to(ROOT).as_posix()
            if rel.endswith(".env") or "/.env" in rel:
                continue
            z.write(p, rel)
            names.append(rel)
    assert "solution.py" in names and not any(n.endswith(".env") for n in names)
    print(f"{dest}\nфайлов: {len(names)}, байт: {dest.stat().st_size}, sha256: {hashlib.sha256(dest.read_bytes()).hexdigest()[:16]}")
    return names


if __name__ == "__main__":
    build(Path(sys.argv[1] if len(sys.argv) > 1 else ROOT / "dist" / "solution.zip").resolve())

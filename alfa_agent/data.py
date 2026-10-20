"""Format-independent file discovery and measured profiles; readers are generated per task."""

import ast
import hashlib
import importlib.util
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import pandas as pd

# Distribution → import root from the organizer's requirements (also used by flat ZIP exports).
LIBRARIES = {name: name for name in (
    'numpy scipy pandas polars pyarrow joblib threadpoolctl catboost lightgbm xgboost statsmodels optuna torch '
    'lightautoml flaml openai httpx tenacity langchain langgraph pydantic orjson jsonschema matplotlib seaborn '
    'tqdm psutil cloudpickle setuptools').split()} | {
    'scikit-learn': 'sklearn', 'imbalanced-learn': 'imblearn', 'category-encoders': 'category_encoders',
    'feature-engine': 'feature_engine', 'autogluon.tabular': 'autogluon', 'langchain-openai': 'langchain_openai',
    'python-dotenv': 'dotenv', 'PyYAML': 'yaml'}


def library_context() -> dict:
    installed = {}
    for name in LIBRARIES:
        try:
            installed[name] = version(name)
        except PackageNotFoundError:
            pass
    return {'allowed_libraries': LIBRARIES, 'installed_libraries': installed}


def within(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"Path outside the permitted directory: {relative}")
    return path


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def input_hashes(directory: Path, files: dict[str, str]) -> dict[str, str]:
    result = {}
    for relative in files.values():
        path = within(directory, relative)
        paths = sorted(p for p in path.rglob('*') if p.is_file()) if path.is_dir() else [path]
        for child in paths:
            result[child.relative_to(directory).as_posix()] = file_hash(within(directory, str(child)))
    return result


def inventory(directory: Path, limit: int = 150) -> list[dict]:
    """Send filenames and sizes, never raw dataset rows or hidden settings, to Qwen."""
    files = []
    for path in sorted(directory.rglob('*')):
        relative = path.relative_to(directory)
        if any(p.startswith('.') or p in {'artifacts', 'outputs', '__pycache__', 'tmp'} for p in relative.parts):
            continue
        if path.is_file() and path.resolve().is_relative_to(directory.resolve()):
            files.append({'path': relative.as_posix(), 'bytes': path.stat().st_size})
            if len(files) == limit:
                break
    return files


def profile(tables: dict) -> dict:
    result = {}
    for role, frame in tables.items():
        if not isinstance(frame, pd.DataFrame):
            raise ValueError(f"load_data must return DataFrames, invalid role: {role}")
        columns = {}
        for column in frame.columns:
            series = frame[column]
            try:
                distinct = int(series.nunique())
            except TypeError:  # Structured cells may be unhashable; profiling must still support them.
                distinct = None
            columns[str(column)] = {'dtype': str(series.dtype), 'missing': int(series.isna().sum()), 'distinct': distinct}
        result[role] = {'rows': len(frame), 'columns': columns}
    return result


def check_code(code: str, functions: tuple[str, ...]):
    """A narrow generated interface, not an OS sandbox. Execution is separately bounded."""
    try:
        tree = ast.parse(code)
    except SyntaxError as error:
        raise ValueError(f'Python syntax error on line {error.lineno}: {error.msg}') from None
    names = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
    if not set(functions).issubset(names):
        raise ValueError('Missing required functions: ' + ', '.join(sorted(set(functions) - names)))
    allowed = {'math', 'statistics', 'collections', 'itertools', 'functools', 'pathlib', 'csv', 'json',
               're', 'datetime', 'typing', 'io', 'zipfile', 'gzip', 'random', 'copy', 'time', 'tempfile',
               'warnings', 'contextlib', 'operator', 'dataclasses', 'enum', '__future__'} | set(LIBRARIES.values())
    for node in ast.walk(tree):
        imports = [a.name for a in node.names] if isinstance(node, ast.Import) else (
            [node.module or ''] if isinstance(node, ast.ImportFrom) else [])
        if any(name.split('.')[0] not in allowed for name in imports):
            raise ValueError('Unsupported generated import: ' + ', '.join(imports))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in {'eval', 'exec', '__import__'}:
            raise ValueError('Dynamic code execution is not part of the generated interface')
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef)):
            continue
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            continue
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            try:
                ast.literal_eval(node.value)
            except (ValueError, TypeError):
                raise ValueError(f'Line {node.lineno}: top-level assignments must be literal constants. '
                                 'Move computed values inside a function. Computations and comprehensions are allowed there.') from None
            continue
        raise ValueError('Generated files must contain imports, functions, classes and literal constants only')


def module(path: Path, expected_hash: str):
    if file_hash(path) != expected_hash:
        raise ValueError('Generated source changed after it was recorded')
    spec = importlib.util.spec_from_file_location('generated_' + path.stem, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result

"""Restricted row-local classification feature functions in a separate process."""
import ast
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import numpy as np
import pandas as pd


def check_code(code):
    try:
        if not isinstance(code, str) or len(code) > 12000:
            raise ValueError('Feature code must have at most 12000 characters')
        tree = ast.parse(code)
        if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
            raise ValueError('Define only add_features(df), without top-level code/imports')
        function = tree.body[0]
        if (function.name != 'add_features' or len(function.args.args) != 1 or function.args.args[0].arg != 'df'
                or function.decorator_list or function.args.defaults or function.args.kwarg or function.args.vararg
                or function.args.kwonlyargs or function.args.posonlyargs or function.returns):
            raise ValueError('Required signature: def add_features(df):')
        allowed = {'copy', 'assign', 'astype', 'fillna', 'replace', 'clip', 'isna', 'notna', 'abs', 'round',
                   'log1p', 'log', 'sqrt', 'sign', 'maximum', 'minimum', 'where', 'isfinite', 'to_numeric',
                   'mean', 'sum', 'min', 'max', 'std', 'nan', 'inf'}
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom, ast.For, ast.While, ast.With, ast.Try, ast.ClassDef,
                ast.Lambda, ast.Global, ast.Nonlocal, ast.ListComp, ast.GeneratorExp, ast.DictComp, ast.SetComp,
                ast.Delete, ast.AugAssign, ast.NamedExpr, ast.Raise)):
                raise ValueError(f'Forbidden syntax: {type(node).__name__}')
            if isinstance(node, ast.FunctionDef) and node is not function:
                raise ValueError('Nested functions are forbidden')
            if isinstance(node, ast.Attribute) and node.attr not in allowed:
                raise ValueError(f'Forbidden feature attribute: {node.attr}')
            if isinstance(node, ast.Name) and node.id.startswith('_'):
                raise ValueError('Private/reflection names are forbidden')
            if isinstance(node, ast.Call):
                if not isinstance(node.func, ast.Attribute):
                    raise ValueError('Only allowed numpy/pandas row-local calls may be used')
                if node.func.attr in {'mean', 'sum', 'min', 'max', 'std'} and not any(
                    arg.arg == 'axis' and isinstance(arg.value, ast.Constant) and arg.value.value == 1 for arg in node.keywords):
                    raise ValueError('Reductions must use axis=1; no global statistics')
                if any(arg.arg is None for arg in node.keywords):
                    raise ValueError('No expanded keyword arguments')
        return []
    except (ValueError, SyntaxError) as error:
        return [str(error)]


def apply_code(code, frame, timeout=25):
    errors = check_code(code)
    if errors:
        raise ValueError('; '.join(errors))
    with tempfile.TemporaryDirectory(prefix='classification_features_') as folder:
        root = Path(folder)
        frame.to_parquet(root/'input.parquet', index=True)
        (root/'code.json').write_text(json.dumps(code), encoding='utf-8')
        env = {key: value for key, value in os.environ.items() if key.upper() in {'SYSTEMROOT', 'WINDIR', 'TEMP', 'TMP', 'TMPDIR'}}
        env.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
        result = subprocess.run([sys.executable, '-I', str(Path(__file__).resolve()), str(root)], cwd=root,
            env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=max(.1, timeout))
        if result.returncode:
            raise ValueError('Custom feature function failed: '+result.stderr.decode('utf-8', errors='replace')[-1200:])
        return pd.read_parquet(root/'output.parquet')


def worker(root):
    code = json.loads((root/'code.json').read_text(encoding='utf-8'))
    if check_code(code):
        raise ValueError('Feature AST rejected')
    frame = pd.read_parquet(root/'input.parquet')
    namespace = {'__builtins__': {}, 'np': np, 'pd': pd}
    exec(compile(code, '<classification_features>', 'exec'), namespace)
    output = namespace['add_features'](frame.copy())
    if (not isinstance(output, pd.DataFrame) or not output.index.equals(frame.index)
            or output.columns.duplicated().any() or not set(frame).issubset(output)
            or not output[frame.columns].equals(frame)):
        raise ValueError('Feature function must preserve all original rows/columns unchanged')
    new = [name for name in output if name not in frame]
    if not 1 <= len(new) <= 32 or any(not pd.api.types.is_numeric_dtype(output[name]) for name in new):
        raise ValueError('Add between 1 and 32 numeric features')
    output[new] = output[new].replace([np.inf, -np.inf], np.nan)
    output.to_parquet(root/'output.parquet', index=True)


if __name__ == '__main__':
    worker(Path(sys.argv[1]))

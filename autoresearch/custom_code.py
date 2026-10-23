"""
Код признаков от Qwen: она пишет функцию, код её проверяет и исполняет в изолированном процессе.

Контракт для Qwen:
    def add_features(df):      # df — все признаки train+test БЕЗ таргета; pd и np уже доступны
        out = pd.DataFrame(index=df.index)
        out["area_per_year"] = df["floor_area"] / (2025 - df["year_built"])
        return out             # только НОВЫЕ числовые колонки, до 25 штук

Проверки до запуска (AST): нет import/exec/eval/open/dunder-атрибутов, нет имени таргета в коде, есть add_features.
Проверки после: DataFrame той же длины, ≤25 колонок, числовые, не константы, не копии существующих колонок,
≥50% конечных значений, время ≤ таймаута. Таргет физически отсутствует во входе — утечка через таргет невозможна.
"""
from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

FORBIDDEN_NAMES = {"open", "exec", "eval", "compile", "__import__", "globals", "locals", "vars", "getattr",
                   "setattr", "delattr", "input", "breakpoint", "exit", "quit", "help", "memoryview"}
FORBIDDEN_ATTRS = {"to_csv", "to_parquet", "to_pickle", "read_csv", "read_parquet", "read_pickle", "system", "popen"}
MAX_COLS = 25


def check_code(code: str, target: str) -> list[str]:
    errs = []
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return [f"синтаксис: {e}"]
    if not any(isinstance(n, ast.FunctionDef) and n.name == "add_features" for n in tree.body):
        errs.append("нужна функция верхнего уровня add_features(df)")
    for n in ast.walk(tree):
        if isinstance(n, (ast.Import, ast.ImportFrom)):
            errs.append("import запрещён: pd и np уже доступны")
        elif isinstance(n, ast.Name) and n.id in FORBIDDEN_NAMES:
            errs.append(f"запрещено: {n.id}")
        elif isinstance(n, ast.Attribute) and (n.attr.startswith("__") or n.attr in FORBIDDEN_ATTRS):
            errs.append(f"запрещённый атрибут: {n.attr}")
        elif isinstance(n, ast.Constant) and isinstance(n.value, str) and n.value == target:
            errs.append(f"код не должен обращаться к таргету '{target}' — его нет во входных данных")
    return sorted(set(errs))


SANDBOX_MAIN = r'''
import sys, json, numpy as np, pandas as pd
code_path, in_path, out_path = sys.argv[1:4]
df = pd.read_parquet(in_path)
ns = {"pd": pd, "np": np, "__builtins__": {k: __builtins__[k] if isinstance(__builtins__, dict) else getattr(__builtins__, k)
      for k in ("abs","min","max","len","range","sum","float","int","str","list","dict","zip","enumerate","round","sorted","set","tuple","bool","isinstance","map","any","all","print","ValueError")}}
exec(compile(open(code_path).read(), "qwen_features", "exec"), ns)
res = ns["add_features"](df.copy())
if not isinstance(res, pd.DataFrame):
    raise TypeError("add_features должна вернуть pandas.DataFrame")
res.reset_index(drop=True).to_parquet(out_path, index=False)
'''


def run_custom(code: str, raw_features: pd.DataFrame, target: str, existing: pd.DataFrame, block_dir: Path,
               name: str, timeout: float = 90) -> tuple[pd.DataFrame | None, list[str]]:
    errs = check_code(code, target)
    if errs:
        return None, errs
    work = Path(block_dir).resolve() / f"{name}_work"
    work.mkdir(parents=True, exist_ok=True)
    (work / "code.py").write_text(code, encoding="utf-8")
    (work / "sandbox.py").write_text(SANDBOX_MAIN, encoding="utf-8")
    inp, out = work / "input.parquet", work / "output.parquet"
    if not inp.exists():
        raw_features.reset_index(drop=True).to_parquet(inp, index=False)
    try:
        r = subprocess.run([sys.executable, "-I", str(work / "sandbox.py"), str(work / "code.py"), str(inp), str(out)],
                           capture_output=True, text=True, timeout=timeout, cwd=work)
    except subprocess.TimeoutExpired:
        return None, [f"превышено время {timeout} с"]
    if r.returncode != 0:
        return None, [f"ошибка выполнения: {r.stderr.strip()[-900:]}"]
    res = pd.read_parquet(out)
    errs = []
    if len(res) != len(raw_features):
        errs.append(f"вернула {len(res)} строк вместо {len(raw_features)}")
    if res.shape[1] == 0 or res.shape[1] > MAX_COLS:
        errs.append(f"нужно 1..{MAX_COLS} новых колонок, получено {res.shape[1]}")
    if errs:
        return None, errs
    keep, notes = {}, []
    sample = existing.sample(min(5000, len(existing)), random_state=0)
    for c in res.columns:
        v = pd.to_numeric(res[c], errors="coerce").astype(float).replace([np.inf, -np.inf], np.nan)
        if v.notna().mean() < 0.5:
            notes.append(f"{c}: <50% конечных значений — отброшена")
            continue
        if v.nunique() <= 1:
            notes.append(f"{c}: константа — отброшена")
            continue
        vs = v.loc[sample.index]
        dup = [e for e in sample.columns if pd.api.types.is_numeric_dtype(sample[e]) and np.allclose(vs.fillna(-9e9), sample[e].astype(float).fillna(-9e9))]
        if dup:
            notes.append(f"{c}: копия {dup[0]} — отброшена")
            continue
        keep[f"{name}__{c}"] = v.to_numpy()
    if not keep:
        return None, ["после проверок не осталось ни одной полезной колонки"] + notes
    return pd.DataFrame(keep), notes

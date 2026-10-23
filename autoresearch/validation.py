"""
Этап 2. Дизайн валидации — решение кода, не LLM: от него зависит, чему верить во всём исследовании.

Принцип: валидация должна воспроизводить то, как устроен test.
  - test в будущем (есть колонка времени) → rolling-origin: фолд k учится на периодах < p_k и проверяется на p_k,
    p_k — последние периоды train. Ровно как test: учимся на прошлом, предсказываем следующий период.
  - есть повторяющиеся объекты без времени → GroupKFold по ключу объекта.
  - иначе → KFold.
Ранняя остановка бустингов — на внутренней отложенной части обучающего фолда (по группам объектов),
никогда на валидационном фолде: иначе оценка завышена.
Фолды фиксируются один раз (npz + sha256) и одинаковы для всех экспериментов.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, KFold, ShuffleSplit


def design_folds(train: pd.DataFrame, time_col: str | None, entity_key: list[str] | None, seed: int,
                 max_time_folds: int = 2, min_train_share: float = 0.3) -> dict:
    n = len(train)
    groups = None
    if entity_key:
        groups = pd.util.hash_pandas_object(train[entity_key].astype(str), index=False).to_numpy()
    folds = []
    if time_col:
        periods = sorted(train[time_col].dropna().unique())
        p = train[time_col].to_numpy()
        for vp in periods[::-1]:
            tr = np.where(p < vp)[0]
            va = np.where(p == vp)[0]
            if len(tr) < min_train_share * n or len(folds) >= max_time_folds:
                break
            folds.append({"train": tr, "valid": va, "label": f"train<{vp} → valid={vp}"})
        folds = folds[::-1]
        kind = "time_rolling_origin"
    if not folds:
        if groups is not None:
            sp = GroupKFold(n_splits=4).split(np.zeros(n), groups=groups)
            kind = "group_kfold"
        else:
            sp = KFold(n_splits=4, shuffle=True, random_state=seed).split(np.zeros(n))
            kind = "kfold"
        folds = [{"train": tr, "valid": va, "label": f"fold{i}"} for i, (tr, va) in enumerate(sp)]
    # внутренняя отложенная часть для ранней остановки — из обучающей части фолда
    for f in folds:
        tr = f["train"]
        if groups is not None:
            a, b = next(GroupShuffleSplit(1, test_size=0.12, random_state=seed).split(tr, groups=groups[tr]))
        else:
            a, b = next(ShuffleSplit(1, test_size=0.12, random_state=seed).split(tr))
        f["fit"], f["early_stop"] = tr[a], tr[b]
    weights = np.array([len(f["valid"]) for f in folds], float)
    return {"kind": kind, "folds": folds, "weights": (weights / weights.sum()).tolist(),
            "labels": [f["label"] for f in folds],
            "sizes": [{"fit": len(f["fit"]), "early_stop": len(f["early_stop"]), "valid": len(f["valid"])} for f in folds]}


def save_folds(design: dict, path: Path) -> str:
    arrays = {}
    for i, f in enumerate(design["folds"]):
        for k in ("train", "valid", "fit", "early_stop"):
            arrays[f"{k}_{i}"] = f[k]
    np.savez(path, **arrays, n_folds=len(design["folds"]))
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def load_folds(path: Path) -> list[dict]:
    z = np.load(path)
    return [{k: z[f"{k}_{i}"] for k in ("train", "valid", "fit", "early_stop")} for i in range(int(z["n_folds"]))]

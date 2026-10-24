"""
Этап 7. Финальное обучение и ответ.

Каждый член ансамбля переобучается на ВСЕХ размеченных строках (число итераций = среднее по фолдам × 1.15:
данных больше, чем в фолде), предсказывает test; смесь по весам. Если на кого-то не хватает времени или он упал —
веса перенормируются по успешным. Предсказания обрезаются диапазоном таргета train.
submission.csv пишется в порядке test.csv и проверяется: строки, id, конечность, дубликаты.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from ar_experiment import ExperimentSpec, run_in_subprocess


def refit_members(weights: dict, mem, ctx: dict, run_dir: Path, deadline: float, log=print,
                  max_seeds: int = 3) -> tuple[np.ndarray | None, dict]:
    exps = {e["id"]: e for e in mem.state["experiments"]}
    preds, used, info = [], [], {}
    members = sorted(weights.items(), key=lambda t: -t[1])
    ests = {i: 1.6 * exps[i]["seconds"] / max(1, len(ctx["fold_weights"])) + 30 for i, _ in members}
    # усреднение по сидам (снижает дисперсию), если времени хватает на 3 сида для всех членов с запасом
    n_seeds = max_seeds if deadline - time.monotonic() > max_seeds * sum(ests.values()) * 1.5 + 60 else 1
    for exp_id, w in members:
        e = exps[exp_id]
        spec = ExperimentSpec.model_validate(e["spec"])
        iters = [i for i in (e.get("best_iters") or []) if i]
        n_rounds = int(np.mean(iters) * 1.15) if iters and spec.model.model_id in ("lightgbm", "catboost", "xgboost") else None
        seed_preds, runs = [], []
        for k in range(n_seeds if spec.model.model_id not in ("ridge", "elasticnet") else 1):
            left = deadline - time.monotonic()
            if left < ests[exp_id]:
                runs.append({"status": "skipped", "reason": f"нет времени: нужно ~{int(ests[exp_id])} с, есть {int(left)} с"})
                break
            d = run_dir / "final" / f"{exp_id}_seed{k}"
            res = run_in_subprocess(spec, {**ctx, "mode": "final", "n_rounds": n_rounds, "seed": ctx["seed"] + k}, d,
                                    timeout=min(left - 20, max(120, ests[exp_id] * 3)))
            runs.append({kk: res.get(kk) for kk in ("status", "error", "seconds", "n_rounds")})
            p = d / "test_pred.npy"
            if res.get("status") == "ok" and p.exists():
                pr = np.load(p)
                if np.isfinite(pr).all():
                    seed_preds.append(pr)
        info[exp_id] = {"seeds_ok": len(seed_preds), "runs": runs}
        if seed_preds:
            preds.append(np.mean(seed_preds, axis=0))
            used.append(w)
        log(f"   финал {exp_id} ({spec.summary()[:60]}): сидов {len(seed_preds)}/{len(runs)}, "
            f"{[r.get('seconds') for r in runs]} с {[r.get('error') for r in runs if r.get('error')][:1]}")
    if not preds:
        return None, info
    wv = np.array(used) / sum(used)
    return np.sum([w * p for w, p in zip(wv, preds)], axis=0), info


def write_submission(pred: np.ndarray, test_ids: pd.DataFrame, id_cols: list[str], target: str, sample_path: str | None,
                     y_train: np.ndarray, out_paths: list[Path]) -> dict:
    lo, hi = float(np.nanmin(y_train)), float(np.nanmax(y_train))
    pred = np.clip(np.nan_to_num(pred, nan=float(np.nanmean(y_train))), lo, hi)
    sub = test_ids[id_cols].copy()
    sub[target] = pred
    cols = list(pd.read_csv(sample_path, nrows=1).columns) if sample_path else id_cols + [target]
    sub = sub[cols]
    checks = {"rows": len(sub), "rows_match_test": len(sub) == len(test_ids), "dup_ids": int(sub.duplicated(id_cols).sum()),
              "finite": bool(np.isfinite(sub[target]).all()), "columns": cols,
              "pred_mean": round(float(sub[target].mean()), 4), "pred_std": round(float(sub[target].std()), 4),
              "train_mean": round(float(np.nanmean(y_train)), 4)}
    ok = checks["rows_match_test"] and checks["dup_ids"] == 0 and checks["finite"]
    checks["ok"] = bool(ok)
    for p in out_paths:
        p.parent.mkdir(parents=True, exist_ok=True)
        sub.to_csv(p, index=False)
    return checks

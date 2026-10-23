"""
Этап 6. Ансамбль: жадный отбор с повторами (Caruana, 2004) по OOF-предсказаниям на тех же фолдах.
Веса = доля выборов. Ансамбль принимается, только если он лучше лучшей одиночной модели больше порога значимости —
иначе в финал идёт одна лучшая модель (проще и надёжнее).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from model_selection.metrics import score


def build_ensemble(mem, exp_dir: Path, y: np.ndarray, metric: str, min_rel: float, max_candidates: int = 12,
                   steps: int = 40) -> dict:
    sign = 1 if mem.direction == "minimize" else -1
    ok = sorted(mem.ok_experiments(), key=lambda e: sign * e["cv"])
    cands, oofs = [], []
    for e in ok:
        p = exp_dir / e["id"] / "oof.npy"
        if p.exists():
            cands.append(e)
            oofs.append(np.load(p))
        if len(cands) >= max_candidates:
            break
    if not cands:
        return {"used": False, "reason": "нет OOF-предсказаний"}
    O = np.vstack(oofs)
    mask = ~np.isnan(O).any(axis=0)
    O, yy = O[:, mask], y[mask]
    single = [score(metric, yy, O[i]) for i in range(len(cands))]
    best_i = int(np.argmin([sign * s for s in single]))
    counts = np.zeros(len(cands))
    cur = np.zeros(mask.sum())
    k = 0
    history = []
    for _ in range(steps):
        trial = [score(metric, yy, (cur * k + O[i]) / (k + 1)) for i in range(len(cands))]
        i = int(np.argmin([sign * t for t in trial]))
        if history and sign * trial[i] >= sign * history[-1] - 1e-12 and k >= 2:
            break
        counts[i] += 1
        cur = (cur * k + O[i]) / (k + 1)
        k += 1
        history.append(trial[i])
    ens_score = score(metric, yy, cur)
    rel = sign * (single[best_i] - ens_score) / abs(single[best_i])
    weights = {cands[i]["id"]: round(float(c / counts.sum()), 4) for i, c in enumerate(counts) if c > 0}
    used = rel >= max(min_rel, 0.003) and len(weights) > 1
    return {"used": bool(used), "weights": weights if used else {cands[best_i]["id"]: 1.0},
            "ensemble_score": round(float(ens_score), 5), "best_single": cands[best_i]["id"],
            "best_single_score": round(float(single[best_i]), 5), "rel_gain_pct": round(rel * 100, 3),
            "candidates": [{"id": c["id"], "summary": c.get("summary"), "oof_score": round(float(s), 5)}
                           for c, s in zip(cands, single)],
            "reason": "ансамбль лучше лучшей модели" if used else "выигрыш ниже порога — берём лучшую одиночную модель"}

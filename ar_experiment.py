"""
Эксперимент = (блоки признаков, модель/лосс/параметры, режим обучения). Исполняется в отдельном процессе
с жёстким таймаутом: зависшая модель не может съесть 3-часовой лимит.

Режим cv:    обучение на фиксированных фолдах, OOF-предсказания, метрика по фолдам, разбор ошибок по сегментам.
Режим final: обучение на всех размеченных строках с числом итераций из cv, предсказание test.
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from ms_metrics import score as metric_score


class ModelSpec(BaseModel):
    model_id: str = "lightgbm"
    loss: str = "l2"
    target_transform: Literal["none", "log1p"] = "none"
    params: dict = Field(default_factory=dict)


class TrainingSpec(BaseModel):
    sample_weight: Literal["none", "recent"] = "none"
    recent_decay: float = Field(0.7, ge=0.2, le=1.0, description="вес периода = decay^(последний−период)")
    drop_patterns: list[str] = Field(default_factory=list, description="подстроки имён колонок, которые исключить")
    target_clip_quantile: float | None = Field(None, ge=0.9, le=1.0, description="обрезать y СВЕРХУ при обучении")


class ExperimentSpec(BaseModel):
    blocks: list[str] = Field(default_factory=lambda: ["base"])
    model: ModelSpec = Field(default_factory=ModelSpec)
    training: TrainingSpec = Field(default_factory=TrainingSpec)

    def key(self) -> str:
        d = self.model_dump()
        d["blocks"] = sorted(set(d["blocks"]) | {"base"})
        d["training"]["drop_patterns"] = sorted(d["training"]["drop_patterns"])
        return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()[:12]

    def summary(self) -> str:
        m, t = self.model, self.training
        extra = []
        if m.params:
            extra.append(",".join(f"{k}={v}" for k, v in m.params.items()))
        if t.sample_weight != "none":
            extra.append(f"w=recent{t.recent_decay}")
        if t.drop_patterns:
            extra.append("drop=" + "|".join(t.drop_patterns))
        if t.target_clip_quantile:
            extra.append(f"clip_q={t.target_clip_quantile}")
        blocks = "+".join(b for b in self.blocks if b != "base") or "base"
        return f"{m.model_id}/{m.loss}/{m.target_transform} [{blocks}] " + " ".join(extra)


# --------------------------------------------------------------------------- исполнение в процессе-воркере
def _load_matrix(ctx: dict, blocks: list[str], drop_patterns: list[str]) -> pd.DataFrame:
    parts = []
    for b in dict.fromkeys(["base", *blocks]):
        parts.append(pd.read_parquet(Path(ctx["cache_dir"]) / f"{b}.parquet"))
    X = pd.concat(parts, axis=1)
    X = X.loc[:, ~X.columns.duplicated()]
    if drop_patterns:
        X = X[[c for c in X.columns if not any(p in c for p in drop_patterns)]]
    return X


def _weights(ctx, idx, spec: ExperimentSpec):
    if spec.training.sample_weight != "recent" or ctx.get("period") is None:
        return None
    p = np.asarray(ctx["period"], float)[idx]
    return spec.training.recent_decay ** (np.nanmax(np.asarray(ctx["period"], float)[: ctx["n_train"]]) - p)


def worker_main(spec_path: str, ctx_path: str, out_dir: str):
    from ar_models import fit_predict
    from ar_validation import load_folds
    spec = ExperimentSpec.model_validate_json(Path(spec_path).read_text())
    ctx = json.loads(Path(ctx_path).read_text())
    out = Path(out_dir)
    t0 = time.time()
    y_all = np.load(ctx["target_npy"])
    n_train = ctx["n_train"]
    if ctx.get("period_npy"):
        ctx["period"] = np.load(ctx["period_npy"])
    X = _load_matrix(ctx, spec.blocks, spec.training.drop_patterns)
    Xtr = X.iloc[:n_train].reset_index(drop=True)
    y = y_all[:n_train]
    m, tr_cfg = spec.model, spec.training
    clip_hi = np.quantile(y, tr_cfg.target_clip_quantile) if tr_cfg.target_clip_quantile else None

    def yfit(idx):
        v = y[idx]
        return np.minimum(v, clip_hi) if clip_hi is not None else v

    res = {"status": "ok", "n_features": int(X.shape[1])}
    if ctx["mode"] == "cv":
        folds = load_folds(Path(ctx["folds_npz"]))
        oof = np.full(n_train, np.nan)
        fold_scores, train_scores, iters, imps = [], [], [], []
        for f in folds:
            fit, es, va = f["fit"], f["early_stop"], f["valid"]
            trs = fit[np.random.default_rng(0).choice(len(fit), min(len(fit), 15000), replace=False)]
            (pv, pt), it, imp = fit_predict(m.model_id, m.loss, m.target_transform, m.params, Xtr.iloc[fit], yfit(fit),
                                            Xtr.iloc[es], y[es], [Xtr.iloc[va], Xtr.iloc[trs]], ctx["n_jobs"], ctx["seed"],
                                            sample_weight=_weights(ctx, fit, spec))
            oof[va] = pv
            fold_scores.append(metric_score(ctx["metric"], y[va], pv))
            train_scores.append(metric_score(ctx["metric"], y[trs], pt))
            iters.append(it)
            if imp is not None:
                imps.append(imp / max(imp.sum(), 1e-12))
        np.save(out / "oof.npy", oof)
        w = np.asarray(ctx["fold_weights"])
        res.update(cv=float(np.dot(w, fold_scores)), folds=[round(s, 5) for s in fold_scores],
                   cv_std=float(np.std(fold_scores)), train_cv=float(np.dot(w, train_scores)), best_iters=iters)
        if imps:
            imp = pd.concat(imps, axis=1).mean(axis=1).sort_values(ascending=False)
            res["top_features"] = [[k, round(float(v), 4)] for k, v in imp.head(12).items()]
            res["n_zero_importance"] = int((imp <= 0).sum())
        res["segments"] = _segments(ctx, y, oof)
    else:  # final
        idx = np.arange(n_train)
        rounds = ctx.get("n_rounds")
        (pt,), it, _ = fit_predict(m.model_id, m.loss, m.target_transform, m.params, Xtr, yfit(idx), None, None,
                                   [X.iloc[n_train:].reset_index(drop=True)], ctx["n_jobs"], ctx["seed"],
                                   sample_weight=_weights(ctx, idx, spec), n_rounds=rounds)
        np.save(out / "test_pred.npy", pt)
        res["n_rounds"] = it
    res["seconds"] = round(time.time() - t0, 1)
    (out / "result.json").write_text(json.dumps(res, ensure_ascii=False, default=float))


def _segments(ctx: dict, y: np.ndarray, oof: np.ndarray) -> dict:
    """Где ошибка: хвост, наличие истории объекта, категории. Это то, что видит аналитик-Qwen."""
    mask = ~np.isnan(oof)
    e2 = (y[mask] - oof[mask]) ** 2
    out = {}
    if len(e2):
        o = np.sort(e2)[::-1]
        out["top1pct_rows_share_of_sq_error"] = round(float(o[: max(1, len(o) // 100)].sum() / o.sum()), 3)
        out["rmse_without_top1pct"] = round(float(np.sqrt(o[max(1, len(o) // 100):].mean())), 3)
        out["mean_bias"] = round(float((oof[mask] - y[mask]).mean()), 3)
    ent = Path(ctx["cache_dir"]) / "entity_history.parquet"
    if ent.exists():
        has = pd.read_parquet(ent, columns=["ent_n"])["ent_n"].to_numpy()[: len(y)] > 0
        for name, s in (("with_history", has), ("without_history", ~has)):
            mm = mask & s
            if mm.sum():
                out[f"rmse_{name}"] = round(float(np.sqrt(((y[mm] - oof[mm]) ** 2).mean())), 3)
                out[f"share_{name}"] = round(float(mm.sum() / mask.sum()), 3)
    if ctx.get("segment_col_npy"):
        g = np.load(ctx["segment_col_npy"], allow_pickle=True)[: len(y)]
        df = pd.DataFrame({"g": g[mask], "e2": e2})
        agg = df.groupby("g")["e2"].agg(["sum", "count", "mean"])
        agg["share"] = agg["sum"] / agg["sum"].sum()
        top = agg.sort_values("share", ascending=False).head(5)
        out["worst_segments"] = [{"segment": str(k), "rows": int(r["count"]), "rmse": round(float(np.sqrt(r["mean"])), 2),
                                  "share_of_sq_error": round(float(r["share"]), 3)} for k, r in top.iterrows()]
    return out


def run_in_subprocess(spec: ExperimentSpec, ctx: dict, out_dir: Path, timeout: float) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "spec.json").write_text(spec.model_dump_json(indent=1))
    (out_dir / "ctx.json").write_text(json.dumps(ctx, default=str))
    root = Path(__file__).resolve().parent
    t = time.time()
    try:
        r = subprocess.run([sys.executable, str(root / "ar_experiment.py"), str(out_dir / "spec.json"),
                            str(out_dir / "ctx.json"), str(out_dir)], capture_output=True, text=True,
                           timeout=max(10, timeout), cwd=root)
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "error": f"превышен таймаут {int(timeout)} с", "seconds": round(time.time() - t, 1)}
    if r.returncode != 0 or not (out_dir / "result.json").exists():
        err = (r.stderr or r.stdout or "").strip()
        tail = "\n".join(err.splitlines()[-6:])
        return {"status": "failed", "error": tail[-1200:], "seconds": round(time.time() - t, 1)}
    return json.loads((out_dir / "result.json").read_text())


if __name__ == "__main__":
    worker_main(*sys.argv[1:4])

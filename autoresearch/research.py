"""
Этапы 4–5. Лаборатория экспериментов и цикл AutoResearch.

Lab        — проверяет спецификацию, исполняет эксперимент (подпроцесс + таймаут), пишет результат в память
             с вычисленным кодом вердиктом (улучшил / не дал / хуже), следит за бюджетом.
Researcher — Qwen: за один вызов разбирает последний эксперимент, обновляет выводы и предлагает следующий.
             Видит всю память. Ответ проверяется; при отказе — один повтор с ошибками.
Heuristic  — исследователь-эвристика: тот же интерфейс, без LLM. Работает, когда Qwen недоступна, отключилась
             по квоте или трижды подряд предложила невалидное. Пайплайн никогда не стоит из-за LLM.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field

from model_selection.registry import LOSSES, METRICS
from model_selection.registry import MODELS as REG
from model_selection.rules import loss_policy

from .config import Budget, Settings
from .custom_code import check_code, run_custom
from .experiment import ExperimentSpec, ModelSpec, TrainingSpec, run_in_subprocess
from .features import FeatureStore, find_families
from .llm import LLM
from .memory import Memory
from .models import DEFAULTS, MODEL_IDS, PARAM_SPACE, check_model_spec, validate_params

DEFAULT_SECONDS = {"lightgbm": 60, "catboost": 300, "xgboost": 150, "extra_trees": 150, "random_forest": 200,
                   "ridge": 20, "elasticnet": 40}


# ============================================================================ Lab
class Lab:
    def __init__(self, mem: Memory, settings: Settings, budget: Budget, store: FeatureStore, ctx: dict,
                 run_dir: Path, raw_features, target: str):
        self.mem, self.s, self.budget, self.store, self.ctx = mem, settings, budget, store, ctx
        self.dir = Path(run_dir) / "experiments"
        self.raw_features, self.target = raw_features, target
        self.n = 0
        self.custom_n = 0
        self.custom_hashes: dict[str, str] = {}

    # ---- проверки
    def validate(self, spec: ExperimentSpec) -> list[str]:
        errs = []
        avail = self.store.available()
        bad = [b for b in spec.blocks if b not in avail]
        if bad:
            errs.append(f"блоки {bad} недоступны; доступны {sorted(avail)}")
        m = spec.model
        errs += check_model_spec(m.model_id, m.loss, m.target_transform)
        errs += validate_params(m.model_id, m.params)
        if m.loss in ("poisson", "gamma", "tweedie") and (self.mem.get("audit.target") or {}).get("min", 0) < 0:
            errs.append(f"лосс {m.loss} требует y ≥ 0")
        if m.target_transform == "log1p" and (self.mem.get("audit.target") or {}).get("min", 0) <= -1:
            errs.append("log1p требует y > -1")
        if spec.training.sample_weight == "recent" and not self.mem.get("audit.time"):
            errs.append("sample_weight=recent возможен только при колонке времени")
        if spec.key() in self.mem.spec_hashes():
            errs.append("такой эксперимент уже был (та же спецификация) — измени что-то по существу")
        est = self.estimate(spec)
        if est > self.time_for_research():
            errs.append(f"не помещается в бюджет: оценка {int(est)} с, доступно {int(self.time_for_research())} с")
        return errs

    def estimate(self, spec: ExperimentSpec) -> float:
        same = [e["seconds"] for e in self.mem.ok_experiments() if e["spec"]["model"]["model_id"] == spec.model.model_id]
        base = float(np.median(same)) if same else DEFAULT_SECONDS.get(spec.model.model_id, 120) * self.ctx.get("time_scale", 1.0)
        return base * 1.25

    def final_reserve(self) -> float:
        """Сколько оставить на ансамбль + финальное переобучение: по фактическому времени лучших экспериментов."""
        ok = sorted(self.mem.ok_experiments(), key=lambda e: e["cv"] * (1 if self.mem.direction == "minimize" else -1))[:5]
        nf = max(1, len(self.ctx["fold_weights"]))
        return 180 + 1.7 * sum(e["seconds"] / nf for e in ok)

    def time_for_research(self) -> float:
        return self.budget.left() - self.final_reserve()

    # ---- код Qwen → новый блок
    def add_custom_block(self, code: str, description: str, source: str) -> tuple[str | None, list[str]]:
        import hashlib
        h = hashlib.sha256(" ".join(code.split()).encode()).hexdigest()[:12]
        if h in self.custom_hashes:
            return self.custom_hashes[h], ["такой код уже исполнялся — переиспользую блок " + self.custom_hashes[h]]
        if self.custom_n >= self.s.max_custom_codes:
            return None, [f"лимит пользовательских блоков ({self.s.max_custom_codes}) исчерпан"]
        name = f"custom_{self.custom_n + 1}"
        df, notes = run_custom(code, self.raw_features, self.target, self.store.get("base"), self.store.dir, name)
        self.mem.log("custom_code", name=name, source=source, description=description, code=code,
                     ok=df is not None, notes=notes)
        if df is None:
            return None, notes
        df.to_parquet(self.store.path(name), index=False)
        self.custom_n += 1
        self.custom_hashes[h] = name
        self.mem.fact(f"block.{name}", {"description": description[:200], "columns": list(df.columns)[:25],
                                        "notes": notes[:5]}, source)
        return name, notes

    # ---- запуск
    def run(self, spec: ExperimentSpec, hypothesis: str, why: str, parent: str | None, source: str) -> dict:
        self.n += 1
        exp_id = f"exp_{self.n:03d}"
        best_before = self.mem.best()
        timeout = min(self.s.experiment_timeout_s, max(60, self.time_for_research()))
        ctx = {**self.ctx, "mode": "cv"}
        res = run_in_subprocess(spec, ctx, self.dir / exp_id, timeout)
        rec = {"id": exp_id, "parent": parent, "source": source, "hypothesis": hypothesis, "why": why,
               "spec": spec.model_dump(), "spec_hash": spec.key(), "summary": spec.summary(), **res}
        if res.get("status") == "ok":
            sign = 1 if self.mem.direction == "minimize" else -1
            if best_before:
                rec["delta_best_pct"] = round(sign * (best_before["cv"] - res["cv"]) / abs(best_before["cv"]) * 100, 3)
            par = next((e for e in self.mem.ok_experiments() if e["id"] == parent), None)
            if par:
                rec["delta_parent_pct"] = round(sign * (par["cv"] - res["cv"]) / abs(par["cv"]) * 100, 3)
            thr = self.s.min_rel_improvement * 100
            d = rec.get("delta_best_pct")
            dp = rec.get("delta_parent_pct")
            if best_before is None:
                rec["verdict"] = "baseline"
            elif d is not None and d >= thr:
                rec["verdict"] = "new_best"
            elif dp is not None and dp >= thr:
                rec["verdict"] = "better_than_parent"
            elif (dp if dp is not None else d) is not None and abs(dp if dp is not None else d) < thr:
                rec["verdict"] = "no_gain"
            else:
                rec["verdict"] = "worse"
        else:
            rec["verdict"] = "failed"
        self.mem.add_experiment(rec)
        return rec

    def stop_reason(self) -> str | None:
        exps = self.mem.state["experiments"]
        if len(exps) >= self.s.max_experiments:
            return "max_experiments"
        if self.time_for_research() < 60:
            return "time_budget"
        ok = self.mem.ok_experiments()
        if len(ok) >= self.s.patience + 3:
            last_best = max(i for i, e in enumerate(exps) if e.get("verdict") in ("new_best", "baseline"))
            if len(exps) - 1 - last_best >= self.s.patience:
                return "patience"
        return None


# ============================================================================ Qwen
class Proposal(BaseModel):
    hypothesis: str = Field(..., description="что проверяем и какой эффект ожидаем")
    why: str = Field(..., description="какие факты/эксперименты на это указывают (с числами)")
    parent: str | None = Field(None, description="id эксперимента, от которого одно изменение")
    spec: ExperimentSpec
    custom_feature_code: str | None = None
    custom_feature_description: str | None = None


class Analysis(BaseModel):
    experiment_id: str
    verdict: Literal["confirmed", "rejected", "inconclusive"]
    lesson: str


class ResearchStep(BaseModel):
    analysis: Analysis | None = None
    new_insights: list[str] = Field(default_factory=list, max_length=3)
    backlog_add: list[str] = Field(default_factory=list, max_length=5)
    decision: Literal["experiment", "stop"] = "experiment"
    proposal: Proposal | None = None


class QwenResearcher:
    def __init__(self, lab: Lab, llm: LLM, method_text: str):
        self.lab, self.llm, self.method = lab, llm, method_text
        self.rejections_in_row = 0

    def catalog(self) -> dict:
        pol = self.lab.mem.get("loss_policy", {})
        return {"AVAILABLE_BLOCKS": self.lab.store.available(),
                "MODELS": {m: {"losses": (["l2", "poisson"] if m in ("extra_trees", "random_forest") else ["l2"]
                                          if m in ("ridge", "elasticnet") else list(REG[m]["losses"])),
                               "param_space": {k: [v[1], v[2]] for k, v in PARAM_SPACE[m].items()},
                               "defaults": DEFAULTS.get(m, {})} for m in MODEL_IDS},
                "LOSS_POLICY_FOR_METRIC": pol,
                "SPEC_EXAMPLE": ExperimentSpec(blocks=["base", "entity_history"]).model_dump()}

    def next(self) -> tuple[ResearchStep | None, str]:
        lab, mem = self.lab, self.lab.mem
        exps = mem.state["experiments"]
        last_id = exps[-1]["id"] if exps else None
        budget = {"seconds_left_for_research": int(lab.time_for_research()),
                  "experiments_done": len(exps), "experiments_max": lab.s.max_experiments,
                  "llm_calls_left": lab.s.llm_max_calls - mem.state["llm"]["calls"],
                  "metric": mem.get("task.metric"), "direction": mem.direction,
                  "significance_threshold_pct": lab.s.min_rel_improvement * 100}
        memory = mem.render(lab.s.memory_chars, budget)

        def validate(step: ResearchStep) -> list[str]:
            errs = []
            if last_id and (step.analysis is None or step.analysis.experiment_id != last_id):
                errs.append(f"analysis.experiment_id должен быть последним экспериментом: {last_id}")
            if step.decision == "stop":
                if len(exps) < 8:
                    errs.append("рано останавливаться: проведено меньше 8 экспериментов")
                return errs
            if step.proposal is None:
                return errs + ["decision=experiment требует proposal"]
            p = step.proposal
            if p.parent and p.parent not in {e["id"] for e in exps}:
                errs.append(f"parent {p.parent} не существует")
            if p.custom_feature_code:
                errs += [f"custom_feature_code: {e}" for e in check_code(p.custom_feature_code, lab.target)]
                return errs  # спецификацию с новым блоком проверим после его сборки
            return errs + lab.validate(p.spec)

        system = self.method + "\n\nКАТАЛОГ:\n" + json.dumps(self.catalog(), ensure_ascii=False, default=str) + \
            "\n\nОтвет — ОДИН JSON по схеме ResearchStep: " + json.dumps(ResearchStep.model_json_schema(), ensure_ascii=False)
        step = self.llm.ask("researcher", system, "MEMORY:\n" + memory, ResearchStep,
                            deadline=lab.budget.deadline - lab.final_reserve(), validator=validate)
        if step is None:
            self.rejections_in_row += 1
            return None, "qwen_unavailable_or_invalid"
        self.rejections_in_row = 0
        return step, "qwen"


# ============================================================================ эвристика
class HeuristicResearcher:
    """Последовательная, но адаптивная стратегия опытного DS: жадный отбор блоков признаков на быстрой модели,
    затем лосс, веса, регуляризация по разрыву train/val, затем разнообразие для ансамбля."""

    def __init__(self, lab: Lab):
        self.lab = lab
        self.done: set[str] = set()

    def _best(self):
        return self.lab.mem.best()

    def _spec(self, base: dict | None = None, **changes) -> ExperimentSpec:
        d = json.loads(json.dumps(base or ExperimentSpec().model_dump()))
        for k, v in changes.items():
            if k == "blocks":
                d["blocks"] = v
            elif k in ("model_id", "loss", "target_transform", "params"):
                d["model"][k] = v
            else:
                d["training"][k] = v
        return ExperimentSpec.model_validate(d)

    def ideas(self):
        lab, mem = self.lab, self.lab.mem
        best = self._best()
        if best is None:
            yield "baseline", "Базовая линия: lightgbm на исходных признаках", "точка отсчёта для всех гипотез", self._spec()
            return
        b = best["spec"]
        avail = lab.store.available()
        blocks = list(b["blocks"])
        # 1. жадный отбор блоков признаков на lightgbm
        if b["model"]["model_id"] == "lightgbm":
            order = ["entity_history", "group_history", "similar_history", "families", "missing", "oof_target_encoding"]
            for blk in order:
                if blk in avail and blk not in blocks:
                    ent = mem.get("audit.entity") or {}
                    why = {"entity_history": f"доля строк test с историей объекта {ent.get('test_rows_with_history')}, корреляция с историей {ent.get('history_corr')}",
                           "group_history": "строки без истории объекта нужно чем-то покрыть: средний таргет группы в прошлом",
                           "similar_history": "строки без своей истории: прошлое похожих объектов (та же категория и размер)",
                           "families": f"семейства колонок {mem.get('audit.families')}: агрегаты устойчивее отдельных месяцев",
                           "missing": "пропуски могут быть информативны", "oof_target_encoding": "категории высокой значимости"}.get(blk, "")
                    yield f"add:{blk}", f"Добавить блок {blk}", why, self._spec(b, blocks=blocks + [blk])
            # 2. сдвиг: сырые колонки семейств как «отпечаток периода»
            adv = mem.get("audit.adversarial") or {}
            fams = find_families(lab.store.num_cols)
            if (adv.get("auc") or 0) > 0.9 and fams and not b["training"]["drop_patterns"]:
                pats = sorted({c for cols in fams.values() for c in cols})
                shift_why = (f"adversarial AUC={adv.get('auc')}, test отличается колонками "
                             f"{[x['feature'] for x in adv.get('top_shift_features', [])[:3]]} — риск выучить «отпечаток периода»")
                yield "drop:raw_families_keep_agg", "Заменить сырые колонки семейств их агрегатами", shift_why, \
                    self._spec(b, blocks=list(dict.fromkeys(blocks + ["families"])), drop_patterns=pats)
                yield "drop:raw_families_all", "Убрать сырые колонки семейств совсем", shift_why, \
                    self._spec(b, drop_patterns=pats)
            # 3. код признаков: взаимодействия топовых числовых признаков (проверяет и песочницу)
            if "custom_interactions" not in self.done and best.get("top_features"):
                nums = [f for f, _ in best["top_features"] if f in lab.store.num_cols][:4]
                if len(nums) >= 2:
                    code = ("def add_features(df):\n    out = pd.DataFrame(index=df.index)\n    cols = " + repr(nums) + "\n"
                            "    for i in range(len(cols)):\n        for j in range(i + 1, len(cols)):\n"
                            "            a, b = df[cols[i]].astype(float), df[cols[j]].astype(float)\n"
                            "            out[cols[i] + '_x_' + cols[j]] = a * b\n"
                            "            out[cols[i] + '_div_' + cols[j]] = a / b.replace(0, np.nan)\n"
                            "    return out\n")
                    yield "custom_interactions", "Попарные произведения и отношения 4 самых важных числовых признаков", \
                        f"топ-признаки {nums} — проверим, не упускает ли модель их взаимодействия", ("code", code, b)
        # 4. лосс, согласованный с метрикой
        pol = mem.get("loss_policy") or {}
        for alt in pol.get("alternatives", []):
            yield f"loss:{alt['loss']}:{alt['transform']}", f"Лосс {alt['loss']} + трансформ {alt['transform']}", \
                ("согласован с метрикой (оценивает ту же статистику)" if alt["consistent"] else
                 "НЕ согласован с метрикой — принимаем только по cv"), \
                self._spec(b, loss=alt["loss"], target_transform=alt["transform"])
        # 5. веса недавних периодов
        if mem.get("audit.time"):
            for d in (0.7, 0.5):
                yield f"recent:{d}", f"Больший вес недавним периодам (decay={d})", \
                    "test — следующий период; недавние периоды ближе к нему", self._spec(b, sample_weight="recent", recent_decay=d)
        # 6. регуляризация по разрыву train/val
        gap = (best["cv"] - best.get("train_cv", best["cv"])) / best["cv"] if best["cv"] else 0
        mid = b["model"]["model_id"]
        if mid == "lightgbm":
            p = dict(b["model"]["params"])
            if gap > 0.2:
                yield "reg:strong", "Сильнее регуляризация: меньше листьев, больше min_child_samples, меньше колонок", \
                    f"разрыв train/val {gap:.0%}: модель переобучается", \
                    self._spec(b, params={**p, "num_leaves": 31, "min_child_samples": 100, "colsample_bytree": 0.5, "reg_lambda": 10.0})
            yield "lr:low", "Меньше learning_rate (0.02) — точнее сходимость", "обычно даёт небольшой выигрыш на финале", \
                self._spec(b, params={**p, "learning_rate": 0.02})
            yield "leaves:more", "Больше листьев (127) — больше ёмкость", "проверяем недообучение", \
                self._spec(b, params={**p, "num_leaves": 127, "min_child_samples": 50})
        # 7. разнообразие для ансамбля на лучших признаках
        for mid2, loss2 in (("catboost", b["model"]["loss"]), ("xgboost", b["model"]["loss"]), ("extra_trees", "l2"), ("ridge", "l2")):
            ok_loss = loss2 if not check_model_spec(mid2, loss2, b["model"]["target_transform"]) else "l2"
            yield f"model:{mid2}", f"{mid2} на лучших признаках", "другое семейство → некоррелированные ошибки для ансамбля", \
                self._spec(b, model_id=mid2, loss=ok_loss, params={})

    def next(self):
        for key, hyp, why, spec in self.ideas():
            if key in self.done:
                continue
            self.done.add(key)
            best = self._best()
            parent = best["id"] if best else None
            if isinstance(spec, tuple) and spec[0] == "code":
                _, code, base = spec
                name, notes = self.lab.add_custom_block(code, hyp, "heuristic")
                if not name:
                    self.lab.mem.reject_proposal(hyp, notes, "heuristic")
                    continue
                spec = self._spec(base, blocks=list(base["blocks"]) + [name])
            errs = self.lab.validate(spec)
            if errs:
                self.lab.mem.reject_proposal(hyp, errs, "heuristic")
                continue
            return {"hypothesis": hyp, "why": why, "parent": parent, "spec": spec}
        return None


# ============================================================================ цикл
def research_loop(lab: Lab, llm: LLM, method_text: str, log=print, on_baseline=None) -> str:
    qwen = QwenResearcher(lab, llm, method_text)
    heur = HeuristicResearcher(lab)
    mem = lab.mem
    # базовая линия — всегда код и всегда запускается (точка отсчёта + страховочный ответ), без проверки бюджета
    heur.done.add("baseline")
    rec = lab.run(ExperimentSpec(), "Базовая линия: lightgbm на исходных признаках", "точка отсчёта для всех гипотез",
                  None, "baseline")
    log(f"   {rec['id']} baseline: {rec.get('cv')} ({rec.get('status')}, {rec.get('seconds')} с) {rec.get('error') or ''}")
    if on_baseline:
        on_baseline(rec)
    stop = None
    while True:
        stop = lab.stop_reason()
        if stop:
            break
        proposal, source = None, "heuristic"
        if llm.enabled and qwen.rejections_in_row < 3:
            step, src = qwen.next()
            if step is not None:
                last = mem.state["experiments"][-1]["id"]
                if step.analysis:
                    mem.annotate_experiment(step.analysis.experiment_id, analysis_verdict=step.analysis.verdict,
                                            analysis=step.analysis.lesson)
                for t in step.new_insights:
                    mem.add_insight(t, "qwen", {"after": last})
                mem.add_backlog(step.backlog_add, "qwen")
                if step.decision == "stop":
                    stop = "qwen_stop"
                    break
                p = step.proposal
                spec = p.spec
                if p.custom_feature_code:
                    name, notes = lab.add_custom_block(p.custom_feature_code, p.custom_feature_description or p.hypothesis, "qwen")
                    if not name:
                        mem.reject_proposal(p.hypothesis, notes, "qwen")
                        qwen.rejections_in_row += 1
                        continue
                    spec = ExperimentSpec.model_validate({**spec.model_dump(), "blocks": list(spec.blocks) + [name]})
                    errs = lab.validate(spec)
                    if errs:
                        mem.reject_proposal(p.hypothesis, errs, "qwen")
                        qwen.rejections_in_row += 1
                        continue
                proposal = {"hypothesis": p.hypothesis, "why": p.why, "parent": p.parent, "spec": spec}
                source = "qwen"
        if proposal is None:
            proposal = heur.next()
            source = "heuristic"
            if proposal is None:
                stop = "ideas_exhausted"
                break
        rec = lab.run(proposal["spec"], proposal["hypothesis"], proposal["why"], proposal["parent"], source)
        best = mem.best()
        log(f"   {rec['id']} [{source}] {rec['summary'][:70]:<70} cv={rec.get('cv') and round(rec['cv'], 4)} "
            f"{rec.get('verdict')} | best={best and round(best['cv'], 4)} | осталось {int(lab.time_for_research())} с")
    mem.log("research_stop", reason=stop)
    return stop

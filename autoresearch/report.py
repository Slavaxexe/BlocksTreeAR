"""Отчёт запуска: report.json, leaderboard.csv, SUMMARY.md и самодостаточный dashboard.html (открывается без сети)."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def write_report(mem, run_dir: Path, task, audit: dict, design: dict, ens: dict, checks: dict | None, budget) -> dict:
    exps = mem.state["experiments"]
    best = mem.best()
    lb = pd.DataFrame([{"id": e["id"], "source": e.get("source"), "summary": e.get("summary"), "cv": e.get("cv"),
                        "cv_std": e.get("cv_std"), "train_cv": e.get("train_cv"), "folds": e.get("folds"),
                        "delta_best_pct": e.get("delta_best_pct"), "verdict": e.get("verdict"),
                        "seconds": e.get("seconds"), "hypothesis": e.get("hypothesis")} for e in exps])
    if len(lb):
        lb.sort_values("cv", ascending=mem.direction == "minimize").to_csv(run_dir / "leaderboard.csv", index=False)
    base = next((e for e in exps if e.get("verdict") == "baseline"), None)
    final_cv = ens.get("ensemble_score") if ens.get("used") else (best and best["cv"])
    report = {
        "task": {"target": task.target, "metric": task.metric, "metric_source": task.metric_source, "id": task.id_cols},
        "stages": mem.state["stages"],
        "facts": {k: v["value"] for k, v in mem.state["facts"].items()},
        "validation": {"kind": design["kind"], "folds": design["labels"], "sizes": design["sizes"]},
        "baseline": base and {"id": base["id"], "cv": base["cv"], "summary": base["summary"]},
        "best_single": best and {"id": best["id"], "cv": best["cv"], "summary": best["summary"], "segments": best.get("segments"),
                                 "top_features": best.get("top_features"), "folds": best.get("folds"), "train_cv": best.get("train_cv")},
        "ensemble": ens,
        "final_cv": final_cv,
        "improvement_vs_baseline_pct": round((base["cv"] - final_cv) / base["cv"] * 100, 2) if base and final_cv and mem.direction == "minimize" else None,
        "submission_checks": checks,
        "experiments": [{k: e.get(k) for k in ("id", "parent", "source", "hypothesis", "why", "summary", "status", "cv", "cv_std",
                                               "folds", "train_cv", "delta_best_pct", "delta_parent_pct", "verdict", "seconds",
                                               "analysis", "analysis_verdict", "error", "segments")} for e in exps],
        "insights": mem.state["insights"], "backlog": mem.state["backlog"],
        "rejected_proposals": mem.state["rejected_proposals"], "llm": mem.state["llm"],
        "elapsed_s": round(budget.elapsed(), 1),
    }
    (run_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    mem.save()
    (run_dir / "SUMMARY.md").write_text(_summary_md(report), encoding="utf-8")
    (run_dir / "dashboard.html").write_text(DASHBOARD.replace("__DATA__", json.dumps(report, ensure_ascii=False, default=str)
                                                              .replace("</", "<\\/")), encoding="utf-8")
    return report


def _summary_md(r: dict) -> str:
    L = [f"# AutoResearch: {r['task']['target']} ({r['task']['metric']})", ""]
    if r["baseline"]:
        L.append(f"- Базовая линия: **{r['baseline']['cv']:.4f}** ({r['baseline']['summary']})")
    if r["best_single"]:
        L.append(f"- Лучшая модель: **{r['best_single']['cv']:.4f}** ({r['best_single']['summary']})")
    L.append(f"- Финал (CV): **{r['final_cv']}**; улучшение к базовой линии: {r['improvement_vs_baseline_pct']}%")
    L.append(f"- Валидация: {r['validation']['kind']} {r['validation']['folds']}")
    L.append(f"- Экспериментов: {len(r['experiments'])}; вызовов Qwen: {r['llm']['calls']}; время {r['elapsed_s']} с")
    L += ["", "| id | источник | эксперимент | cv | Δ к лучшему % | вердикт |", "|---|---|---|---|---|---|"]
    for e in r["experiments"]:
        L.append(f"| {e['id']} | {e['source']} | {e['summary']} | {e['cv'] and round(e['cv'], 4)} | {e['delta_best_pct']} | {e['verdict']} |")
    L += ["", "## Выводы", *[f"- {i['text']} _({i['source']})_" for i in r["insights"]]]
    return "\n".join(L) + "\n"


DASHBOARD = r"""<!doctype html><html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>AutoResearch run</title><style>
:root{--bg:#f7f7f5;--card:#fff;--ink:#1d1d1b;--mute:#6b6b66;--line:#e4e3de;--acc:#2f5bea;--good:#1f8a4c;--bad:#c2410c;--warn:#b7791f}
@media(prefers-color-scheme:dark){:root{--bg:#141413;--card:#1e1e1c;--ink:#ecebe6;--mute:#9b9a93;--line:#33332f;--acc:#7c9cff;--good:#4ade80;--bad:#fb923c;--warn:#fbbf24}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,-apple-system,Segoe UI,sans-serif}
main{max-width:1180px;margin:0 auto;padding:24px 16px 60px}h1{font-size:22px;margin:0 0 4px}h2{font-size:15px;margin:28px 0 10px;text-transform:uppercase;letter-spacing:.06em;color:var(--mute)}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:10px;margin-top:16px}.k{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px}
.k b{display:block;font-size:22px;font-variant-numeric:tabular-nums}.k span{color:var(--mute);font-size:12px}
.stages{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:8px}.st{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:10px;font-size:12px}
.st h4{margin:0 0 4px;font-size:13px}.st .s{color:var(--mute)}
table{width:100%;border-collapse:collapse;background:var(--card);border:1px solid var(--line);border-radius:10px;overflow:hidden;font-size:12.5px}
th,td{padding:7px 8px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}th{color:var(--mute);font-weight:600}td.n{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
.v-new_best{color:var(--good);font-weight:600}.v-worse,.v-failed,.v-timeout{color:var(--bad)}.v-no_gain{color:var(--mute)}.v-better_than_parent{color:var(--warn)}
.chip{display:inline-block;padding:1px 7px;border-radius:99px;border:1px solid var(--line);font-size:11px;color:var(--mute)}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px}.grid2{display:grid;grid-template-columns:1fr 1fr;gap:12px}
@media(max-width:760px){.grid2{grid-template-columns:1fr}} svg text{fill:var(--mute);font-size:11px}.small{color:var(--mute);font-size:12px}
</style></head><body><main id="app"></main><script>
const R=__DATA__;const $=s=>document.querySelector(s);const f=(x,d=3)=>x==null?'—':(+x).toFixed(d);
const esc=s=>String(s??'').replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
const ex=R.experiments.filter(e=>e.cv!=null);const mn=(R.facts.direction||'minimize')==='minimize';
let h=`<h1>AutoResearch · ${esc(R.task.target)} · ${esc(R.task.metric).toUpperCase()}</h1><div class=small>валидация: ${esc(R.validation.kind)} — ${esc(R.validation.folds.join('; '))}</div>
<div class=kpis><div class=k><span>Финал (CV)</span><b>${f(R.final_cv)}</b></div><div class=k><span>Базовая линия</span><b>${f(R.baseline&&R.baseline.cv)}</b></div>
<div class=k><span>Улучшение к базовой</span><b>${R.improvement_vs_baseline_pct==null?'—':R.improvement_vs_baseline_pct+'%'}</b></div>
<div class=k><span>Экспериментов</span><b>${R.experiments.length}</b></div><div class=k><span>Вызовов Qwen</span><b>${R.llm.calls}</b></div><div class=k><span>Время</span><b>${Math.round(R.elapsed_s/60)} мин</b></div></div>`;
h+=`<h2>Этапы</h2><div class=stages>`+Object.entries(R.stages).map(([k,v])=>`<div class=st><h4>${esc(k)}</h4><div class=s>${esc(v.status)} · ${v.seconds??'—'} с</div><div>${esc(JSON.stringify(v.summary).slice(0,160))}</div></div>`).join('')+`</div>`;
// график
if(ex.length){const W=1100,H=260,P=40;const ys=ex.map(e=>e.cv);const lo=Math.min(...ys),hi=Math.max(...ys);const sx=i=>P+i*(W-2*P)/Math.max(1,ex.length-1);const sy=v=>H-P+(-(v-lo)/(hi-lo||1))*(H-2*P);
let best=[],b=null;ex.forEach(e=>{b=b==null?e.cv:(mn?Math.min(b,e.cv):Math.max(b,e.cv));best.push(b)});
const col=v=>v==='new_best'||v==='baseline'?'var(--good)':v==='worse'?'var(--bad)':'var(--mute)';
h+=`<h2>Ход исследования</h2><div class=card><svg viewBox="0 0 ${W} ${H}" width="100%"><polyline fill="none" stroke="var(--acc)" stroke-width="2" points="${best.map((v,i)=>sx(i)+','+sy(v)).join(' ')}"/>`+
ex.map((e,i)=>`<circle cx="${sx(i)}" cy="${sy(e.cv)}" r="4.5" fill="${col(e.verdict)}"><title>${esc(e.id)} ${esc(e.summary)} cv=${f(e.cv,4)}</title></circle>`).join('')+
`<text x="4" y="${sy(hi)+4}">${f(hi,2)}</text><text x="4" y="${sy(lo)+4}">${f(lo,2)}</text></svg><div class=small>линия — лучший результат на момент; точки — эксперименты (зелёный — новый лучший, красный — хуже)</div></div>`}
// где ошибка
const bs=R.best_single||{};const sg=bs.segments||{};
h+=`<h2>Где ошибка лучшей модели</h2><div class=grid2><div class=card><table><tr><th>срез</th><th>значение</th></tr>`+
Object.entries(sg).filter(([k])=>k!=='worst_segments').map(([k,v])=>`<tr><td>${esc(k)}</td><td class=n>${esc(v)}</td></tr>`).join('')+`</table></div>
<div class=card><table><tr><th>сегмент</th><th>строк</th><th>RMSE</th><th>доля ошибки</th></tr>`+(sg.worst_segments||[]).map(s=>`<tr><td>${esc(s.segment)}</td><td class=n>${s.rows}</td><td class=n>${f(s.rmse,2)}</td><td class=n>${f(s.share_of_sq_error*100,1)}%</td></tr>`).join('')+`</table></div></div>`;
// эксперименты
h+=`<h2>Все эксперименты</h2><table><tr><th>id</th><th>кто</th><th>гипотеза → результат</th><th>cv</th><th>Δ к лучшему</th><th>вердикт</th></tr>`+
R.experiments.map(e=>`<tr><td>${esc(e.id)}<div class=small>← ${esc(e.parent||'')}</div></td><td><span class=chip>${esc(e.source)}</span></td><td><b>${esc(e.hypothesis)}</b><div class=small>${esc(e.summary)}</div>${e.why?`<div class=small>почему: ${esc(e.why)}</div>`:''}${e.analysis?`<div class=small>урок Qwen: ${esc(e.analysis)}</div>`:''}${e.error?`<div class="small v-failed">${esc(e.error).slice(0,300)}</div>`:''}</td><td class=n>${f(e.cv,4)}<div class=small>±${f(e.cv_std,2)} · train ${f(e.train_cv,2)}</div></td><td class=n>${e.delta_best_pct==null?'—':e.delta_best_pct+'%'}</td><td class="v-${esc(e.verdict)}">${esc(e.verdict)}</td></tr>`).join('')+`</table>`;
// ансамбль, выводы, факты
const en=R.ensemble||{};h+=`<h2>Ансамбль и ответ</h2><div class=grid2><div class=card><b>${en.used?'Ансамбль':'Одна модель'}</b> · OOF ${f(en.ensemble_score,4)} vs лучшая ${f(en.best_single_score,4)} (${en.rel_gain_pct}%)<div class=small>${esc(en.reason)}</div><div>${esc(JSON.stringify(en.weights))}</div></div>
<div class=card><b>submission.csv</b><div class=small>${esc(JSON.stringify(R.submission_checks))}</div></div></div>`;
h+=`<h2>Выводы в памяти</h2><div class=card>`+(R.insights.map(i=>`<div>• ${esc(i.text)} <span class=chip>${esc(i.source)}</span></div>`).join('')||'—')+`</div>`;
h+=`<h2>Ключевые факты</h2><div class=card><pre style="white-space:pre-wrap;font-size:12px;margin:0">${esc(JSON.stringify({time:R.facts['audit.time'],entity:R.facts['audit.entity'],target:R.facts['audit.target'],adversarial:R.facts['audit.adversarial'],loss_policy:R.facts['loss_policy']},null,1))}</pre></div>`;
$('#app').innerHTML=h;</script></body></html>"""

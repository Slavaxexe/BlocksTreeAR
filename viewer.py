"""Read-only local viewer: python viewer.py [outputs|run-dir] [--export snapshot.html]."""

import argparse
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import time
from urllib.parse import urlsplit


def collect(root: Path) -> dict:
    """Only known experiment JSON files are read; no CSV, .env or API access."""
    root = root.resolve()
    folders = [root] if (root / "task.json").is_file() else [p.parent for p in root.glob("**/task.json")]
    runs = []
    for folder in folders:
        warnings, modified = [], []
        def read(path):
            if not path.is_file():
                return None
            try:
                if not path.resolve().is_relative_to(root):
                    raise ValueError("Path outside the selected root")
                value = json.loads(path.read_text(encoding="utf-8"), parse_constant=lambda _: None)
                if not isinstance(value, dict):
                    raise ValueError("Expected a JSON object")
                modified.append(path.stat().st_mtime)
                return value
            except (OSError, ValueError):
                warnings.append(f"{path.relative_to(folder)}: JSON недоступен или некорректен")
                return None
        task = read(folder / "task.json")
        if task is None:
            continue
        run = {key: read(folder / f"{key}.json") for key in ("profile", "validation", "best", "delta", "report")}
        nodes = []
        for directory in sorted((folder / "nodes").glob("node_*")):
            start = read(directory / "start.json")
            if start:
                nodes.append({"id": directory.name, "start": start, "finish": read(directory / "finish.json"),
                              "delta": read(directory / "delta.json")})
        run.update(id=folder.relative_to(root).as_posix() if folder != root else folder.name,
                   task=task, nodes=nodes, warnings=warnings, modified=max(modified, default=0))
        runs.append(run)
    return {"runs": sorted(runs, key=lambda r: r["modified"], reverse=True), "captured": time.time()}


HTML = r'''<!doctype html>
<html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Alfa · Схема и эксперименты</title><link rel="icon" href="data:,">
<style>
:root{color-scheme:light dark;--bg:light-dark(#f6f7fa,#11141a);--paper:light-dark(#fff,#1a1f28);--ink:light-dark(#202638,#e2e7ef);--muted:light-dark(#647084,#a3aec0);--line:light-dark(#dce1eb,#353d4d);--purple:light-dark(#6947c4,#bea7ff);--purple-bg:light-dark(#f2edff,#2c2444);--teal:light-dark(#087c72,#75d7c1);--teal-bg:light-dark(#eaf7f2,#17362f);--red:light-dark(#b34343,#f4a2a2);--red-bg:light-dark(#fff0ef,#3d272a);--amber:light-dark(#95650b,#e5bf74)}

*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
button,select{font:inherit;color:inherit}
button,select{background:var(--paper);border:1px solid var(--line);border-radius:7px;padding:8px 13px}
button{cursor:pointer}
button:focus-visible,select:focus-visible,[role=button]:focus-visible{outline:3px solid var(--purple);outline-offset:3px}
button[aria-selected=true]{background:var(--ink);color:var(--paper);border-color:var(--ink)}
header{max-width:1400px;margin:auto;padding:28px 32px 20px;display:flex;align-items:center;justify-content:space-between;gap:20px}
h1{font-size:23px;line-height:1.25;letter-spacing:-.5px;margin:0 0 5px;font-weight:650}
h2{font-size:17px;margin:0 0 12px}
h3{font-size:14px;margin:22px 0 8px}
p{margin:0 0 12px}
small,.muted{color:var(--muted)}
small{font-size:12px}
.mono,pre,.metric{font-family:ui-monospace,SFMono-Regular,Consolas,monospace}
.metric{font-variant-numeric:tabular-nums}
nav{display:flex;gap:8px}
.topline{max-width:1400px;margin:auto;padding:0 32px 16px;display:flex;align-items:center;gap:16px;flex-wrap:wrap}
.topline label{display:flex;gap:10px;align-items:center}
select{max-width:min(650px,70vw)}
main{max-width:1400px;margin:auto;padding:0 32px 28px}
.panel{background:var(--paper);border:1px solid var(--line);border-radius:12px;padding:24px}
.legend{display:flex;gap:22px;flex-wrap:wrap;font-size:12px;color:var(--muted);margin-bottom:12px}
.legend span{display:flex;align-items:center;gap:8px}
.dot{width:9px;height:9px;border-radius:50%;background:var(--muted);display:inline-block}
.dot.llm{background:var(--purple)}
.dot.cpu{background:var(--teal)}
.dot.error{background:var(--red)}
.diagram-scroll{overflow:auto}
.architecture{width:100%;min-width:920px;display:block}
.architecture text,.tree text{fill:var(--ink);font:13px -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
.architecture .caption,.tree .caption{fill:var(--muted);font-size:12px}
.architecture .title,.tree .title{font-weight:650}
.box{fill:var(--paper);stroke:var(--line);stroke-width:1.3}
.box.llm{fill:var(--purple-bg)}
.box.cpu{fill:var(--teal-bg)}
.edge{stroke:var(--muted);stroke-width:1.3;fill:none}
.feedback{stroke:var(--purple);stroke-dasharray:5 4}
.frame{fill:var(--purple-bg);opacity:.5;stroke:none}
.tree-shell{display:grid;grid-template-columns:minmax(0,1fr) 350px;gap:20px}
.tree{display:block}
.tree-container{overflow:auto;min-height:300px}
.tree-node{cursor:pointer}
.tree-node.selected .box{stroke:var(--ink);stroke-width:2}
.tree-node.best .box{stroke:var(--purple);stroke-width:2.3}
.tree-node.completed .box{fill:var(--teal-bg)}
.tree-node.failed .box,.tree-node.timeout .box{fill:var(--red-bg)}
.tree .score{font-family:ui-monospace,SFMono-Regular,Consolas,monospace;font-size:17px;font-variant-numeric:tabular-nums}
.tree .best-label{fill:var(--purple);font-size:11px;font-weight:650}
.summary{display:flex;gap:22px;flex-wrap:wrap;padding:0 0 16px;color:var(--muted);font-size:12px}
.summary strong{display:block;color:var(--ink);font-size:16px;font-weight:600;margin-top:3px}
.detail{border-left:1px solid var(--line);padding-left:20px;overflow-wrap:anywhere}
.detail pre{background:var(--bg);border-radius:7px;padding:12px;white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px;max-height:460px;overflow:auto}
.json-tabs{display:flex;gap:6px;margin-top:20px}
.json-tabs button{font-size:12px;padding:5px 10px}
.comparison{padding:8px 0;border-bottom:1px solid var(--line);font-size:12px}
.comparison b{font-weight:600}
.changes{margin:6px 0 0;padding-left:16px}
.error-text{color:var(--red)}
[hidden]{display:none!important}
.footnote{margin:14px 0 0;font-size:12px;color:var(--muted)}
#connection{font-size:12px;color:var(--muted)}

@media(max-width:1000px){.tree-shell{grid-template-columns:1fr}
.detail{border-left:0;border-top:1px solid var(--line);padding:20px 0 0}
header{align-items:flex-start}
header,main{padding-left:16px;padding-right:16px}
.topline{padding-left:16px;padding-right:16px}
.panel{padding:16px}
nav{flex-wrap:wrap}
}

</style>
<header><div><h1>Как работает ML-агент</h1><small>Текущая реализация · JSON → эксперимент → измерение → следующая гипотеза</small></div><nav aria-label="Разделы"><button id="architecture-tab" aria-selected="true">Схема</button><button id="tree-tab" aria-selected="false">Эксперименты</button></nav></header>
<div class="topline"><label id="run-picker" hidden>Запуск <select id="runs" aria-label="Запуск эксперимента"></select></label><span id="connection" role="status"></span></div>
<main>
<section id="architecture-panel" class="panel"><div class="legend"><span><i class="dot"></i>Детерминированный Python</span><span><i class="dot llm"></i>Вызов Qwen API</span><span><i class="dot cpu"></i>Локальное обучение на CPU</span></div><div class="diagram-scroll"><svg id="architecture" class="architecture" viewBox="0 0 1120 820" role="img" aria-label="Схема фактической работы приложения: baseline, проверка Qwen, гипотезы, два worker, сравнение и сохранение лучшего результата"></svg></div><p class="footnote">Baseline сохраняется до первого обращения к Qwen. Summarizer — обычный код. Worker исполняет только заранее написанные sklearn pipeline; текст модели не запускается как Python.</p></section>
<section id="tree-panel" class="panel" hidden><div id="empty" hidden></div><div id="experiment"><div id="summary" class="summary"></div><div class="legend"><span><i class="dot cpu"></i>Завершён</span><span><i class="dot error"></i>Ошибка / таймаут</span><span><i class="dot llm"></i>Лучший результат</span><span>Связь = parent_id · нажмите ноду для деталей</span></div><div class="tree-shell"><div class="tree-container"><svg id="tree" class="tree" role="img" aria-label="Дерево происхождения ML-гипотез"></svg></div><aside id="detail" class="detail" aria-label="Детали эксперимента"></aside></div></div><p class="footnote">Показаны записанные JSON, а не скрытое состояние процесса. Нода без finish могла ещё выполняться или быть прервана. Сам запрос baker виден только после появления start.json; без report.json завершение запуска не подтверждено.</p></section>
</main><script id="initial-data" type="application/json">__DATA__</script>
<script>
const $=id=>document.getElementById(id), NS='http://www.w3.org/2000/svg';
const element=(tag,cls,text)=>{const e=document.createElement(tag);if(cls)e.className=cls;if(text!==undefined)e.textContent=text;return e};
const svg=(tag,attrs={},text)=>{const e=document.createElementNS(NS,tag);for(const [k,v]of Object.entries(attrs))e.setAttribute(k,v);if(text!==undefined)e.textContent=text;return e};
const fmt=n=>typeof n==='number'&&Number.isFinite(n)?n.toFixed(6):'—';
const families={dummy:'Dummy baseline',extra_trees:'ExtraTrees',hist_gradient_boosting:'HistGradientBoosting'};
const statuses={completed:'Завершён',failed:'Ошибка',timeout:'Таймаут'};
const stops={max_experiments:'Лимит экспериментов',time_budget:'Лимит времени',proposal_limit:'Лимит предложений',llm_unavailable_or_contract_conflict:'API / конфликт контракта',llm_unavailable_or_invalid_hypothesis:'API / невалидная гипотеза'};
let state=JSON.parse($('initial-data').textContent), active=null, selected=null, jsonTab='start', signature='', live=__LIVE__;
function marker(canvas){const d=svg('defs'),m=svg('marker',{id:canvas.id+'-arrow',viewBox:'0 0 10 10',refX:9,refY:5,markerWidth:7,markerHeight:7,orient:'auto-start-reverse'});m.append(svg('path',{d:'M 0 0 L 10 5 L 0 10 z',fill:'var(--muted)'}));d.append(m);canvas.append(d)}
function edge(canvas,d,feedback=false){canvas.append(svg('path',{d,class:'edge'+(feedback?' feedback':''),'marker-end':`url(#${canvas.id}-arrow)`}))}
function box(canvas,x,y,w,h,title,lines=[],kind=''){
 const g=svg('g');g.append(svg('rect',{x,y,width:w,height:h,rx:9,class:'box '+kind}));g.append(svg('text',{x:x+16,y:y+27,class:'title'},title));
 lines.forEach((line,i)=>g.append(svg('text',{x:x+16,y:y+49+i*18,class:'caption'},line)));canvas.append(g);return g;
}
function architecture(){
 const c=$('architecture');marker(c);c.append(svg('rect',{x:40,y:184,width:640,height:160,rx:12,class:'frame'}));
 const paths=['M 340 75 H 410','M 710 75 H 780','M 560 120 V 157 H 360 V 184','M 560 120 V 157 H 940 V 226','M 160 304 V 365 H 360 V 402','M 360 304 V 402','M 560 304 V 365 H 360 V 402','M 360 472 V 500 H 180 V 535','M 320 577 H 420','M 700 577 H 800','M 940 306 V 420 H 560 V 535','M 940 620 V 692','M 800 738 H 700','M 420 738 H 320'];
 paths.forEach(p=>edge(c,p));edge(c,'M 560 692 V 657 H 180 V 620',true);
 box(c,40,30,300,90,'01  Условие + проверка CSV',['Заголовки, target, ID, основная метрика','Фиксируем split 80/20 · seed 42']);
 box(c,410,30,300,90,'02  node_000 · Dummy baseline',['start → обучение → finish','Среднее / самый частый класс'],'cpu');
 box(c,780,30,300,90,'Сразу готовый submission.csv',['Проверка строк, ID и предсказаний','Первый best.json до LLM']);
 c.append(svg('text',{x:60,y:211,class:'caption'},'03  Режим Qwen · три независимых асинхронных запроса'));
 box(c,60,226,200,78,'Problem',['Тип задачи + target'],'llm');box(c,280,226,160,78,'DataFiles',['Пути CSV + ID'],'llm');box(c,460,226,200,78,'Metric',['Метрика + направление'],'llm');
 box(c,800,226,280,80,'Режим offline',['Фиксированная очередь моделей','Без запросов к API']);
 box(c,220,402,280,70,'Pydantic + сверка с CSV',['Конфликт → остановить поиск']);
 box(c,40,535,280,85,'04  Baker · Qwen',['Профиль + лучшие результаты + Δ','Одна новая гипотеза по схеме'],'llm');
 box(c,420,535,280,85,'05  start.json',['Родитель + pipeline + hashes','Валидация; отсев дублей']);
 box(c,800,535,280,85,'06  Worker A / Worker B',['До двух отдельных CPU-процессов','Fit → validation → refit → test'],'cpu');
 box(c,800,692,280,92,'07  finish.json + submission',['Реальная метрика / длительность','Контроллер проверяет предсказания','Ошибка / таймаут сохраняет best']);
 box(c,420,692,280,92,'08  Summarizer · Python',['Δ к baseline / родителю / лидеру','Leaderboard + ошибки → delta.json']);
 box(c,40,692,280,92,'09  Лучший результат',['Только улучшение меняет best.json','Атомарная замена submission.csv']);
 c.append(svg('text',{x:269,y:648,class:'caption'},'feedback для следующей гипотезы Qwen'));
}
function showTab(name) {
  const tree = name === 'tree';
  $('architecture-panel').hidden = tree; $('tree-panel').hidden = !tree; $('run-picker').hidden = !tree;
  $('architecture-tab').setAttribute('aria-selected', !tree); $('tree-tab').setAttribute('aria-selected', tree);
  location.hash = tree ? 'tree' : 'architecture';
  if (tree) render();
}
function modeOf(run) {
  if (run.report?.mode) return run.report.mode;
  return run.nodes.some(n => n.start.source === 'qwen') ? 'qwen'
    : run.nodes.some(n => n.start.source === 'offline') ? 'offline' : 'режим не записан';
}
function updatePicker() {
  const select = $('runs'); select.replaceChildren();
  for (const r of state.runs) {
    const label = `${r.id.includes('synthetic') ? 'Синтетика' : 'Данные'} · ${modeOf(r)} · ${r.id}`;
    const option = element('option', '', label); option.value = r.id; select.append(option);
  }
  if (!state.runs.some(r => r.id === active)) active = (state.runs.find(r => !r.id.includes('synthetic')) ?? state.runs[0])?.id;
  select.value = active ?? '';
}
function render() {
  const r = state.runs.find(r => r.id === active);
  $('empty').hidden = !!r; $('experiment').hidden = !r;
  if (!r) { $('empty').textContent = 'Запусков пока нет. Запустите solution.py; новые JSON появятся здесь автоматически.'; return; }
  if (!r.nodes.some(n => n.id === selected)) selected = r.nodes.find(n => n.id === r.best?.node_id)?.id ?? r.nodes[0]?.id;
  const mode = modeOf(r), metric = {rmse: 'RMSE ↓', macro_f1: 'Macro F1 ↑'}[r.task.metric?.name] ?? 'Не записана';
  const info = [
    ['Источник гипотез', mode === 'offline' ? 'Фиксированная очередь' : mode === 'qwen' ? 'Qwen API' : 'Режим не записан'],
    ['Основная метрика', metric], ['Лучший score', fmt(r.best?.value)],
    ['Запуск', r.report ? (stops[r.report.stop_reason] ?? r.report.stop_reason) : 'report.json ещё не записан']
  ];
  $('summary').replaceChildren();
  for (const [label, value] of info) {
    const item = element('div', '', label); item.append(element('strong', '', value)); $('summary').append(item);
  }
  drawTree(r); drawDetail(r);
}
function drawTree(r) {
  const c = $('tree'); c.replaceChildren(); marker(c);
  const nodes = r.nodes, map = new Map(nodes.map(n => [n.id, n])), children = new Map(nodes.map(n => [n.id, []]));
  const roots = [], positions = new Map(), visiting = new Set(); let leaf = 0;
  for (const n of nodes) {
    if (map.has(n.start.parent_id) && n.start.parent_id !== n.id) children.get(n.start.parent_id).push(n);
    else roots.push(n);
  }
  // Parents sit between their children; interrupted or malformed forests remain inspectable.
  function place(n, depth) {
    if (visiting.has(n.id) || positions.has(n.id)) return;
    visiting.add(n.id);
    const kids = children.get(n.id).filter(k => !visiting.has(k.id)); kids.forEach(k => place(k, depth + 1));
    const ys = kids.map(k => positions.get(k.id)?.y).filter(y => y !== undefined);
    positions.set(n.id, {x: 24 + depth * 290, y: ys.length ? ys.reduce((a, b) => a + b, 0) / ys.length : 30 + leaf++ * 150});
    visiting.delete(n.id);
  }
  roots.forEach(n => place(n, 0)); nodes.filter(n => !positions.has(n.id)).forEach(n => place(n, 0));
  const width = Math.max(650, ...Array.from(positions.values(), p => p.x + 270));
  const height = Math.max(270, ...Array.from(positions.values(), p => p.y + 145));
  c.setAttribute('viewBox', `0 0 ${width} ${height}`); c.style.width = width + 'px'; c.style.height = height + 'px';
  for (const n of nodes) {
    const p = positions.get(n.start.parent_id), q = positions.get(n.id);
    if (p && n.start.parent_id !== n.id) edge(c, `M ${p.x+240} ${p.y+55} C ${p.x+265} ${p.y+55} ${q.x-25} ${q.y+55} ${q.x} ${q.y+55}`);
  }
  for (const n of nodes) {
    const p = positions.get(n.id), f = n.finish, best = r.best?.node_id === n.id;
    const status = statuses[f?.status] ?? 'Нет finish';
    const duration = typeof f?.duration_seconds === 'number' ? ' · ' + f.duration_seconds.toFixed(2) + ' с' : '';
    const g = svg('g', {
      class: `tree-node ${f?.status??'pending'} ${n.id===selected?'selected':''} ${best?'best':''}`,
      role: 'button', tabindex: 0, 'aria-label': `${n.id}, ${status}, ${fmt(f?.value)}`, 'data-node': n.id
    });
    g.append(svg('rect', {x:p.x, y:p.y, width:240, height:110, rx:9, class:'box'}));
    g.append(svg('text', {x:p.x+15, y:p.y+24, class:'caption'}, n.id + ' · ' + n.start.source));
    if (best) g.append(svg('text', {x:p.x+189, y:p.y+24, class:'best-label'}, 'ЛИДЕР'));
    g.append(svg('text', {x:p.x+15, y:p.y+46, class:'title'}, families[n.start.hypothesis?.pipeline?.model?.family] ?? 'Неизвестная модель'));
    g.append(svg('text', {x:p.x+15, y:p.y+73, class:'score'}, fmt(f?.value)));
    g.append(svg('text', {x:p.x+15, y:p.y+95, class:'caption'}, status + duration));
    const select = () => { selected = n.id; render(); };
    g.addEventListener('click', select);
    g.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); select(); } });
    c.append(g);
  }
}
function drawDetail(r) {
  const n = r.nodes.find(n => n.id === selected), d = $('detail'); d.replaceChildren();
  if (!n) { d.append(element('p', 'muted', 'start.json ещё не записан.')); return; }
  d.append(element('h2', '', n.id), element('small', '', `Родитель: ${n.start.parent_id ?? 'нет · baseline'}`));
  d.append(element('h3', '', 'Гипотеза'), element('p', '', n.start.hypothesis?.rationale ?? '—'));
  d.append(element('small', '', n.start.hypothesis?.expected_effect ?? ''));
  if (n.finish?.error) d.append(element('p', 'error-text', n.finish.error));
  const comparisons = n.delta?.comparisons ?? [];
  if (comparisons.length) d.append(element('h3', '', 'Сравнения'));
  for (const row of comparisons) {
    const item = element('div', 'comparison'), details = element('details'), list = element('ul', 'changes');
    const direction = row.improvement > 0 ? 'улучшение' : row.improvement < 0 ? 'ухудшение' : 'без изменений';
    item.append(element('b', '', `К ${row.reference_id}: ${direction} ${fmt(Math.abs(row.improvement))}`));
    details.append(element('summary', '', `Изменений параметров: ${row.changes.length}`));
    for (const change of row.changes) list.append(element('li', 'mono', `${change.path}: ${change.before} → ${change.after}`));
    details.append(list); item.append(details); d.append(item);
  }
  const tabs = element('div', 'json-tabs');
  for (const key of ['start', 'finish', 'delta']) {
    const button = element('button', '', key + '.json'); button.setAttribute('aria-selected', key === jsonTab);
    button.onclick = () => { jsonTab = key; drawDetail(r); }; tabs.append(button);
  }
  d.append(tabs);
  const value = n[jsonTab]; d.append(element('pre', '', value ? JSON.stringify(value, null, 2) : jsonTab + '.json ещё не записан.'));
  for (const warning of [...r.warnings, ...(r.report?.warnings ?? [])]) d.append(element('p', 'error-text', warning));
}
async function refresh() {
  try {
    const response = await fetch('/data.json', {cache:'no-store'});
    if (!response.ok) throw Error('HTTP ' + response.status);
    const data = await response.json(), next = JSON.stringify(data.runs);
    if (next !== signature) { state = data; signature = next; updatePicker(); render(); }
    $('connection').textContent = 'Локальный live · обновление каждые 2 с · ' + new Date(data.captured * 1000).toLocaleTimeString('ru-RU');
  } catch (error) { $('connection').textContent = 'Нет связи с локальным viewer. Показан последний снимок.'; }
  setTimeout(refresh, 2000);
}
$('architecture-tab').onclick=()=>showTab('architecture');$('tree-tab').onclick=()=>showTab('tree');$('runs').onchange=()=>{active=$('runs').value;selected=null;render()};
architecture();updatePicker();showTab(location.hash==='#tree'?'tree':'architecture');$('connection').textContent='Снимок · '+new Date(state.captured*1000).toLocaleString('ru-RU');if(live)refresh();
</script></html>'''


def page(data: dict, live: bool = False) -> bytes:
    # JSON is embedded as data, never HTML; a model's text cannot close this script tag.
    payload = json.dumps(data, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c")
    return HTML.replace("__LIVE__", str(live).lower()).replace("__DATA__", payload).encode("utf-8")


class Handler(BaseHTTPRequestHandler):
    def __init__(self, *args, root: Path, **kwargs):
        self.root = root
        super().__init__(*args, **kwargs)

    def do_GET(self):
        route = urlsplit(self.path).path
        if route not in ("/", "/data.json"):
            self.send_error(404)
            return
        data = collect(self.root)
        content = page(data, live=True) if route == "/" else json.dumps(data, ensure_ascii=False, allow_nan=False).encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8" if route == "/" else "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(content)

    def log_message(self, *args):
        pass


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", type=Path, default=Path("outputs"))
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--export", type=Path, help="Write a self-contained snapshot instead of starting the server")
    args = parser.parse_args(argv)
    if args.export:
        args.export.parent.mkdir(parents=True, exist_ok=True)
        args.export.write_bytes(page(collect(args.root)))
        print(args.export.resolve())
        return
    with ThreadingHTTPServer(("127.0.0.1", args.port), partial(Handler, root=args.root.resolve())) as server:
        print(f"Viewer: http://127.0.0.1:{server.server_port} · root: {args.root.resolve()}", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()

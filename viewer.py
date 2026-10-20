"""Read-only local viewer: python viewer.py [JSON directories ...] [--export snapshot.html]."""

import argparse
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import time
from urllib.parse import urlsplit


def collect(root: Path, *extra_roots: Path) -> dict:
    """Read known JSON, including partial extractions; no CSV, .env or API access."""
    roots, runs = (root, *extra_roots), []
    anchors = ("task.json", "problem.json", "data.json", "metric.json", "output.json", "extraction.json", "report.json")
    for root in roots:
        root = root.resolve()
        folders = [root] if any((root / name).is_file() for name in anchors) else sorted(
            {p.parent for name in anchors for p in root.glob(f"**/{name}")})
        folders = sorted({p.parent if p.name == 'extraction' and p != root else p for p in folders})
        runs.extend(collect_folders(root, folders, prefix=root.name + "/" if len(roots) > 1 else ""))
    return {"runs": sorted(runs, key=lambda r: r["modified"], reverse=True), "captured": time.time()}


def collect_folders(root: Path, folders: list[Path], prefix: str) -> list[dict]:
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
        def source(path):
            if not path.is_file():
                return None
            try:
                if not path.resolve().is_relative_to(root) or path.stat().st_size > 100000:
                    raise ValueError('Source outside root or too large')
                modified.append(path.stat().st_mtime)
                return path.read_text(encoding='utf-8')
            except (OSError, ValueError):
                warnings.append(f'{path.name}: исходник недоступен')
                return None
        task = read(folder / "task.json")
        run = {key: read(folder / f"{key}.json") for key in ("profile", "validation", "best", "delta", "report", "preparation")}
        block_directory = folder / 'extraction' if (folder / 'extraction').is_dir() else folder
        blocks = {key: read(block_directory / f"{key}.json") for key in ("problem", "data", "metric", "output")}
        extraction = read(folder / "extraction.json")
        is_extraction = extraction is not None or any(b is not None and "evidence" in b for b in blocks.values())
        is_extraction |= "task_sha256" in (run["report"] or {}) and "status" in (run["report"] or {})
        if task is None and not is_extraction:
            continue
        nodes = []
        for directory in sorted((folder / "nodes").glob("node_*")):
            start = read(directory / "start.json")
            if start:
                nodes.append({"id": directory.name, "start": start, "finish": read(directory / "finish.json"),
                              "delta": read(directory / "delta.json"), "code": source(directory / 'experiment.py')})
        if task is None and extraction:
            task = {key: extraction.get(key, {}) for key in ('problem', 'data', 'metric')}
        run.update(id=prefix + (folder.relative_to(root).as_posix() if folder != root else folder.name),
                   directory=str(folder), kind="extraction" if is_extraction else "experiment", blocks=blocks, extraction=extraction,
                   block_directory=str(block_directory), preparation_code=source(folder / 'prepare.py'),
                   task=task, nodes=nodes, warnings=warnings, modified=max(modified, default=0))
        runs.append(run)
    return runs


HTML = r'''<!doctype html>
<html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Alfa · Извлечение и эксперименты</title><link rel="icon" href="data:,">
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
.tree-node.partial .box{fill:var(--purple-bg)}
.tree .score{font-family:ui-monospace,SFMono-Regular,Consolas,monospace;font-size:17px;font-variant-numeric:tabular-nums}
.tree .best-label{fill:var(--purple);font-size:11px;font-weight:650}
.summary{display:flex;gap:22px;flex-wrap:wrap;padding:0 0 16px;color:var(--muted);font-size:12px}
.summary strong{display:block;color:var(--ink);font-size:16px;font-weight:600;margin-top:3px}
.detail{border-left:1px solid var(--line);padding-left:20px;overflow-wrap:anywhere}
.detail pre{background:var(--bg);border-radius:7px;padding:12px;white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px;max-height:460px;overflow:auto}
.facts{max-height:510px;overflow:auto;margin:16px 0}
.facts p{margin:4px 0 8px}
.facts summary{cursor:pointer;color:var(--muted)}
blockquote{margin:8px 0;padding:0 10px;border-left:2px solid var(--line);color:var(--muted)}
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
<header><div><h1>Как работает ML-агент</h1><small>Условие → подготовка → код → baseline</small></div><nav aria-label="Разделы"><button id="extraction-tab" aria-selected="true">Извлечение</button><button id="architecture-tab" aria-selected="false">Схема</button><button id="tree-tab" aria-selected="false">Эксперименты</button></nav></header>
<div class="topline"><label id="run-picker" hidden>Запуск <select id="runs" aria-label="Запуск"></select></label><span id="connection" role="status"></span></div>
<main>
<section id="architecture-panel" class="panel" hidden><p class="muted">Текущий путь до первого baseline. Состояния и JSON конкретного запуска — во вкладке «Извлечение».</p><div class="legend"><span><i class="dot"></i>Детерминированный Python</span><span><i class="dot llm"></i>Вызов Qwen API</span><span><i class="dot cpu"></i>Локальное обучение на CPU</span></div><div class="diagram-scroll"><svg id="architecture" class="architecture" viewBox="0 0 1120 820" role="img" aria-label="Текущая архитектура: извлечение, подготовка, генерация кода и первый baseline"></svg></div><p class="footnote">Qwen пишет prepare.py и experiment.py. Общая обвязка измеряет профиль, фиксирует split и вычисляет score. Цикл улучшения гипотез ещё не подключён.</p></section>
<section id="tree-panel" class="panel" hidden><div id="empty" hidden></div><div id="experiment"><div id="summary" class="summary"></div><p id="tree-legend" class="legend"></p><div class="tree-shell"><div class="tree-container"><svg id="tree" class="tree" role="img" aria-label="Граф сохранённых JSON"></svg></div><aside id="detail" class="detail" aria-label="Детали выбранного блока"></aside></div></div><p id="tree-note" class="footnote"></p></section>
</main><script id="initial-data" type="application/json">__DATA__</script>
<script>
const $=id=>document.getElementById(id), NS='http://www.w3.org/2000/svg';
const element=(tag,cls,text)=>{const e=document.createElement(tag);if(cls)e.className=cls;if(text!==undefined)e.textContent=text;return e};
const svg=(tag,attrs={},text)=>{const e=document.createElementNS(NS,tag);for(const [k,v]of Object.entries(attrs))e.setAttribute(k,v);if(text!==undefined)e.textContent=text;return e};
const fmt=n=>typeof n==='number'&&Number.isFinite(n)?n.toFixed(6):'—';
const families={dummy:'Dummy baseline',extra_trees:'ExtraTrees',hist_gradient_boosting:'HistGradientBoosting'};
const statuses={completed:'Завершён',failed:'Ошибка',timeout:'Таймаут'};
const stops={max_experiments:'Лимит экспериментов',time_budget:'Лимит времени',proposal_limit:'Лимит предложений',llm_unavailable_or_contract_conflict:'API / конфликт контракта',llm_unavailable_or_invalid_hypothesis:'API / невалидная гипотеза'};
const roles={problem:'Задача',data:'Данные',metric:'Метрика',output:'Формат ответа'};
const extractionStatus=r=>({running:'Выполняется',ready:'Готово',partial:'Частично',failed:'Ошибка'}[r.report?.status]??'Нет итогового отчёта');
let state=JSON.parse($('initial-data').textContent), active=null, selected=null, jsonTab='start', signature='', view='extraction', live=__LIVE__;
function marker(canvas){const d=svg('defs'),m=svg('marker',{id:canvas.id+'-arrow',viewBox:'0 0 10 10',refX:9,refY:5,markerWidth:7,markerHeight:7,orient:'auto-start-reverse'});m.append(svg('path',{d:'M 0 0 L 10 5 L 0 10 z',fill:'var(--muted)'}));d.append(m);canvas.append(d)}
function edge(canvas,d,feedback=false){canvas.append(svg('path',{d,class:'edge'+(feedback?' feedback':''),'marker-end':`url(#${canvas.id}-arrow)`}))}
function box(canvas,x,y,w,h,title,lines=[],kind=''){
 const g=svg('g');g.append(svg('rect',{x,y,width:w,height:h,rx:9,class:'box '+kind}));g.append(svg('text',{x:x+16,y:y+27,class:'title'},title));
 lines.forEach((line,i)=>g.append(svg('text',{x:x+16,y:y+49+i*18,class:'caption'},line)));canvas.append(g);return g;
}
function architecture(){
 const c=$('architecture'); marker(c); c.setAttribute('viewBox','0 0 1120 700');
 ['M 270 75 H 330','M 670 75 H 760','M 900 120 V 175 H 180 V 230',
  'M 320 275 H 420','M 700 275 H 800','M 940 320 V 370 H 180 V 420',
  'M 320 465 H 420','M 700 465 H 800','M 940 510 V 570 H 560 V 590',
  'M 420 635 H 320'].forEach(p=>edge(c,p));
 box(c,40,30,230,90,'01  Markdown',['Текст условия','Файлы остаются локально']);
 box(c,330,30,340,90,'02  Четыре extractor’а Qwen',['Problem · Data · Metric · Output','Произвольные имена, форматы и метрики'],'llm');
 box(c,760,30,320,90,'03  extraction.json',['Схема + проверка исходных цитат','Не проверяем поддержку исполнителем']);
 box(c,40,230,280,90,'04  Подготовка · Qwen',['Условие + инвентаризация файлов','Чтение, split, scorer → prepare.py'],'llm');
 box(c,420,230,280,90,'05  Исполнение Python',['Читаем реальные данные','Измеряем профиль; сохраняем split'],'cpu');
 box(c,800,230,280,90,'06  preparation.json',['Общие условия всех экспериментов','Профиль, split, метрика, проблемы']);
 box(c,40,420,280,90,'07  Baker · Qwen',['План простейшего baseline','Данные, preprocessing, модель, fit'],'llm');
 box(c,420,420,280,90,'08  Реализатор · Qwen',['Один experiment.py','Короткая проверка на части данных'],'llm');
 box(c,800,420,280,90,'09  start.json',['План + ссылки + хеш исходника','После записи start код фиксируется']);
 box(c,420,590,280,90,'10  Отдельный процесс',['Fit → predictions → общий scorer','finish.json + предсказания'],'cpu');
 box(c,40,590,280,90,'11  Готовый baseline',['Файл ответа рядом с условием','Best — только при вычисленном score']);
}
function showTab(name) {
  view = name; const diagram = name === 'architecture';
  $('architecture-panel').hidden = !diagram; $('tree-panel').hidden = diagram; $('run-picker').hidden = diagram;
  for (const tab of ['extraction', 'architecture', 'tree']) $(tab+'-tab').setAttribute('aria-selected', tab === name);
  location.hash = name;
  if (!diagram) { selected = null; updatePicker(); render(); }
}
function modeOf(run) {
  if (run.report?.mode) return run.report.mode;
  return run.nodes.some(n => n.start.source === 'qwen') ? 'qwen'
    : run.nodes.some(n => n.start.source === 'offline') ? 'offline' : 'режим не записан';
}
function updatePicker() {
  const select = $('runs'), runs = state.runs.filter(r => view === 'extraction' ? r.kind === 'extraction' : r.nodes.length || r.kind === 'experiment'); select.replaceChildren();
  for (const r of runs) {
    const label = r.kind === 'extraction' ? `${extractionStatus(r)} · ${r.id}`
      : `${r.id.includes('synthetic') ? 'Синтетика' : 'Данные'} · ${modeOf(r)} · ${r.id}`;
    const option = element('option', '', label); option.value = r.id; select.append(option);
  }
  if (!runs.some(r => r.id === active)) active = (runs.find(r => !r.id.includes('synthetic')) ?? runs[0])?.id;
  select.value = active ?? '';
}
function render() {
  const r = state.runs.find(r => r.id === active);
  $('empty').hidden = !!r; $('experiment').hidden = !r;
  $('tree-legend').textContent = view === 'extraction' ? 'Четыре extractor’а → сборка → подготовка → baseline · зелёный = готово · красный = ошибка · нажмите блок'
    : 'Связь = parent_id · зелёный = завершён · красный = ошибка · фиолетовая рамка = лидер';
  $('tree-note').textContent = view === 'extraction' ? (live ? 'Граф читает сохранённые файлы каждые 2 с. ' : 'Снимок сохранённых JSON. ')+ 'Evidence подтверждает источник фактов. preparation.json содержит измеренный профиль. Статусы читаются из последних записанных файлов и не являются проверкой, что процесс ещё жив.'
    : 'Показаны записанные JSON. Нода без finish могла выполняться или быть прервана. Без report.json завершение запуска не подтверждено.';
  if (!r) { $('empty').textContent = view === 'extraction' ? 'Результатов извлечения пока нет. Запустите python -m alfa_agent extract; JSON появятся здесь автоматически.' : 'Экспериментов пока нет.'; return; }
  if (r.kind === 'extraction' && view === 'extraction') { drawExtraction(r); drawExtractionDetail(r); return; }
  if (!['start', 'finish', 'delta', 'code'].includes(jsonTab)) jsonTab = 'start';
  if (!r.nodes.some(n => n.id === selected)) selected = r.nodes.find(n => n.id === r.best?.node_id)?.id ?? r.nodes[0]?.id;
  const mode = modeOf(r), metric = r.preparation?.metric?.name ?? r.task?.metric?.name ?? 'Не записана';
  const info = [
    ['Источник гипотез', mode === 'offline' ? 'Фиксированная очередь' : mode === 'qwen' ? 'Qwen API' : 'Режим не записан'],
    ['Основная метрика', metric], ['Лучший score', fmt(r.best?.value)],
    ['Запуск', r.report?.status ? extractionStatus(r) : r.report ? (stops[r.report.stop_reason] ?? r.report.stop_reason) : 'report.json ещё не записан']
  ];
  $('summary').replaceChildren();
  for (const [label, value] of info) {
    const item = element('div', '', label); item.append(element('strong', '', value)); $('summary').append(item);
  }
  drawTree(r); drawDetail(r);
}
function drawExtraction(r) {
  if (!(selected in roles) && !['result', 'preparation', 'node_000'].includes(selected)) selected = Object.keys(roles).find(k => r.blocks[k]) ?? 'result';
  const c = $('tree'), report = r.report, accepted = Object.values(r.blocks).filter(Boolean).length;
  c.replaceChildren(); marker(c); c.setAttribute('viewBox', '0 0 735 540'); c.style.width = '735px'; c.style.height = '540px';
  $('summary').replaceChildren();
  for (const [label, value] of [['Статус запуска', extractionStatus(r)], ['Сохранено блоков', accepted+'/4'],
    ['Этап', {extraction:'Извлечение',preparation:'Подготовка',baseline:'Baseline',complete:'Завершён'}[report?.stage]??'Извлечение'],
    ['Вызовов в отчёте', report?.llm_calls?.length ?? '—']]) {
    const item = element('div', '', label); item.append(element('strong', '', value)); $('summary').append(item);
  }
  const brief = {
    problem: [r.blocks.problem?.task_type, r.blocks.problem?.target_column].filter(Boolean).join(' · '),
    data: [r.blocks.data?.train_file, r.blocks.data?.test_file].filter(Boolean).join(' · '),
    metric: r.blocks.metric ? (r.blocks.metric.name??'Имя не указано') + ({minimize:' ↓',maximize:' ↑'}[r.blocks.metric.direction]??'') : '',
    output: r.blocks.output?.columns?.join(', ')
  };
  function card(id, x, y, w, title, lines, status) {
    const g = box(c, x, y, w, 110, title, lines);
    g.setAttribute('class', `tree-node ${status} ${selected===id?'selected':''}`);
    g.setAttribute('role', 'button'); g.setAttribute('tabindex', 0); g.setAttribute('data-node', id);
    g.setAttribute('aria-label', {result:'Итоговый extraction.json',preparation:'preparation.json',node_000:'node_000 baseline'}[id] ?? id+'.json');
    g.onclick = () => { selected = id; render(); };
    g.onkeydown = e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); g.onclick(); } };
  }
  Object.keys(roles).forEach((key, i) => {
    const y = 20+i*130, calls = report?.llm_calls?.filter(call => call.role === key) ?? [];
    const saved = !!r.blocks[key], failed = !saved && calls.some(call => call.status !== 'completed');
    const status = saved ? 'JSON сохранён'+(calls.length>1?' · исправлен':'') : failed ? 'Ошибка · JSON не сохранён' : 'JSON ещё нет';
    edge(c, `M 167 270 H 185 V ${y+55} H 210`); edge(c, `M 460 ${y+55} H 480 V 140 H 500`);
    card(key, 210, y, 250, roles[key], [(brief[key] ?? '').slice(0, 33), key+'.json', status], saved?'completed':failed?'failed':'pending');
  });
  box(c, 12, 215, 155, 110, 'Условие', ['Текст задания', 'SHA256 '+(report?.task_sha256?.slice(0, 10) ?? '—')]);
  card('result', 500, 85, 215, 'Объединение JSON', ['Схема + исходные цитаты',
    r.extraction ? 'extraction.json сохранён' : 'Итоговый JSON ещё нет'], r.extraction?'completed':report?.status==='failed'?'failed':'pending');
  edge(c, 'M 607 195 V 280'); edge(c, 'M 607 390 V 410');
  const p = r.preparation, n = r.nodes.find(n=>n.id==='node_000'), f = n?.finish;
  card('preparation', 500, 280, 215, 'Подготовка · Qwen + Python', [
    'preparation.json', p?.status??'Файл ещё не записан', p?.metric?.name??''],
    p?.status==='blocked'?'failed':p?.status==='ready'?'completed':p?.status==='partial'?'partial':'pending');
  card('node_000', 500, 410, 215, 'node_000 · Baseline', [
    n ? 'start + experiment.py' : 'Start ещё не записан', f ? statuses[f.status]??f.status : 'Finish ещё не записан',
    f?.value !== null && f?.value !== undefined ? 'Score '+fmt(f.value) : 'Score не вычислен'], f?.status??'pending');
}
function drawExtractionDetail(r) {
  const d = $('detail'), key = selected, report = r.report; d.replaceChildren();
  if (key === 'node_000') {
    if (r.nodes.some(n=>n.id===key)) { drawDetail(r); return; }
    d.append(element('h2', '', 'Baseline'), element('p', '', 'start.json появится после генерации кода и короткой проверки запуска.')); return;
  }
  d.append(element('h2', '', key === 'result' ? 'Извлечение и сборка' : key==='preparation'?'Подготовка данных':roles[key]+' · '+key+'.json'));
  d.append(element('small', 'mono', (key in roles?r.block_directory:r.directory)+'/'+(key === 'result' ? 'extraction.json' : key+'.json')));
  if (key === 'result' || key === 'preparation') {
    d.append(element('p', '', key==='result' ? 'extraction.json объединяет ответы с evidence. report.json хранит этап, ошибки и попытки API.'
      : 'Qwen предложил чтение, split и scorer в prepare.py. Python измерил профиль и сохранил общие условия экспериментов.'));
    for (const error of report?.errors ?? []) d.append(element('p', 'error-text', error));
    if (report?.checks?.length) {
      const list = element('ul', 'changes'); for (const check of report.checks) list.append(element('li', 'mono', check));
      const checks = element('details'); checks.append(element('summary', '', 'Пройденные проверки: '+report.checks.length), list); d.append(checks);
    }
    const files = key==='result'?['extraction','report']:['preparation','preparation_code'];
    if (!files.includes(jsonTab)) jsonTab = files[0];
    const tabs = element('div', 'json-tabs');
    for (const name of files) {
      const button = element('button', '', name==='preparation_code'?'prepare.py':name+'.json'); button.setAttribute('aria-selected', name === jsonTab);
      button.onclick = () => { jsonTab = name; drawExtractionDetail(r); }; tabs.append(button);
    }
    const value = r[jsonTab];
    d.append(tabs, element('pre', '', typeof value==='string'?value:value?JSON.stringify(value, null, 2):'Файл ещё не записан.'));
  } else {
    const block = r.blocks[key], facts = element('div', 'facts');
    d.append(element('p', '', block ? 'Принятый ответ Qwen: поля, типы и цитаты прошли проверку.' : 'Файл не сохранён. Принятого ответа этого extractor’а пока нет.'));
    for (const [field, value] of Object.entries(block ?? {}).filter(([field]) => field !== 'evidence')) {
      const row = element('div', 'comparison'); row.append(element('b', 'mono', field),
        element('p', '', value === null ? 'Не указано в условии (null)' : typeof value==='object' ? JSON.stringify(value) : String(value)));
      const quotes = block.evidence?.[field] ?? [];
      if (quotes.length) {
        const evidence = element('details'); evidence.append(element('summary', '', 'Цитаты из условия: '+quotes.length));
        for (const quote of quotes) evidence.append(element('blockquote', '', quote)); row.append(evidence);
      }
      facts.append(row);
    }
    d.append(facts);
    const calls = report?.llm_calls?.filter(call => call.role === key) ?? [];
    for (const call of calls) d.append(element('small', '', `Попытка ${call.attempt}: ${call.status}. `));
    const raw = element('details'); raw.append(element('summary', '', 'JSON · '+key+'.json'),
      element('pre', '', block ? JSON.stringify(block, null, 2) : 'Файл ещё не записан.')); d.append(raw);
  }
  for (const warning of r.warnings) d.append(element('p', 'error-text', warning));
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
    const model = n.start.hypothesis?.pipeline?.model;
    const label = families[model?.family] ?? model?.family ?? model?.name ?? model?.type ?? 'Описание модели в JSON';
    g.append(svg('text', {x:p.x+15, y:p.y+46, class:'title'}, String(label).slice(0, 30)));
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
  const files = ['start', 'finish', ...(n.code?['code']:[]), ...(n.delta?['delta']:[])];
  if (!files.includes(jsonTab)) jsonTab = 'start';
  d.append(element('h2', '', n.id), element('small', '', `Родитель: ${n.start.parent_id ?? 'нет · baseline'}`));
  if (n.finish) d.append(element('p', 'metric', `${n.finish.metric?.name??'Метрика'}: ${fmt(n.finish.value)} · ${n.finish.evaluation_status??n.finish.status}`));
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
  for (const key of files) {
    const button = element('button', '', key==='code'?'experiment.py':key+'.json'); button.setAttribute('aria-selected', key === jsonTab);
    button.onclick = () => { jsonTab = key; drawDetail(r); }; tabs.append(button);
  }
  d.append(tabs);
  const value = n[jsonTab]; d.append(element('pre', '', typeof value==='string'?value:value?JSON.stringify(value, null, 2):jsonTab+'.json ещё не записан.'));
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
for (const tab of ['extraction','architecture','tree']) $(tab+'-tab').onclick=()=>showTab(tab);
$('runs').onchange=()=>{active=$('runs').value;selected=null;render()};
architecture();showTab(['#tree','#architecture'].includes(location.hash)?location.hash.slice(1):'extraction');$('connection').textContent='Снимок · '+new Date(state.captured*1000).toLocaleString('ru-RU');if(live)refresh();
</script></html>'''


def page(data: dict, live: bool = False) -> bytes:
    # JSON is embedded as data, never HTML; a model's text cannot close this script tag.
    payload = json.dumps(data, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c")
    return HTML.replace("__LIVE__", str(live).lower()).replace("__DATA__", payload).encode("utf-8")


class Handler(BaseHTTPRequestHandler):
    def __init__(self, *args, roots: list[Path], **kwargs):
        self.roots = roots
        super().__init__(*args, **kwargs)

    def do_GET(self):
        route = urlsplit(self.path).path
        if route not in ("/", "/data.json"):
            self.send_error(404)
            return
        data = collect(*self.roots)
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
    parser.add_argument("roots", nargs="*", type=Path, default=[Path("artifacts")])
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--export", type=Path, help="Write a self-contained snapshot instead of starting the server")
    args = parser.parse_args(argv)
    if args.export:
        args.export.parent.mkdir(parents=True, exist_ok=True)
        args.export.write_bytes(page(collect(*args.roots)))
        print(args.export.resolve())
        return
    with ThreadingHTTPServer(("127.0.0.1", args.port), partial(Handler, roots=args.roots)) as server:
        print(f"Viewer: http://127.0.0.1:{server.server_port} · roots: {', '.join(str(p.resolve()) for p in args.roots)}", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()

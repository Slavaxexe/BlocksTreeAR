"""Read-only live viewer for a single training-component run (stdlib only)."""
import argparse
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import re


_ALLOWED = re.compile(r"(?:report|validation_decision|training_plan|model_templates|final_analysis|analysis_\d+|plan_\d+|repair_(?:(?:analysis|plan)_)?\d+)\.json\Z")
_SECRET_KEY = re.compile(r"api.?key|authorization|password|secret|access.?token", re.I)
_TOKEN = re.compile(r"\bsk-[A-Za-z0-9_-]{10,}\b")


def sanitize(value):
    if isinstance(value, dict):
        return {key: ('[скрыто]' if _SECRET_KEY.search(key) else sanitize(item))
                for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize(item) for item in value]
    return _TOKEN.sub('[ключ скрыт]', value) if isinstance(value, str) else value


def snapshot(root):
    documents, pending = {}, []
    if root.name == 'training':
        pipeline_report = root.parent/'pipeline_report.json'
        if pipeline_report.is_file():
            try:
                documents['pipeline_report.json'] = sanitize(json.loads(pipeline_report.read_text(encoding='utf-8-sig')))
            except (OSError, ValueError):
                pending.append('pipeline_report.json: ожидаем завершения записи')
    if root.is_dir():
        for path in sorted(root.iterdir()):
            if not _ALLOWED.fullmatch(path.name) or not path.is_file() or path.is_symlink():
                continue
            try:
                if path.stat().st_size > 8_000_000:
                    pending.append(path.name + ': файл слишком большой')
                    continue
                documents[path.name] = sanitize(json.loads(path.read_text(encoding='utf-8-sig')))
            except (OSError, ValueError):
                pending.append(path.name + ': ожидаем завершения записи')
    return {'directory': str(root), 'exists': root.is_dir(), 'documents': documents,
            'pending': pending}


HTML = r'''<!doctype html>
<html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Обучение и валидация · Qwen</title><style>
:root{color-scheme:dark;--bg:#101522;--card:#1a2234;--muted:#a5b1c9;--line:#33405a;--green:#57d8ab;--blue:#86adff;--red:#ff9393;--yellow:#efd187}
*{box-sizing:border-box}body{margin:0;font:15px/1.5 system-ui,sans-serif;background:var(--bg);color:#edf2ff}main{max-width:1500px;margin:auto;padding:30px 28px}h1{font-size:27px;margin:0 0 5px}h2{font-size:18px;margin:0 0 14px}p{margin:6px 0}.muted{color:var(--muted)}header{display:flex;justify-content:space-between;gap:18px;align-items:center}#status{padding:8px 16px;border:1px solid var(--line);border-radius:24px;white-space:nowrap}#directory{font:12px ui-monospace,monospace;overflow-wrap:anywhere;margin-top:14px}.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:24px 0}.metric,.panel{border:1px solid var(--line);border-radius:16px;background:var(--card);padding:18px}.metric strong{display:block;font-size:23px;margin-top:5px}.metric span{color:var(--muted);font-size:13px}.flow{display:flex;align-items:stretch;gap:8px;overflow:auto;padding:6px 0 18px}.step{min-width:125px;flex:1;padding:15px 12px;border:1px solid var(--line);border-radius:12px;position:relative;background:#151d2d}.step+.step:before{content:'→';position:absolute;left:-13px;top:34px;color:var(--muted);z-index:1;background:var(--card)}.step b{display:block;font-size:13px;line-height:1.45}.step small{display:block;font-size:11px;color:var(--muted);margin-top:10px}.step.done{border-color:var(--green)}.step.done small{color:var(--green)}.step.active{border-color:var(--blue);background:#21345a;box-shadow:0 0 0 2px #86adff20}.step.active small{color:var(--blue)}.step.error{border-color:var(--red)}.step.error small{color:var(--red)}.columns{display:grid;grid-template-columns:1.2fr 1fr;gap:18px;margin-top:18px}.panel{min-width:0}table{width:100%;border-collapse:collapse;font-size:13px}th,td{text-align:left;border-bottom:1px solid var(--line);padding:10px 7px}th{color:var(--muted);font-weight:500}tbody tr{cursor:pointer}tbody tr:hover{background:#25334d}button{font:inherit;color:inherit;background:#24314b;border:1px solid #435575;border-radius:9px;padding:7px 11px;cursor:pointer}.tabs{display:flex;flex-wrap:wrap;gap:7px;margin-bottom:12px}.tabs button.selected{border-color:var(--blue);color:var(--blue)}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:12px/1.65 ui-monospace,monospace;max-height:530px;overflow:auto;margin:0}.note{padding:13px 16px;border-radius:10px;border:1px solid var(--line);font-size:13px;margin-top:15px}.error-text{color:var(--red)}.good{color:var(--green)}.active-text{color:var(--blue)}#submission{overflow-wrap:anywhere}.footer{margin-top:18px;color:var(--muted);font-size:12px}@media(max-width:950px){.columns{grid-template-columns:1fr}.metrics{grid-template-columns:repeat(2,1fr)}main{padding:20px 14px}h1{font-size:23px}.step{min-width:138px}header{align-items:flex-start}}
</style></head><body><main>
<header><div><h1>Обучение и валидация</h1><p class="muted">Qwen выбирает параметры → код обучает → метрика определяет победителя</p></div><div id="status">Ожидание</div></header>
<div id="directory" class="muted"></div>
<div class="metrics"><div class="metric"><span>Этап</span><strong id="stage">Ожидание</strong></div><div class="metric"><span>Лучшая валидация</span><strong id="score">—</strong></div><div class="metric"><span>Эксперименты: успешные / ошибки</span><strong id="counts">0 / 0</strong></div><div class="metric"><span>Прошло / общий бюджет</span><strong id="time">—</strong></div></div>
<section class="panel"><h2>Аналитик · планировщик · координатор</h2><div class="tabs" id="roles"></div><p class="muted">Аналитик: факты, гипотезы и риски → планировщик: следующие параметры → координатор: допуск, таймер, победитель и полный refit.</p><div id="flow" class="flow"></div><p id="activity" class="muted"></p><div id="scenario" class="note" hidden></div><div id="notice" class="note"></div><div id="coordinator" class="note"></div></section>
<div class="columns"><section class="panel"><h2>Результаты экспериментов</h2><table><thead><tr><th>Попытка / модель</th><th>Статус</th><th>Метрика</th><th>Train</th></tr></thead><tbody id="trials"></tbody></table><p id="empty" class="muted">Данные появятся после начала запуска.</p><div id="submission" class="note"></div></section><section class="panel"><h2>Решения Qwen и детали</h2><div id="tabs" class="tabs"></div><pre id="details">Ожидание отчёта…</pre></section></div>
<p id="connection" class="footer">Обновление каждую секунду. Страница только читает файлы запуска.</p></main><script>
const $=id=>document.getElementById(id);let selected='report.json',latest={},lastSuccess=0;
const labels={preparation:'Margo: подготовка данных',handoff:'Проверка выхода margo',validation_decision:'Планировщик: валидация',training_plan:'Планировщик: первый план',training:'Обучение',repair_review:'Qwen исправляет ошибку',repair_analysis:'Аналитик: ошибка',repair_planning:'Планировщик: исправление',result_analysis:'Аналитик: результаты',next_plan:'Планировщик: следующие опыты',coordinator_select:'Координатор: выбор победителя',final_analysis:'Аналитик: итог',final_refit:'Полная выборка',complete:'Завершено',input_validation:'Проверка входа'};
const stopLabels={max_trials:'Лимит экспериментов',search_time_limit:'Закончился бюджет поиска',no_improvement:'Нет существенного улучшения',planner_stop:'Планировщик предложил остановиться',no_new_candidates:'Нет новых конфигураций',planning_unavailable:'Анализ или планирование недоступны',no_affordable_candidates:'Нет экспериментов, помещающихся в бюджет',insufficient_time_for_planning_and_trial:'Оставлено время на полный refit',planning_round_limit:'Лимит раундов планирования'};
const statuses={running:'Выполняется',completed:'Готово',failed:'Ошибка',partial:'Частичный результат',pending:'Ожидает',reviewing:'Проверяет',repaired:'Исправлено',skipped:'Пропущено'};
const number=x=>typeof x==='number'&&Number.isFinite(x)?x.toLocaleString('ru-RU',{maximumFractionDigits:6}):'—';
const seconds=x=>typeof x==='number'?`${Math.floor(x/60)}м ${Math.floor(x%60)}с`:'—';
function text(tag,value,cls){let e=document.createElement(tag);e.textContent=value;if(cls)e.className=cls;return e}
function detail(name){selected=name;renderDetails()}
function renderDetails(){const tabs=$('tabs');tabs.replaceChildren();Object.keys(latest).forEach(name=>{let b=text('button',name.replace('.json',''));if(name===selected)b.className='selected';b.onclick=()=>detail(name);tabs.append(b)});$('details').textContent=latest[selected]===undefined?'Выберите решение или эксперимент.':JSON.stringify(latest[selected],null,2)}
function render(data){const d=data.documents||{},r=d['report.json']||d['pipeline_report.json']||{},trials=Array.isArray(r.trials)?r.trials:[],repairs=Array.isArray(r.repairs)?r.repairs:[],winner=r.winner||{},final=r.final_refit||{},running=r.status==='running',stage=r.stage||'';
latest={...d};trials.forEach((t,i)=>latest[t.trial_id||`попытка_${i}`]=t);if(selected!=='report.json'&&!(selected in latest))selected='report.json';
$('directory').textContent=data.directory;$('status').textContent=statuses[r.status]||'Ожидание запуска';$('status').className=r.status==='failed'?'error-text':r.status==='completed'?'good':running?'active-text':'';
const controlled=String(data.directory||'').includes('training-qwen-live');$('scenario').hidden=!controlled;$('scenario').textContent='Тест восстановления: сценарий намеренно задаёт Ridge alpha=-1, план и разбиение. Диагностику падения, исправленный план и итоговый анализ выполняет настоящий Qwen. Python-код обучения фиксирован; Qwen исправляет параметры.';
$('stage').textContent=labels[stage]||'Ожидание';$('score').textContent=winner.score===undefined?'—':`${number(winner.score)} ${String(r.primary_metric||'').toUpperCase()}`;
const ok=trials.filter(t=>t.status==='completed').length,errors=trials.filter(t=>t.status==='failed').length;
$('counts').textContent=`${ok} / ${errors}`;$('time').textContent=`${seconds(r.elapsed_seconds)} / ${seconds(r.time_budget_seconds)}`;
const hasRepair=repairs.length>0||Object.keys(d).some(n=>/^repair_/.test(n));
const history=Array.isArray(r.role_history)?r.role_history:[],state=r.coordinator||{};
$('roles').replaceChildren();[['analyst','Аналитик · Qwen'],['planner','Планировщик · Qwen'],['coordinator','Координатор · код']].forEach(([key,label])=>{let role=text('span',label,running&&r.active_role===key?'active-text':'muted');$('roles').append(role)});
const specs=history.length?[['Вход + шаблоны',!!d['model_templates.json'],false],['Планировщик: валидация',!!d['validation_decision.json'],stage==='validation_decision'],['Планировщик: первый план',!!d['training_plan.json'],stage==='training_plan'],['Worker: обучение',trials.length>0,stage==='training'],['Аналитик',history.some(e=>e.role==='analyst'&&e.status==='completed'),['result_analysis','repair_analysis','final_analysis'].includes(stage)],['Планировщик: следующий план',history.some(e=>['next_plan','repair_planning'].includes(e.stage)&&e.status==='completed'),['next_plan','repair_planning'].includes(stage)],['Координатор: выбор',!!r.search_stop_reason,stage==='coordinator_select'],['Полная выборка',final.status==='completed',stage==='final_refit'],['Submission',r.status==='completed'&&!!r.submission,false]]:[['Входной JSON',!!r.status,false],['Выбор валидации',!!d['validation_decision.json'],stage==='validation_decision'],['План обучения',!!d['training_plan.json'],stage==='training_plan'],['Обучение',trials.length>0,stage==='training'],['Ошибка',errors>0,false,'error'],['Qwen: исправление',hasRepair,stage==='repair_review'],['Сравнение',!!d['final_analysis.json']||stage==='final_refit'||r.status==='completed',['result_analysis','final_analysis'].includes(stage)],['Полная выборка',final.status==='completed',stage==='final_refit'],['Submission',r.status==='completed'&&!!r.submission,false]];
$('flow').replaceChildren();specs.forEach(([name,done,active,kind],i)=>{let e=text('div','','step');if(active&&running)e.classList.add('active');else if(done)e.classList.add(kind||'done');e.append(text('b',`${i+1}. ${name}`));let state=active&&running?'Выполняется':done?(kind==='error'?'Обнаружена':'Пройдено'):'Ожидает';if([4,5].includes(i)&&!done&&!active)state='При необходимости';e.append(text('small',state));$('flow').append(e)});
const active=r.active_trial;let activity=running?`Сейчас: ${labels[stage]||stage}.`:(r.status==='completed'?'Запуск завершён.':r.error||'Ожидаем появления report.json в папке запуска.');if(active)activity+=' '+(typeof active==='string'?active:JSON.stringify(active));$('activity').textContent=activity;
$('notice').textContent=r.error||`Победитель выбирается по ${String(r.primary_metric||'основной метрике').toUpperCase()} на общей валидации (${r.direction==='maximize'?'больше — лучше':'меньше — лучше'}). Финальное обучение использует все размеченные строки. Скрытый тест не используется для выбора модели.`;$('notice').className='note'+(r.error?' error-text':'');
$('coordinator').hidden=!history.length;$('coordinator').textContent=`Раундов без улучшения: ${state.rounds_without_improvement||0} / ${state.patience_rounds||'—'} · Резерв на финал: ${seconds(r.final_reserved_seconds)} · Пропущено конфигураций: ${state.skipped_candidates?.length||0}`+(r.search_stop_reason?` · Причина остановки: ${stopLabels[r.search_stop_reason]||r.search_stop_reason}`:'');
$('trials').replaceChildren();trials.forEach((t,i)=>{let row=document.createElement('tr'),id=t.trial_id||`попытка_${i}`,name=t.candidate?.name||t.family||'—';row.onclick=()=>detail(id);row.append(text('td',`${winner.trial_id===id?'★ ':''}${id} · ${name}`),text('td',statuses[t.status]||t.status||'—',t.status==='failed'?'error-text':'good'),text('td',number(t.score)),text('td',number(t.training_score)));$('trials').append(row)});
$('empty').hidden=trials.length>0;
let submission=r.submission?`Файл: ${r.submission}`:'Submission ещё не готов.';if(final.full_training_rows)submission+=` · Финальное обучение: ${final.full_training_rows} строк.`;if(r.status==='partial')submission+=' Внимание: полное переобучение не завершилось, возможен резервный результат.';$('submission').textContent=submission;
renderDetails();lastSuccess=Date.now();$('connection').textContent=`Обновлено ${new Date().toLocaleTimeString('ru-RU')}. Состояния берутся из файлов; время фиксируется координатором на контрольных точках.`+(data.pending?.length?' '+data.pending.join('; '):'');}
async function poll(){try{let res=await fetch('/data.json',{cache:'no-store'});if(!res.ok)throw Error(`HTTP ${res.status}`);render(await res.json())}catch(e){$('connection').textContent=`Нет связи с локальным сервером: ${e.message}. Повтор через секунду.`}finally{setTimeout(poll,1000)}}poll();
</script></body></html>'''


class Handler(BaseHTTPRequestHandler):
    def __init__(self, *args, root, **kwargs):
        self.root = root
        super().__init__(*args, **kwargs)

    def do_GET(self):
        route = self.path.split('?', 1)[0]
        if route in ('/', '/index.html'):
            body, content_type = HTML.encode('utf-8'), 'text/html; charset=utf-8'
        elif route == '/data.json':
            body = json.dumps(snapshot(self.root), ensure_ascii=False, allow_nan=False).encode('utf-8')
            content_type = 'application/json; charset=utf-8'
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args):
        pass


def main():
    parser = argparse.ArgumentParser(description='Live read-only training and validation diagram')
    parser.add_argument('root', type=Path, help='Run directory, may not exist yet')
    parser.add_argument('--port', type=int, default=8766)
    args = parser.parse_args()
    server = ThreadingHTTPServer(('127.0.0.1', args.port), partial(Handler, root=args.root.resolve()))
    print(f'Training viewer: http://127.0.0.1:{args.port}/', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()

"""Read-only localhost console and research viewer for Vitaly's solution."""
import argparse
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import re


TOKEN = re.compile(r'\bsk-[A-Za-z0-9_-]{8,}\b|Bearer\s+[A-Za-z0-9_.-]+', re.I)
SECRET = re.compile(r'api.?key|authorization|password|secret|access.?token', re.I)


def clean(value):
    if isinstance(value, dict):
        return {key: '[скрыто]' if SECRET.search(key) else clean(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clean(item) for item in value]
    return TOKEN.sub('[ключ скрыт]', value) if isinstance(value, str) else value


def read(path, limit=200000):
    try:
        with Path(path).open('rb') as stream:
            stream.seek(max(0, Path(path).stat().st_size-limit))
            return stream.read().decode('utf-8', errors='replace')
    except OSError:
        return ''


def document(path):
    try:
        return json.loads(read(path, 5000000))
    except ValueError:
        return {}


def snapshot(root, manifest_path=None):
    manifest = document(manifest_path or root/'artifacts/latest_launch.json')
    artifact = Path(manifest['artifacts']) if manifest.get('artifacts') else None
    if artifact and not artifact.resolve().is_relative_to(root.resolve()):
        artifact = None
    state = document(artifact/'state.json') if artifact else {}
    journal = []
    if artifact:
        for line in read(artifact/'journal.jsonl').splitlines()[-120:]:
            try:
                journal.append(json.loads(line))
            except ValueError:
                pass
    console = Path(manifest['console_log']) if manifest.get('console_log') else None
    logs = read(console) if console and console.resolve().is_relative_to(root.resolve()) else ''
    active_id = (state.get('active_experiment') or {}).get('id', '')
    if artifact and re.fullmatch(r'exp_\d+', active_id):
        worker_logs = read(artifact/'experiments'/active_id/'execution.log', 30000)
        logs += '\n--- Active experiment '+active_id+' ---\n'+(worker_logs or 'Preparing experiment; waiting for worker output...')
    report = document(artifact/'report.json') if artifact else {}
    return clean({'launch': manifest, 'state': state, 'report': report, 'journal': journal, 'console': logs})


HTML = r'''<!doctype html><html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Vitaly · AutoResearch</title><style>
body{background:#101722;color:#e8edf7;font:15px/1.5 system-ui;margin:0;padding:25px}main{max-width:1450px;margin:auto}h1{margin:0}small{color:#aebdd2}.cards{display:flex;flex-wrap:wrap;gap:12px;margin:20px 0}.card,section{background:#192438;border:1px solid #34425b;border-radius:12px;padding:16px}.card{flex:1;min-width:160px}.card b{display:block;font-size:23px}section{margin-top:15px}pre{font:12px/1.65 Consolas,monospace;white-space:pre-wrap;overflow-wrap:anywhere;max-height:520px;overflow:auto}table{width:100%;border-collapse:collapse;font-size:13px}td,th{text-align:left;padding:9px;border-bottom:1px solid #34425b}tr{cursor:pointer}button{padding:7px;background:#253851;color:inherit;border:1px solid #57719a;border-radius:6px}#path{overflow-wrap:anywhere}.error{color:#ffaaaa}.good{color:#77e1b7}
</style><main><h1>Vitaly · AutoResearch</h1><small id="subtitle">AutoResearch · этапы, эксперименты, Qwen и живые логи</small><div class="cards"><div class="card">Статус<b id="status">Ожидание</b></div><div class="card">Прошло / бюджет<b id="time">—</b></div><div class="card"><span id="score-label">Лучшая CV во время поиска</span><b id="best">—</b><small id="score-note"></small></div><div class="card">Эксперименты / Qwen<b id="counts">—</b></div></div><p id="path"></p><section><h2>Этапы</h2><div id="stages"></div></section><section><h2>Консоль <button id="follow">Прокрутка: вкл.</button></h2><pre id="console"></pre></section><section><h2>Эксперименты</h2><table><thead><tr><th>ID</th><th>Модель и признаки</th><th>Метрика CV</th><th>Статус</th><th>Время</th></tr></thead><tbody id="trials"></tbody></table></section><section><h2>События и ответы Qwen</h2><small>Последние 120 событий; ключи скрываются. Оценки относятся к валидации, не к скрытому тесту.</small><pre id="events"></pre></section><section><h2>Выводы исследователя</h2><pre id="insights"></pre></section><p id="connection"></p></main><script>
const $=x=>document.getElementById(x);let follow=true;const fmt=x=>typeof x==='number'?x.toLocaleString('ru-RU',{maximumFractionDigits:6}):'—';const tm=x=>`${Math.floor((x||0)/60)}м ${Math.floor((x||0)%60)}с`;
$('follow').onclick=()=>{follow=!follow;$('follow').textContent='Прокрутка: '+(follow?'вкл.':'выкл.')};
function renderScore(d, rank){
 const report=d.report||{}, state=d.state||{}, metric=report.task?.metric||state.facts?.['task.metric']?.value||'';
 const finalStage=report.stages?.['7_final']||state.stages?.['7_final'];
 const finalReady=report.submission_checks?.ok===true && (report.final_status||finalStage?.status)==='ok';
 const hasReport=Object.keys(report).length>0;
 if(hasReport || finalStage){
  $('score-label').textContent='Итоговый submission · CV';
  $('best').textContent=finalReady && Number.isFinite(report.final_cv)?fmt(report.final_cv)+' '+metric:'Оценка недоступна';
  const ensemble=report.ensemble||{}, count=Object.keys(ensemble.weights||{}).length;
  $('score-note').textContent=finalReady && Number.isFinite(report.final_cv)?(ensemble.used?'Ансамбль из '+count+' моделей. ':'Одиночная модель. ')+'Оценка на валидации; скор скрытого теста неизвестен.':'Для сохранённого submission нет подтверждённой CV-оценки.';
 }else{
  $('score-label').textContent='Лучшая CV во время поиска';
  $('best').textContent=fmt(rank[0]?.cv)+' '+metric;
  $('score-note').textContent='Итоговая модель ещё не выбрана.';
 }
}

async function poll(){try{let res=await fetch('/data.json',{cache:'no-store'});if(!res.ok)throw Error(res.status);let d=await res.json(),r=d.launch||{},s=d.state||{},ex=s.experiments||[],facts=s.facts||{};let ok=ex.filter(e=>e.status==='ok'),maximize=facts.direction?.value==='maximize',rank=[...ok].sort((a,b)=>maximize?b.cv-a.cv:a.cv-b.cv);$('status').textContent=({running:'Выполняется',completed:'Завершён',failed:'Ошибка',time_limit:'Лимит времени',stopped_by_user:'Остановлен пользователем'})[r.status]||'Ожидание';$('status').className=r.status==='failed'?'error':r.status==='completed'?'good':'';$('time').textContent=tm(r.elapsed_seconds)+' / '+tm(r.budget_seconds);renderScore(d, rank);$('counts').textContent=ex.length+' / '+(s.llm?.calls||0);$('subtitle').textContent=(r.task_type==='classification'?'Классификация':'Регрессия')+' · этапы, эксперименты, Qwen и живые логи';$('path').textContent=r.artifacts||r.launch_directory||'';$('stages').textContent=Object.entries(s.stages||{}).map(([name,v])=>name+': '+v.status+' ('+tm(v.seconds)+')').join(' → ')||'Начало запуска';$('console').textContent=d.console||'Ожидание первых логов…';$('events').textContent=(d.journal||[]).map(x=>JSON.stringify(x,null,2)).join('\n\n');$('insights').textContent=JSON.stringify({insights:s.insights||[],backlog:s.backlog||[],llm:s.llm||{}},null,2);$('trials').replaceChildren();ex.forEach(e=>{let tr=document.createElement('tr');[e.id,e.summary,fmt(e.cv),e.status+' / '+e.verdict,tm(e.seconds)].forEach(v=>{let td=document.createElement('td');td.textContent=v||'—';tr.append(td)});tr.onclick=()=>{$('insights').textContent=JSON.stringify(e,null,2)};$('trials').append(tr)});if(follow){$('console').scrollTop=$('console').scrollHeight;$('events').scrollTop=$('events').scrollHeight}$('connection').textContent='Обновлено '+new Date().toLocaleTimeString('ru-RU')}catch(e){$('connection').textContent='Соединение: '+e.message}finally{setTimeout(poll,2000)}}poll();
</script></html>'''


class Handler(BaseHTTPRequestHandler):
    def __init__(self, *args, root, manifest_path=None, **kwargs):
        self.root = root
        self.manifest_path = manifest_path
        super().__init__(*args, **kwargs)

    def do_GET(self):
        if self.path.split('?', 1)[0] == '/':
            body, mime = HTML.encode('utf-8'), 'text/html; charset=utf-8'
        elif self.path.split('?', 1)[0] == '/data.json':
            body, mime = json.dumps(snapshot(self.root, self.manifest_path), ensure_ascii=False).encode('utf-8'), 'application/json; charset=utf-8'
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header('Content-Type', mime)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args):
        pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--port', type=int, default=8768)
    parser.add_argument('--manifest', type=Path, default=None)
    args = parser.parse_args()
    server = ThreadingHTTPServer(('127.0.0.1', args.port), partial(Handler, root=args.root.resolve(), manifest_path=args.manifest.resolve() if args.manifest else None))
    print(f'Vitaly viewer: http://127.0.0.1:{args.port}/', flush=True)
    server.serve_forever()

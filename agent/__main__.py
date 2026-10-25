"""Qwen diagnostics, source extraction, baseline and hypothesis search."""

import argparse
import asyncio
from datetime import datetime, timezone
import getpass
import json
from pathlib import Path
import sys

from .config import ConfigurationError, Settings
from .qwen import QwenError, ask_qwen


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Qwen: извлечение условия, baseline и поиск гипотез.")
    parser.add_argument("command", choices=["check-config", "ping", "ask", "extract", "baseline", "search"])
    parser.add_argument("--env-file", type=Path, help="Файл настроек; по умолчанию .env в корне проекта.")
    parser.add_argument("--prompt", help="Текст запроса для команды ask.")
    parser.add_argument("--task", type=Path, default=Path("task.md"),
                        help="Markdown с условием; пути к данным отсчитываются от его каталога.")
    parser.add_argument("--output-dir", type=Path, help="Новый каталог запуска; существующий не перезаписывается.")
    parser.add_argument("--time-budget", type=float, default=600)
    parser.add_argument("--final-refit-seconds", type=float, default=300, help="Резерв на refit лучшего решения; 0 отключает. Входит в общий time-budget.")
    parser.add_argument("--experiment-timeout", type=float, default=240)
    parser.add_argument("--max-experiments", type=int, default=10, help="Всего нод, включая baseline.")
    parser.add_argument('--unlimited-nodes', action='store_true', help='Не ограничивать количество модельных и исследовательских нод.')
    parser.add_argument("--parallel", type=int, default=6, help="Одновременных нод основного дерева: по умолчанию 6.")
    parser.add_argument("--research-parallel", type=int, default=2, help="Одновременных нод исследования данных; по умолчанию 2, 0 отключает.")
    parser.add_argument("--eda-fraction", type=float, default=.15, help="Доля общего времени для начального EDA после сохранения baseline; по умолчанию 0.15.")
    parser.add_argument("--eda-seconds", type=float, help="Явный лимит начального EDA в секундах; действует и с --research-unlimited. Затем запускаются эксперименты.")
    parser.add_argument("--research-call-fraction", type=float, default=.3, help="Доля всех API-вызовов для EDA, включая повторы; по умолчанию 0.3. Остальное остаётся моделям и подготовке.")
    parser.add_argument("--research-unlimited", action="store_true", help="Убрать отдельные лимиты EDA по нодам, времени и последовательным ошибкам; квота API, явное окно EDA и общий бюджет остаются.")
    parser.add_argument("--max-research-nodes", type=int, default=12, help="Всего исследовательских нод на запуск.")
    parser.add_argument("--research-timeout", type=float, default=60, help="Лимит выполнения одного исследования, секунд.")
    parser.add_argument("--parallel-agents", type=int, default=8, help="Одновременных обращений к Qwen для всех ролей.")
    parser.add_argument("--max-llm-calls", type=int, default=80, help="Лимит всех запросов Qwen, включая повторы; 0 — без ограничения.")
    parser.add_argument("--automl", action=argparse.BooleanOptionalAction, default=True,
                        help="Выделенные FLAML/AutoGluon-эксперименты после EDA; настройки выбирает Qwen; отключение: --no-automl.")
    parser.add_argument("--max-code-attempts", "--max-config-attempts", type=int, default=5,
                        help="Версий конфигурации блока: первая и исправления; по умолчанию 5.")
    parser.add_argument("--key-stdin", action="store_true", help="Скрыто запросить ключ в памяти процесса.")
    args = parser.parse_args(argv)
    if args.command == "ask" and not args.prompt:
        parser.error("Для ask нужен --prompt.")
    settings = None
    try:
        settings = Settings.from_env(args.env_file)
        if args.key_stdin:
            from dataclasses import replace
            settings = replace(settings, api_key=getpass.getpass("Qwen API key (hidden): "))
        if args.command in {"extract", "baseline", "search"}:
            from .agents import extract_task
            from .runner import Options, run
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
            output = args.output_dir or Path("artifacts") / f"{args.task.resolve().parent.name}-{stamp}"
            if args.command == "extract":
                report = asyncio.run(extract_task(args.task.resolve(), output, settings))
            else:
                options = Options(time_budget=args.time_budget, experiment_timeout=args.experiment_timeout,
                    output_dir=output, max_experiments=(None if args.unlimited_nodes else args.max_experiments) if args.command == "search" else 1,
                    parallel=args.parallel, parallel_agents=args.parallel_agents, max_llm_calls=None if args.max_llm_calls == 0 else args.max_llm_calls,
                    max_code_attempts=args.max_code_attempts, research_parallel=args.research_parallel,
                    max_research_nodes=None if args.unlimited_nodes else args.max_research_nodes,
                    research_timeout=args.research_timeout, automl=args.automl,
                    final_refit_seconds=args.final_refit_seconds, eda_fraction=args.eda_fraction,
                    eda_seconds=args.eda_seconds,
                    research_call_fraction=args.research_call_fraction,
                    research_unlimited=args.research_unlimited)
                output, report = asyncio.run(run(args.task, options, settings))
            print(report.model_dump_json(indent=2))
            print(f"Artifacts: {output.resolve()}")
            return 2 if report.status == "failed" else 0
        if args.command == "check-config":
            status = settings.public_status()
            try:
                settings.validate()
            except ConfigurationError as error:
                status.update(ready=False, message=str(error))
                print(json.dumps(status, ensure_ascii=False, indent=2))
                return 2
            status.update(ready=True, message="Настройки заполнены. Доступность API ещё не проверена.")
            print(json.dumps(status, ensure_ascii=False, indent=2))
            return 0
        prompt = "Ответь одним словом: готов." if args.command == "ping" else args.prompt
        answer = ask_qwen(settings, prompt)
        payload = {"answer": answer.text, "model": answer.model, "usage": answer.usage}
        output = json.dumps(payload, ensure_ascii=False, indent=2)
        print(output.replace(settings.api_key, "[REDACTED]"))
        return 0
    except (ConfigurationError, QwenError, ValueError, OSError) as error:
        message = str(error)
        if settings and settings.api_key:
            message = message.replace(settings.api_key, "[REDACTED]")
        print(message, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

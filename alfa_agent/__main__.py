"""Qwen diagnostics, extraction and bounded experiment search."""

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
    parser = argparse.ArgumentParser(description="Подключение Qwen и отдельная проверка извлечения условия.")
    parser.add_argument("command", choices=["check-config", "ping", "ask", "extract", "baseline"])
    parser.add_argument("--env-file", type=Path, help="Файл настроек; по умолчанию .env в корне проекта.")
    parser.add_argument("--prompt", help="Текст запроса для команды ask.")
    parser.add_argument("--task", type=Path, default=Path("task.md"), help="Markdown с условием задачи.")
    parser.add_argument("--output-dir", type=Path, help="Новый каталог запуска; существующий не перезаписывается.")
    parser.add_argument("--time-budget", type=float, default=600)
    parser.add_argument("--experiment-timeout", type=float, default=240)
    parser.add_argument("--max-nodes", type=int, default=10, help="Число нод вместе с baseline; 1 — только baseline.")
    parser.add_argument("--platform-output", action="store_true", help="Требовать submission.csv для платформы.")
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
        if args.command in {"extract", "baseline"}:
            from .agents import extract_task
            from .runner import Options, run
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
            output = args.output_dir or Path("artifacts") / f"{args.task.resolve().parent.name}-{stamp}"
            if args.command == "extract":
                report = asyncio.run(extract_task(args.task.resolve(), output, settings))
            else:
                options = Options(time_budget=args.time_budget, experiment_timeout=args.experiment_timeout,
                                  output_dir=output, max_nodes=args.max_nodes, platform_output=args.platform_output)
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

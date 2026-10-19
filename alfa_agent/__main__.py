"""Qwen diagnostics and the standalone, training-free extraction stage."""

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
    parser.add_argument("command", choices=["check-config", "ping", "ask", "extract"])
    parser.add_argument("--env-file", type=Path, help="Файл настроек; по умолчанию .env в корне проекта.")
    parser.add_argument("--prompt", help="Текст запроса для команды ask.")
    parser.add_argument("--task", type=Path, default=Path("task.md"), help="Условие для extract; без обучения.")
    parser.add_argument("--output-dir", type=Path, help="Новый каталог результатов extract.")
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
        if args.command == "extract":
            from .agents import extract_task
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
            output = args.output_dir or Path("artifacts/extraction") / args.task.parent.name / stamp
            report = asyncio.run(extract_task(args.task.resolve(), output, settings))
            print(report.model_dump_json(indent=2))
            print(f"Artifacts: {output.resolve()}")
            return 0 if report.status == "ready" else 2
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
    except (ConfigurationError, QwenError, ValueError) as error:
        message = str(error)
        if settings and settings.api_key:
            message = message.replace(settings.api_key, "[REDACTED]")
        print(message, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

"""Configuration check and a deliberately small connection smoke test."""

import argparse
import json
from pathlib import Path
import sys

from .config import ConfigurationError, Settings
from .qwen import QwenError, ask_qwen


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Заготовка подключения к Qwen API организаторов.")
    parser.add_argument("command", choices=["check-config", "ping", "ask"])
    parser.add_argument("--env-file", type=Path, help="Файл настроек; по умолчанию .env в корне проекта.")
    parser.add_argument("--prompt", help="Текст запроса для команды ask.")
    args = parser.parse_args(argv)
    if args.command == "ask" and not args.prompt:
        parser.error("Для ask нужен --prompt.")
    settings = None
    try:
        settings = Settings.from_env(args.env_file)
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

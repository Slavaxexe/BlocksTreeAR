"""Проверка доступа к Qwen до полного прогона: python -m autoresearch.ping"""
import sys
import tempfile
from pathlib import Path

from .config import Settings, load_env
from .llm import LLM
from .memory import Memory

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    load_env(ROOT)
    s = Settings()
    if not s.api_key:
        print("❌ OPENAI_API_KEY не задан: впиши ключ в .env рядом с solution.py")
        sys.exit(2)
    ok, msg = LLM(s, Memory(Path(tempfile.mkdtemp()))).ping()
    print(("✅ Qwen отвечает: " if ok else "❌ Qwen не ответила: ") + msg)
    sys.exit(0 if ok else 1)

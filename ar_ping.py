"""Проверка доступа к Qwen до полного прогона: python -m autoresearch.ping"""
import sys
import tempfile
from pathlib import Path

from ar_config import Settings, load_env
from ar_llm import LLM
from ar_memory import Memory

ROOT = Path(__file__).resolve().parent

if __name__ == "__main__":
    load_env(ROOT)
    s = Settings()
    if not s.api_key:
        print("❌ OPENAI_API_KEY не задан: впиши ключ в .env рядом с solution.py")
        sys.exit(2)
    ok, msg = LLM(s, Memory(Path(tempfile.mkdtemp()))).ping()
    print(("✅ Qwen отвечает: " if ok else "❌ Qwen не ответила: ") + msg)
    sys.exit(0 if ok else 1)

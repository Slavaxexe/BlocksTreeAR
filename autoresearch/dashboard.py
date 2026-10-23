"""Пересобрать dashboard.html для готового запуска: python -m autoresearch.dashboard artifacts/autoresearch_<время>

--artifact  — без обёртки <html>, для публикации как страницы claude.ai (Artifact).
"""
import json
import sys
import tempfile
from pathlib import Path

from .config import Settings
from .llm import LLM
from .memory import Memory
from .report import render_dashboard
from .task import read_task


def rebuild(run_dir: Path, standalone: bool = True, extra: dict | None = None) -> str:
    run_dir = Path(run_dir)
    report = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
    mem = Memory(Path(tempfile.mkdtemp()))
    mem.state = json.loads((run_dir / "state.json").read_text(encoding="utf-8"))
    task_md = Path(report["facts"]["task.submission_path"]).with_name("task.md")
    scratch = Memory(Path(tempfile.mkdtemp()))
    task = read_task(task_md, scratch, LLM(Settings(llm_enabled=False), scratch), 1e18)
    return render_dashboard(report, mem, task, extra, standalone)


if __name__ == "__main__":
    rd = Path(sys.argv[1])
    art = "--artifact" in sys.argv
    out = rd / ("dashboard_artifact.html" if art else "dashboard.html")
    out.write_text(rebuild(rd, standalone=not art), encoding="utf-8")
    print(out)

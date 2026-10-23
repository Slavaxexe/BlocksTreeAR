"""Плоский ZIP для платформы (в корне только .py и requirements.txt — подпапки платформа не копирует).
autoresearch/X.py → ar_X.py, model_selection/X.py → ms_X.py, agent/X.py → max_X.py (запасной агент),
.md/.html ресурсы → строки в ar_assets.py.  python tools/build_flat.py OUT_DIR
"""
import ast
import re
import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MS = ["registry", "builders", "metrics", "diagnostics", "rules", "probe"]


def ar_code(src: str) -> str:
    s = re.sub(r"\bfrom \.(\w+) import", r"from ar_\1 import", src)
    s = re.sub(r"\bfrom autoresearch\.(\w+) import", r"from ar_\1 import", s)
    s = re.sub(r"\bfrom model_selection\.(\w+) import", r"from ms_\1 import", s)
    s = s.replace("Path(__file__).resolve().parents[1]", "Path(__file__).resolve().parent")
    s = s.replace('[sys.executable, "-m", "autoresearch.experiment",', '[sys.executable, str(root / "ar_experiment.py"),')
    s = s.replace('(Path(__file__).with_name("METHOD.md")).read_text(encoding="utf-8")', '__import__("ar_assets").METHOD')
    s = s.replace('TEMPLATE.read_text(encoding="utf-8")', '__import__("ar_assets").DASHBOARD')
    return s


def build(out: Path):
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    files = {}
    for p in sorted((ROOT / "autoresearch").glob("*.py")):
        if p.stem not in ("__init__", "dashboard"):
            files[f"ar_{p.stem}.py"] = ar_code(p.read_text(encoding="utf-8"))
    for m in MS:
        files[f"ms_{m}.py"] = ar_code(re.sub(r"\bfrom \.(\w+) import", r"from ms_\1 import", (ROOT / "model_selection" / f"{m}.py").read_text(encoding="utf-8")))
    files["ar_assets.py"] = ("METHOD = " + repr((ROOT / "autoresearch/METHOD.md").read_text(encoding="utf-8")) +
                             "\nDASHBOARD = " + repr((ROOT / "autoresearch/dashboard_template.html").read_text(encoding="utf-8")) + "\n")
    # запасной агент — тем же способом, что tools/build_platform.py Макса
    mods = {f.stem: ("max_cli" if f.stem == "__main__" else f"max_{f.stem}") for f in (ROOT / "agent").glob("*.py") if f.stem != "__init__"}
    for name, flat in mods.items():
        code = (ROOT / "agent" / f"{name}.py").read_text(encoding="utf-8")
        for o, r in mods.items():
            code = re.sub(r"\bfrom \." + re.escape(o) + r" import\b", f"from {r} import", code)
            code = code.replace(f"agent.{o}", r)
        files[flat + ".py"] = code.replace("Path(__file__).resolve().parents[1]", "Path(__file__).resolve().parent")
    sol = (ROOT / "solution.py").read_text(encoding="utf-8")
    files["solution.py"] = ar_code(sol).replace("from agent.__main__ import", "from max_cli import")
    for name, code in files.items():
        ast.parse(code, filename=name)
        assert "from ." not in code or name.startswith("max_"), name
        (out / name).write_text(code, encoding="utf-8")
    shutil.copy(ROOT / "requirements.txt", out / "requirements.txt")
    z = out.with_suffix(".zip")
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(out.iterdir()):
            zf.write(p, p.name)
    print(f"{z}: {len(list(out.iterdir()))} файлов, {z.stat().st_size} байт, папок нет")


if __name__ == "__main__":
    build(Path(sys.argv[1]).resolve())

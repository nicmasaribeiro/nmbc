# helpers_docs.py
import os, shutil, subprocess
from pathlib import Path
from flask import current_app
import markdown as md

def _storage_root():
    root = os.path.join(current_app.instance_path, "docs")
    Path(root).mkdir(parents=True, exist_ok=True)
    return root

def render_markdown_safe(text: str) -> str:
    return md.markdown(text or "", extensions=["fenced_code", "tables"])

def compile_latex_to_pdf(tex_source: str, out_dir: str) -> str:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tex_file = out / "document.tex"
    tex_file.write_text(tex_source, encoding="utf-8")

    # require pdflatex; fail gracefully if missing
    pdflatex = shutil.which("pdflatex")
    if not pdflatex:
        raise FileNotFoundError("pdflatex not found in PATH")

    # Run twice for refs/TOC
    for _ in range(2):
        proc = subprocess.run(
            [pdflatex, "-interaction=nonstopmode", "-halt-on-error", tex_file.name],
            cwd=out, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
        if proc.returncode != 0:
            raise RuntimeError(f"LaTeX compile failed:\n{proc.stdout}")
    pdf_path = out / "document.pdf"
    return str(pdf_path)

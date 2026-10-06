"""Build report/DATA266_Lab1_Report_Team_21.pdf from the Markdown next to it.

    cd report && python build_pdf.py

Needs pandoc and xelatex. The Markdown stays readable on GitHub; this script only adapts it for LaTeX: Times New Roman, tables with
full borders, a shaded header row and padded cells, arrows in math mode, and break points inside long code paths so they wrap.
"""
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "DATA266_Lab1_Report_Team_21.md"
OUT = HERE / "DATA266_Lab1_Report_Team_21.pdf"
HEADER = r"""
\usepackage{fontspec}
\setmainfont{Times New Roman}
\setmonofont{Courier New}[Scale=0.85]
\usepackage{etoolbox}
\usepackage{colortbl}
\newcommand{\tablesize}{\small}
\AtBeginEnvironment{longtable}{\tablesize}
\setlength{\tabcolsep}{5pt}
\renewcommand{\arraystretch}{1.3}
\setlength{\extrarowheight}{1.5pt}
\usepackage{microtype}
\setlength{\emergencystretch}{3em}
\usepackage{caption}
\captionsetup{font=small, labelfont=bf, skip=6pt}
\renewcommand{\topfraction}{0.9}
\renewcommand{\bottomfraction}{0.8}
\renewcommand{\textfraction}{0.07}
\renewcommand{\floatpagefraction}{0.75}
\usepackage{float}
\floatplacement{figure}{H}
"""
TEX_ESCAPES = {"\\": r"\textbackslash{}", "{": r"\{", "}": r"\}", "_": r"\_\allowbreak{}", "%": r"\%", "$": r"\$",
               "&": r"\&", "#": r"\#", "^": r"\^{}", "~": r"\textasciitilde{}", "/": r"/\allowbreak{}", ".": r".\allowbreak{}"}
TITLE = r"""```{=latex}
\begin{center}
{\LARGE\bfseries DATA 266 Lab 1: Team Report\par}
\vspace{10pt}
{\large PairProgramming\_Team\_21\par}
\vspace{4pt}
{\large Ayush Sunil Gawai\enspace$\cdot$\enspace Sneha Tumkur Narendra\par}
\end{center}
\vspace{8pt}
```
"""                                                              # a compact title block instead of pandoc's \maketitle
HEAD_CELL = r"\begin{minipage}[b]{\linewidth}\raggedright"


def texttt(code: str) -> str:
    """Escape a code span for LaTeX, with break points after / _ . but none at its very end (no stray ")" on a new line)."""
    out = "".join(TEX_ESCAPES.get(c, c) for c in code)
    return out[: -len(r"\allowbreak{}")] if out.endswith(r"\allowbreak{}") else out


def wide_tables(m: re.Match) -> str:
    """Tables with six or more columns get a smaller font so their cells don't wrap into tall rows."""
    table = m.group(1) + (m.group(2) or "")                    # the table and its "Table:" caption, kept together
    if m.group(1).splitlines()[0].count("|") - 1 < 6:
        return table
    return ("```{=latex}\n\\renewcommand{\\tablesize}{\\footnotesize}\\setlength{\\tabcolsep}{3pt}\n```\n\n" + table
            + "\n```{=latex}\n\\renewcommand{\\tablesize}{\\small}\\setlength{\\tabcolsep}{5pt}\n```\n")


def for_latex(md: str) -> str:
    _, yaml, body = md.split("---\n", 2)                      # the title block between the first two "---" lines
    body = re.sub(r"(?<!`)`([^`\n]+)`(?!`)",
                  lambda m: "`\\texttt{" + texttt(m.group(1)) + "}`{=latex}", body)
    body = re.sub(r"((?:^\|.*\n)+)(\nTable: .*\n)?", wide_tables, body, flags=re.M)
    body = body.replace("→", "$\\rightarrow$").replace("↔", "$\\leftrightarrow$")
    return TITLE + body


def grid(table: str) -> str:
    """One pandoc longtable -> full borders, a shaded header row and padded header cells."""
    spec_end = table.index("@{}}\n") + len("@{}}\n")
    spec, rest = table[:spec_end], table[spec_end:]
    n = spec.count("p{(")
    spec = re.sub(r"\(\\linewidth - \d+\\tabcolsep\)", rf"(\\linewidth - {2 * n}\\tabcolsep - {n + 1}\\arrayrulewidth)", spec)
    spec = spec.replace("{@{}\n", "{|\n").replace("@{}}\n", "|}\n").replace("}}\n  >{", "}}|\n  >{")
    rest = rest.replace("\\toprule\\noalign{}\n", "\\hline\n\\rowcolor{gray!15}\n").replace("\\midrule\\noalign{}\n", "\\hline\n")
    rest = rest.replace("\\bottomrule\\noalign{}\n", "")
    rest = rest.replace(HEAD_CELL + "\n", HEAD_CELL + "\\rule{0pt}{2.6ex}").replace("\n\\end{minipage}", "\\rule[-1.1ex]{0pt}{0pt}\n\\end{minipage}")
    head, body = rest.split("\\endlastfoot\n", 1)
    body = body.replace(" \\\\\n", " \\\\ \\hline\n")                # a rule under every body row
    return spec + head + "\\endlastfoot\n" + body


with tempfile.TemporaryDirectory() as tmp:
    md, hdr, tex = Path(tmp) / "report.md", Path(tmp) / "header.tex", Path(tmp) / "report.tex"
    md.write_text(for_latex(SRC.read_text(encoding="utf-8")), encoding="utf-8")
    hdr.write_text(HEADER, encoding="utf-8")
    subprocess.run(["pandoc", str(md), "-s", "-o", str(tex), "-H", str(hdr), "-V", "geometry:margin=2cm", "-V", "fontsize=11pt",
                    "-V", "colorlinks=true", "-V", "linkcolor=black", "-V", "urlcolor=blue", "--columns=110"], check=True)
    src = tex.read_text(encoding="utf-8")
    src = re.sub(r"\\begin\{longtable\}\[\]\{@\{\}.*?\\end\{longtable\}", lambda m: grid(m.group(0)), src, flags=re.S)
    tex.write_text(src, encoding="utf-8")
    for _ in range(2):                                          # twice, so longtable settles its column widths
        r = subprocess.run(["xelatex", "-interaction=nonstopmode", "-halt-on-error", f"-output-directory={tmp}", str(tex)],
                           cwd=HERE, capture_output=True, text=True)
        if r.returncode:
            raise SystemExit(r.stdout[-3000:])
    shutil.copy(Path(tmp) / "report.pdf", OUT)
print(f"wrote {OUT.name}")

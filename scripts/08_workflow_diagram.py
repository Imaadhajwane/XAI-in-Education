"""
08 - Figure 1: methodology workflow (Graphviz)
=============================================
Requires the Graphviz `dot` binary (https://graphviz.org/download/) and the
python package `graphviz`. Writes Fig01_workflow.{png,pdf,svg}.
The diagram reflects the REVISED protocol (nested repeated CV, imbalance
comparison, synthetic-data sensitivity, PREX-Edu validation).
"""
from _common import C, get_logger

import shutil
import subprocess

log = get_logger("08_workflow")

COL = {"design": ("#EEF2FF", "#4338CA"), "data": ("#ECFDF5", "#047857"), "model": ("#EFF6FF", "#1D4ED8"),
       "xai": ("#FEF2F2", "#B91C1C"), "report": ("#F7FEE7", "#4D7C0F")}


def node(name, label, kind, shape="box"):
    fill, line = COL[kind]
    return (f'  {name} [label=<{label}>, shape={shape}, style="rounded,filled", fillcolor="{fill}", '
            f'color="{line}", fontcolor="#111827", penwidth=1.3];')


lanes = {
    "design": ("1 · Research design", [
        ("d1", "<b>Problem &amp; RQ1–RQ4</b><br/><font point-size='9'>leakage-free early warning</font>"),
        ("d2", "<b>Literature review</b><br/><font point-size='9'>XAI-in-education gaps</font>"),
        ("d3", "<b>Primary data collection</b><br/><font point-size='9'>N = 1,208 students, 13 predictors</font>")]),
    "data": ("2 · Data preparation", [
        ("p1", "<b>Audit &amp; cleaning</b><br/><font point-size='9'>missing → 'Unknown', types, ranges</font>"),
        ("p2", "<b>Target construction</b><br/><font point-size='9'>At-Risk = last-term % ≤ 50 (5.5%)</font>"),
        ("p3", "<b>Leakage removal</b><br/><font point-size='9'>drop LastTerm%, Grade</font>"),
        ("p4", "<b>Encoding</b><br/><font point-size='9'>one-hot nominal · ordinal · Min–Max</font>")]),
    "model": ("3 · Modelling &amp; evaluation", [
        ("m1", "<b>Six classifiers</b><br/><font point-size='9'>LR · SVM · RF · DT · GB · XGB</font>"),
        ("m2", "<b>Nested 10×10 repeated CV</b><br/><font point-size='9'>inner GridSearch (PR-AUC) + threshold</font>"),
        ("m3", "<b>Imbalance &amp; synthetic-data<br/>sensitivity</b><br/><font point-size='9'>7 strategies · CTGAN/TVAE/Copula (train-only)</font>"),
        ("m4", "<b>Statistical comparison</b><br/><font point-size='9'>corrected t · Friedman–Nemenyi · DeLong</font>"),
        ("m5", "<b>Calibration · DCA · fairness</b>")]),
    "xai": ("4 · Explainability (PREX-Edu)", [
        ("x1", "<b>SHAP · LIME · PDP · PFI</b><br/><font point-size='9'>stability, agreement</font>"),
        ("x2", "<b>PREX-Edu aggregation</b><br/><font point-size='9'>4 pedagogical dimensions (Σ|SHAP|)</font>"),
        ("x3", "<b>Faithfulness &amp; actionability</b><br/><font point-size='9'>ablation · AOPC · counterfactuals</font>"),
        ("x4", "<b>Educator validation</b><br/><font point-size='9'>planned, powered teacher study</font>")]),
    "report": ("5 · Reporting", [
        ("r1", "<b>Results &amp; discussion</b>"), ("r2", "<b>Limitations &amp; future work</b>")]),
}
lines = ["digraph G {", '  graph [rankdir=TB, nodesep=0.35, ranksep=0.35, newrank=true, fontname="Liberation Serif", bgcolor="white", pad=0.2];',
         '  node [fontname="Liberation Serif", fontsize=11, margin="0.15,0.07", width=2.3];',
         '  edge [color="#6B7280", arrowsize=0.7, penwidth=1.1];']
for kind, (title, items) in lanes.items():
    fill, line = COL[kind]
    lines.append(f'  subgraph cluster_{kind} {{ label=<<b>{title}</b>>; fontsize=13; fontcolor="{line}"; '
                 f'style="rounded"; color="{line}"; penwidth=1.0;')
    for n, lab in items:
        lines.append("  " + node(n, lab, kind))
    for (a, _), (b, _) in zip(items, items[1:]):
        lines.append(f"    {a} -> {b};")
    lines.append("  }")
lines += ["  {rank=same; d1; p1; m1; x1; r1;}",
          "  d3 -> p1 [constraint=false];", "  p4 -> m1 [constraint=false];", "  m5 -> x1 [constraint=false];",
          "  x4 -> r1 [constraint=false];",
          '  m4 -> m2 [style=dashed, constraint=false];',
          '  x3 -> x2 [style=dashed, constraint=false];', "}"]
dot_src = "\n".join(lines)
(C.FIG_DIR / "Fig01_workflow.dot").write_text(dot_src, encoding="utf-8")

if shutil.which("dot"):
    for fmt in ["png", "pdf", "svg"]:
        args = ["dot", f"-T{fmt}", str(C.FIG_DIR / "Fig01_workflow.dot"), "-o", str(C.FIG_DIR / f"Fig01_workflow.{fmt}")]
        if fmt == "png":
            args.insert(1, "-Gdpi=400")
        subprocess.run(args, check=True)
    log.info("Figure 1 written")
else:
    log.info("Graphviz 'dot' not found - .dot source written; render at https://dreampuf.github.io/GraphvizOnline")

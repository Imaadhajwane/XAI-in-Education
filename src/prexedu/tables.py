"""
Table export: every table is written as CSV (machine-readable), LaTeX
(booktabs) and Markdown, and appended to a registry that the final step turns
into a single formatted Excel workbook + Word compendium.
"""
from __future__ import annotations

import json

import pandas as pd

from . import config as C

REGISTRY = C.TAB_DIR / "_registry.json"


def save_table(df: pd.DataFrame, name: str, caption: str, note: str = "", index: bool = False):
    df.to_csv(C.TAB_DIR / f"{name}.csv", index=index)
    try:
        latex = df.to_latex(index=index, escape=True, caption=caption, label=f"tab:{name}",
                            position="htbp")
        latex = latex.replace("\\begin{table}", "\\begin{table}\n\\small")
        (C.TAB_DIR / f"{name}.tex").write_text(latex, encoding="utf-8")
    except Exception:
        pass
    try:
        md = f"**{caption}**\n\n" + df.to_markdown(index=index) + (f"\n\n*Note.* {note}" if note else "")
        (C.TAB_DIR / f"{name}.md").write_text(md, encoding="utf-8")
    except Exception:
        pass
    reg = json.loads(REGISTRY.read_text()) if REGISTRY.exists() else {}
    reg[name] = {"caption": caption, "note": note, "index": index}
    REGISTRY.write_text(json.dumps(reg, indent=2))
    return df


def r(x, nd=3):
    """Round helper that leaves strings alone."""
    try:
        return round(float(x), nd)
    except Exception:
        return x


def mean_sd(a, nd=3, pct=False):
    import numpy as np
    a = np.asarray(a, dtype=float)
    k = 100 if pct else 1
    return f"{k * np.nanmean(a):.{nd}f} ± {k * np.nanstd(a, ddof=1):.{nd}f}"

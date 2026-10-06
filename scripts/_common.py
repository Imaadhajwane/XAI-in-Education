"""Shared bootstrap for every script: put ``src`` on the path, set logging."""
import logging
import os
import sys
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")
for _s in (sys.stdout, sys.stderr):           # Windows consoles: avoid UnicodeEncodeError on ✓ ▶ τ
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
os.environ.setdefault("PYTHONWARNINGS", "ignore")  # silence joblib workers too

from prexedu import config as C  # noqa: E402


def get_logger(name: str) -> logging.Logger:
    log = logging.getLogger(name)
    if not log.handlers:
        log.setLevel(logging.INFO)
        fmt = logging.Formatter("%(asctime)s | %(name)s | %(message)s", "%H:%M:%S")
        for h in (logging.StreamHandler(sys.stdout),
                  logging.FileHandler(C.LOG_DIR / f"{name}.log", mode="w", encoding="utf-8")):
            h.setFormatter(fmt)
            log.addHandler(h)
    if C.FAST:
        log.info("*** FAST MODE: smoke-test settings - do NOT report these numbers ***")
    return log


class Timer:
    def __init__(self, log, what):
        self.log, self.what = log, what

    def __enter__(self):
        self.t = time.time()
        self.log.info(f"▶ {self.what}")

    def __exit__(self, *a):
        self.log.info(f"✓ {self.what} ({time.time() - self.t:.1f}s)")

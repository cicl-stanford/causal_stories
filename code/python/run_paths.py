import os
import sys
import time

# Limit BLAS threads and use a dummy display before numpy/pygame load.
for _key in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_key, "1")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")

PHYSICS_DIR = os.path.join(DATA_DIR, "physics_simulations")
KDE_DIR = os.path.join(DATA_DIR, "kde_results")
BESTFIT_DIR = os.path.join(DATA_DIR, "bestfit")
RUN_DIR = os.path.join(DATA_DIR, "grid_search")
FIGURES_DIR = os.path.join(SCRIPT_DIR, "figures")
STATUS_PATH = os.path.join(DATA_DIR, "STATUS")
PROGRESS_LOG_PATH = os.path.join(DATA_DIR, "progress.log")

_last_status_write = 0.0
_last_progress_bucket = -1


def ensure_run_dirs():
    for path in (DATA_DIR, PHYSICS_DIR, KDE_DIR, BESTFIT_DIR, RUN_DIR, FIGURES_DIR):
        os.makedirs(path, exist_ok=True)
    return DATA_DIR


def write_run_status(line, force=False, percent=None):
    """Overwrite STATUS with the current line. Append progress.log every 5%."""
    global _last_status_write, _last_progress_bucket
    text = str(line).replace("\r", "").strip()
    if not text:
        return
    now = time.time()
    if not force and now - _last_status_write < 0.4:
        return
    _last_status_write = now
    os.makedirs(DATA_DIR, exist_ok=True)
    tmp = STATUS_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        handle.write(text + "\n")
    os.replace(tmp, STATUS_PATH)
    if percent is None:
        return
    bucket = min(100, int(percent // 5) * 5)
    if bucket == _last_progress_bucket and not force:
        return
    _last_progress_bucket = bucket
    with open(PROGRESS_LOG_PATH, "a", encoding="utf-8") as handle:
        handle.write(text + "\n")


_progress_width = 0


def _fmt_duration(seconds):
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}m"
    return f"{seconds / 3600:.1f}h"


def print_progress(current, total, start_time, label="progress", extra="", done_this_run=None):
    """One terminal line: label  [bar]  n/N  pct%  eta Xs."""
    global _progress_width
    if total <= 0:
        return
    current = max(0, min(int(current), int(total)))
    elapsed = time.time() - start_time
    counted = current if done_this_run is None else max(0, int(done_this_run))
    remaining = total - current
    if remaining <= 0:
        tail = _fmt_duration(elapsed)
    elif counted > 0:
        tail = f"eta {_fmt_duration(elapsed / counted * remaining)}"
    else:
        tail = "eta --"
    pct = int(round(100.0 * current / total))
    width = 16
    filled = int(width * current / total)
    bar = "#" * filled + "-" * (width - filled)
    extra_text = f"  {extra}" if extra else ""
    line = f"{label}  [{bar}]  {current}/{total}  {pct}%  {tail}{extra_text}"
    pad = max(_progress_width, len(line))
    sys.stdout.write("\r" + line.ljust(pad))
    sys.stdout.flush()
    _progress_width = len(line)
    write_run_status(line, force=(current >= total), percent=100.0 * current / total)
    if current >= total:
        sys.stdout.write("\n")
        _progress_width = 0

"""Shared config resolution, tag-folder, and timing helpers for mixture_of_trends_following."""
import json
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

STUDY_NAME = "mixture_of_trends_following"
STUDY_DIR = Path(__file__).resolve().parent

# cache column layout (spec 4.4)
COL_TS = 0
SLOPE_COLS = (1, 2, 3, 4)
COL_SLOPES = 5
IMB_COLS = (6, 7, 8, 9)
COL_IMBALANCES = 10
COL_TRIGGER = 11
LABEL_COLS = (12, 13, 14, 15)
COL_LSLOPES = 16
N_COLS = 17


def cli_tag(argv: list) -> str:
    return argv[1] if len(argv) > 1 else "default"


def data_dir(tag: str) -> Path:
    """data/mixture_of_trends_following/{tag}/, created if missing."""
    d = Path("data") / STUDY_NAME / tag
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_params(tag: str = "default") -> dict:
    """Tag-folder config.json if present, else the study-folder config.json. Adds params['tag']."""
    tag_cfg = data_dir(tag) / "config.json"
    cfg_path = tag_cfg if tag_cfg.exists() else STUDY_DIR / "config.json"
    params = json.loads(cfg_path.read_text())
    params["tag"] = tag
    return params


def assets_of(params: dict) -> list:
    a = params["assets"]
    return a if isinstance(a, list) else [a]


def parse_ts_ms(value: str) -> int:
    """'YYYY-MM-DD HH:MM:SS' (UTC) -> epoch ms."""
    dt = datetime.strptime(value, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def window_label(w: int) -> str:
    """10080 -> '7d', 1440 -> '1d', 240 -> '4h', 60 -> '1h', else '{w}m'."""
    if w % 1440 == 0 and w >= 1440:
        return f"{w // 1440}d"
    if w % 60 == 0 and w >= 60:
        return f"{w // 60}h"
    return f"{w}m"


@contextmanager
def timed(label: str):
    t0 = time.perf_counter()
    yield
    print(f"{label}: {time.perf_counter() - t0:.2f}s")

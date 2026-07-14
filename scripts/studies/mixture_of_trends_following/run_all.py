"""Full pipeline: build -> report -> draw. Build step uses multiprocessing across assets
when len(assets) > 1 (assets are independent). Run: python3 run_all.py [tag]"""
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

from scripts.studies.mixture_of_trends_following.build_cache import build_asset_cache
from scripts.studies.mixture_of_trends_following.common import assets_of, cli_tag, load_params, timed
from scripts.studies.mixture_of_trends_following.draw import draw
from scripts.studies.mixture_of_trends_following.report import report


def run_build(params: dict) -> None:
    assets = assets_of(params)
    if len(assets) > 1:
        with timed("build (parallel)"):
            with ProcessPoolExecutor(max_workers=len(assets)) as ex:
                list(ex.map(build_asset_cache, assets, [params] * len(assets)))
    else:
        with timed("build"):
            for asset in assets:
                build_asset_cache(asset, params)


def run_all(params: dict) -> None:
    run_build(params)
    report(params)
    draw(params)


if __name__ == "__main__":
    tag = cli_tag(sys.argv)
    run_all(load_params(tag))

"""Reproduce every number and figure in the report.

    python run_all.py --data-dir /path/to/ml-1m

Outputs go to results/ (see README.md). Run with --help for the options.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import logging
import sys
from pathlib import Path

from recsys.config import load_config
from recsys.data import download, load_movielens
from recsys.data.movielens import FILES
from recsys.experiments import PARTS, RunContext
from recsys.utils import REPO_ROOT, OutputPaths, environment_info, seed_everything, setup_logging, timed, write_json

DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"
TEMPLATE = REPO_ROOT / "templates" / "Recommender_Systems_Results_Template.xlsm"
WORKBOOK_NAME = "Recommender_Systems_Results_Group2.xlsm"

log = logging.getLogger("run_all")


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the MovieLens 1M experiments and write all results.")
    parser.add_argument(
        "--data-dir", type=Path, default=REPO_ROOT / "data" / "ml-1m",
        help="folder that contains ratings.dat, movies.dat and users.dat (default: data/ml-1m)",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="YAML config (default: configs/default.yaml)")
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="where to write results (default: results/, or results-quick/ with --quick)",
    )
    parser.add_argument(
        "--parts", type=int, nargs="+", choices=sorted(PARTS), default=sorted(PARTS),
        help="project parts to run (default: all implemented parts)",
    )
    parser.add_argument("--download", action="store_true", help="download MovieLens 1M from GroupLens if it is missing")
    parser.add_argument(
        "--quick", action="store_true",
        help="smaller search grids to check that everything runs; these numbers are not used in the report",
    )
    return parser.parse_args(argv)


def data_checksums(data_dir: Path) -> dict[str, str]:
    return {name: hashlib.md5((data_dir / name).read_bytes()).hexdigest() for name in FILES}


def _repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.name


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = args.output_dir or REPO_ROOT / ("results-quick" if args.quick else "results")
    out = OutputPaths(output_dir.resolve()).create()
    setup_logging(out.logs / "run_all.log")

    config = load_config(args.config)
    seed_everything(config.seed)
    started = dt.datetime.now(dt.timezone.utc)

    if args.download and not (args.data_dir / "ratings.dat").is_file():
        download(args.data_dir)
    timings: dict[str, float] = {}
    try:
        with timed("loading data", timings):
            data = load_movielens(args.data_dir)
    except FileNotFoundError as err:
        log.error("%s", err)
        return 2

    ctx = RunContext(config=config, data=data, out=out, quick=args.quick, timings=timings)
    for part in args.parts:
        with timed(f"part {part}", timings):
            PARTS[part](ctx)

    ctx.results.write_csv(out.root / "results.csv")
    ctx.results.write_workbook(TEMPLATE, out.root / WORKBOOK_NAME, config.team)
    # Local paths are left out on purpose: run_info.json is committed, and the
    # MD5 sums identify the data better than a folder name would.
    write_json(out.root / "run_info.json", {
        "config_file": _repo_relative(args.config),
        "parts": args.parts,
        "quick": args.quick,
        "seed": config.seed,
        "started_utc": started.isoformat(timespec="seconds"),
        "finished_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "timings_seconds": timings,
        "data_md5": data_checksums(args.data_dir),
        "environment": environment_info(),
        "config": config.to_dict(),
    })
    log.info("Wrote %d result rows to %s", len(ctx.results), out.root)
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Small helpers: output folders, logging, seeding and run metadata."""

from __future__ import annotations

import json
import logging
import platform
import random
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd

log = logging.getLogger("recsys")

REPO_ROOT = Path(__file__).resolve().parent.parent
TRACKED_PACKAGES = ("numpy", "pandas", "scipy", "matplotlib", "openpyxl", "PyYAML", "scikit-learn")


@dataclass(frozen=True)
class OutputPaths:
    root: Path

    @property
    def figures(self) -> Path:
        return self.root / "figures"

    @property
    def tables(self) -> Path:
        return self.root / "tables"

    @property
    def logs(self) -> Path:
        return self.root / "logs"

    def create(self) -> OutputPaths:
        for folder in (self.root, self.figures, self.tables, self.logs):
            folder.mkdir(parents=True, exist_ok=True)
        return self


def setup_logging(log_file: Path | None = None, level: int = logging.INFO) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file is not None:
        handlers.append(logging.FileHandler(log_file, mode="w", encoding="utf-8"))
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers,
        force=True,
    )


def seed_everything(seed: int) -> None:
    """Seed the global generators. Code that needs randomness should still
    create its own np.random.default_rng(seed) and pass it along."""
    random.seed(seed)
    np.random.seed(seed)


@contextmanager
def timed(label: str, timings: dict[str, float] | None = None) -> Iterator[None]:
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    log.info("%s took %.1f s", label, elapsed)
    if timings is not None:
        timings[label] = round(elapsed, 2)


def git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        )
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.stdout.strip() + ("-dirty" if dirty.stdout.strip() else "")


def environment_info() -> dict[str, Any]:
    versions = {}
    for name in TRACKED_PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": versions,
        "git_commit": git_commit(),
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8", newline="\n")


def save_table(df: pd.DataFrame, path: Path, **kwargs: Any) -> None:
    """Write a CSV with LF line endings on every platform."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, lineterminator="\n", **kwargs)

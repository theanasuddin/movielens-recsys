"""Experiment configuration, read from a YAML file into frozen dataclasses."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class TeamConfig:
    name: str = "Group 2"
    members: tuple[str, ...] = ()
    repository: str = ""
    part3_track: str = ""


@dataclass(frozen=True)
class SplitConfig:
    mode: str = "calendar"
    val_start: str = "2001-01-01"
    test_start: str = "2001-07-01"
    quantiles: tuple[float, float] = (0.8, 0.9)


@dataclass(frozen=True)
class FilterConfig:
    min_item_train_ratings: int = 1
    min_user_history: int = 1


@dataclass(frozen=True)
class EvaluationConfig:
    k: int = 10
    relevance_threshold: int = 4
    bootstrap_samples: int = 1000
    confidence: float = 0.95
    batch_size: int = 1024


@dataclass(frozen=True)
class Config:
    seed: int = 42
    team: TeamConfig = field(default_factory=TeamConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    filtering: FilterConfig = field(default_factory=FilterConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    # Model grids and other per-part settings stay as plain dicts. Each part's
    # runner reads what it needs from here.
    part1: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_SECTIONS = {
    "team": TeamConfig,
    "split": SplitConfig,
    "filtering": FilterConfig,
    "evaluation": EvaluationConfig,
}


def _section(cls: type, values: dict[str, Any] | None) -> Any:
    values = dict(values or {})
    allowed = {f.name for f in fields(cls)}
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError(f"Unknown keys in config section '{cls.__name__}': {unknown}")
    for key, value in values.items():
        if isinstance(value, list):
            values[key] = tuple(value)
    return cls(**values)


def load_config(path: str | Path) -> Config:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    allowed = {f.name for f in fields(Config)}
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(f"Unknown top-level config keys: {unknown}")

    kwargs: dict[str, Any] = {name: _section(cls, raw.get(name)) for name, cls in _SECTIONS.items()}
    kwargs["seed"] = int(raw.get("seed", Config.seed))
    kwargs["part1"] = dict(raw.get("part1") or {})
    config = Config(**kwargs)

    if config.split.mode not in {"calendar", "quantile"}:
        raise ValueError(f"split.mode must be 'calendar' or 'quantile', got {config.split.mode!r}")
    return config

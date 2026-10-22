"""State shared by the experiment runners of all three parts."""

from __future__ import annotations

from dataclasses import dataclass, field

from recsys.config import Config
from recsys.data import IdMap, MovieLens, TemporalSplit, temporal_split
from recsys.reporting import ResultsTable
from recsys.utils import OutputPaths


@dataclass
class RunContext:
    config: Config
    data: MovieLens
    out: OutputPaths
    results: ResultsTable = field(default_factory=ResultsTable)
    quick: bool = False
    timings: dict[str, float] = field(default_factory=dict)
    _split: TemporalSplit | None = None
    _ids: IdMap | None = None

    @property
    def ids(self) -> IdMap:
        if self._ids is None:
            self._ids = IdMap.from_movielens(self.data)
        return self._ids

    @property
    def split(self) -> TemporalSplit:
        """The main temporal split from the config, computed once and shared."""
        if self._split is None:
            self._split = temporal_split(self.data.ratings, self.config.split)
        return self._split

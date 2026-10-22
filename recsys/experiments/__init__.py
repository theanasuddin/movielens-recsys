"""One runner per project part. run_all.py calls them in order."""

from recsys.experiments import part1
from recsys.experiments.context import RunContext

PARTS = {1: part1.run}

__all__ = ["PARTS", "RunContext"]

"""
Utilities for loading Solitaire training data from the Java engine logs.

The Java engine (in the `cards` repo) emits lines like:

    EPISODE_STEP {"type":"step", ...}
    EPISODE_SUMMARY {"type":"summary", ...}

This module parses those lines into Python dictionaries so that other
modules can:

* Turn board states into tensors.
* Build supervised or RL datasets for model training.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional
import json


EPISODE_STEP_PREFIX = "EPISODE_STEP "
EPISODE_SUMMARY_PREFIX = "EPISODE_SUMMARY "


@dataclass
class EpisodeStep:
    """Single logged step in a Solitaire game."""

    raw: dict

    @property
    def game_index(self) -> Optional[int]:
        value = self.raw.get("game_index")
        return int(value) if value is not None else None

    @property
    def solver(self) -> str:
        return str(self.raw.get("solver", "unknown"))

    @property
    def step_index(self) -> int:
        return int(self.raw.get("step_index", -1))

    @property
    def chosen_command(self) -> str:
        return str(self.raw.get("chosen_command", ""))

    @property
    def legal_moves(self) -> List[str]:
        return list(self.raw.get("legal_moves", []))

    @property
    def recommended_moves(self) -> List[str]:
        return list(self.raw.get("recommended_moves", []))

    @property
    def tableau_visible(self) -> List[List[str]]:
        return list(self.raw.get("tableau_visible", []))

    @property
    def tableau_face_down(self) -> List[int]:
        return list(self.raw.get("tableau_face_down", []))

    @property
    def foundation(self) -> List[List[str]]:
        return list(self.raw.get("foundation", []))

    @property
    def talon(self) -> List[str]:
        return list(self.raw.get("talon", []))

    @property
    def stock_size(self) -> int:
        return int(self.raw.get("stock_size", 0))


@dataclass
class EpisodeSummary:
    """High-level summary of a single Solitaire game."""

    raw: dict

    @property
    def game_index(self) -> Optional[int]:
        value = self.raw.get("game_index")
        return int(value) if value is not None else None

    @property
    def solver(self) -> str:
        return str(self.raw.get("solver", "unknown"))

    @property
    def won(self) -> bool:
        return bool(self.raw.get("won", False))

    @property
    def iterations(self) -> int:
        return int(self.raw.get("iterations", 0))

    @property
    def successful_moves(self) -> int:
        return int(self.raw.get("successful_moves", 0))


@dataclass
class Episode:
    """Full episode: ordered steps plus a final summary."""

    steps: List[EpisodeStep]
    summary: Optional[EpisodeSummary]


def _parse_json_after_prefix(line: str, prefix: str) -> Optional[dict]:
    """Strip the prefix and parse the JSON payload, if present."""
    if not line.startswith(prefix):
        return None
    payload = line[len(prefix) :].strip()
    if not payload:
        return None
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        return None


def iter_log_events(lines: Iterable[str]) -> Iterable[tuple[str, dict]]:
    """
    Yield (event_type, payload_dict) for each parsed log event.

    event_type is either "step" or "summary".
    """
    for line in lines:
        line = line.rstrip("\n")
        if EPISODE_STEP_PREFIX in line:
            _, _, tail = line.partition(EPISODE_STEP_PREFIX)
            payload = _parse_json_after_prefix(EPISODE_STEP_PREFIX + tail, EPISODE_STEP_PREFIX)
            if payload is not None:
                yield "step", payload
        elif EPISODE_SUMMARY_PREFIX in line:
            _, _, tail = line.partition(EPISODE_SUMMARY_PREFIX)
            payload = _parse_json_after_prefix(EPISODE_SUMMARY_PREFIX + tail, EPISODE_SUMMARY_PREFIX)
            if payload is not None:
                yield "summary", payload


def load_episodes_from_log(path: Path | str) -> List[Episode]:
    """
    Parse a log file produced by the Java engine into a list of episodes.

    The current implementation assumes that:
    * Steps for a given game appear in order.
    * Each game ends with a single summary event.
    * Games are not interleaved.
    """
    p = Path(path)
    steps: List[EpisodeStep] = []
    summary: Optional[EpisodeSummary] = None
    episodes: List[Episode] = []

    with p.open("r", encoding="utf-8") as f:
        for event_type, payload in iter_log_events(f):
            if event_type == "step":
                steps.append(EpisodeStep(payload))
            elif event_type == "summary":
                summary = EpisodeSummary(payload)
                episodes.append(Episode(steps=steps, summary=summary))
                steps = []
                summary = None

    # If the file ended without a summary, keep the partial episode.
    if steps:
        episodes.append(Episode(steps=steps, summary=summary))

    return episodes


__all__ = [
    "EpisodeStep",
    "EpisodeSummary",
    "Episode",
    "iter_log_events",
    "load_episodes_from_log",
]


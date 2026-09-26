"""Analyze LLM Solitaire episode logs using the screening rubric.

The tool uses only the Python standard library. It emits objective per-game and aggregate metrics,
then applies the documented heuristic to suggest one of the four screening ratings. A reviewer
must confirm or override that suggestion in the research notebook.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_JSON_OUTPUT = REPO_ROOT / "experiments" / "runtime" / "work" / "llm_episode_analysis.json"
DEFAULT_MARKDOWN_OUTPUT = REPO_ROOT / "experiments" / "runtime" / "work" / "llm_episode_analysis.md"
MARKERS = ("EPISODE_GAME ", "EPISODE_STEP ", "EPISODE_SUMMARY ")


@dataclass
class GameRecords:
    """Records associated with one game in log order."""

    run_id: str | None
    solver: str
    game_index: int
    game_total: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    steps: list[dict[str, Any]] = field(default_factory=list)
    summary: dict[str, Any] | None = None


def _extract_record(line: str) -> dict[str, Any] | None:
    for marker in MARKERS:
        marker_index = line.find(marker)
        if marker_index >= 0:
            return json.loads(line[marker_index + len(marker) :])
    return None


def load_records(paths: Iterable[Path]) -> list[dict[str, Any]]:
    """Load episode JSON objects while tolerating prefixes from mixed log encoders."""
    records: list[dict[str, Any]] = []
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not any(marker in line for marker in MARKERS):
                    continue
                try:
                    record = _extract_record(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid episode JSON in {path}:{line_number}: {exc}") from exc
                if record is not None:
                    records.append(record)
    return records


def group_games(records: Iterable[dict[str, Any]]) -> list[GameRecords]:
    """Associate ordered metadata, step, and summary records with game instances."""
    games: list[GameRecords] = []
    active: dict[tuple[str | None, str, int], GameRecords] = {}

    for record in records:
        run_id = record.get("run_id")
        solver = str(record.get("solver", "unknown"))
        game_index = int(record.get("game_index", 1))
        key = (run_id, solver, game_index)
        record_type = record.get("type")

        if record_type == "game":
            game = GameRecords(
                run_id=run_id,
                solver=solver,
                game_index=game_index,
                game_total=record.get("game_total"),
                metadata=dict(record.get("metadata") or {}),
            )
            games.append(game)
            active[key] = game
            continue

        game = active.get(key)
        if game is None:
            game = GameRecords(
                run_id=run_id,
                solver=solver,
                game_index=game_index,
                game_total=record.get("game_total"),
            )
            games.append(game)
            active[key] = game

        if record_type == "step":
            game.steps.append(record)
        elif record_type == "summary":
            game.summary = record
            active.pop(key, None)

    return games


def _foundation_count(step: dict[str, Any]) -> int:
    return sum(len(pile) for pile in step.get("foundation", []))


def _face_down_count(step: dict[str, Any]) -> int:
    return sum(int(count) for count in step.get("tableau_face_down", []))


def _move_endpoints(command: str) -> tuple[str, str] | None:
    parts = command.split()
    if len(parts) not in (3, 4) or parts[0].lower() != "move":
        return None
    return parts[1], parts[-1]


def analyze_game(game: GameRecords) -> dict[str, Any]:
    """Calculate objective progress and coherence metrics for one game."""
    state_keys = [step.get("state_key") for step in game.steps if step.get("state_key") is not None]
    commands = [str(step.get("chosen_command", "")) for step in game.steps]
    foundation_counts = [_foundation_count(step) for step in game.steps]
    face_down_counts = [_face_down_count(step) for step in game.steps]
    summary = game.summary or {}

    final_foundation = int(summary.get("final_foundation_cards", foundation_counts[-1] if foundation_counts else 0))
    final_face_down = int(summary.get("final_tableau_face_down", face_down_counts[-1] if face_down_counts else 28))
    max_foundation = max([final_foundation, *foundation_counts])
    minimum_face_down = min([final_face_down, *face_down_counts])
    initial_face_down = face_down_counts[0] if face_down_counts else 28
    repeated_states = len(state_keys) - len(set(state_keys))

    immediate_reversals = 0
    previous_endpoints: tuple[str, str] | None = None
    for command in commands:
        endpoints = _move_endpoints(command)
        if endpoints is not None and previous_endpoints == (endpoints[1], endpoints[0]):
            immediate_reversals += 1
        previous_endpoints = endpoints

    illegal_selections = sum(
        1
        for step in game.steps
        if step.get("chosen_command") not in step.get("legal_moves", [])
    )
    repetition_ratio = repeated_states / len(state_keys) if state_keys else 0.0

    return {
        "run_id": game.run_id,
        "solver": game.solver,
        "game_index": game.game_index,
        "game_total": game.game_total,
        "metadata": game.metadata,
        "completed": game.summary is not None,
        "won": bool(summary.get("won", False)),
        "termination_reason": summary.get("termination_reason", "unknown"),
        "iterations": int(summary.get("iterations", len(game.steps))),
        "successful_moves": int(summary.get("successful_moves", 0)),
        "duration_seconds": float(summary.get("duration_nanos", 0)) / 1_000_000_000,
        "recorded_steps": len(game.steps),
        "unique_states": len(set(state_keys)),
        "repeated_states": repeated_states,
        "repetition_ratio": repetition_ratio,
        "immediate_reversals": immediate_reversals,
        "stock_turns": sum(command == "turn" for command in commands),
        "foundation_to_tableau_moves": sum(command.startswith("move F") for command in commands),
        "illegal_selections": illegal_selections,
        "maximum_foundation_cards": max_foundation,
        "final_foundation_cards": final_foundation,
        "initial_tableau_face_down": initial_face_down,
        "minimum_tableau_face_down": minimum_face_down,
        "tableau_cards_revealed": max(0, initial_face_down - minimum_face_down),
        "strong_late_position": max_foundation >= 24 or minimum_face_down <= 3,
        "pathological_repetition": len(state_keys) >= 20 and repetition_ratio >= 0.30,
    }


def _median(games: list[dict[str, Any]], field_name: str) -> float:
    values = [float(game[field_name]) for game in games]
    return statistics.median(values) if values else 0.0


def suggest_rating(games: list[dict[str, Any]], expected_games: int) -> tuple[str, str]:
    """Apply the ordered thresholds from LLM_EVALUATION.md."""
    completed = [game for game in games if game["completed"]]
    if expected_games <= 0:
        return "technical_failure", "No games were found for the selected run."
    required = math.ceil(expected_games * 0.80)
    if len(completed) < required:
        return (
            "technical_failure",
            f"Only {len(completed)} of {expected_games} planned games have summaries; {required} are required.",
        )

    wins = sum(game["won"] for game in completed)
    strong_positions = sum(game["strong_late_position"] for game in completed)
    if wins >= 1 or strong_positions >= math.ceil(len(completed) / 2):
        return (
            "promising",
            f"The batch recorded {wins} win(s) and {strong_positions} strong late position(s).",
        )

    pathological = sum(game["pathological_repetition"] for game in completed)
    low_progress = (
        _median(completed, "maximum_foundation_cards") <= 4
        and _median(completed, "tableau_cards_revealed") <= 4
    )
    if wins == 0 and (low_progress or pathological >= math.ceil(len(completed) / 2)):
        return (
            "not_viable",
            "No wins and either low median progress or pathological repetition in at least half the games.",
        )

    return (
        "needs_prompting",
        "The batch is technically valid and shows progress, but it did not meet the promising threshold.",
    )


def analyze_records(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Build a complete single-batch analysis payload."""
    games = [analyze_game(game) for game in group_games(records)]
    expected_games = max(
        [int(game["game_total"]) for game in games if game.get("game_total") is not None]
        or [len(games)]
    )
    completed = [game for game in games if game["completed"]]
    rating, rationale = suggest_rating(games, expected_games)
    metadata = next((game["metadata"] for game in games if game["metadata"]), {})

    aggregate = {
        "expected_games": expected_games,
        "observed_games": len(games),
        "completed_games": len(completed),
        "wins": sum(game["won"] for game in completed),
        "completion_percent": 100.0 * len(completed) / expected_games if expected_games else 0.0,
        "median_maximum_foundation_cards": _median(completed, "maximum_foundation_cards"),
        "median_tableau_cards_revealed": _median(completed, "tableau_cards_revealed"),
        "median_repetition_ratio": _median(completed, "repetition_ratio"),
        "pathological_repetition_games": sum(game["pathological_repetition"] for game in completed),
        "strong_late_position_games": sum(game["strong_late_position"] for game in completed),
        "total_immediate_reversals": sum(game["immediate_reversals"] for game in completed),
        "total_illegal_selections": sum(game["illegal_selections"] for game in completed),
    }
    return {
        "rubric_version": "1",
        "metadata": metadata,
        "suggested_rating": rating,
        "rating_rationale": rationale,
        "aggregate": aggregate,
        "games": games,
    }


def render_markdown(report: dict[str, Any]) -> str:
    """Render a compact report suitable for linking from the research notebook."""
    aggregate = report["aggregate"]
    metadata = report["metadata"]
    lines = [
        "# LLM Episode Analysis",
        "",
        f"- **Model:** {metadata.get('model', 'unknown')}",
        f"- **Provider:** {metadata.get('provider', 'unknown')}",
        f"- **Prompt:** {metadata.get('prompt_version', 'unknown')}",
        f"- **Suggested rating:** `{report['suggested_rating']}`",
        f"- **Rationale:** {report['rating_rationale']}",
        "",
        "## Batch Summary",
        "",
        "| Planned | Completed | Wins | Median max foundation | Median cards revealed | Pathological repetition |",
        "|---:|---:|---:|---:|---:|---:|",
        (
            f"| {aggregate['expected_games']} | {aggregate['completed_games']} | {aggregate['wins']} | "
            f"{aggregate['median_maximum_foundation_cards']:.1f} | "
            f"{aggregate['median_tableau_cards_revealed']:.1f} | "
            f"{aggregate['pathological_repetition_games']} |"
        ),
        "",
        "## Games",
        "",
        "| Game | Result | End | Steps | Max foundation | Revealed | Repeat % | Reversals |",
        "|---:|---|---|---:|---:|---:|---:|---:|",
    ]
    for game in report["games"]:
        result = "win" if game["won"] else "loss"
        lines.append(
            f"| {game['game_index']} | {result} | {game['termination_reason']} | "
            f"{game['recorded_steps']} | {game['maximum_foundation_cards']} | "
            f"{game['tableau_cards_revealed']} | {game['repetition_ratio'] * 100:.1f}% | "
            f"{game['immediate_reversals']} |"
        )
    lines.extend(
        [
            "",
            "## Human Review",
            "",
            "- **Confirmed rating:**",
            "- **Representative games:**",
            "- **Dominant failure mode:**",
            "- **Evidence:**",
            "- **Next action:**",
            "",
            "Use `experiments/LLM_EVALUATION.md` for the qualitative checklist and override rules.",
        ]
    )
    return "\n".join(lines) + "\n"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode_logs", nargs="+", type=Path)
    parser.add_argument("--run-id", help="analyze one run from append-only episode logs")
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    parser.add_argument("--markdown-output", type=Path, default=DEFAULT_MARKDOWN_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    records = load_records(args.episode_logs)
    if args.run_id:
        records = [record for record in records if record.get("run_id") == args.run_id]
    else:
        run_ids = {record.get("run_id") for record in records if record.get("run_id") is not None}
        if len(run_ids) > 1:
            raise SystemExit(
                "Episode logs contain multiple run IDs; select one with --run-id: "
                + ", ".join(sorted(run_ids))
            )
    report = analyze_records(records)
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    args.markdown_output.write_text(render_markdown(report), encoding="utf-8")
    print(f"Wrote {args.json_output}")
    print(f"Wrote {args.markdown_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

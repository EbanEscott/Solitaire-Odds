#!/usr/bin/env python3
"""Run a recorded, search-only matched-deal benchmark without external services."""

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys

ENGINE = Path(__file__).resolve().parents[1]
PLAYERS = ("rule", "greedy", "hill", "beam", "monteCarlo", "astar")


def positive(value):
    """Parse a strictly positive game count or iteration cap."""
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def summarize(output, expected_players, games):
    """Validate complete matched results and write a compact machine-readable summary."""
    with (output / "games.csv").open() as source:
        rows = list(csv.DictReader(source))
    summary = {}
    for player in expected_players:
        selected = sorted((row for row in rows if row["player"] == player), key=lambda row: int(row["game"]))
        if [int(row["game"]) for row in selected] != list(range(1, games + 1)):
            raise ValueError(f"Incomplete or duplicate results for {player}")
        wins = sum(row["won"] == "true" for row in selected)
        seconds = sum(float(row["seconds"]) for row in selected)
        streak = best = 0
        terminations = {}
        for row in selected:
            streak = streak + 1 if row["won"] == "true" else 0
            best = max(best, streak)
            reason = row["termination"]
            terminations[reason] = terminations.get(reason, 0) + 1
        summary[player] = dict(games=games, wins=wins, winPercent=100 * wins / games,
                               totalSeconds=seconds, averageSeconds=seconds / games,
                               averageMoves=sum(int(row["moves"]) for row in selected) / games,
                               bestWinStreak=best, terminations=terminations)
    if len(rows) != games * len(expected_players):
        raise ValueError("Unexpected result rows")
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main():
    """Capture configuration/source evidence, run one opt-in test, and validate output."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=positive, default=100)
    parser.add_argument("--players", nargs="+", choices=PLAYERS, default=list(PLAYERS))
    parser.add_argument("--draw-count", type=int, choices=(1, 3), default=3)
    parser.add_argument("--stock-visibility", choices=("topOnly", "allDrawn"), default="topOnly")
    parser.add_argument("--max-stock-passes", choices=("1", "3", "null"), default="null")
    parser.add_argument("--thoughtful", action="store_true")
    parser.add_argument("--max-moves", type=positive, default=1000)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--player-seed", type=int, default=20261017)
    parser.add_argument("--property", action="append", default=[], metavar="KEY=VALUE",
                        help="A* or Monte Carlo search override (astar.* or montecarlo.*)")
    parser.add_argument("--output", type=Path, help="fresh output directory; existing directories are rejected")
    args = parser.parse_args()
    if len(set(args.players)) != len(args.players):
        parser.error("players must be unique")
    for seed in (args.seed_start, args.player_seed):
        if not -(2**63) <= seed <= 2**63 - args.games:
            parser.error("seed range exceeds signed Java long")
    overrides = {}
    for item in args.property:
        key, separator, value = item.partition("=")
        if not separator or not key.startswith(("astar.", "montecarlo.")) or key in overrides:
            parser.error("each property must be a unique astar.* or montecarlo.* KEY=VALUE")
        overrides[key] = value
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = (args.output or ENGINE / "logs" / "profiles" / f"search-sweep-{stamp}").resolve()
    output.mkdir(parents=True, exist_ok=False)
    run_id = f"search-sweep-{stamp}"
    properties = {
        "search.sweep": "true", "search.output": str(output),
        "search.players": ",".join(args.players), "test.games": str(args.games),
        "test.max.moves.per.game": str(args.max_moves), "search.seedStart": str(args.seed_start),
        "search.playerSeed": str(args.player_seed), "game.drawCount": str(args.draw_count),
        "game.stockVisibility": args.stock_visibility, "game.maxStockPasses": args.max_stock_passes,
        "game.thoughtful": str(args.thoughtful).lower(), "training.mode": "false",
        "guidance.mode": "false", "log.episodes": "false", "experiment.run.id": run_id,
        **overrides,
    }
    command = [str(ENGINE / ("gradlew.bat" if os.name == "nt" else "gradlew")), "test",
               "--tests", "ai.games.results.SearchSweepTest", "--rerun-tasks", "--console=plain",
               *(f"-D{key}={value}" for key, value in properties.items())]
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ENGINE, text=True).strip()
    manifest = dict(runId=run_id, startedUtc=datetime.now(timezone.utc).isoformat(), revision=revision,
                    platform=platform.platform(), command=command, properties=properties, status="running")
    # Retain actual sources, including uncommitted code, so revision alone is not misleading.
    snapshot = output / "source"
    shutil.copytree(ENGINE / "src", snapshot / "src")
    shutil.copytree(ENGINE / "scripts", snapshot / "scripts", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(ENGINE / "gradle", snapshot / "gradle")
    for name in ("build.gradle", "settings.gradle", "gradlew", "gradlew.bat"):
        shutil.copy2(ENGINE / name, snapshot / name)
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    environment = os.environ.copy()
    environment["SOLITAIRE_LOG_DIR"] = str(output / "logs")
    print(f"Run directory: {output}", flush=True)
    try:
        with (output / "console.log").open("w") as log:
            result = subprocess.run(command, cwd=ENGINE, env=environment, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f"Gradle failed ({result.returncode}); see {output / 'console.log'}")
        summary = summarize(output, args.players, args.games)
        manifest["status"] = "complete"
        print(json.dumps(summary, indent=2))
    except BaseException as error:
        manifest["status"] = "failed"
        manifest["error"] = str(error)
        raise
    finally:
        manifest["finishedUtc"] = datetime.now(timezone.utc).isoformat()
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()

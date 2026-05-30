"""Generate cumulative MLP frontier experiment specs.

This produces one spec per combination of:

- training frontier: Levels 2 through N, where N ranges from 2 to 10
- MLP size ladder: (128,1), (256,2), (512,3), (1024,3)

Each generated spec trains once on archived logs through the chosen frontier and then
evaluates that checkpoint across a fixed ladder that covers both easy levels and the
historical cliff region seen in the deleted Level 6-10 findings.
"""

from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
ARCHIVE_ROOT = REPO_ROOT / "engine" / "logs" / "archive"
OUTPUT_DIR = REPO_ROOT / "experiments" / "specs"

# Keep dense coverage on easier opponents, then probe the historical cliff region
# where prior archived-model runs broke near Levels 19, 30-31, and 42.
EVALUATION_LEVELS = list(range(2, 21)) + [22, 24, 26, 28, 29, 30, 31, 35, 40, 41, 42, 45, 50]
FRONTIER_LEVELS = list(range(2, 11))
# Use a single shard per level. Deterministic seeded evaluation can yield fewer
# actual games than requested, which makes later shards empty and causes the run
# to fail before the ladder completes.
EVALUATION_REQUESTED_GAMES = 200
MLP_CONFIGS = (
    {"slug": "hd128_nl1", "hidden_dim": 128, "num_layers": 1},
    {"slug": "hd256_nl2", "hidden_dim": 256, "num_layers": 2},
    {"slug": "hd512_nl3", "hidden_dim": 512, "num_layers": 3},
    {"slug": "hd1024_nl3", "hidden_dim": 1024, "num_layers": 3},
)


def _relative_to_repo(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT).as_posix()


def _latest_archived_episode_log(level: int) -> Path:
    level_dir = ARCHIVE_ROOT / f"level{level}"
    candidates = sorted(level_dir.glob("*/episode.log"))
    if not candidates:
        raise FileNotFoundError(f"could not find archived episode log for level {level} under {level_dir}")
    return max(candidates, key=lambda path: (path.parent.name, path.stat().st_mtime_ns))


def _build_sources(frontier_level: int) -> list[str]:
    return [_relative_to_repo(_latest_archived_episode_log(level)) for level in range(2, frontier_level + 1)]


def _build_service_port(config_index: int, frontier_level: int) -> int:
    return 8100 + (config_index * 20) + (frontier_level - 2)


def _build_spec(frontier_level: int, config_index: int, config: dict[str, int | str]) -> dict[str, object]:
    slug = str(config["slug"])
    hidden_dim = int(config["hidden_dim"])
    num_layers = int(config["num_layers"])
    frontier_tag = f"upto{frontier_level:02d}"
    experiment_id = f"mlp-frontier-{frontier_tag}-{slug.replace('_', '-')}-dev"
    checkpoint_prefix = f"policy_value_{experiment_id.replace('-', '_')}"

    return {
        "api_version": "v1",
        "experiment_id": experiment_id,
        "description": (
            f"Development cumulative MLP frontier sweep run with hidden_dim={hidden_dim} "
            f"and num_layers={num_layers}, trained on archived Levels 2 through {frontier_level} "
            f"and evaluated across a fixed ladder spanning Levels 2 through 50."
        ),
        "architecture": {
            "family": "mlp",
            "params": {
                "hidden_dim": hidden_dim,
                "num_layers": num_layers,
                "batch_norm": False,
                "residual": False,
            },
        },
        "dataset": {
            "kind": "archived_episode_logs",
            "sources": _build_sources(frontier_level),
        },
        "training": {
            "kind": "mlp_policy_value",
            "epochs": 5,
            "batch_size": 64,
            "learning_rate": 0.001,
            "checkpoint_every_epochs": 1,
            "checkpoint_prefix": checkpoint_prefix,
            "metrics_filename": "epoch_metrics.jsonl",
            "resume_from": None,
        },
        "evaluation": {
            "kind": "alpha_level_suite",
            "levels": EVALUATION_LEVELS,
            "games_per_shard": EVALUATION_REQUESTED_GAMES,
            "seed_start": 0,
            "seed_end": EVALUATION_REQUESTED_GAMES,
            "service_host": "127.0.0.1",
            "service_port": _build_service_port(config_index, frontier_level),
            "mcts_simulations": 64,
            "mcts_max_depth": 16,
            "mcts_cpuct": 1.5,
        },
    }


def _write_spec(path: Path, payload: dict[str, object]) -> bool:
    rendered = json.dumps(payload, indent=2)
    if path.exists() and path.read_text(encoding="utf-8") == rendered + "\n":
        return False
    path.write_text(rendered + "\n", encoding="utf-8")
    return True


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    written = 0
    total = 0

    for config_index, config in enumerate(MLP_CONFIGS):
        for frontier_level in FRONTIER_LEVELS:
            payload = _build_spec(frontier_level, config_index, config)
            file_name = f"mlp_frontier_upto{frontier_level:02d}_{config['slug']}_dev.json"
            output_path = OUTPUT_DIR / file_name
            if _write_spec(output_path, payload):
                written += 1
            total += 1

    print(f"Generated {total} MLP frontier specs ({written} written or updated).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
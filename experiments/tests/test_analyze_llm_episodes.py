from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from experiments.tools.analyze_llm_episodes import analyze_records, load_records


def batch_records(
    completed_games: int = 10,
    *,
    wins: int = 0,
    maximum_foundation: int = 0,
    cards_revealed: int = 0,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for game_index in range(1, completed_games + 1):
        records.append(
            {
                "type": "game",
                "run_id": "test-run",
                "game_index": game_index,
                "game_total": 10,
                "solver": "OllamaPlayer",
                "metadata": {"provider": "Ollama", "model": "test-model", "prompt_version": "P0"},
            }
        )
        records.append(
            {
                "type": "step",
                "run_id": "test-run",
                "game_index": game_index,
                "game_total": 10,
                "solver": "OllamaPlayer",
                "step_index": 1,
                "state_key": game_index,
                "chosen_command": "turn",
                "legal_moves": ["turn", "quit"],
                "foundation": [[str(index) for index in range(maximum_foundation)], [], [], []],
                "tableau_face_down": [28 - cards_revealed, 0, 0, 0, 0, 0, 0],
            }
        )
        records.append(
            {
                "type": "summary",
                "run_id": "test-run",
                "game_index": game_index,
                "game_total": 10,
                "solver": "OllamaPlayer",
                "iterations": 1,
                "successful_moves": 1,
                "won": game_index <= wins,
                "duration_nanos": 1_000_000_000,
                "termination_reason": "won" if game_index <= wins else "quit",
                "final_foundation_cards": maximum_foundation,
                "final_tableau_face_down": 28 - cards_revealed,
            }
        )
    return records


class AnalyzeLlmEpisodesTest(unittest.TestCase):
    def test_suggests_each_outcome_from_documented_thresholds(self) -> None:
        technical = analyze_records(batch_records(completed_games=7))
        not_viable = analyze_records(batch_records())
        needs_prompting = analyze_records(batch_records(maximum_foundation=10, cards_revealed=8))
        promising = analyze_records(batch_records(wins=1))

        self.assertEqual("technical_failure", technical["suggested_rating"])
        self.assertEqual("not_viable", not_viable["suggested_rating"])
        self.assertEqual("needs_prompting", needs_prompting["suggested_rating"])
        self.assertEqual("promising", promising["suggested_rating"])

    def test_loads_plain_and_prefixed_episode_lines(self) -> None:
        records = batch_records(completed_games=1)
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "episode.log"
            path.write_text(
                "ignored\n"
                + "EPISODE_GAME " + json.dumps(records[0]) + "\n"
                + "2026-09-25 INFO logger - EPISODE_STEP " + json.dumps(records[1]) + "\n"
                + "EPISODE_SUMMARY " + json.dumps(records[2]) + "\n",
                encoding="utf-8",
            )

            loaded = load_records([path])

        self.assertEqual(records, loaded)


if __name__ == "__main__":
    unittest.main()

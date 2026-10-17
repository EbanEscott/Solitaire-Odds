"""Verify sweep aggregation without launching Gradle or playing games."""

import csv
import importlib.util
from pathlib import Path
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "search_sweep.py"
spec = importlib.util.spec_from_file_location("search_sweep", SCRIPT)
sweep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sweep)


class SearchSweepSummaryTest(unittest.TestCase):
    def write_rows(self, output, rows):
        with (output / "games.csv").open("w") as target:
            writer = csv.DictWriter(target, fieldnames=("player", "game", "won", "seconds", "moves", "termination"))
            writer.writeheader()
            writer.writerows(rows)

    def test_aggregates_in_deal_order_and_keeps_caps_separate(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            self.write_rows(output, [
                dict(player="greedy", game=3, won="false", seconds=4, moves=1000, termination="move_cap"),
                dict(player="greedy", game=1, won="true", seconds=1, moves=100, termination="won"),
                dict(player="greedy", game=2, won="true", seconds=1, moves=100, termination="won"),
            ])
            result = sweep.summarize(output, ["greedy"], 3)["greedy"]
            self.assertEqual(2, result["wins"])
            self.assertEqual(2, result["bestWinStreak"])
            self.assertEqual(2, result["averageSeconds"])
            self.assertEqual(400, result["averageMoves"])
            self.assertEqual({"won": 2, "move_cap": 1}, result["terminations"])
            self.assertTrue((output / "summary.json").exists())

    def test_rejects_missing_or_duplicate_games(self):
        for indices in ([1], [1, 1]):
            with self.subTest(indices=indices), tempfile.TemporaryDirectory() as directory:
                output = Path(directory)
                self.write_rows(output, [dict(player="greedy", game=index, won="false", seconds=1,
                                             moves=1, termination="quit") for index in indices])
                with self.assertRaises(ValueError):
                    sweep.summarize(output, ["greedy"], 2)
                self.assertFalse((output / "summary.json").exists())


if __name__ == "__main__":
    unittest.main()
